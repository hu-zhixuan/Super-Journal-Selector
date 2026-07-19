import os
import logging
import requests
import urllib3.util.connection as urllib3_cn
import gradio as gr
from dotenv import load_dotenv

# 引入文档解析库
import docx
import pypdf

# 导入流水线模块
from fetch_papers import OpenAlexFetcher
from extract_features import FeatureExtractor
from aggregate import ProfileAggregator
from generate_profile import ProfileGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

# ==================== Socket 层 DNS 劫持补丁 ====================
def patched_create_connection(address, *args, **kwargs):
    host, port = address
    if host == "fxb.supa.net.cn":
        return urllib3_cn._orig_create_connection(("114.80.15.146", port), *args, **kwargs)
    return urllib3_cn._orig_create_connection(address, *args, **kwargs)

if not hasattr(urllib3_cn, "_orig_create_connection"):
    urllib3_cn._orig_create_connection = urllib3_cn.create_connection
    urllib3_cn.create_connection = patched_create_connection
    logger.info("已成功加载 Socket DNS 直连补丁：fxb.supa.net.cn -> 114.80.15.146")
# ===============================================================


def parse_docx(file_path: str) -> str:
    """
    解析 Word 文档，只读取前 50 个非空段落（包含 Title/Abstract/Intro，避免 Token 浪费）
    """
    try:
        doc = docx.Document(file_path)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        # 截取前 50 段
        limited_text = "\n".join(paragraphs[:50])
        logger.info(f"成功解析 Word，截取前 50 段，共 {len(limited_text)} 字。")
        return limited_text
    except Exception as e:
        logger.error(f"解析 Word 失败: {e}")
        return f"[Word 解析失败]: {str(e)}"


def parse_pdf(file_path: str) -> str:
    """
    解析 PDF 文档，只读取前 4 页（包含 Title/Abstract/Intro，避免 Token 浪费）
    """
    try:
        reader = pypdf.PdfReader(file_path)
        pages_to_read = min(len(reader.pages), 4)  # 只读前 4 页
        text_list = []
        for i in range(pages_to_read):
            page_text = reader.pages[i].extract_text()
            if page_text:
                text_list.append(page_text)
        limited_text = "\n".join(text_list)
        logger.info(f"成功解析 PDF，读取前 {pages_to_read} 页，共 {len(limited_text)} 字。")
        return limited_text
    except Exception as e:
        logger.error(f"解析 PDF 失败: {e}")
        return f"[PDF 解析失败]: {str(e)}"


def search_journals(query: str):
    """
    调用 OpenAlex 的 Autocomplete API 联想检索期刊名
    """
    if not query or len(query.strip()) < 3:
        return gr.Dropdown(choices=[])
    
    url = "https://api.openalex.org/autocomplete/sources"
    params = {"q": query}
    try:
        resp = requests.get(url, params=params, proxies={"http": None, "https": None}, timeout=5)
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            choices = [item.get("display_name") for item in results if item.get("display_name")]
            choices = list(dict.fromkeys(choices))  # 去重
            if choices:
                return gr.Dropdown(choices=choices, value=choices[0])
    except Exception as e:
        logger.warning(f"期刊联想搜索失败: {e}")
    
    return gr.Dropdown(choices=[])


def run_pipeline(journal_name: str, years: int, max_papers: int, user_draft: str, file_obj):
    """
    网页端调用的生成函数。支持文件解析优先逻辑。
    """
    journal_name = journal_name.strip() if journal_name else ""
    if not journal_name:
        yield "❌ 错误：请先在上方输入期刊关键词并选择一个目标期刊！", ""
        return

    # 优先解析上传的文档文件
    final_draft_text = ""
    if file_obj is not None:
        file_path = file_obj.name
        ext = os.path.splitext(file_path)[1].lower()
        yield f"⏳ 正在解析上传的 {ext} 文档（仅读取前4页/段落以防浪费 Token）...", ""
        
        if ext == ".docx":
            final_draft_text = parse_docx(file_path)
        elif ext == ".pdf":
            final_draft_text = parse_pdf(file_path)
        else:
            yield f"❌ 错误：不支持的文档格式 {ext}，仅支持 .docx 和 .pdf 格式！", ""
            return
    else:
        final_draft_text = user_draft.strip() if user_draft else ""

    try:
        # Layer ①: 抓取数据
        yield "⏳ [1/4] 正在连接 OpenAlex 检索期刊 ID 并拉取近年发文摘要...", ""
        fetcher = OpenAlexFetcher()
        papers = fetcher.fetch_recent_papers(
            journal_name=journal_name,
            years=int(years),
            max_papers=int(max_papers),
        )
        if not papers:
            yield f"❌ 错误：未在 OpenAlex 中检索到期刊 '{journal_name}' 或近几年该刊无有效发文。", ""
            return

        # Layer ②: LLM 结构化特征提取
        total_papers = len(papers)
        yield f"⏳ [2/4] 成功抓取 {total_papers} 篇有效论文。正在拉起大模型并发提取学术特征...", ""
        
        extractor = FeatureExtractor()
        extracted_features = []
        
        for idx, paper in enumerate(papers):
            progress_msg = f"⏳ [2/4] 正在使用大模型解析文献特征 (进度: {idx + 1}/{total_papers})...\n文献: 《{paper.title[:45]}...》"
            yield progress_msg, ""
            
            feat = extractor.extract_paper(paper)
            if feat:
                feat_dict = feat.model_dump()
                feat_dict["title"] = paper.title
                feat_dict["cited_by_count"] = paper.cited_by_count
                feat_dict["publication_year"] = paper.publication_year
                extracted_features.append(feat_dict)
                
        if not extracted_features:
            yield "❌ 错误：大模型未成功从摘要中抽取出任何结构化特征！请检查接口连接。", ""
            return

        # Layer ③: 纯代码统计聚合
        yield "⏳ [3/4] 特征抽取完成！正在启动 Python 统计引擎计算范式分布与中位数...", ""
        aggregator = ProfileAggregator()
        aggregated_stats = aggregator.aggregate(extracted_features)

        # Layer ④: LLM 生成偏好画像与策略报告
        yield "⏳ [4/4] 统计聚合完毕。正在调用大模型撰写深度学术画像与对标修改策略书...", ""
        generator = ProfileGenerator()
        
        draft_text = final_draft_text.strip() if final_draft_text and final_draft_text.strip() else None
        
        report_markdown = generator.generate_report(
            journal_name=journal_name,
            aggregated_stats=aggregated_stats,
            user_draft_text=draft_text
        )

        os.makedirs("output", exist_ok=True)
        safe_journal_filename = "".join(c if c.isalnum() else "_" for c in journal_name)
        output_path = os.path.join("output", f"{safe_journal_filename}_WebUI_Report.md")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_markdown)

        success_msg = f"🎉 画像报告生成成功！已保存至本地: {output_path}"
        yield success_msg, report_markdown

    except Exception as e:
        logger.error(f"网页端运行异常: {str(e)}")
        yield f"❌ 运行失败，错误原因: {str(e)}", ""


# ==================== GRADIO 网页界面布局 ====================
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
).set(
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
)

with gr.Blocks(title="期刊选稿画像助手 - WebUI") as demo:
    gr.Markdown(
        """
        # 📊 期刊选稿画像助手 (Journal Profile Assistant)
        **打通成果撰写与投稿优化全链路的“学术品味诊断与策略改造器”**
        """
    )
    
    with gr.Row():
        # 左侧输入控制区
        with gr.Column(scale=2):
            gr.Markdown("### ⚙️ 投稿对标参数配置")
            
            search_input = gr.Textbox(
                label="🔍 输入期刊关键词进行联想（输入3个字母以上自动检索）",
                placeholder="例如: computers 或 strategic",
                value=""
            )
            
            journal_input = gr.Dropdown(
                label="🎯 选择目标期刊全称 (从下方匹配的候选列表中选择)",
                choices=["Computers in Human Behavior"],
                value="Computers in Human Behavior",
                allow_custom_value=True,
                interactive=True
            )
            
            with gr.Row():
                years_input = gr.Slider(
                    minimum=1, maximum=5, value=3, step=1,
                    label="数据回溯年份"
                )
                max_papers_input = gr.Slider(
                    minimum=5, maximum=40, value=15, step=5,
                    label="最大采样文献篇数"
                )
                
            # 输入方式卡片：提供粘贴文本和文件上传两种选择
            with gr.Tab("📝 选项 A：手动粘贴摘要/草稿"):
                draft_input = gr.Textbox(
                    label="粘贴拟投稿论文的 Title/Abstract/大纲",
                    placeholder="在此粘贴，系统将给出字面级的手术重构方案...",
                    lines=8
                )
                
            with gr.Tab("📁 选项 B：上传草稿文件 (解析前4页)"):
                file_input = gr.File(
                    label="选择你的 Word (.docx) 或 PDF (.pdf) 文件",
                    file_types=[".docx", ".pdf"]
                )
            
            submit_btn = gr.Button("🚀 一键生成期刊选稿画像", variant="primary")
            
        # 右侧报告输出区
        with gr.Column(scale=3):
            gr.Markdown("### 📝 期刊选稿画像与手术修稿报告")
            status_output = gr.Textbox(
                label="系统运行状态",
                value="就绪。等待输入并点击生成...",
                interactive=False
            )
            report_output = gr.Markdown(
                value="*报告生成后将在此处以精美 Markdown 格式自动渲染展示。*"
            )

    # 事件流绑定：输入关键词时，实时触发下拉框备选项更新
    search_input.input(
        fn=search_journals,
        inputs=search_input,
        outputs=journal_input
    )

    # 按钮点击事件绑定 (将 file_input 接入输入列表)
    submit_btn.click(
        fn=run_pipeline,
        inputs=[journal_input, years_input, max_papers_input, draft_input, file_input],
        outputs=[status_output, report_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, theme=theme)
