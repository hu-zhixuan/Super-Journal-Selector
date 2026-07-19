import os
import logging
import requests
import urllib3.util.connection as urllib3_cn
import gradio as gr
from dotenv import load_dotenv
from datetime import datetime

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
    解析 Word 文档，读取完整内容进行高精度对标与诊断
    """
    try:
        doc = docx.Document(file_path)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        full_text = "\n".join(paragraphs)
        logger.info(f"成功解析 Word，读取完整文档，共 {len(full_text)} 字。")
        return full_text
    except Exception as e:
        logger.error(f"解析 Word 失败: {e}")
        return f"[Word 解析失败]: {str(e)}"


def parse_pdf(file_path: str) -> str:
    """
    解析 PDF 文档，读取完整内容进行高精度对标与诊断
    """
    try:
        reader = pypdf.PdfReader(file_path)
        text_list = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_list.append(page_text)
        full_text = "\n".join(text_list)
        logger.info(f"成功解析 PDF，读取完整 {len(reader.pages)} 页，共 {len(full_text)} 字。")
        return full_text
    except Exception as e:
        logger.error(f"解析 PDF 失败: {e}")
        return f"[PDF 解析失败]: {str(e)}"


# 学术圈高频缩写与中文俗称/别名智能映射字典
JOURNAL_ALIASES = {
    "pra": "Physical Review A",
    "prl": "Physical Review Letters",
    "prb": "Physical Review B",
    "prd": "Physical Review D",
    "prx": "Physical Review X",
    "jacs": "Journal of the American Chemical Society",
    "tpami": "IEEE Transactions on Pattern Analysis and Machine Intelligence",
    "pami": "IEEE Transactions on Pattern Analysis and Machine Intelligence",
    "cvpr": "IEEE/CVF Conference on Computer Vision and Pattern Recognition",
    "chb": "Computers in Human Behavior",
    "nature": "Nature",
    "自然": "Nature",
    "science": "Science",
    "科学": "Science",
    "cell": "Cell",
    "细胞": "Cell",
    "lancet": "The Lancet",
    "柳叶刀": "The Lancet",
    "nejm": "New England Journal of Medicine",
    "physica a": "Physica A: Statistical Mechanics and its Applications",
    "amj": "Academy of Management Journal",
    "amr": "Academy of Management Review",
    "misq": "MIS Quarterly",
    "isr": "Information Systems Research",
}


def search_journals(query: str):
    """
    智能期刊联想与索引引擎：
    1. 支持英文缩写/中文别名智能映射；
    2. 调用 OpenAlex 向量检索并统计近3年活跃发文量，剔除历史停更死链；
    3. 生成带被引与发文标签的 [(Label, Value)] 结构，直观又准确。
    """
    q_clean = query.strip().lower() if query else ""
    if not q_clean or len(q_clean) < 2:
        return gr.Dropdown(choices=[], value=None)

    search_terms = []
    # 1. 如果命中缩写或别名，优先将其作为核心检索词
    if q_clean in JOURNAL_ALIASES:
        search_terms.append(JOURNAL_ALIASES[q_clean])
    search_terms.append(query.strip())

    candidates = []
    seen_ids = set()
    current_year = datetime.now().year

    for term in search_terms:
        url = "https://api.openalex.org/sources"
        try:
            resp = requests.get(
                url, 
                params={"search": term, "per-page": 6}, 
                proxies={"http": None, "https": None}, 
                timeout=5
            )
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                for item in results:
                    sid = item.get("id")
                    if sid in seen_ids:
                        continue
                    seen_ids.add(sid)

                    display_name = item.get("display_name", "Unknown")
                    counts_by_year = item.get("counts_by_year", [])
                    recent_works = sum(
                        c.get("works_count", 0) 
                        for c in counts_by_year 
                        if c.get("year", 0) >= (current_year - 2)
                    )
                    
                    summary_stats = item.get("summary_stats", {})
                    if_est = summary_stats.get("2yr_mean_citedness", "N/A")
                    if isinstance(if_est, float):
                        if_est = f"{if_est:.1f}"

                    # 过滤掉完全停更且在多结果干扰项中的历史条目
                    if recent_works == 0 and len(results) > 2:
                        continue

                    # 构造富信息展示标签
                    label = f"✨ {display_name} (近3年发文: {recent_works}篇 | 估算IF: {if_est})"
                    # 如果正好是缩写别名对标上的正牌，赋予最高加权
                    score = recent_works * (3 if q_clean in JOURNAL_ALIASES and display_name.lower() == JOURNAL_ALIASES[q_clean].lower() else 1)
                    candidates.append((score, label, display_name))
        except Exception as e:
            logger.warning(f"智能联想检索异常: {e}")

    # 按活跃权重降序排列，确保活的、顶级的、匹配准的期刊排在第一项
    candidates.sort(key=lambda x: x[0], reverse=True)
    dropdown_choices = [(item[1], item[2]) for item in candidates[:6]]
    
    if dropdown_choices:
        return gr.Dropdown(choices=dropdown_choices, value=dropdown_choices[0][1])
    return gr.Dropdown(choices=[], value=None)


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
        yield f"⏳ 正在解析上传的 {ext} 完整学术文档，即将开始高精度对标与诊断...", ""
        
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
        papers, journal_metadata = fetcher.fetch_recent_papers(
            journal_name=journal_name,
            years=int(years),
            max_papers=int(max_papers),
        )
        if not papers:
            yield f"❌ 错误：未在 OpenAlex 中检索到期刊 '{journal_name}' 或近几年该刊无有效发文。", ""
            return

        # Layer ②: LLM 结构化特征提取
        total_papers = len(papers)
        yield f"⏳ [2/4] 成功建立 {total_papers} 篇大样本有效论文池。正在开启多线程池并发高速提取特征 (Workers=10)...", ""
        
        extractor = FeatureExtractor()
        extracted_features = extractor.extract_batch(papers, max_workers=10)
                
        if not extracted_features:
            yield "❌ 错误：大模型未成功从摘要中抽取出任何结构化特征！请检查接口连接。", ""
            return

        draft_text = final_draft_text.strip() if final_draft_text and final_draft_text.strip() else None

        # Layer ③: 纯代码统计聚合
        yield "⏳ [3/4] 特征抽取完成！正在启动 Python 统计引擎计算范式分布并执行文献相似度诊断...", ""
        aggregator = ProfileAggregator()
        aggregated_stats = aggregator.aggregate(extracted_features, user_draft_text=draft_text)

        # Layer ④: LLM 生成偏好画像与策略报告
        yield "⏳ [4/4] 统计聚合完毕。正在调用大模型撰写深度学术画像与对标修改策略书...", ""
        generator = ProfileGenerator()
        
        report_markdown = generator.generate_report(
            journal_name=journal_name,
            aggregated_stats=aggregated_stats,
            journal_metadata=journal_metadata,
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
                    minimum=20, maximum=200, value=100, step=10,
                    label="大样本并发采样文献数 (100+高特异性指向)"
                )
                
            # 输入方式卡片：提供粘贴文本和文件上传两种选择
            with gr.Tab("📝 选项 A：手动粘贴摘要/草稿"):
                draft_input = gr.Textbox(
                    label="粘贴拟投稿论文的 Title/Abstract/大纲",
                    placeholder="在此粘贴，系统将给出字面级的手术重构方案...",
                    lines=8
                )
                
            with gr.Tab("📁 选项 B：上传草稿文件 (支持整篇全量对标)"):
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
