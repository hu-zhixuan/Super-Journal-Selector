import os
import logging
import gradio as gr
from dotenv import load_dotenv

# 引入我们写好的四层流水线模块
from fetch_papers import OpenAlexFetcher
from extract_features import FeatureExtractor
from aggregate import ProfileAggregator
from generate_profile import ProfileGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()


def run_pipeline(journal_name: str, years: int, max_papers: int, user_draft: str):
    """
    网页端调用的生成函数。使用 yield 机制，将运行进度实时吐给前端网页显示。
    """
    journal_name = journal_name.strip()
    if not journal_name:
        yield "❌ 错误：请输入有效的期刊名称！", ""
        return

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
            yield f"❌ 错误：未在 OpenAlex 中检索到期刊 '{journal_name}' 或近几年无发文摘要。", ""
            return

        # Layer ②: LLM 结构化特征提取
        total_papers = len(papers)
        yield f"⏳ [2/4] 成功抓取 {total_papers} 篇有效论文。正在拉起大模型并发提取学术特征...", ""
        
        # 实例化特征提取层
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
        
        # 如果用户没有填摘要草稿，传入 None
        draft_text = user_draft.strip() if user_draft and user_draft.strip() else None
        
        report_markdown = generator.generate_report(
            journal_name=journal_name,
            aggregated_stats=aggregated_stats,
            user_draft_text=draft_text
        )

        # 保存一份到本地 output 目录下
        os.makedirs("output", exist_ok=True)
        safe_journal_filename = "".join(c if c.isalnum() else "_" for c in journal_name)
        output_path = os.path.join("output", f"{safe_journal_filename}_WebUI_Report.md")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_markdown)

        success_msg = f"🎉 画像报告生成成功！已保存至本地: {output_path}"
        yield success_msg, report_markdown

    except Exception as e:
        logger.error(f"网页端运行流水线发生异常: {str(e)}")
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

with gr.Blocks(theme=theme, title="期刊选稿画像助手 - WebUI") as demo:
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
            journal_input = gr.Textbox(
                label="目标期刊名称 (英文全称)",
                placeholder="例如: Computers in Human Behavior 或 Strategic Management Journal",
                value="Computers in Human Behavior"
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
                
            draft_input = gr.Textbox(
                label="你的论文摘要/草稿 (可选，用于定制化改写对标)",
                placeholder="在此粘贴拟投稿论文的 Title/Abstract/大纲，系统将给出字面级的手术重构方案...",
                lines=8
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

    # 按钮点击事件绑定
    submit_btn.click(
        fn=run_pipeline,
        inputs=[journal_input, years_input, max_papers_input, draft_input],
        outputs=[status_output, report_output]
    )

if __name__ == "__main__":
    # 启动本地服务，自动打开浏览器
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
