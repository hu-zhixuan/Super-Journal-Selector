import logging
import os
from datetime import datetime
from typing import Any, Dict

# 引入文档解析库
import docx
import gradio as gr
import pypdf
import requests
from dotenv import load_dotenv

import network_config  # noqa: F401  # 必须最先导入：配置 HF 镜像，避免模型下载卡死
from aggregate import ProfileAggregator
from extract_features import FeatureExtractor

# 导入流水线模块
from fetch_papers import OpenAlexFetcher
from generate_profile import ProfileGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

# ==================== Socket 层 DNS 劫持补丁 ====================
from llm_client import install_dns_patch  # noqa: E402  # 须在 network_config 初始化后调用

install_dns_patch()
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
    解析 PDF 文档，读取完整内容进行高精度对标与诊断。
    优先使用 PyMuPDF (fitz) 以获得更精准的学术排版与公式格式保留，若未安装则降级使用 pypdf。
    """
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(file_path)
        text_list = []
        for page in doc:
            page_text = page.get_text("layout")
            if page_text:
                text_list.append(page_text)
        full_text = "\n".join(text_list)
        logger.info(f"成功使用 PyMuPDF 解析 PDF，共 {len(doc)} 页，共 {len(full_text)} 字。")
        return full_text
    except ImportError:
        logger.info("未检测到 PyMuPDF (fitz)，将降级使用 pypdf 进行解析。")
        try:
            reader = pypdf.PdfReader(file_path)
            text_list = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_list.append(page_text)
            full_text = "\n".join(text_list)
            logger.info(f"使用 pypdf 解析 PDF 成功，共 {len(reader.pages)} 页，共 {len(full_text)} 字。")
            return full_text
        except Exception as e_pypdf:
            logger.error(f"pypdf 解析 PDF 失败: {e_pypdf}")
            return f"[PDF 解析失败]: {str(e_pypdf)}"
    except Exception as e:
        logger.error(f"PyMuPDF 解析 PDF 失败: {e}")
        return f"[PDF 解析失败]: {str(e)}"


def parse_draft_input(draft_text: str, file_obj: Any) -> str:
    """
    统一解析用户输入的草稿文本或上传的文件（支持 .docx 与 .pdf）
    """
    if file_obj is not None:
        file_path = getattr(file_obj, "name", str(file_obj))
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".docx":
            return parse_docx(file_path)
        elif ext == ".pdf":
            return parse_pdf(file_path)
    return draft_text.strip() if draft_text else ""


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
            params: Dict[str, Any] = {"search": term, "per-page": 6}
            resp = requests.get(
                url,
                params=params,
                proxies={"http": None, "https": None},  # type: ignore[dict-item]
                timeout=5,
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
                        c.get("works_count", 0) for c in counts_by_year if c.get("year", 0) >= (current_year - 2)
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
                    score = recent_works * (
                        3
                        if q_clean in JOURNAL_ALIASES and display_name.lower() == JOURNAL_ALIASES[q_clean].lower()
                        else 1
                    )
                    candidates.append((score, label, display_name))
        except Exception as e:
            logger.warning(f"智能联想检索异常: {e}")

    # 按活跃权重降序排列，确保活的、顶级的、匹配准的期刊排在第一项
    candidates.sort(key=lambda x: x[0], reverse=True)
    dropdown_choices = [(item[1], item[2]) for item in candidates[:6]]

    if dropdown_choices:
        return gr.Dropdown(choices=dropdown_choices, value=dropdown_choices[0][1])
    return gr.Dropdown(choices=[], value=None)


def run_pipeline(journal_name: str, years: int, max_papers: int, user_draft: str, file_obj, progress=gr.Progress()):
    """
    网页端调用的生成函数。支持文件解析优先逻辑与实时进度条推流。
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

    if final_draft_text:
        from aggregate import clean_and_truncate_draft

        final_draft_text = clean_and_truncate_draft(final_draft_text)

    # 提取草稿检索关键词，触发双通道主题匹配检索（与 CLI/SDK 入口对齐）
    search_query = None
    if final_draft_text:
        try:
            progress(0.02, desc="正在提取草稿检索关键词以开启双通道主题匹配...")
            yield "⏳ [1/4] 正在从草稿中提取学术检索关键词，开启双通道动态对标检索...", ""
            from llm_client import LLMClient
            from main import extract_search_keywords

            keywords = extract_search_keywords(LLMClient(), final_draft_text)
            if keywords:
                search_query = " ".join(keywords)
        except Exception as e_kw:
            logger.warning(f"草稿关键词提取失败，将降级为纯高引热门检索: {e_kw}")

    try:
        # Layer ①: 抓取数据
        progress(0.05, desc="正在连接 OpenAlex 检索期刊 ID 并拉取近年发文摘要...")
        yield "⏳ [1/4] 正在连接 OpenAlex 检索期刊 ID 并拉取近年发文摘要...", ""
        fetcher = OpenAlexFetcher()
        papers, journal_metadata = fetcher.fetch_recent_papers(
            journal_name=journal_name,
            years=int(years),
            max_papers=int(max_papers),
            search_query=search_query,
        )
        if not papers:
            yield f"❌ 错误：未在 OpenAlex 中检索到期刊 '{journal_name}' 或近几年该刊无有效发文。", ""
            return

        # Layer ②: LLM 结构化特征提取与实时进度条
        total_papers = len(papers)
        workers = FeatureExtractor._get_max_workers()
        progress(0.20, desc=f"[2/4] 正在拉起 {workers} 线程并发抽取 0/{total_papers}...")
        yield (
            f"⏳ [2/4] 成功建立 {total_papers} 篇大样本有效论文池。已开启多线程并发网络抽取 (Workers={workers})...",
            "",
        )

        extractor = FeatureExtractor()
        extracted_features = []

        for completed, total, p_item, current_results in extractor.extract_batch_iter(papers):
            extracted_features = current_results
            sub_pct = completed / total
            overall_pct = 0.20 + 0.55 * sub_pct
            title_preview = (p_item.title[:25] + "...") if len(p_item.title) > 25 else p_item.title

            # 保持顶部原生进度条描述长度固定，防止页面高度跳动抖动
            progress(overall_pct, desc=f"[2/4] 抽取特征 ({completed}/{total}) - {int(sub_pct * 100)}%")

            # 动态渲染控制台文本 ASCII 进度条
            filled_len = int(completed * 20 // total)
            bar_str = "█" * filled_len + "░" * (20 - filled_len)
            yield (
                f"⏳ [2/4] 正在并发抽取特征 ({completed}/{total} 篇 - {int(sub_pct * 100)}%)\n"
                f"进度: [{bar_str}]\n"
                f"最新完成: 《{title_preview}》",
                "",
            )

        if not extracted_features:
            yield "❌ 错误：大模型未成功从摘要中抽取出任何结构化特征！请检查接口连接。", ""
            return

        draft_text = final_draft_text.strip() if final_draft_text and final_draft_text.strip() else None

        # Layer ③: 纯代码统计聚合
        progress(0.80, desc="[3/4] 特征抽取完成！正在启动统计引擎计算余弦相似度...")
        yield "⏳ [3/4] 特征抽取完成！正在启动 Python 统计引擎计算范式分布并执行文献相似度诊断...", ""
        aggregator = ProfileAggregator()
        aggregated_stats = aggregator.aggregate(extracted_features, user_draft_text=draft_text)

        # Layer ④: LLM 生成偏好画像与策略报告
        progress(0.90, desc="[4/4] 正在根据大样本事实底图撰写对标报告与发表概率预测...")
        yield "⏳ [4/4] 统计聚合完毕。正在调用大模型撰写深度学术画像与对标修改策略书...", ""
        generator = ProfileGenerator()

        report_markdown = generator.generate_report(
            journal_name=journal_name,
            aggregated_stats=aggregated_stats,
            journal_metadata=journal_metadata,
            user_draft_text=draft_text,
        )

        progress(1.0, desc="[4/4] 画像报告与发表概率预测生成成功！")

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


def chat_with_reviewer(message: str, history: list, report_text: str):
    """
    与模拟主编(AE)/审稿人在线对线辩论的后端逻辑。
    兼容 Gradio 新版 dict/ChatMessage 格式与旧版 tuple 格式。
    """
    # 智能探测当前 Gradio 历史的格式
    is_dict_format = True
    if history and isinstance(history[0], (list, tuple)):
        is_dict_format = False

    # 错误消息处理
    err_reply = "⚠️ 【系统提示】请先在“📊 选稿画像与循证诊断”标签页中成功生成一份期刊诊断报告，然后再在此处与模拟审稿人在线对线。"
    if not report_text or not report_text.strip() or "报告生成后将在此处" in report_text:
        if is_dict_format:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": err_reply})
        else:
            history.append((message, err_reply))
        return history, ""

    from llm_client import LLMClient

    llm = LLMClient()

    system_prompt = (
        "You are an Associate Editor and senior Peer Reviewer for the selected academic journal. "
        "You have just generated the diagnostic report (context provided below). "
        "The author (user) is chatting with you to defend their manuscript, clarify your points, or suggest revisions. "
        "Be professional, direct, academic, strict, but constructively critical. Avoid generic AI fluff. "
        "Always respond in professional Chinese. ground your arguments in the report context and published papers.\n\n"
        f"--- Diagnostic Report Context ---\n{report_text}\n--- End Context ---"
    )

    # 格式化对话历史
    conversation_history = ""
    if is_dict_format:
        for msg in history:
            role = "Author" if msg.get("role") == "user" else "AE/Reviewer"
            content = msg.get("content", "")
            conversation_history += f"{role}: {content}\n"
    else:
        for user_msg, bot_msg in history:
            conversation_history += f"Author: {user_msg}\nAE/Reviewer: {bot_msg}\n"

    prompt = f"""
Below is the history of your discussion with the author, followed by the author's new message. Respond directly, using your professional Associate Editor identity.

{conversation_history}
Author (New Message): {message}
AE/Reviewer:
"""
    try:
        reply = llm.call(prompt=prompt, system_prompt=system_prompt, temperature=0.3)
        if is_dict_format:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": reply})
        else:
            history.append((message, reply))
    except Exception as e:
        logger.error(f"模拟审稿人对话异常: {e}")
        fail_msg = f"❌ 审稿人开小差了，回复失败，原因为: {str(e)}"
        if is_dict_format:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": fail_msg})
        else:
            history.append((message, fail_msg))

    return history, ""


def run_journal_router(router_draft_text: str, router_file_obj: Any, main_draft_text: str, main_file_obj: Any):
    final_text = ""
    # 优先使用路由 Tab 自带的上传文件或粘贴文本
    if router_file_obj is not None:
        final_text = parse_draft_input(router_draft_text, router_file_obj)
    elif router_draft_text and router_draft_text.strip():
        final_text = router_draft_text.strip()
    # 若路由 Tab 为空，自动共享复用主大厅【选稿画像与循证诊断】中的文件或文本
    elif main_file_obj is not None:
        final_text = parse_draft_input(main_draft_text, main_file_obj)
    elif main_draft_text and main_draft_text.strip():
        final_text = main_draft_text.strip()

    if not final_text:
        return "❌ 错误：未检测到任何草稿内容！请在上方粘贴草稿/上传 Word 或 PDF 文件，或者在主大厅【📊 选稿画像与循证诊断】中上传草稿！"

    from journal_router import JournalRouter

    router = JournalRouter()
    res = router.route_journals(final_text)

    note = res.get("draft_summary_note", "论文摘要解析完成")
    tiers = res.get("recommended_tiers", [])

    md_lines = [
        "### 🧭 全网学术期刊投递梯队路由诊断大盘",
        f"> **稿件定位摘要**：{note}\n",
        "| 投递梯队 | 推荐期刊名称 | 综合契合度得分 | 预估基准录用率 | 决策路由理由 |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for item in tiers:
        tier = item.get("tier", "")
        name = item.get("journal_name", "")
        score = item.get("fit_score", 0)
        rate = item.get("estimated_acceptance_rate", "")
        reason = item.get("reason", "")
        md_lines.append(f"| **{tier}** | `{name}` | **{score} 分** | `{rate}` | {reason} |")

    return "\n".join(md_lines)


custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ===== Force Harmonized Modern Light Mode (Fix Dark Input Mismatch) ===== */
:root, body, .dark, [data-theme="dark"], .gradio-container, .gradio-container.dark {
    --body-background-fill: #f8fafc !important;
    --background-fill-primary: #f8fafc !important;
    --background-fill-secondary: #ffffff !important;
    --block-background-fill: #ffffff !important;
    --block-border-color: #e2e8f0 !important;
    --body-text-color: #1e293b !important;
    --body-text-color-subdued: #64748b !important;
    --color-accent: #4f46e5 !important;
    --color-accent-soft: #eef2ff !important;

    /* Input Variables Fix */
    --input-background-fill: #ffffff !important;
    --input-background-fill-focus: #ffffff !important;
    --input-border-color: #cbd5e1 !important;
    --input-border-color-focus: #6366f1 !important;
    --input-text-color: #0f172a !important;

    /* Label & Badge Variables Fix */
    --block-label-background-fill: #eef2ff !important;
    --block-label-text-color: #4338ca !important;
    --block-label-border-color: #c7d2fe !important;
    --block-title-text-color: #1e293b !important;

    /* Neutral Palette Force Light */
    --neutral-50: #f8fafc !important;
    --neutral-100: #f1f5f9 !important;
    --neutral-200: #e2e8f0 !important;
    --neutral-300: #cbd5e1 !important;
    --neutral-400: #94a3b8 !important;
    --neutral-500: #64748b !important;
    --neutral-600: #475569 !important;
    --neutral-700: #334155 !important;
    --neutral-800: #1e293b !important;
    --neutral-900: #0f172a !important;
    --neutral-950: #020617 !important;

    color-scheme: light !important;
}

/* ===== Hard Override for Textareas, Inputs, Dropdowns ===== */
.gradio-container textarea,
.gradio-container input[type="text"],
.gradio-container input[type="number"],
.gradio-container select,
.gradio-container .gr-input,
.gradio-container .gr-text-input,
.gradio-container .input-container,
.gradio-container .secondary-wrap,
.gradio-container [data-testid="textbox"],
.gradio-container [data-testid="dropdown"],
.gradio-container [data-testid="number-input"],
.gradio-container fieldset {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
}

.gradio-container textarea:focus,
.gradio-container input[type="text"]:focus,
.gradio-container input[type="number"]:focus,
.gradio-container select:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
    outline: none !important;
}

.gradio-container textarea::placeholder,
.gradio-container input::placeholder {
    color: #94a3b8 !important;
}

/* ===== Labels & Badges Fix ===== */
.gradio-container span[data-testid="block-label"],
.gradio-container label > span,
.gradio-container .label-text,
.gradio-container .group-label {
    background-color: #eef2ff !important;
    color: #4338ca !important;
    font-weight: 600 !important;
    border: 1px solid #c7d2fe !important;
    border-radius: 6px !important;
}

/* ===== Global Background & Typography ===== */
html, body, .gradio-container, .gradio-container.dark {
    background-color: #f8fafc !important;
    background-image: none !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    color: #1e293b !important;
}

/* Force all Gradio blocks/panels to clean white */
.gradio-container .block, .gradio-container .form,
.gradio-container [data-testid="column"],
.gradio-container .gr-panel,
.gradio-container .gr-box {
    background-color: #ffffff !important;
    border-color: #e2e8f0 !important;
    color: #1e293b !important;
}

/* ===== Hero Banner ===== */
.hero-banner {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 32px 40px;
    margin-bottom: 24px;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.03);
}

.hero-title {
    font-size: 1.85rem;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 8px;
    letter-spacing: -0.02em;
}

.hero-subtitle {
    color: #475569;
    font-size: 0.98rem;
    font-weight: 400;
    line-height: 1.6;
}

/* ===== Buttons ===== */
.gr-button-primary, button.primary {
    background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    box-shadow: 0 2px 4px rgba(79, 70, 229, 0.25) !important;
    transition: all 0.2s ease !important;
}

.gr-button-primary:hover, button.primary:hover {
    background: linear-gradient(135deg, #4338ca 0%, #3730a3 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 16px rgba(79, 70, 229, 0.3) !important;
}

/* ===== Tabs ===== */
.tab-nav button, div.tab-nav button {
    font-weight: 500 !important;
    color: #64748b !important;
    border-radius: 6px !important;
    padding: 8px 16px !important;
    transition: all 0.15s ease !important;
}

.tab-nav button.selected, div.tab-nav button.selected {
    background: #eef2ff !important;
    color: #4f46e5 !important;
    border-bottom: 2px solid #4f46e5 !important;
    font-weight: 600 !important;
}

/* ===== Tables ===== */
.markdown-body table, .prose table {
    width: 100% !important;
    border-collapse: collapse !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    overflow: hidden !important;
    margin: 16px 0 !important;
    font-size: 0.875rem !important;
}

.markdown-body th, .prose th {
    background: #f1f5f9 !important;
    color: #334155 !important;
    font-weight: 600 !important;
    padding: 12px 16px !important;
    text-align: left !important;
    border-bottom: 1px solid #e2e8f0 !important;
}

.markdown-body td, .prose td {
    padding: 10px 16px !important;
    border-bottom: 1px solid #f1f5f9 !important;
    color: #475569 !important;
}

.markdown-body tr:nth-child(even) td, .prose tr:nth-child(even) td {
    background: #f8fafc !important;
}

/* ===== Chatbot ===== */
.gr-chatbot, [data-testid="chatbot"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    background: #ffffff !important;
}

/* ===== Status Box ===== */
.status-box textarea {
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    color: #475569 !important;
    font-size: 0.85rem !important;
}

/* ===== Sliders ===== */
.gr-slider .slider-input {
    accent-color: #4f46e5 !important;
}

/* ===== File Upload ===== */
.gr-file-upload, [data-testid="file"] {
    border: 2px dashed #cbd5e1 !important;
    border-radius: 10px !important;
    background: #fafbfc !important;
    transition: border-color 0.2s ease !important;
}

.gr-file-upload:hover, [data-testid="file"]:hover {
    border-color: #4f46e5 !important;
}

/* ===== Markdown Output ===== */
.markdown-body, .prose {
    color: #334155 !important;
    line-height: 1.7 !important;
}

.markdown-body h1, .markdown-body h2, .markdown-body h3 {
    color: #1e293b !important;
    font-weight: 700 !important;
}

/* ===== Scrollbar ===== */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}

::-webkit-scrollbar-track {
    background: transparent;
}

::-webkit-scrollbar-thumb {
    background: #cbd5e1;
    border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
    background: #94a3b8;
}
"""

app_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="#f8fafc",
    block_background_fill="#ffffff",
    block_border_width="1px",
    block_border_color="#e2e8f0",
    block_radius="10px",
    button_primary_background_fill="#4f46e5",
    button_primary_background_fill_hover="#4338ca",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#f1f5f9",
    button_secondary_background_fill_hover="#e2e8f0",
    input_background_fill="#ffffff",
    input_border_color="#cbd5e1",
    input_border_color_focus="#6366f1",
)


with gr.Blocks(title="期刊选稿画像助手 - WebUI", css=custom_css, theme=app_theme) as demo:
    gr.HTML(
        """
        <div class="hero-banner">
            <div class="hero-title">期刊选稿画像助手</div>
            <div class="hero-subtitle">百篇大样本驱动的学术品味诊断、全网多期刊智能路由与投稿策略系统</div>
        </div>
        """
    )

    with gr.Tabs():
        with gr.Tab("选稿画像与循证诊断"):
            with gr.Row():
                # 左侧输入控制区
                with gr.Column(scale=2, min_width=380):
                    gr.Markdown("### 投稿对标参数配置")

                    search_input = gr.Textbox(
                        label="输入期刊关键词进行联想（输入 2 个字母以上自动检索）",
                        placeholder="例如: computers 或 strategic",
                        value="",
                    )

                    journal_input = gr.Dropdown(
                        label="选择目标期刊全称",
                        choices=["Computers in Human Behavior"],
                        value="Computers in Human Behavior",
                        allow_custom_value=True,
                        interactive=True,
                    )

                    with gr.Row():
                        years_input = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="数据回溯年份")
                        max_papers_input = gr.Slider(
                            minimum=20, maximum=200, value=100, step=10, label="大样本并发采样文献数 (100+高特异性指向)"
                        )

                    # 输入方式卡片：提供粘贴文本和文件上传两种选择
                    with gr.Tab("粘贴摘要/草稿"):
                        draft_input = gr.Textbox(
                            label="粘贴拟投稿论文的 Title / Abstract / 大纲",
                            placeholder="在此粘贴，系统将给出手术级重构方案...",
                            lines=8,
                        )

                    with gr.Tab("上传草稿文件"):
                        file_input = gr.File(label="上传 Word (.docx) 或 PDF (.pdf) 文件", file_types=[".docx", ".pdf"])

                    submit_btn = gr.Button("一键生成期刊选稿画像与对标报告", variant="primary", size="lg")

                # 右侧报告输出区
                with gr.Column(scale=3, min_width=480):
                    gr.Markdown("### 期刊选稿画像与修稿报告")
                    status_output = gr.Textbox(
                        label="运行状态",
                        value="就绪，等待输入并点击生成...",
                        interactive=False,
                        elem_classes=["status-box"],
                    )
                    report_output = gr.Markdown(value="*报告生成后将在此处以精美 Markdown 格式自动渲染展示。*")

        with gr.Tab("多期刊梯队智能路由"):
            gr.Markdown("### 论文草稿多期刊投递阵列路由")
            gr.Markdown(
                "系统将自动解析你的论文草稿（支持粘贴文本、上传 Word/PDF 文件，并共享主大厅的草稿与文件），对比候选期刊池评定 **冲刺 (Reaching)**、**主投 (Target)** 与 **保底 (Safe)** 三级投递梯队。"
            )

            with gr.Tabs():
                with gr.Tab("粘贴摘要/草稿"):
                    router_draft_input = gr.Textbox(
                        label="粘贴你的论文草稿 (Title / Abstract / 全文)",
                        lines=6,
                        placeholder="在此粘贴论文草稿（若在主大厅已粘贴或上传文件，此处可留空，系统自动复用）...",
                    )
                with gr.Tab("上传草稿文件"):
                    router_file_input = gr.File(
                        label="上传 Word (.docx) 或 PDF (.pdf) 文件", file_types=[".docx", ".pdf"]
                    )

            router_btn = gr.Button("一键路由生成多期刊投递阵列", variant="primary", size="lg")
            router_output = gr.Markdown(value="*决策阵列生成后将在此处渲染展示。*")

        with gr.Tab("模拟审稿人在线对答"):
            gr.Markdown(
                """
                ### 模拟 AE / 审稿人对话舱
                在主大厅生成诊断报告后，你可以在此与模拟 Associate Editor / 审稿人展开在线答辩与交流。
                审稿人将继承本期刊的学术品味，对你的修改思路与稳健性方案进行审核把关。
                """
            )
            chatbot = gr.Chatbot(label="与 Associate Editor / 审稿人对话中", height=480)
            msg_input = gr.Textbox(
                label="输入你的疑问或辩词",
                placeholder="例如: 关于第2点样本量劣势，如果我补充二期追踪数据达到 N=750，可以吗？",
                lines=2,
            )
            with gr.Row():
                send_btn = gr.Button("发送", variant="primary")
                clear_btn = gr.Button("清除历史对话")

    # 事件流绑定：输入关键词时，实时触发下拉框备选项更新
    search_input.input(fn=search_journals, inputs=search_input, outputs=journal_input)

    # 按钮点击事件绑定 (将 file_input 接入输入列表，并更新状态)
    submit_btn.click(
        fn=run_pipeline,
        inputs=[journal_input, years_input, max_papers_input, draft_input, file_input],
        outputs=[status_output, report_output],
    )

    # 路由大脑点击事件绑定 (全量接入本 Tab 输入与主大厅输入)
    router_btn.click(
        fn=run_journal_router,
        inputs=[router_draft_input, router_file_input, draft_input, file_input],
        outputs=router_output,
    )

    # 聊天消息发送绑定 (直接绑定 report_output 作为 report_text 输入，解决 gr.State 缓存问题)
    send_btn.click(fn=chat_with_reviewer, inputs=[msg_input, chatbot, report_output], outputs=[chatbot, msg_input])
    msg_input.submit(fn=chat_with_reviewer, inputs=[msg_input, chatbot, report_output], outputs=[chatbot, msg_input])

    # 清空对话
    clear_btn.click(fn=lambda: ([], ""), inputs=None, outputs=[chatbot, msg_input])

if __name__ == "__main__":
    print("\n[WebUI 启动成功] 请打开浏览器访问: http://127.0.0.1:7860\n", flush=True)
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
