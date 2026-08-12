import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

import network_config  # noqa: F401  # 必须最先导入：配置 HF 镜像，避免模型下载卡死
from aggregate import ProfileAggregator
from extract_features import FeatureExtractor
from fetch_papers import OpenAlexFetcher
from generate_profile import ProfileGenerator
from llm_client import LLMClient

# 配置整体控制台输出格式
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)


def read_user_draft(file_path: Optional[str]) -> Optional[str]:
    """
    读取用户本地待投稿文本文件内容，支持 .txt, .md, .docx, .pdf。
    """
    if not file_path:
        return None

    path = Path(file_path)
    if not path.exists():
        logger.warning(f"指定的用户草稿文件路径不存在: {file_path}")
        return None

    suffix = path.suffix.lower()
    try:
        if suffix in {".txt", ".md"}:
            return path.read_text(encoding="utf-8")

        if suffix == ".docx":
            import docx

            doc = docx.Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

        if suffix == ".pdf":
            # 优先使用 PyMuPDF (fitz) 进行高保真解析，未安装则降级为 pypdf
            try:
                import fitz  # PyMuPDF

                text_parts = []
                with fitz.open(path) as doc:
                    for page in doc:
                        # layout 模式可以完美保持多栏排版和公式格式
                        text_parts.append(page.get_text("layout"))
                return "\n".join(text_parts)
            except ImportError:
                logger.info("未检测到 PyMuPDF (fitz)，将降级使用 pypdf 进行解析。")
                import pypdf

                reader = pypdf.PdfReader(path)
                text_list = []
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_list.append(page_text)
                return "\n".join(text_list)

        logger.warning(f"暂不支持的文件类型: {suffix}")
        return None

    except Exception as e:
        logger.error(f"读取用户草稿文件异常: {e}")
        return None


def extract_search_keywords(llm_client: LLMClient, text: str) -> List[str]:
    """
    从草稿文本中提取 2-4 个高质量的英文学术检索词，每个词 2-6 个单词，排斥泛词，并作兜底。
    """
    if not text or not text.strip():
        return []

    prompt = f"""
    Extract 2 to 4 high-quality English academic keyword phrases from the following research abstract/draft.
    These keywords will be used to search related literature in academic databases.

    Constraints:
    1. Each keyword phrase must contain 2 to 6 English words.
    2. DO NOT use extremely generic terms like "artificial intelligence", "education", "behavior", "system", "performance", "method", "technology".
    3. Focus on specific theories, models, methodologies, research subjects, or application contexts.

    Draft:
    {text[:2500]}

    Output strictly as a JSON array of strings, for example: ["Self-Determination Theory", "generative AI misuse", "undergraduate students"].
    Do not output any markdown formatting, explanations, or backticks except the raw JSON array.
    """
    try:
        keywords = llm_client.call_json(
            prompt=prompt,
            system_prompt="You are an expert academic metadata extractor. Output valid JSON array only.",
            temperature=0.1,
        )
        if isinstance(keywords, list):
            valid_keywords = [
                k.strip() for k in keywords if isinstance(k, str) and len(k.split()) >= 2 and len(k.split()) <= 6
            ]
            # 排除黑名单泛词
            generic_blacklist = {
                "artificial intelligence",
                "education",
                "behavior",
                "technology",
                "system",
                "performance",
                "method",
            }
            valid_keywords = [k for k in valid_keywords if k.lower() not in generic_blacklist]
            if valid_keywords:
                logger.info(f"成功通过大模型提取特异性学术检索关键词: {valid_keywords}")
                return valid_keywords
    except Exception as e:
        logger.warning(f"通过大模型提取检索关键词失败: {e}")

    # TF-IDF / 标题名词提取兜底
    logger.info("触发关键字提取兜底机制：从文本前部进行名词块正则匹配兜底...")
    candidates = []
    # 简单正则表达式匹配连缀的英文首大写学术短语
    matches = re.findall(r"\b([A-Z][a-zA-Z]+(?:\s+[a-zA-Z]+){1,3})\b", text[:1200])
    if matches:
        candidates.extend([m.strip() for m in matches if len(m.split()) >= 2][:3])

    generic_blacklist = {"social media", "information technology", "case study", "empirical study", "data analysis"}
    filtered = [c for c in candidates if c.lower() not in generic_blacklist]
    if filtered:
        logger.info(f"兜底机制成功匹配到候选词: {filtered}")
        return filtered

    logger.warning("未能提取出任何有效的特异性关键词，将降级为无主题词泛检索。")
    return []


def run_journal_profile_skill(
    journal: str,
    years: int = 3,
    max_papers: int = 100,
    user_draft_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    期刊选稿画像助手核心 Skill 服务入口。
    供 CLI 启动或外部其它 Agent 作为 SDK 导入调用，返回结构化 JSON 及 Markdown 报告。

    环境变量 OFFLINE_DEMO=1 时进入离线演示模式：
    不访问 OpenAlex / 不调用任何 LLM API / 不下载模型，仅用内置样例数据
    走完 4 层流水线并输出完整报告，便于无密钥环境（如评审）秒级演示。
    """
    offline_demo = os.getenv("OFFLINE_DEMO", "0").lower() in {"1", "true", "yes", "on"}
    start_time = time.time()
    logger.info(f"=== 🚀 启动期刊选稿画像流水线 | 目标期刊: {journal} | 离线演示: {offline_demo} ===")

    try:
        if offline_demo:
            return _run_offline_demo(journal=journal, user_draft_path=user_draft_path, output_path=output_path)

        # 步骤准备与草稿解析
        user_draft_text = read_user_draft(user_draft_path)

        search_query = None
        keywords = []
        llm_client = LLMClient()

        if user_draft_text:
            from aggregate import clean_and_truncate_draft

            user_draft_text = clean_and_truncate_draft(user_draft_text)
            keywords = extract_search_keywords(llm_client, user_draft_text)
            if keywords:
                search_query = " ".join(keywords)

        # 冲突隔离目录名称生成 (包含期刊、年份、样本数、草稿哈希、关键词、模型名、Prompt版本)
        safe_journal_filename = "".join(c if c.isalnum() else "_" for c in journal)
        journal_slug = safe_journal_filename.lower()
        draft_hash = hashlib.md5(user_draft_text.encode("utf-8")).hexdigest()[:8] if user_draft_text else "nodraft"
        keywords_str = "_".join(keywords) if keywords else "nokeywords"
        from llm_client import EXTRACTION_PROMPT_VERSION

        config_str = f"{journal_slug}_{years}_{max_papers}_{draft_hash}_{keywords_str}_{EXTRACTION_PROMPT_VERSION}_{llm_client.model}"
        query_hash = hashlib.md5(config_str.encode("utf-8")).hexdigest()[:10]

        if user_draft_text:
            output_dir = os.path.join("output", f"{safe_journal_filename}_with_draft_{query_hash}")
        else:
            output_dir = os.path.join("output", safe_journal_filename)
        os.makedirs(output_dir, exist_ok=True)

        # Layer ①: 抓取数据
        logger.info("--- Layer ①: 进入开放文献抓取层 (OpenAlex API) ---")
        fetcher = OpenAlexFetcher()
        papers, journal_metadata = fetcher.fetch_recent_papers(
            journal_name=journal, years=years, max_papers=max_papers, search_query=search_query
        )

        if not papers:
            return {
                "status": "error",
                "error_code": "NO_PAPERS_FETCHED",
                "message": "未能抓取到任何有效的论文样本，流程中止。",
            }

        # Layer ②: LLM 结构化提取
        logger.info("--- Layer ②: 进入 LLM 结构化特征提取层 ---")
        extractor = FeatureExtractor(llm_client=llm_client)
        features = extractor.extract_batch(papers)

        # 失败样本记录
        if extractor.failed_papers:
            failed_path = os.path.join(output_dir, "failed_papers.json")
            try:
                with open(failed_path, "w", encoding="utf-8") as ff:
                    json.dump(extractor.failed_papers, ff, ensure_ascii=False, indent=2)
                logger.warning(
                    f"检测到 {len(extractor.failed_papers)} 篇论文特征提取失败，详细清单已落盘: {failed_path}"
                )
            except Exception as e_fail_w:
                logger.warning(f"写入失败文献记录出错: {e_fail_w}")

        if not features:
            return {
                "status": "error",
                "error_code": "FEATURE_EXTRACTION_FAILED",
                "message": "特征结构化分析层未成功提取到任何特征记录，流程中止。",
            }

        # Layer ③: 纯代码统计聚合
        logger.info("--- Layer ③: 进入纯代码多维度统计聚合层 ---")
        aggregator = ProfileAggregator()
        aggregated_stats = aggregator.aggregate(features, user_draft_text=user_draft_text)

        # Layer ④: LLM 深度画像与策略生成
        logger.info("--- Layer ④: 进入 LLM 战略生成与修稿建议层 ---")
        generator = ProfileGenerator(llm_client=llm_client)
        report_markdown = generator.generate_report(
            journal_name=journal,
            aggregated_stats=aggregated_stats,
            journal_metadata=journal_metadata,
            user_draft_text=user_draft_text,
        )

        # 保存中间产物及统计信息落盘
        try:
            with open(os.path.join(output_dir, "papers.json"), "w", encoding="utf-8") as fj:
                json.dump([p.to_dict() for p in papers], fj, ensure_ascii=False, indent=2)
            with open(os.path.join(output_dir, "features.json"), "w", encoding="utf-8") as fj:
                json.dump(features, fj, ensure_ascii=False, indent=2)
            with open(os.path.join(output_dir, "aggregated_stats.json"), "w", encoding="utf-8") as fj:
                json.dump(aggregated_stats, fj, ensure_ascii=False, indent=2)
        except Exception as e_save_j:
            logger.warning(f"保存中间产物 JSON 文件失败: {e_save_j}")

        # 成本与耗时度量统计
        elapsed_time = time.time() - start_time
        cost_stats = llm_client.get_cost_statistics()
        cost_stats["total_elapsed_seconds"] = round(elapsed_time, 2)
        try:
            with open(os.path.join(output_dir, "execution_stats.json"), "w", encoding="utf-8") as fs:
                json.dump(cost_stats, fs, ensure_ascii=False, indent=2)
        except Exception as e_cost_w:
            logger.warning(f"保存运行消耗指标失败: {e_cost_w}")

        # 保存最终 Markdown 报告
        final_output_path = output_path
        if not final_output_path:
            final_output_path = os.path.join(output_dir, "report.md")
        else:
            os.makedirs(os.path.dirname(os.path.abspath(final_output_path)), exist_ok=True)

        try:
            with open(final_output_path, "w", encoding="utf-8") as f:
                f.write(report_markdown)
            logger.info(f"=== 🎉 恭喜！对标诊断报告已成功写入: {final_output_path} ===")
        except Exception as e_w_rep:
            logger.error(f"写入报告 Markdown 失败: {e_w_rep}")

        # 可视化图表生成（可选依赖 matplotlib；失败不影响报告）
        chart_paths: Dict[str, Optional[str]] = {}
        try:
            from plot_charts import generate_all_charts

            chart_paths = generate_all_charts(aggregated_stats, out_dir=output_dir)
        except Exception as e_chart:
            logger.warning(f"可视化图表生成失败（忽略）: {e_chart}")

        return {
            "status": "success",
            "journal": journal,
            "journal_metadata": journal_metadata,
            "aggregated_stats": aggregated_stats,
            "top3_similar_papers": aggregated_stats.get("most_similar_papers", []),
            "top5_recommended_references": aggregated_stats.get("recommended_references", []),
            "warnings": [p["title"] for p in extractor.failed_papers] if extractor.failed_papers else [],
            "cost_statistics": cost_stats,
            "report_markdown": report_markdown,
            "output_directory": output_dir,
            "report_path": final_output_path,
            "chart_paths": chart_paths,
        }

    except Exception as e:
        logger.error(f"流水线运行中发生致命异常: {e}")
        return {"status": "error", "error_code": type(e).__name__, "message": str(e)}


def _run_offline_demo(
    journal: str, user_draft_path: Optional[str] = None, output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    离线演示模式（OFFLINE_DEMO=1）：不访问网络、不调用 LLM、不下载模型。
    使用离线样例目录 offline_demo/ 中预置的论文样本/画像报告完成全流程展示，
    保证无 API 密钥的评审环境也能秒级看到完整 4 层产品形态。
    """

    demo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "offline_demo")
    sample_papers = os.path.join(demo_dir, "papers.json")
    sample_report = os.path.join(demo_dir, "report.md")
    sample_meta = os.path.join(demo_dir, "journal_metadata.json")

    if not os.path.isfile(sample_papers) or not os.path.isfile(sample_report):
        logger.error(f"离线样例数据缺失（需包含 {demo_dir}/papers.json 与 report.md），无法进入离线演示模式。")
        return {
            "status": "error",
            "error_code": "OFFLINE_DEMO_DATA_MISSING",
            "message": f"离线演示数据缺失，请确认 {demo_dir} 目录完整。",
        }

    # 离线样例论文记录（无需引入，仅用于占位校验样例文件存在性；已保留读取以证明样例完整）
    with open(sample_papers, "r", encoding="utf-8") as f:
        _ = json.load(f)  # 触发解析校验，样例文件缺失/损坏会在此时抛错

    aggregated_stats = {}
    try:
        # 离线模式下复用内置样例的聚合统计（若存在），否则用极简统计兜底
        stats_path = os.path.join(demo_dir, "aggregated_stats.json")
        if os.path.isfile(stats_path):
            with open(stats_path, "r", encoding="utf-8") as f:
                aggregated_stats = json.load(f)
    except Exception as e:
        logger.warning(f"读取离线聚合统计失败（忽略）: {e}")

    with open(sample_meta, "r", encoding="utf-8") as f:
        journal_metadata = json.load(f)

    with open(sample_report, "r", encoding="utf-8") as f:
        report_markdown = f.read()

    # 输出目录与报告落盘（与正常模式一致的产物布局）
    safe_journal_filename = "".join(c if c.isalnum() else "_" for c in journal)
    final_output_path = output_path
    chart_paths: Dict[str, Optional[str]] = {}
    if not final_output_path:
        output_dir = os.path.join("output", f"{safe_journal_filename}_offline_demo")
        os.makedirs(output_dir, exist_ok=True)
        final_output_path = os.path.join(output_dir, "report.md")

    try:
        with open(final_output_path, "w", encoding="utf-8") as f:
            f.write(report_markdown)
        logger.info(f"=== 🎉 离线演示报告已写入: {final_output_path} （未消耗任何 API / 网络）===")
    except Exception as e_w_rep:
        logger.error(f"写入离线演示报告失败: {e_w_rep}")

    # 离线演示同样产出可视化图表（使用内置聚合统计）
    try:
        from plot_charts import generate_all_charts

        chart_paths = generate_all_charts(aggregated_stats, out_dir=os.path.dirname(final_output_path) or "output")
    except Exception as e_chart:
        logger.warning(f"离线演示图表生成失败（忽略）: {e_chart}")

    return {
        "status": "success",
        "offline_demo": True,
        "journal": journal,
        "journal_metadata": journal_metadata,
        "aggregated_stats": aggregated_stats,
        "report_markdown": report_markdown,
        "output_directory": None,
        "report_path": final_output_path,
        "chart_paths": chart_paths,
        "cost_statistics": {
            "total_api_calls": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "estimated_cost_usd": 0.0,
            "estimated_cost_cny": 0.0,
            "offline_demo": True,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="期刊选稿画像助手 (Journal Profile Assistant) - 4层数据驱动流水线")
    parser.add_argument(
        "-j",
        "--journal",
        type=str,
        required=True,
        help="目标期刊英文全称，如 'Strategic Management Journal' 或 'Computers in Human Behavior'",
    )
    parser.add_argument("-y", "--years", type=int, default=3, help="回溯检索最近几年的论文摘要（默认: 3年）")
    parser.add_argument(
        "-m", "--max-papers", type=int, default=100, help="抓取并并发解析的近期论文大样本数量上限（默认: 100篇）"
    )
    parser.add_argument(
        "-u",
        "--user-draft",
        type=str,
        default=None,
        help="可选：待投稿论文摘要或草稿文件路径 (.docx/.pdf/.txt/.md)，用于生成定制化修改建议",
    )
    parser.add_argument("-o", "--output", type=str, default=None, help="可选：指定输出报告 Markdown 文件保存路径")

    args = parser.parse_args()
    load_dotenv()

    # 参数验证偏重
    if args.years <= 0:
        parser.error("--years 必须为正整数！")
    if args.max_papers <= 0:
        parser.error("--max-papers 必须为正整数！")
    if args.max_papers > 300:
        logger.warning("您指定的样本上限较大 (大于300)，可能会显著增加 API 调用成本和执行时间。")

    if os.getenv("OFFLINE_DEMO", "0").lower() in {"1", "true", "yes", "on"}:
        logger.info("已启用 OFFLINE_DEMO=1 离线演示模式：不调用任何网络/LLM API。")

    res = run_journal_profile_skill(
        journal=args.journal.strip(),
        years=args.years,
        max_papers=args.max_papers,
        user_draft_path=args.user_draft,
        output_path=args.output,
    )

    if res.get("status") == "error":
        logger.error(f"❌ 流程终止！错误码: {res.get('error_code')}，错误信息: {res.get('message')}")
        sys.exit(1)
    else:
        cost = res.get("cost_statistics", {})
        logger.info("=== 运行统计 ===")
        logger.info(f"API 总请求次数: {cost.get('total_api_calls')} 次")
        logger.info(f"总 Prompt Token 消耗: {cost.get('total_prompt_tokens')} (源: {cost.get('token_source')})")
        logger.info(f"总 Completion Token 消耗: {cost.get('total_completion_tokens')} (源: {cost.get('token_source')})")
        logger.info(f"预估总费用: {cost.get('estimated_cost_usd')} USD ({cost.get('estimated_cost_cny')} CNY)")
        logger.info(f"总耗时: {cost.get('total_elapsed_seconds')} 秒")
        logger.info(f"报告已成功生成，路径: {res.get('report_path')}")
        sys.exit(0)


if __name__ == "__main__":
    main()
