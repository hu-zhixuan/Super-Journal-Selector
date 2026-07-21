import os
import sys
import re
import argparse
import logging
from typing import Optional, List
from pathlib import Path
from dotenv import load_dotenv

from fetch_papers import OpenAlexFetcher
from extract_features import FeatureExtractor
from aggregate import ProfileAggregator
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
            doc = docx.Document(path)
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
            temperature=0.1
        )
        if isinstance(keywords, list):
            valid_keywords = [
                k.strip() for k in keywords 
                if isinstance(k, str) and len(k.split()) >= 2 and len(k.split()) <= 6
            ]
            # 排除黑名单泛词
            generic_blacklist = {"artificial intelligence", "education", "behavior", "technology", "system", "performance", "method"}
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


def main():
    parser = argparse.ArgumentParser(
        description="期刊选稿画像助手 (Journal Profile Assistant) - 4层数据驱动流水线"
    )
    parser.add_argument(
        "-j", "--journal",
        type=str,
        required=True,
        help="目标期刊英文全称，如 'Strategic Management Journal' 或 'Computers in Human Behavior'"
    )
    parser.add_argument(
        "-y", "--years",
        type=int,
        default=3,
        help="回溯检索最近几年的论文摘要（默认: 3年）"
    )
    parser.add_argument(
        "-m", "--max-papers",
        type=int,
        default=100,
        help="抓取并并发解析的近期论文大样本数量上限（默认: 100篇）"
    )
    parser.add_argument(
        "-u", "--user-draft",
        type=str,
        default=None,
        help="可选：待投稿论文摘要或草稿文件路径 (.docx/.pdf/.txt/.md)，用于生成定制化修改建议"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="可选：指定输出报告 Markdown 文件保存路径"
    )

    args = parser.parse_args()
    load_dotenv()

    # 参数验证偏重
    if args.years <= 0:
        parser.error("--years 必须为正整数！")
    if args.max_papers <= 0:
        parser.error("--max-papers 必须为正整数！")
    if args.max_papers > 300:
        logger.warning("您指定的样本上限较大 (大于300)，可能会显著增加 API 调用成本和执行时间。")

    journal_name = args.journal.strip()
    logger.info(f"=== 🚀 启动期刊选稿画像流水线 | 目标期刊: {journal_name} ===")

    # 步骤准备与草稿解析
    user_draft_text = read_user_draft(args.user_draft)
    
    search_query = None
    if user_draft_text:
        from aggregate import clean_and_truncate_draft
        user_draft_text = clean_and_truncate_draft(user_draft_text)
        
        # 初始化 LLM 客户端提取主题词
        llm_client = LLMClient()
        keywords = extract_search_keywords(llm_client, user_draft_text)
        if keywords:
            search_query = " ".join(keywords)

    # Layer ①: 抓取数据（传入 search_query 激活双通道动态检索）
    logger.info("--- Layer ①: 进入开放文献抓取层 (OpenAlex API) ---")
    fetcher = OpenAlexFetcher()
    try:
        papers, journal_metadata = fetcher.fetch_recent_papers(
            journal_name=journal_name,
            years=args.years,
            max_papers=args.max_papers,
            search_query=search_query
        )
    except Exception as e:
        logger.error(f"抓取文献大样本失败: {e}")
        sys.exit(1)

    if not papers:
        logger.error("未能抓取到足够或有效的带摘要论文，流程提前停止。")
        sys.exit(1)

    # Layer ②: LLM 结构化提取
    logger.info("--- Layer ②: 进入 LLM 结构化特征提取层 ---")
    extractor = FeatureExtractor()
    features = extractor.extract_batch(papers)
    if not features:
        logger.error("LLM 结构化解析未返回有效特征记录，流程提前停止。")
        sys.exit(1)

    # Layer ③: 纯代码统计聚合
    logger.info("--- Layer ③: 进入纯代码多维度统计聚合层 ---")
    aggregator = ProfileAggregator()
    aggregated_stats = aggregator.aggregate(features, user_draft_text=user_draft_text)

    # Layer ④: LLM 深度画像与修稿策略生成
    logger.info("--- Layer ④: 进入 LLM 战略生成与修稿建议层 ---")
    generator = ProfileGenerator()
    report_markdown = generator.generate_report(
        journal_name=journal_name,
        aggregated_stats=aggregated_stats,
        journal_metadata=journal_metadata,
        user_draft_text=user_draft_text,
    )

    # 保存报告与中间结果落盘 (落盘至 output/{journal_slug}/ 保证生产级可观测性)
    safe_journal_filename = "".join(c if c.isalnum() else "_" for c in journal_name)
    output_dir = os.path.join("output", safe_journal_filename)
    os.makedirs(output_dir, exist_ok=True)

    # 保存中间产物落盘
    try:
        import json
        with open(os.path.join(output_dir, "papers.json"), "w", encoding="utf-8") as fj:
            json.dump([p.to_dict() for p in papers], fj, ensure_ascii=False, indent=2)
        with open(os.path.join(output_dir, "features.json"), "w", encoding="utf-8") as fj:
            json.dump(features, fj, ensure_ascii=False, indent=2)
        with open(os.path.join(output_dir, "aggregated_stats.json"), "w", encoding="utf-8") as fj:
            json.dump(aggregated_stats, fj, ensure_ascii=False, indent=2)
        logger.info(f"中间产物 (papers.json, features.json, aggregated_stats.json) 已成功落盘保存至: {output_dir}")
    except Exception as e_save_j:
        logger.warning(f"保存中间产物 JSON 失败: {e_save_j}")

    # 保存 Markdown 报告
    output_path = args.output
    if not output_path:
        output_path = os.path.join(output_dir, "report.md")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_markdown)
        logger.info(f"=== 🎉 恭喜！完整的期刊画像与修稿建议已成功写入: {output_path} ===")
    except Exception as e:
        logger.error(f"保存最终 Markdown 文件失败: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
