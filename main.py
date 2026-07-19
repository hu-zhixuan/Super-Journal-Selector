import os
import argparse
import logging
from typing import Optional
from dotenv import load_dotenv

from fetch_papers import OpenAlexFetcher
from extract_features import FeatureExtractor
from aggregate import ProfileAggregator
from generate_profile import ProfileGenerator

# 配置整体控制台输出格式
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)


def read_user_draft(file_path: Optional[str]) -> Optional[str]:
    """
    读取用户本地待投稿文本文件内容（可选）
    """
    if not file_path:
        return None
    if not os.path.exists(file_path):
        logger.warning(f"指定的用户草稿文件路径不存在: {file_path}")
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"读取用户草稿文件异常: {str(e)}")
        return None


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
        default=30,
        help="抓取并解析的近期论文样本数量上限（默认: 30篇）"
    )
    parser.add_argument(
        "-u", "--user-draft",
        type=str,
        default=None,
        help="可选：待投稿论文摘要或草稿文本文件路径，用于生成定制化修改建议"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="可选：指定输出报告 Markdown 文件保存路径"
    )

    args = parser.parse_args()
    load_dotenv()

    journal_name = args.journal.strip()
    logger.info(f"=== 🚀 启动期刊选稿画像流水线 | 目标期刊: {journal_name} ===")

    # 步骤准备
    user_draft_text = read_user_draft(args.user_draft)

    # Layer ①: 抓取数据
    logger.info("--- Layer ①: 进入开放文献抓取层 (OpenAlex API) ---")
    fetcher = OpenAlexFetcher()
    papers, journal_metadata = fetcher.fetch_recent_papers(
        journal_name=journal_name,
        years=args.years,
        max_papers=args.max_papers,
    )
    if not papers:
        logger.error("未能抓取到足够或有效的带摘要论文，流程提前停止。")
        return

    # Layer ②: LLM结构化提取
    logger.info("--- Layer ②: 进入 LLM 结构化特征提取层 ---")
    extractor = FeatureExtractor()
    features = extractor.extract_batch(papers)
    if not features:
        logger.error("LLM 结构化解析未返回有效特征记录，流程提前停止。")
        return

    # Layer ③: 纯代码统计聚合
    logger.info("--- Layer ③: 进入纯代码多维度统计聚合层 ---")
    aggregator = ProfileAggregator()
    aggregated_stats = aggregator.aggregate(features)

    # Layer ④: LLM深度画像与修稿策略生成
    logger.info("--- Layer ④: 进入 LLM 战略生成与修稿建议层 ---")
    generator = ProfileGenerator()
    report_markdown = generator.generate_report(
        journal_name=journal_name,
        aggregated_stats=aggregated_stats,
        journal_metadata=journal_metadata,
        user_draft_text=user_draft_text,
    )

    # 保存报告
    output_path = args.output
    if not output_path:
        os.makedirs("output", exist_ok=True)
        safe_journal_filename = "".join(c if c.isalnum() else "_" for c in journal_name)
        output_path = os.path.join("output", f"{safe_journal_filename}_Profile_Report.md")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_markdown)
        logger.info(f"=== 🎉 恭喜！完整的期刊画像与修稿建议已成功写入: {output_path} ===")
    except Exception as e:
        logger.error(f"保存最终 Markdown 文件失败: {str(e)}")


if __name__ == "__main__":
    main()
