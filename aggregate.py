import logging
from typing import List, Dict, Any
from collections import Counter
import statistics

logger = logging.getLogger(__name__)


class ProfileAggregator:
    """
    层③：纯代码统计聚合层。不依赖 LLM，仅基于纯粹的统计计算与数据清洗，将数十篇单文结构化特征聚合为高浓度的维度分布指标。
    """

    @staticmethod
    def aggregate(features_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算方法学占比、高频理论矩阵、分析模型热度榜以及样本阈值分布
        """
        total_count = len(features_list)
        if total_count == 0:
            raise ValueError("传入的结构化特征列表为空，无法进行统计聚合。")

        logger.info(f"正在对 {total_count} 篇结构化数据进行多维度统计聚合计算...")

        # 1. 研究范式（分类统计）分布与占比
        method_counts = Counter([f.get("method_category", "Other") for f in features_list])
        method_distribution = {
            category: {
                "count": count,
                "percentage": round((count / total_count) * 100, 1),
            }
            for category, count in method_counts.most_common()
        }

        # 2. 各研究范式下的平均被引权重（展示哪类文章在该刊最常产出高引爆款）
        method_citations: Dict[str, List[int]] = {}
        for f in features_list:
            cat = f.get("method_category", "Other")
            if cat not in method_citations:
                method_citations[cat] = []
            method_citations[cat].append(f.get("cited_by_count", 0))

        method_avg_citations = {
            cat: round(statistics.mean(c_list), 1) if c_list else 0.0
            for cat, c_list in method_citations.items()
        }

        # 3. 样本量级门槛统计（仅针对 Quantitative & Computational 中有效 numeric 样本量 > 0 的）
        numeric_samples = [
            f["sample_size_approx"]
            for f in features_list
            if f.get("sample_size_approx", -1) > 0
        ]
        sample_size_stats = {}
        if numeric_samples:
            numeric_samples.sort()
            sample_size_stats = {
                "min_sample": min(numeric_samples),
                "median_sample": int(statistics.median(numeric_samples)),
                "max_sample": max(numeric_samples),
                "valid_numeric_count": len(numeric_samples),
            }
        else:
            sample_size_stats = {
                "min_sample": "N/A",
                "median_sample": "N/A",
                "max_sample": "N/A",
                "valid_numeric_count": 0,
            }

        # 4. 高频核心理论框架排行 (Top 12)
        all_theories = []
        for f in features_list:
            for t in f.get("theoretical_frameworks", []):
                cleaned_t = t.strip()
                if cleaned_t and len(cleaned_t) > 2:
                    all_theories.append(cleaned_t)
        top_theories = Counter(all_theories).most_common(12)

        # 5. 高频分析统计工具排行 (Top 12)
        all_tools = []
        for f in features_list:
            for tool in f.get("analytical_tools", []):
                cleaned_tool = tool.strip()
                if cleaned_tool and len(cleaned_tool) > 1:
                    all_tools.append(cleaned_tool)
        top_tools = Counter(all_tools).most_common(12)

        # 6. 挑选中近期被引次最高、具代表性的论文创新点金句作为示范 (Top 5)
        sorted_by_citations = sorted(
            features_list, key=lambda x: x.get("cited_by_count", 0), reverse=True
        )
        representative_novelties = [
            {
                "title": item.get("title", ""),
                "cited_by_count": item.get("cited_by_count", 0),
                "method": item.get("method_category", ""),
                "novelty_highlight": item.get("novelty_highlight", ""),
            }
            for item in sorted_by_citations[:5]
        ]

        aggregated_data = {
            "total_papers_analyzed": total_count,
            "method_distribution": method_distribution,
            "method_avg_citations": method_avg_citations,
            "sample_size_stats": sample_size_stats,
            "top_theories": [{"name": name, "count": cnt} for name, cnt in top_theories],
            "top_tools": [{"name": name, "count": cnt} for name, cnt in top_tools],
            "representative_novelties": representative_novelties,
        }

        logger.info("统计聚合完成。")
        return aggregated_data
