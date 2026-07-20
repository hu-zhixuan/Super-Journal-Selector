import logging
import re
from typing import List, Dict, Any, Optional
from collections import Counter
import statistics

logger = logging.getLogger(__name__)


class ProfileAggregator:
    """
    层③：纯代码统计聚合层。不依赖 LLM，以 0-Token 消耗的纯 Python 统计函数进行指标计算，
    并内置文本余弦相似度算法对标用户草稿，筛选出 Top 3 最相似的近年发表文献。
    """

    @staticmethod
    def calculate_cosine_similarity(text1: str, text2: str) -> float:
        """
        基于词频（Bag of Words）计算两个文本段落的余弦相似度
        """
        if not text1 or not text2:
            return 0.0
        
        def get_words(t: str) -> List[str]:
            # 转小写并提取单词，过滤标点
            return re.findall(r'\b\w+\b', t.lower())

        w1 = get_words(text1)
        w2 = get_words(text2)
        if not w1 or not w2:
            return 0.0

        c1 = Counter(w1)
        c2 = Counter(w2)
        
        all_words = set(c1.keys()).union(set(c2.keys()))
        
        # 向量点积与模长计算
        dot_product = sum(c1.get(w, 0) * c2.get(w, 0) for w in all_words)
        mag1 = sum(v ** 2 for v in c1.values()) ** 0.5
        mag2 = sum(v ** 2 for v in c2.values()) ** 0.5
        
        if not mag1 or not mag2:
            return 0.0
        return dot_product / (mag1 * mag2)

    @staticmethod
    def aggregate(features_list: List[Dict[str, Any]], user_draft_text: Optional[str] = None) -> Dict[str, Any]:
        """
        计算方法学占比、高频理论矩阵、分析模型热度榜、样本阈值分布，并基于余弦相似度找出 Top 3 最对标的已发表论文。
        """
        total_count = len(features_list)
        if total_count == 0:
            raise ValueError("传入的结构化特征列表为空，无法进行统计聚合。")

        logger.info(f"正在对 {total_count} 篇文献的特征进行多维度统计聚合与对标相似度计算...")

        # 1. 研究范式（分类统计）分布与占比
        method_counts = Counter([f.get("method_category", "Other") for f in features_list])
        method_distribution = {
            category: {
                "count": count,
                "percentage": round((count / total_count) * 100, 1),
            }
            for category, count in method_counts.most_common()
        }

        # 2. 各研究范式下的平均被引权重
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

        # 5b. 方法论审计：开源科学实践聚合统计 (Open Science Practices)
        open_science_counter = Counter()
        for f in features_list:
            practices = f.get("open_science_practices", [])
            # 处理 list[str]
            if not practices:
                open_science_counter["None"] += 1
            else:
                for p in practices:
                    cleaned_p = p.strip()
                    # 规范化命名，如 Open Data, Open Code, Preregistration
                    if not cleaned_p or cleaned_p.lower() == "none":
                        open_science_counter["None"] += 1
                    else:
                        # 转成首字母大写便于统一聚合显示
                        title_p = cleaned_p.title()
                        if "Data" in title_p:
                            title_p = "Open Data"
                        elif "Code" in title_p:
                            title_p = "Open Code"
                        elif "Pre" in title_p or "Reg" in title_p:
                            title_p = "Preregistration"
                        open_science_counter[title_p] += 1
        
        # 计算百分比
        open_science_stats = {
            p_name: {
                "count": cnt,
                "percentage": round((cnt / total_count) * 100, 1)
            }
            for p_name, cnt in open_science_counter.most_common()
        }

        # 5c. 方法论审计：统计汇报风格分布统计 (Statistical Reporting Style)
        reporting_styles = []
        for f in features_list:
            style = f.get("statistical_reporting_style", "None")
            if style and style.strip().lower() != "none":
                style_clean = style.strip()
                # 简单清洗聚合
                if "p-value" in style_clean.lower() or "p value" in style_clean.lower():
                    reporting_styles.append("Significance Testing (P-values)")
                elif "bootstrap" in style_clean.lower() or "mediation" in style_clean.lower():
                    reporting_styles.append("Mediation & Bootstrapping CIs")
                elif "bayesian" in style_clean.lower():
                    reporting_styles.append("Bayesian Analysis")
                else:
                    reporting_styles.append(style_clean[:40] + ("..." if len(style_clean) > 40 else ""))
            else:
                reporting_styles.append("None / Qualitative Description")
        
        top_reporting_styles = Counter(reporting_styles).most_common(5)

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

        # 7. 基于余弦相似度，在大样本真实发表池中计算与用户当前论文草稿最相似的 Top 3 篇已发表标杆
        most_similar_papers = []
        if user_draft_text and user_draft_text.strip():
            similarities = []
            for f in features_list:
                # 将标题、摘要、理论构念、分析工具与 OpenAlex 概念关键词融合，构建高维度语义向量文本
                semantic_elements = [
                    f.get("title", ""),
                    f.get("abstract", ""),
                    " ".join(f.get("theoretical_frameworks", [])),
                    " ".join(f.get("analytical_tools", [])),
                    " ".join(f.get("concepts", [])),
                ]
                paper_content = " ".join([elem for elem in semantic_elements if elem]).strip()
                sim = ProfileAggregator.calculate_cosine_similarity(user_draft_text, paper_content)
                similarities.append((sim, f))
            
            # 按相似度降序排列
            similarities.sort(key=lambda x: x[0], reverse=True)
            for sim, f in similarities[:3]:
                most_similar_papers.append({
                    "title": f.get("title", ""),
                    "similarity_score": round(sim, 3),
                    "method_category": f.get("method_category", ""),
                    "sample_description": f.get("sample_description", ""),
                    "theoretical_frameworks": f.get("theoretical_frameworks", []),
                    "analytical_tools": f.get("analytical_tools", []),
                    "novelty_highlight": f.get("novelty_highlight", ""),
                    "publication_year": f.get("publication_year", 0),
                    "cited_by_count": f.get("cited_by_count", 0),
                    "concepts": f.get("concepts", [])
                })

        aggregated_data = {
            "total_papers_analyzed": total_count,
            "method_distribution": method_distribution,
            "method_avg_citations": method_avg_citations,
            "sample_size_stats": sample_size_stats,
            "top_theories": [{"name": name, "count": cnt} for name, cnt in top_theories],
            "top_tools": [{"name": name, "count": cnt} for name, cnt in top_tools],
            "open_science_stats": open_science_stats,
            "top_reporting_styles": [{"style": style, "count": cnt} for style, cnt in top_reporting_styles],
            "representative_novelties": representative_novelties,
            "most_similar_papers": most_similar_papers,
        }

        logger.info(f"统计与对标相似度聚合计算完成。最相似文献匹配数: {len(most_similar_papers)}")
        return aggregated_data
