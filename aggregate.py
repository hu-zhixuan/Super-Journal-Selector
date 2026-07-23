import os
import json
import re
import logging
import hashlib
from typing import List, Dict, Any, Optional
from collections import Counter
import statistics
from datetime import datetime

# 须在 sentence_transformers 之前导入，确保 HF 镜像设置在 huggingface_hub 固化常量前生效
import network_config

# 尝试导入 sentence-transformers 以支持高精度语义向量计算
try:
    from sentence_transformers import SentenceTransformer, util
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

logger = logging.getLogger(__name__)


def clean_and_truncate_draft(text: str, max_chars: int = 150000) -> str:
    """
    清洗并截断用户草稿：
    1. 自动识别并切除文末无用的参考文献 (References / Bibliography / 参考文献)，腾出有效上下文。
    2. 如果文本仍超长，执行保留头部与尾部的智能截断，压缩中间内容。
    """
    if not text:
        return ""
    
    text_len = len(text)
    # 在文章后 40% 的位置查找参考文献标识进行截断
    search_start = int(text_len * 0.6)
    tail_text = text[search_start:]
    
    # 匹配各类参考文献标题的正则式
    ref_patterns = [
        r"\n\s*(?:==+\s*)?(?:References|Bibliography|REFERENCES|BIBLIOGRAPHY|参考文献)\s*(?:\n|=)",
        r"\n\s*\[\s*(?:References|REFERENCES)\s*\]\s*\n"
    ]
    
    cutoff_idx = -1
    for pat in ref_patterns:
        match = re.search(pat, tail_text)
        if match:
            cutoff_idx = search_start + match.start()
            break
            
    if cutoff_idx != -1:
        logger.info(f"✨ 智能草稿清洗：检测到文末参考文献章节，执行切除（切除位置: {cutoff_idx}/{text_len}，节省了 {text_len - cutoff_idx} 字符）")
        text = text[:cutoff_idx].strip()
    
    # 截断限制
    if len(text) > max_chars:
        half_limit = max_chars // 2
        logger.info(f"⚠️ 草稿文本超长 ({len(text)} 字符)，执行智能保留头部和尾部截断...")
        text = text[:half_limit] + "\n\n... [中间超长内容已被智能截断压缩以优化大模型上下文窗口] ...\n\n" + text[-half_limit:]
        
    return text


class ProfileAggregator:
    """
    层③：统计聚合层。不调用 LLM，纯用 Python 代码将百篇级文献的结构化特征进行高维聚合，计算各种分布与相似度对标。
    """

    @staticmethod
    def calculate_cosine_similarity(text1: str, text2: str) -> float:
        """
        BoW 词频余弦相似度计算，内置英文常用停用词（Stop Words）过滤降噪
        """
        # 内置中英文学术常见停用词表
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'else', 'when', 'at', 'by', 'from', 'in', 'on', 'to',
            'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'of', 'up', 'down', 'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'having', 'do', 'does', 'did', 'doing', 'can', 'could', 'should', 'would', 'will', 'i', 'me', 'my', 'we',
            'our', 'us', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'it', 'its', 'they', 'them', 'their', 'this',
            'that', 'these', 'those', 'which', 'who', 'whom', 'as', 'than', 'such', 'both', 'each', 'either', 'neither',
            '的', '了', '在', '是', '和', '与', '于', '对', '等', '及', '中', '或', '有', '为', '以', '上', '下'
        }
        
        def get_word_freq(t: str) -> Dict[str, int]:
            # 提取英文单词
            words = re.sub(r"[^\w\s]", " ", t.lower()).split()
            filtered_words = [w for w in words if w and w not in stop_words and not w.isdigit()]
            # 提取 CJK 中文字符 bigrams
            cjk_chars = re.findall(r'[\u4e00-\u9fff]', t)
            cjk_bigrams = [cjk_chars[i] + cjk_chars[i+1] for i in range(len(cjk_chars)-1)]
            filtered_cjk = [b for b in cjk_bigrams if not any(w in stop_words for w in b)]
            return Counter(filtered_words + filtered_cjk)

        freq1 = get_word_freq(text1)
        freq2 = get_word_freq(text2)
        
        all_words = set(freq1.keys()).union(set(freq2.keys()))
        if not all_words:
            return 0.0
            
        dot_product = sum(freq1.get(w, 0) * freq2.get(w, 0) for w in all_words)
        magnitude1 = sum(v ** 2 for v in freq1.values()) ** 0.5
        magnitude2 = sum(v ** 2 for v in freq2.values()) ** 0.5
        
        if not magnitude1 or not magnitude2:
            return 0.0
        return dot_product / (magnitude1 * magnitude2)

    def aggregate(self, features_list: List[Dict[str, Any]], user_draft_text: Optional[str] = None) -> Dict[str, Any]:
        """
        多维度聚合特征，集成 Sentence-Transformers 个性化相似度对标，以及代码打分排序的 Top 5 推荐引用文献生成。
        """
        total_count = len(features_list)
        if total_count == 0:
            return {}

        current_year = datetime.now().year

        # 1. 研究方法范式分布 (Method Categories)
        method_counts = Counter([f.get("method_category") for f in features_list])
        method_distribution = {
            m: {
                "count": cnt,
                "percentage": round((cnt / total_count) * 100, 1)
            }
            for m, cnt in method_counts.most_common()
        }

        # 2. 统计各方法的平均被引次数 (Method Citation Impact)
        method_citations: Dict[str, List[int]] = {}
        for f in features_list:
            m = f.get("method_category", "Unknown")
            citations = f.get("cited_by_count", 0)
            method_citations.setdefault(m, []).append(citations)
            
        method_avg_citations = {
            m: round(statistics.mean(cits), 1) if cits else 0
            for m, cits in method_citations.items()
        }

        # 3. 定量样本量范围的统计属性 (Sample Size Stats)
        sample_sizes = [
            f.get("sample_size_approx", -1)
            for f in features_list
            if f.get("sample_size_approx", -1) > 0
        ]
        sample_size_stats = {
            "min": min(sample_sizes) if sample_sizes else "N/A",
            "median": int(statistics.median(sample_sizes)) if sample_sizes else "N/A",
            "max": max(sample_sizes) if sample_sizes else "N/A"
        }

        # 4. 高频理论框架/构念排行 (Top 12)
        all_theories = []
        for f in features_list:
            for theory in f.get("theoretical_frameworks", []):
                cleaned_theory = theory.strip()
                if cleaned_theory and len(cleaned_theory) > 1:
                    all_theories.append(cleaned_theory)
        top_theories = Counter(all_theories).most_common(12)

        # 5. 高频分析统计工具排行 (Top 12)
        all_tools = []
        for f in features_list:
            for tool in f.get("analytical_tools", []):
                cleaned_tool = tool.strip()
                if cleaned_tool and len(cleaned_tool) > 1:
                    all_tools.append(cleaned_tool)
        top_tools = Counter(all_tools).most_common(12)

        # 5b. 开源科学实践聚合统计
        open_science_counter = Counter()
        for f in features_list:
            practices = f.get("open_science_practices", [])
            if not practices:
                open_science_counter["None"] += 1
            else:
                for p in practices:
                    cleaned_p = p.strip()
                    if not cleaned_p or cleaned_p.lower() == "none":
                        open_science_counter["None"] += 1
                    else:
                        title_p = cleaned_p.title()
                        if "Data" in title_p:
                            title_p = "Open Data"
                        elif "Code" in title_p:
                            title_p = "Open Code"
                        elif "Pre" in title_p or "Reg" in title_p:
                            title_p = "Preregistration"
                        open_science_counter[title_p] += 1
        
        open_science_stats = {
            p_name: {
                "count": cnt,
                "percentage": round((cnt / total_count) * 100, 1)
            }
            for p_name, cnt in open_science_counter.most_common()
        }

        # 5c. 统计汇报风格统计
        reporting_styles = []
        for f in features_list:
            style = f.get("statistical_reporting_style", "None")
            if style and style.strip().lower() != "none":
                style_clean = style.strip()
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

        # 6. 被引频次最高的代表性论文 (Top 5)
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

        # 7. 动态双表计算 (用户草稿传入时生效)
        most_similar_papers = []
        recommended_references = []

        if user_draft_text and user_draft_text.strip():
            similarities = []
            use_semantic = HAS_SENTENCE_TRANSFORMERS
            
            # 使用 sentence-transformers 计算高维编码
            if use_semantic:
                try:
                    # 本地缓存（ModelScope/自定义路径）优先，避免 huggingface 下载卡死
                    model_path = network_config.resolve_embedding_model_path('all-MiniLM-L6-v2')
                    logger.info(f"正在初始化 SentenceTransformer ('{model_path}') 提取语义特征对标向量...")
                    model = SentenceTransformer(model_path)
                    draft_emb = model.encode(user_draft_text, convert_to_tensor=True)
                    
                    from llm_client import EMBEDDING_MODEL_NAME
                    model_slug = "".join(c if c.isalnum() else "_" for c in EMBEDDING_MODEL_NAME.lower())
                    os.makedirs("cache", exist_ok=True)
                    for f in features_list:
                        paper_id_clean = f.get("paper_id") or f.get("title", "")
                        paper_hash = hashlib.md5(paper_id_clean.encode("utf-8")).hexdigest()[:12]
                        emb_cache_file = os.path.join("cache", f"embedding_{paper_hash}_{model_slug}.json")
                        
                        semantic_elements = [
                            f.get("title", ""),
                            f.get("abstract", ""),
                            " ".join(f.get("theoretical_frameworks", [])),
                            " ".join(f.get("analytical_tools", [])),
                            " ".join(f.get("concepts", [])),
                        ]
                        content = " ".join([elem for elem in semantic_elements if elem]).strip()
                        
                        paper_emb = None
                        if os.path.exists(emb_cache_file):
                            try:
                                with open(emb_cache_file, "r") as fec:
                                    paper_emb_list = json.load(fec)
                                import torch
                                paper_emb = torch.tensor(paper_emb_list).to(draft_emb.device)
                            except Exception as e_emb_c:
                                logger.warning(f"读取 embedding 缓存失败: {e_emb_c}")
                                
                        if paper_emb is None:
                            paper_emb_tensor = model.encode(content, convert_to_tensor=True)
                            paper_emb = paper_emb_tensor
                            try:
                                with open(emb_cache_file, "w") as fec:
                                    json.dump(paper_emb_tensor.tolist(), fec)
                            except Exception as e_w_emb_c:
                                logger.warning(f"写入 embedding 缓存失败: {e_w_emb_c}")
                                
                        sim = float(util.cos_sim(draft_emb, paper_emb)[0][0])
                        similarities.append((sim, f))
                except Exception as e_sem:
                    logger.warning(f"使用 sentence-transformers 对标异常，将降级为词频余弦对标: {e_sem}")
                    use_semantic = False
            
            if not use_semantic:
                for f in features_list:
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
            
            # --- 列表 1：诊断差距的最相似 Top 3 标杆文献 (Strictly sorted by similarity) ---
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

            # --- 列表 2：最强学术增补推荐引用 Top 5 文献 (Code-Driven Weighted Formula) ---
            max_citations = max([f.get("cited_by_count", 0) for f in features_list]) if features_list else 0
            
            scored_papers = []
            draft_words = set(re.sub(r"\W+", " ", user_draft_text.lower()).split())
            
            for sim, f in similarities:
                # 1. 被引分 (max-min scale)
                cit_score = (f.get("cited_by_count", 0) / max_citations) if max_citations > 0 else 0.0
                
                # 2. 新鲜度分
                diff = current_year - f.get("publication_year", current_year)
                recency_score = max(0.2, 1.0 - diff * 0.2)
                
                # 3. 关键词重合度分
                paper_words = set()
                for c in f.get("concepts", []):
                    paper_words.update(re.sub(r"\W+", " ", c.lower()).split())
                for t in f.get("theoretical_frameworks", []):
                    paper_words.update(re.sub(r"\W+", " ", t.lower()).split())
                overlap = len(draft_words.intersection(paper_words))
                keyword_overlap_score = min(overlap / 5.0, 1.0)
                
                # 加权打分公式
                final_score = 0.45 * sim + 0.25 * cit_score + 0.20 * recency_score + 0.10 * keyword_overlap_score
                scored_papers.append((final_score, sim, f))
            
            # 按最终得分降序排序，取前 5 篇作为推荐引用
            scored_papers.sort(key=lambda x: x[0], reverse=True)
            for score, sim, f in scored_papers[:5]:
                recommended_references.append({
                    "title": f.get("title", ""),
                    "final_score": round(score, 3),
                    "similarity_score": round(sim, 3),
                    "publication_year": f.get("publication_year", current_year),
                    "cited_by_count": f.get("cited_by_count", 0),
                    "theoretical_frameworks": f.get("theoretical_frameworks", []),
                    "analytical_tools": f.get("analytical_tools", []),
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
            "recommended_references": recommended_references
        }

        logger.info(f"统计与对标相似度聚合计算完成。对标标杆文献: {len(most_similar_papers)} 篇，推荐引用文献: {len(recommended_references)} 篇。")
        return aggregated_data
