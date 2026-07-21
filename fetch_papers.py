import os
import json
import re
import requests
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PaperRecord:
    id: str
    doi: str
    title: str
    abstract: str
    publication_year: int
    cited_by_count: int
    source_title: str
    concepts: List[str] = None

    def __post_init__(self):
        if self.concepts is None:
            self.concepts = []

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def deduplicate_papers(papers: List[PaperRecord]) -> List[PaperRecord]:
    """
    强物理去重逻辑：
    1. DOI 去重
    2. ID 去重
    3. 标题归一化去重（去除所有标点、小写、除空格）
    4. 年份 + 标题前 60 字符对齐去重
    """
    seen_dois = set()
    seen_ids = set()
    seen_normalized_titles = set()
    seen_year_titles = set()

    deduped = []
    for p in papers:
        # 标题归一化
        normalized_title = re.sub(r"\W+", " ", p.title.lower()).strip()
        year_title_prefix = f"{p.publication_year}_{normalized_title[:60]}"

        # Check DOI
        if p.doi and p.doi.strip():
            clean_doi = p.doi.strip().lower()
            if clean_doi in seen_dois:
                continue
            seen_dois.add(clean_doi)

        # Check ID
        if p.id and p.id.strip():
            clean_id = p.id.strip().lower()
            if clean_id in seen_ids:
                continue
            seen_ids.add(clean_id)

        # Check Normalized Title
        if normalized_title in seen_normalized_titles:
            continue
        seen_normalized_titles.add(normalized_title)

        # Check Year + Title prefix (first 60 chars)
        if year_title_prefix in seen_year_titles:
            continue
        seen_year_titles.add(year_title_prefix)

        deduped.append(p)
    return deduped


class OpenAlexFetcher:
    """
    层①：数据抓取层。封装 OpenAlex 开放 API（无需 key），获取目标期刊最近N年的论文摘要文本并结构化。
    最新升级：支持双通道（热门高引 + 主题词检索）动态检索和 Europe PMC 检索的多路召回与 Fallback 降级。
    """

    BASE_URL = "https://api.openalex.org"

    def __init__(self, email: Optional[str] = None):
        self.email = email or os.getenv("OPENALEX_EMAIL")
        self.headers = {
            "User-Agent": f"JournalProfileSkill/1.0 ({self.email or 'mailto:researcher@example.com'})"
        }

    def resolve_journal_source(self, journal_name: str) -> Optional[Dict[str, Any]]:
        """
        匹配或检索 OpenAlex 中的 Journal Source ID。
        """
        url = f"{self.BASE_URL}/sources"
        params = {"search": journal_name, "per-page": 5}
        try:
            resp = requests.get(url, params=params, proxies={"http": None, "https": None}, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if not results:
                logger.error(f"未在 OpenAlex 中找到期刊: '{journal_name}'")
                return None

            def clean_name(n: str) -> str:
                return "".join(c for c in n.lower() if c.isalnum()).strip()

            target_clean = clean_name(journal_name)
            current_year = datetime.now().year
            candidates = []

            for res in results:
                display_name = res.get("display_name", "")
                res_clean = clean_name(display_name)
                
                counts_by_year = res.get("counts_by_year", [])
                recent_works_sum = sum(
                    c.get("works_count", 0)
                    for c in counts_by_year
                    if c.get("year", 0) >= (current_year - 2)
                )
                
                is_name_match = (target_clean in res_clean) or (res_clean in target_clean)
                candidates.append((res, recent_works_sum, is_name_match))

            matched_candidates = [c for c in candidates if c[2]]
            if matched_candidates:
                matched_candidates.sort(key=lambda x: x[1], reverse=True)
                best_match = matched_candidates[0][0]
                logger.info(f"匹配到活跃期刊: '{best_match.get('display_name')}' (ID: {best_match.get('id')}), 近3年发文: {matched_candidates[0][1]} 篇")
                return best_match

            candidates.sort(key=lambda x: x[1], reverse=True)
            best_match = candidates[0][0]
            logger.info(f"采用相似推荐最活跃期刊: '{best_match.get('display_name')}' (ID: {best_match.get('id')}), 近3年发文: {candidates[0][1]} 篇")
            return best_match

        except Exception as e:
            logger.error(f"检索期刊 Source ID 异常: {str(e)}")
            return None

    @staticmethod
    def _reconstruct_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> str:
        """
        OpenAlex 返回的摘要倒排索引还原为连贯英文段落。
        """
        if not inverted_index or not isinstance(inverted_index, dict):
            return ""
        word_list: List[Dict[str, Any]] = []
        for word, pos_list in inverted_index.items():
            for pos in pos_list:
                word_list.append({"word": word, "pos": pos})
        word_list.sort(key=lambda x: x["pos"])
        return " ".join([item["word"] for item in word_list])

    def fetch_recent_papers(
        self, journal_name: str, years: int = 3, max_papers: int = 100, search_query: Optional[str] = None
    ) -> Tuple[List[PaperRecord], Dict[str, Any]]:
        """
        核心方法：获取带摘要大样本。集成本地 Caching、双通道动态配比检索和 Europe PMC 多路 fallback。
        """
        current_year = datetime.now().year
        min_year = current_year - years

        # 1. 本地缓存读取检查
        q_str = search_query if search_query else "none"
        query_hash = hashlib.md5(q_str.encode("utf-8")).hexdigest()[:8]
        cache_dir = "cache"
        os.makedirs(cache_dir, exist_ok=True)
        journal_slug = "".join(c if c.isalnum() else "_" for c in journal_name.lower().strip())
        cache_file = os.path.join(cache_dir, f"papers_{journal_slug}_{years}_{max_papers}_{query_hash}.json")

        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as fc:
                    cached_data = json.load(fc)
                cached_papers = [PaperRecord(**p) for p in cached_data.get("papers", [])]
                cached_meta = cached_data.get("journal_metadata", {})
                if cached_papers:
                    logger.info(f"✨ 命中本地缓存，成功加载 {len(cached_papers)} 篇论文及元数据: {cache_file}")
                    return cached_papers, cached_meta
            except Exception as e_cache:
                logger.warning(f"读取本地缓存文件失败 (将重新抓取): {e_cache}")

        # 2. 定位期刊元数据
        source_info = self.resolve_journal_source(journal_name)
        if not source_info:
            raise ValueError(f"无法定位期刊 Source: {journal_name}")

        source_id = source_info["id"]
        source_display_name = source_info["display_name"]
        summary_stats = source_info.get("summary_stats", {})
        x_concepts = source_info.get("x_concepts", [])

        # 3. 加载本地分区（防呆清洗）
        local_partition = {
            "jcr_zone": "未知",
            "cas_zone": "未知",
            "cas_sub_categories": "N/A",
            "is_top": "未知"
        }
        try:
            partitions_path = os.path.join(os.path.dirname(__file__), "journal_partitions.json")
            if os.path.exists(partitions_path):
                with open(partitions_path, "r", encoding="utf-8") as f:
                    raw_db = json.load(f)
                
                # 对加载的数据做首尾空格和大小写清洗
                db = {}
                for k, v in raw_db.items():
                    db[k.strip().lower()] = {
                        sub_k.strip().lower(): sub_v.strip()
                        for sub_k, sub_v in v.items()
                    }

                q_clean = journal_name.lower().strip()
                match_found = False
                for k, v in db.items():
                    if (k in q_clean) or (q_clean in k) or (source_display_name.lower() in k) or (k in source_display_name.lower()):
                        local_partition = {
                            "jcr_zone": v.get("jcr_zone", "未知"),
                            "cas_zone": v.get("cas_zone", "未知"),
                            "cas_sub_categories": v.get("cas_sub_categories", "N/A"),
                            "is_top": v.get("is_top", "未知")
                        }
                        match_found = True
                        break

                if not match_found:
                    try:
                        from llm_client import LLMClient
                        llm = LLMClient()
                        prompt = f"""
Given the academic journal name: "{source_display_name}" (Estimated Impact Factor: {summary_stats.get("2yr_mean_citedness", "N/A")}), please estimate its standard academic rankings.
Output strictly in JSON format matching this schema:
{{
    "jcr_zone": "JCR zone (e.g. Q1, Q2, Q3, Q4)",
    "cas_zone": "Chinese Academy of Sciences partition zone (e.g. 1区, 2区, 3区, 4区)",
    "cas_sub_categories": "Sub-disciplines/Categories in Chinese (e.g. 物理:多学科, 计算机:信息系统)",
    "is_top": "Whether it is a CAS Top Journal (e.g. 是 (Top 期刊), 否)"
}}
Only return JSON.
"""
                        resp_json = llm.call_json(prompt=prompt)
                        local_partition = {
                            "jcr_zone": resp_json.get("jcr_zone", "未知"),
                            "cas_zone": resp_json.get("cas_zone", "未知"),
                            "cas_sub_categories": resp_json.get("cas_sub_categories", "未知"),
                            "is_top": resp_json.get("is_top", "未知")
                        }
                    except Exception as e_llm:
                        logger.warning(f"智能大模型估算分区出错: {e_llm}")
        except Exception as e:
            logger.warning(f"加载本地分区数据字典异常: {e}")

        journal_metadata = {
            "display_name": source_display_name,
            "issn": source_info.get("issn", ["Unknown"])[0] if source_info.get("issn") else "Unknown",
            "h_index": summary_stats.get("h_index", "N/A"),
            "estimated_impact_factor": summary_stats.get("2yr_mean_citedness", "N/A"),
            "works_count": source_info.get("works_count", "N/A"),
            "cited_by_count": source_info.get("cited_by_count", "N/A"),
            "categories": [c.get("name") for c in x_concepts[:6] if c.get("name")],
            "jcr_zone": local_partition.get("jcr_zone"),
            "cas_zone": local_partition.get("cas_zone"),
            "cas_sub_categories": local_partition.get("cas_sub_categories"),
            "is_top": local_partition.get("is_top")
        }

        # 4. 双通道配比逻辑
        papers_channel_a: List[PaperRecord] = []
        papers_channel_b: List[PaperRecord] = []

        if not search_query or not search_query.strip():
            # 100% 走通道 A (高引热门底图)
            target_a = max_papers
            target_b = 0
            logger.info("未提供论文草稿，系统开启 100% 期刊热门高引文献检索。")
        else:
            # 默认配比：A 通道 60%，B 通道 40%
            target_b = int(max_papers * 0.40)
            target_a = max_papers - target_b
            logger.info(f"开启双通道动态对标：通道 A (高引基准) 计划 {target_a} 篇，通道 B (主题匹配) 计划 {target_b} 篇。检索词: {search_query}")

        # ===== 通道 A：热门高引文献检索 =====
        if target_a > 0:
            filter_str = f"primary_location.source.id:{source_id},publication_year:>{min_year},has_abstract:true"
            url = f"{self.BASE_URL}/works"
            params = {
                "filter": filter_str,
                "sort": "cited_by_count:desc",
                "per-page": min(target_a * 2, 200),
            }
            try:
                resp = requests.get(url, params=params, headers=self.headers, proxies={"http": None, "https": None}, timeout=25)
                resp.raise_for_status()
                results = resp.json().get("results", [])
                for item in results:
                    abstract_text = self._reconstruct_abstract(item.get("abstract_inverted_index"))
                    if len(abstract_text.split()) < 40:
                        continue
                    concept_names = [c.get("display_name") for c in item.get("concepts", [])[:6] if c.get("display_name")]
                    paper = PaperRecord(
                        id=item.get("id", ""),
                        doi=item.get("doi", "") or "",
                        title=item.get("title", "Untitled"),
                        abstract=abstract_text,
                        publication_year=item.get("publication_year", current_year),
                        cited_by_count=item.get("cited_by_count", 0),
                        source_title=source_display_name,
                        concepts=concept_names,
                    )
                    papers_channel_a.append(paper)
                    if len(papers_channel_a) >= target_a:
                        break
                logger.info(f"通道 A (高引热门) 实际获取: {len(papers_channel_a)} 篇。")
            except Exception as e_a:
                logger.error(f"通道 A 获取失败: {e_a}")

        # ===== 通道 B：多路主题文献检索与 Fallback =====
        if target_b > 0:
            # 路线 B1：OpenAlex 期刊内 search 检索
            filter_str = f"primary_location.source.id:{source_id},publication_year:>{min_year},has_abstract:true"
            url = f"{self.BASE_URL}/works"
            params = {
                "filter": filter_str,
                "search": search_query,
                "per-page": min(target_b * 2, 100),
            }
            try:
                resp = requests.get(url, params=params, headers=self.headers, proxies={"http": None, "https": None}, timeout=25)
                resp.raise_for_status()
                results = resp.json().get("results", [])
                for item in results:
                    abstract_text = self._reconstruct_abstract(item.get("abstract_inverted_index"))
                    if len(abstract_text.split()) < 40:
                        continue
                    concept_names = [c.get("display_name") for c in item.get("concepts", [])[:6] if c.get("display_name")]
                    paper = PaperRecord(
                        id=item.get("id", ""),
                        doi=item.get("doi", "") or "",
                        title=item.get("title", "Untitled"),
                        abstract=abstract_text,
                        publication_year=item.get("publication_year", current_year),
                        cited_by_count=item.get("cited_by_count", 0),
                        source_title=source_display_name,
                        concepts=concept_names,
                    )
                    papers_channel_b.append(paper)
                logger.info(f"路线 B1 (OpenAlex Search) 获取: {len(papers_channel_b)} 篇。")
            except Exception as e_b1:
                logger.warning(f"路线 B1 (OpenAlex Search) 异常: {e_b1}")

            # 路线 B2：若召回不足，在 OpenAlex 期刊内尝试概念标题模糊词匹配 (作为 B1 Fallback)
            if len(papers_channel_b) < target_b:
                logger.info("路线 B1 召回不足，触发路线 B2 (OpenAlex 标题短语检索) 进行补充...")
                # 将关键词拆开，通过 title.search 检索
                params["filter"] = f"{filter_str},title.search:{search_query}"
                if "search" in params:
                    del params["search"]
                try:
                    resp = requests.get(url, params=params, headers=self.headers, proxies={"http": None, "https": None}, timeout=25)
                    resp.raise_for_status()
                    results = resp.json().get("results", [])
                    b2_count = 0
                    for item in results:
                        abstract_text = self._reconstruct_abstract(item.get("abstract_inverted_index"))
                        if len(abstract_text.split()) < 40:
                            continue
                        concept_names = [c.get("display_name") for c in item.get("concepts", [])[:6] if c.get("display_name")]
                        paper = PaperRecord(
                            id=item.get("id", ""),
                            doi=item.get("doi", "") or "",
                            title=item.get("title", "Untitled"),
                            abstract=abstract_text,
                            publication_year=item.get("publication_year", current_year),
                            cited_by_count=item.get("cited_by_count", 0),
                            source_title=source_display_name,
                            concepts=concept_names,
                        )
                        papers_channel_b.append(paper)
                        b2_count += 1
                    logger.info(f"路线 B2 (Title Search) 补齐了 {b2_count} 篇，累计 B 通道达到: {len(papers_channel_b)} 篇。")
                except Exception as e_b2:
                    logger.warning(f"路线 B2 检索异常: {e_b2}")

            # 路线 B3：若依然不足，触发 Europe PMC 进行跨源主题检索 (作为 B2 Fallback)
            if len(papers_channel_b) < target_b:
                logger.info("路线 B2 召回仍不足，触发路线 B3 (Europe PMC 主题检索) 进行补充...")
                try:
                    # 组合 Europe PMC 检索语句
                    epmc_query = f'JOURNAL:"{source_display_name}" AND ({search_query}) AND PUB_YEAR:[{min_year} TO {current_year}] AND HAS_ABSTRACT:Y'
                    epmc_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
                    epmc_params = {
                        "query": epmc_query,
                        "format": "json",
                        "pageSize": min(target_b * 2, 50),
                        "resultType": "core",
                        "sort": "CITED desc"
                    }
                    resp_epmc = requests.get(epmc_url, params=epmc_params, proxies={"http": None, "https": None}, timeout=20)
                    if resp_epmc.status_code == 200:
                        epmc_results = resp_epmc.json().get("resultList", {}).get("result", [])
                        b3_count = 0
                        for item in epmc_results:
                            title = item.get("title", "Untitled")
                            abstract = item.get("abstractText", "")
                            if len(abstract.split()) < 40:
                                continue
                            keywords = item.get("keywordList", {}).get("keyword", [])
                            paper = PaperRecord(
                                id=item.get("id", ""),
                                doi=item.get("doi", "") or "",
                                title=title,
                                abstract=abstract,
                                publication_year=int(item.get("pubYear", current_year)),
                                cited_by_count=item.get("citedByCount", 0),
                                source_title=source_display_name,
                                concepts=keywords
                            )
                            papers_channel_b.append(paper)
                            b3_count += 1
                        logger.info(f"路线 B3 (Europe PMC) 补齐了 {b3_count} 篇，累计 B 通道达到: {len(papers_channel_b)} 篇。")
                except Exception as e_b3:
                    logger.warning(f"路线 B3 跨源检索异常: {e_b3}")

        # ===== 动态合并与通道 B 召回不足补齐机制 =====
        # 合并去重
        combined_papers = papers_channel_a + papers_channel_b
        deduped_papers = deduplicate_papers(combined_papers)

        # 统计去重后各通道各自的有效供给数 (通过 ID 追溯)
        channel_b_ids = {p.id for p in papers_channel_b if p.id}
        deduped_b = [p for p in deduped_papers if p.id in channel_b_ids]
        deduped_a = [p for p in deduped_papers if p.id not in channel_b_ids]

        logger.info(f"物理去重后：有效通用样本 {len(deduped_a)} 篇，有效相关主题样本 {len(deduped_b)} 篇。")

        # 检查通道 B 实际贡献是否达标，如果不达标或总数不足，使用通道 A 的剩余热门文章进行垫底补齐
        total_valid = len(deduped_papers)
        if total_valid < max_papers and len(deduped_a) < (max_papers - len(deduped_b)):
            # 如果去重后总量依然不够，且有检索接口完全失灵的情况，尝试降级做纯热门召回填补空缺
            logger.warning(f"文献样本库总数 ({total_valid} 篇) 仍未达到计划的 {max_papers} 篇。触发全热门召回扩容填补...")
            # 拉大分页抓取热门
            url = f"{self.BASE_URL}/works"
            filter_str = f"primary_location.source.id:{source_id},publication_year:>{min_year},has_abstract:true"
            params = {
                "filter": filter_str,
                "sort": "cited_by_count:desc",
                "per-page": 200,
            }
            try:
                resp = requests.get(url, params=params, headers=self.headers, proxies={"http": None, "https": None}, timeout=25)
                if resp.status_code == 200:
                    results = resp.json().get("results", [])
                    for item in results:
                        abstract_text = self._reconstruct_abstract(item.get("abstract_inverted_index"))
                        if len(abstract_text.split()) < 40:
                            continue
                        concept_names = [c.get("display_name") for c in item.get("concepts", [])[:6] if c.get("display_name")]
                        paper = PaperRecord(
                            id=item.get("id", ""),
                            doi=item.get("doi", "") or "",
                            title=item.get("title", "Untitled"),
                            abstract=abstract_text,
                            publication_year=item.get("publication_year", current_year),
                            cited_by_count=item.get("cited_by_count", 0),
                            source_title=source_display_name,
                            concepts=concept_names,
                        )
                        deduped_papers.append(paper)
                    deduped_papers = deduplicate_papers(deduped_papers)
            except Exception as e_pad:
                logger.warning(f"热门垫底补齐抓取异常: {e_pad}")

        # 截断限制
        final_papers = deduped_papers[:max_papers]
        logger.info(f"最终输出去重对齐后的精选大样本库共: {len(final_papers)} 篇。")

        if not final_papers:
            raise ValueError(f"无法为期刊 '{source_display_name}' 抓取到任何有效的带摘要论文样本")

        # 5. 保存至本地 Caching 目录以供复用
        try:
            cache_data = {
                "papers": [p.to_dict() for p in final_papers],
                "journal_metadata": journal_metadata
            }
            with open(cache_file, "w", encoding="utf-8") as fc:
                json.dump(cache_data, fc, ensure_ascii=False, indent=2)
            logger.info(f"💾 文献抓取成功落盘缓存：{cache_file}")
        except Exception as e_w_cache:
            logger.warning(f"写入缓存文件异常: {e_w_cache}")

        return final_papers, journal_metadata
