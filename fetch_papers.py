import hashlib
import json
import logging
import os
import random
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# ============ 网络健壮性配置 ============
try:
    import urllib.request
except ImportError:  # pragma: no cover
    pass

# 每次对 OpenAlex 请求前的串行礼貌延迟（秒），配合 mailto 进入 Polite Pool 规避 429 冷启动
_OA_PER_REQUEST_DELAY = float(os.getenv("OA_REQUEST_DELAY", "2.5"))
# OpenAlex 重试次数与初始退避基数（指数退避：2.5 -> 5 -> 10s ...）
_OA_MAX_RETRIES = int(os.getenv("OA_MAX_RETRIES", "5"))
# 串行延迟锁（进程级）
_sleep_lock = random.Random()

# arXiv API 兜底抓取（主源 OpenAlex 不可用时使用）
_ARXIV_BASE = os.getenv("ARXIV_BASE", "http://export.arxiv.org/api/query")
_ARXIV_TOPICS = [  # 与金融方法学语义强相关的固定主题
    "cat:q-fin.TR", "cat:q-fin.ST",
    "ti:machine learning AND abs:trading OR abs:stock OR abs:asset",
    "ti:deep learning AND abs:portfolio OR abs:return OR abs:market",
    "ti:reinforcement learning AND abs:trading OR abs:portfolio",
    "ti:forecasting AND abs:stock OR abs:volatility OR abs:return",
]
# 若某次抓取启用了 arXiv 兜底，置 True 以跳过主源依赖的通道统计/热门扩容
_arxiv_source_active = False


@dataclass
class PaperRecord:
    id: str
    doi: str
    title: str
    abstract: str
    publication_year: int
    cited_by_count: int
    source_title: str
    concepts: List[str] = field(default_factory=list)

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


# 显式禁用 requests 的系统代理（规避代理工具对学术 API 的 SSL 连接干扰）；
# 标注为 Any 以兼容 requests 类型存根（其声明的 proxies 不接受 None 值）
_NO_SYSTEM_PROXY: Any = {"http": None, "https": None}


class OpenAlexFetcher:
    """
    层①：数据抓取层。封装 OpenAlex 开放 API（无需 key），获取目标期刊最近N年的论文摘要文本并结构化。
    最新升级：支持双通道（热门高引 + 主题词检索）动态检索和 Europe PMC 检索的多路召回与 Fallback 降级。
    """

    BASE_URL = "https://api.openalex.org"

    def __init__(self, email: Optional[str] = None):
        self.email = email or os.getenv("OPENALEX_EMAIL")
        if not self.email or self.email == "your_email@example.com":
            logger.info(
                "未在 .env 中检测到有效的 OPENALEX_EMAIL。建议配置真实邮箱以加入 OpenAlex Polite Pool，享受更高频的学术检索响应速度。"
            )
            self.email = "researcher@example.com"
        self.headers = {"User-Agent": f"JournalProfileSkill/1.0 (mailto:{self.email})"}

    @staticmethod
    def _polite_wait() -> None:
        """进程级串行礼貌延迟：所有 OpenAlex 请求间至少间隔 OA_REQUEST_DELAY 秒，
        显著降低 429 冷启动命中率（OpenAlex 官方建议 100ms-3s 礼貌间隔）。"""
        time.sleep(_OA_PER_REQUEST_DELAY)

    @staticmethod
    def _oa_request(url: str, params: Dict[str, Any], timeout: int = 25) -> requests.Response:
        """OpenAlex 请求统一出口：串行延迟 + 指数退避重试（附带 mailto 进入礼貌池）。"""
        for attempt in range(_OA_MAX_RETRIES):
            OpenAlexFetcher._polite_wait()
            try:
                resp = requests.get(
                    url, params=params, headers=OpenAlexFetcher._current_headers(), proxies=_NO_SYSTEM_PROXY, timeout=timeout
                )
                if resp.status_code == 429:
                    wait = _OA_PER_REQUEST_DELAY * (2 ** attempt)
                    logger.warning(f"OpenAlex 429 限流 (第 {attempt+1}/{_OA_MAX_RETRIES} 次)，退避等待 {wait:.1f}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp
            except requests.exceptions.HTTPError as e_http:
                if e_http.response is not None and e_http.response.status_code == 429 and attempt < _OA_MAX_RETRIES - 1:
                    wait = _OA_PER_REQUEST_DELAY * (2 ** attempt)
                    logger.warning(f"OpenAlex 429 限流异常 (第 {attempt+1}/{_OA_MAX_RETRIES} 次)，退避等待 {wait:.1f}s...")
                    time.sleep(wait)
                    continue
                raise
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e_net:
                if attempt < _OA_MAX_RETRIES - 1:
                    wait = _OA_PER_REQUEST_DELAY * (2 ** attempt)
                    logger.warning(f"OpenAlex 网络异常 (第 {attempt+1}/{_OA_MAX_RETRIES} 次): {e_net}，等待 {wait:.1f}s 重试...")
                    time.sleep(wait)
                    continue
                raise
        # 理论不可达，防御性抛出
        resp = requests.Response()
        resp.status_code = 429
        resp.url = url
        raise requests.exceptions.HTTPError(f"OpenAlex 请求持续失败 ({url})", response=resp)

    @staticmethod
    def _current_headers() -> Dict[str, str]:
        return {"User-Agent": "JournalProfileSkill/1.0 (mailto:researcher@example.com)"}

    @classmethod
    def _arxiv_fallback(cls, journal_name: str, years: int, max_papers: int, search_query: Optional[str] = None) -> List[PaperRecord]:
        """主源 OpenAlex 不可用时，从 arXiv 高质量金融主题抓取带摘要论文作为兜底样本。

        注意：arXiv 不是该期刊的精确题录，而是方法与主题相近的学术前沿样本。
        返回的 PaperRecord.source_title 标注为 'arXiv (近似主题兜底)'，确保下游明白数据来源。
        """
        import xml.etree.ElementTree as ET

        ns = {"a": "http://www.w3.org/2005/Atom"}
        topics = _ARXIV_TOPICS
        if search_query:
            topics = [f"all:{search_query}"] + topics  # 用草稿关键词优先检索
        seen: set = set()
        papers: List[PaperRecord] = []
        current_year = datetime.now().year
        for topic in topics[:8]:
            if len(papers) >= max_papers:
                break
            url = f"{_ARXIV_BASE}?search_query={urllib.parse.quote(topic)}&start=0&max_results=40&sortBy=submittedDate&sortOrder=descending"
            try:
                xml_text = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "JournalProfileSkill/1.0 (mailto:researcher@example.com)"}), timeout=40).read().decode("utf-8", "replace")
                root = ET.fromstring(xml_text)
                for e in root.findall("a:entry", ns):
                    title = re.sub(r"\s+", " ", e.findtext("a:title", "", ns) or "").strip()
                    summary = re.sub(r"\s+", " ", e.findtext("a:summary", "", ns) or "").strip()
                    key = re.sub(r"[^a-z0-9]+", "", title.lower())
                    if not title or key in seen or len(summary.split()) < 40:
                        continue
                    seen.add(key)
                    # 用 id+title 构造唯一 ID，杜绝哈希冲突
                    arx_id = ""
                    for link_el in e.findall("a:link", ns):
                        if link_el.get("title") == "abs":
                            arx_id = (link_el.get("href") or "").split("/abs/")[-1]
                            break
                    unique_id = f"arxiv:{arx_id or title}"
                    year = current_year
                    try:
                        year = int((e.findtext("a:published", "") or "")[:4]) or current_year
                    except ValueError:
                        pass
                    papers.append(PaperRecord(
                        id=unique_id, doi="", title=title, abstract=summary,
                        publication_year=year, cited_by_count=0,
                        source_title=f"arXiv {journal_name} (近似主题兜底)",
                        concepts=["arXiv"],  # 语义占位
                    ))
                    if len(papers) >= max_papers:
                        break
                logger.info(f"arXiv 兜底检索 [{topic[:45]}]：累计 {len(papers)} 篇")
            except Exception as e_arx:
                logger.warning(f"arXiv 兜底检索失败 [{topic[:45]}]: {e_arx}")
            time.sleep(2.0)  # arXiv 礼貌延迟
        logger.info(f"arXiv 兜底总计获取 {len(papers)} 篇（来源标注为近似主题）")
        return papers

    def resolve_journal_source(self, journal_name: str) -> Optional[Dict[str, Any]]:
        """
        匹配或检索 OpenAlex 中的 Journal Source ID。
        """
        url = f"{self.BASE_URL}/sources"
        params: Dict[str, Any] = {"search": journal_name, "per-page": 5}
        try:
            resp = self._oa_request(url, params, timeout=15)
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
                    c.get("works_count", 0) for c in counts_by_year if c.get("year", 0) >= (current_year - 2)
                )

                is_name_match = (target_clean in res_clean) or (res_clean in target_clean)
                candidates.append((res, recent_works_sum, is_name_match))

            matched_candidates = [c for c in candidates if c[2]]
            if matched_candidates:
                # 精确匹配优先：规范化后名称完全相等的源优先于子串匹配，
                # 避免 "Computers in Human Behavior" 被同名子刊 "Reports" 等挤掉；多条精确命中再以活跃度排序
                exact_candidates = [c for c in matched_candidates if clean_name(c[0]["display_name"]) == target_clean]
                pool = exact_candidates if exact_candidates else matched_candidates
                pool.sort(key=lambda x: x[1], reverse=True)
                best_match = pool[0][0]
                match_kind = "精确" if exact_candidates else "活跃"
                logger.info(
                    f"匹配到{match_kind}期刊: '{best_match.get('display_name')}' (ID: {best_match.get('id')}), 近3年发文: {pool[0][1]} 篇"
                )
                return best_match

            candidates.sort(key=lambda x: x[1], reverse=True)
            best_match = candidates[0][0]
            logger.info(
                f"采用相似推荐最活跃期刊: '{best_match.get('display_name')}' (ID: {best_match.get('id')}), 近3年发文: {candidates[0][1]} 篇"
            )
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

    @staticmethod
    def _build_paper_from_openalex(
        item: Dict[str, Any], source_display_name: str, current_year: int
    ) -> Optional[PaperRecord]:
        """从 OpenAlex 单条记录构建 PaperRecord，自动还原倒排摘要并过滤过短摘要。"""
        abstract_text = OpenAlexFetcher._reconstruct_abstract(item.get("abstract_inverted_index"))
        if len(abstract_text.split()) < 40:
            return None
        concept_names = [c.get("display_name") for c in item.get("concepts", [])[:6] if c.get("display_name")]
        return PaperRecord(
            id=item.get("id", ""),
            doi=item.get("doi", "") or "",
            title=item.get("title", "Untitled"),
            abstract=abstract_text,
            publication_year=item.get("publication_year", current_year),
            cited_by_count=item.get("cited_by_count", 0),
            source_title=source_display_name,
            concepts=concept_names,
        )

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
        source_id, source_display_name = None, journal_name
        if source_info:
            source_id = source_info["id"]
            source_display_name = source_info["display_name"]
        else:
            # 主源定位失败：降级为“按名称近似”画像，走 arXiv 兜底
            logger.warning(f"无法定位期刊 Source '{journal_name}'（OpenAlex 可能限流），降级使用 arXiv 近似主题兜底样本。")
            arxiv_papers = self._arxiv_fallback(journal_name, years, max_papers, search_query)
            if not arxiv_papers:
                raise ValueError(f"无法为期刊 '{journal_name}' 抓取到任何有效的带摘要论文样本（主源+arXiv 兜底均失败）")
            pdf_cache_file = os.path.join("cache", f"papers_{journal_slug}_{years}_{max_papers}_{query_hash}.json")
            try:
                os.makedirs("cache", exist_ok=True)
                with open(pdf_cache_file, "w", encoding="utf-8") as fc:
                    json.dump({"papers": [p.to_dict() for p in arxiv_papers],
                               "journal_metadata": {"display_name": journal_name,
                                                     "source": "arXiv 近似主题兜底",
                                                     "caveat": "非该期刊确切题录，仅方法与主题相近"}},
                              fc, ensure_ascii=False, indent=2)
            except Exception as e_cache2:
                logger.warning(f"写入 arXiv 兜底缓存失败: {e_cache2}")
            return arxiv_papers[:max_papers], {"display_name": journal_name, "source": "arXiv 近似主题兜底"}

        summary_stats = source_info.get("summary_stats", {})
        x_concepts = source_info.get("x_concepts", [])

        # 3. 加载本地分区（防呆清洗）
        local_partition = {"jcr_zone": "未知", "cas_zone": "未知", "cas_sub_categories": "N/A", "is_top": "未知"}
        try:
            partitions_path = os.path.join(os.path.dirname(__file__), "journal_partitions.json")
            if os.path.exists(partitions_path):
                with open(partitions_path, "r", encoding="utf-8") as f:
                    raw_db = json.load(f)

                # 对加载的数据做首尾空格和大小写清洗
                db = {}
                for k, v in raw_db.items():
                    db[k.strip().lower()] = {sub_k.strip().lower(): sub_v.strip() for sub_k, sub_v in v.items()}

                q_clean = journal_name.lower().strip()
                match_found = False
                for k, v in db.items():
                    if (
                        (k in q_clean)
                        or (q_clean in k)
                        or (source_display_name.lower() in k)
                        or (k in source_display_name.lower())
                    ):
                        local_partition = {
                            "jcr_zone": v.get("jcr_zone", "未知"),
                            "cas_zone": v.get("cas_zone", "未知"),
                            "cas_sub_categories": v.get("cas_sub_categories", "N/A"),
                            "is_top": v.get("is_top", "未知"),
                        }
                        match_found = True
                        break

                if not match_found:
                    logger.info(f"期刊 '{source_display_name}' 未匹配到本地分区数据，默认标记为未知。")
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
            "is_top": local_partition.get("is_top"),
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
            logger.info(
                f"开启双通道动态对标：通道 A (高引基准) 计划 {target_a} 篇，通道 B (主题匹配) 计划 {target_b} 篇。检索词: {search_query}"
            )

        # ===== 通道 A：热门高引文献检索 =====
        if target_a > 0:
            filter_str = f"primary_location.source.id:{source_id},publication_year:>{min_year},has_abstract:true"
            url = f"{self.BASE_URL}/works"
            params: Dict[str, Any] = {
                "filter": filter_str,
                "sort": "cited_by_count:desc",
                "per-page": min(target_a * 2, 200),
            }
            try:
                resp = self._oa_request(url, params, timeout=25)
                results = resp.json().get("results", [])
                for item in results:
                    paper = self._build_paper_from_openalex(item, source_display_name, current_year)
                    if not paper:
                        continue
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
                resp = self._oa_request(url, params, timeout=25)
                results = resp.json().get("results", [])
                for item in results:
                    paper = self._build_paper_from_openalex(item, source_display_name, current_year)
                    if not paper:
                        continue
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
                    resp = self._oa_request(url, params, timeout=25)
                    results = resp.json().get("results", [])
                    b2_count = 0
                    for item in results:
                        paper = self._build_paper_from_openalex(item, source_display_name, current_year)
                        if not paper:
                            continue
                        papers_channel_b.append(paper)
                        b2_count += 1
                    logger.info(
                        f"路线 B2 (Title Search) 补齐了 {b2_count} 篇，累计 B 通道达到: {len(papers_channel_b)} 篇。"
                    )
                except Exception as e_b2:
                    logger.warning(f"路线 B2 检索异常: {e_b2}")

            # 路线 B3：若依然不足，触发 Europe PMC 进行跨源主题检索 (作为 B2 Fallback)
            if len(papers_channel_b) < target_b:
                logger.info("路线 B2 召回仍不足，触发路线 B3 (Europe PMC 主题检索) 进行补充...")
                try:
                    # 组合 Europe PMC 检索语句
                    epmc_query = f'JOURNAL:"{source_display_name}" AND ({search_query}) AND PUB_YEAR:[{min_year} TO {current_year}] AND HAS_ABSTRACT:Y'
                    epmc_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
                    epmc_params: Dict[str, Any] = {
                        "query": epmc_query,
                        "format": "json",
                        "pageSize": min(target_b * 2, 50),
                        "resultType": "core",
                        "sort": "CITED desc",
                    }
                    resp_epmc = requests.get(
                        epmc_url,
                        params=epmc_params,
                        proxies=_NO_SYSTEM_PROXY,
                        timeout=20,
                    )
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
                                concepts=keywords,
                            )
                            papers_channel_b.append(paper)
                            b3_count += 1
                        logger.info(
                            f"路线 B3 (Europe PMC) 补齐了 {b3_count} 篇，累计 B 通道达到: {len(papers_channel_b)} 篇。"
                        )
                except Exception as e_b3:
                    logger.warning(f"路线 B3 跨源检索异常: {e_b3}")

        # ===== 动态合并与通道 B 召回不足补齐机制 =====
        # 合并去重
        combined_papers = papers_channel_a + papers_channel_b
        deduped_papers = deduplicate_papers(combined_papers)

        # 主源全部失败（OpenAlex 限流/网络故障）时，降级到 arXiv 近似主题兜底
        if not deduped_papers:
            logger.warning("OpenAlex 主源没有返回任何有效论文，降级使用 arXiv 近似主题兜底样本...")
            try:
                arxiv_papers = self._arxiv_fallback(journal_name, years, max_papers, search_query)
            except Exception as e_arx2:
                logger.error(f"arXiv 兜底也失败: {e_arx2}")
                arxiv_papers = []
            if arxiv_papers:
                deduped_papers = arxiv_papers[:max_papers]
                journal_metadata = {
                    "display_name": source_display_name,
                    "source": "arXiv 近似主题兜底",
                    "caveat": "非该期刊确切题录，仅方法与主题相近（OpenAlex 主源不可用）",
                }
                logger.info(f"arXiv 兜底提供 {len(deduped_papers)} 篇论文用于画像（来源标注为近似主题）。")
        if not deduped_papers:
            raise ValueError(f"无法为期刊 '{source_display_name}' 抓取到任何有效的带摘要论文样本")
        # 若确有主源结果，仍按原逻辑合并统计通道贡献；否则直达兜底缓存写盘
        if not (papers_channel_a or papers_channel_b) or (len(deduped_papers) >= max_papers and not arxiv_papers):
            pass  # 走下方统一提交逻辑

        # 统计去重后各通道各自的有效供给数 (通过 ID 追溯)
        channel_b_ids = {p.id for p in papers_channel_b if p.id}
        deduped_b = [p for p in deduped_papers if p.id in channel_b_ids]
        deduped_a = [p for p in deduped_papers if p.id not in channel_b_ids]
        logger.info(f"物理去重后：有效通用样本 {len(deduped_a)} 篇，有效相关主题样本 {len(deduped_b)} 篇。")

        # 若当前样本全部来自 arXiv 兜底，跳过通道统计与热门扩容（它们依赖主源 source_id）
        if _arxiv_source_active or all(
            (p.source_title or "").startswith("arXiv") for p in deduped_papers
        ):
            final_papers = deduped_papers[:max_papers]
            logger.info(f"使用 arXiv 兜底样本 {len(final_papers)} 篇（来源标注为近似主题）。")
        else:
            # 检查通道 B 实际贡献是否达标，如果不达标或总数不足，使用通道 A 的剩余热门文章进行垫底补齐
            total_valid = len(deduped_papers)
            if total_valid < max_papers and len(deduped_a) < (max_papers - len(deduped_b)):
                # 如果去重后总量依然不够，且有检索接口完全失灵的情况，尝试降级做纯热门召回填补空缺
                logger.warning(
                    f"文献样本库总数 ({total_valid} 篇) 仍未达到计划的 {max_papers} 篇。触发全热门召回扩容填补..."
                )
                # 拉大分页抓取热门
                url = f"{self.BASE_URL}/works"
                filter_str = f"primary_location.source.id:{source_id},publication_year:>{min_year},has_abstract:true"
                params = {
                    "filter": filter_str,
                    "sort": "cited_by_count:desc",
                    "per-page": 200,
                }
                try:
                    resp = self._oa_request(url, params, timeout=25)
                    if resp.status_code == 200:
                        results = resp.json().get("results", [])
                        for item in results:
                            paper = self._build_paper_from_openalex(item, source_display_name, current_year)
                            if not paper:
                                continue
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
            cache_data = {"papers": [p.to_dict() for p in final_papers], "journal_metadata": journal_metadata}
            with open(cache_file, "w", encoding="utf-8") as fc:
                json.dump(cache_data, fc, ensure_ascii=False, indent=2)
            logger.info(f"💾 文献抓取成功落盘缓存：{cache_file}")
        except Exception as e_w_cache:
            logger.warning(f"写入缓存文件异常: {e_w_cache}")

        return final_papers, journal_metadata
