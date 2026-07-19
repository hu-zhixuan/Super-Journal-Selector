import os
import requests
import logging
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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OpenAlexFetcher:
    """
    层①：数据抓取层。封装 OpenAlex 开放 API（无需 key），获取目标期刊最近N年的论文摘要文本并结构化。
    最新升级：支持清洗符号后的名字匹配，并根据 counts_by_year 统计近期发文活跃度，优先选择最新的活跃数据库条目。
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
        清理标点符号，并优先选择近年发文最活跃的数据库记录，避开已停更的历史记录。
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
            
            # 清洗字符串函数：转小写，去除标点与斜杠等，便于模糊对标
            def clean_name(n: str) -> str:
                return "".join(c for c in n.lower() if c.isalnum()).strip()

            target_clean = clean_name(journal_name)
            current_year = datetime.now().year
            candidates = []

            for res in results:
                display_name = res.get("display_name", "")
                res_clean = clean_name(display_name)
                
                # 计算近 3 年的发文总数 (用以判断是否活跃)
                counts_by_year = res.get("counts_by_year", [])
                recent_works_sum = sum(
                    c.get("works_count", 0)
                    for c in counts_by_year
                    if c.get("year", 0) >= (current_year - 2)
                )
                
                # 判断名字是否高度相似
                is_name_match = (target_clean in res_clean) or (res_clean in target_clean)
                candidates.append((res, recent_works_sum, is_name_match))

            # 筛选名字匹配成功的候选人
            matched_candidates = [c for c in candidates if c[2]]
            
            if matched_candidates:
                # 按照近 3 年发文活跃度降序排列，选择发文量最大、最新最活跃的条目
                matched_candidates.sort(key=lambda x: x[1], reverse=True)
                best_match = matched_candidates[0][0]
                logger.info(f"匹配到活跃期刊: '{best_match.get('display_name')}' (ID: {best_match.get('id')}), 近3年发文: {matched_candidates[0][1]} 篇")
                return best_match
            
            # 如果没有完全匹配的名字，降级直接选检索候选列表里发文最活跃的一个
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
        OpenAlex 返回的摘要格式为倒排索引（单词->位置列表），此处纯代码还原为连贯英文段落
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
        self, journal_name: str, years: int = 3, max_papers: int = 30
    ) -> Tuple[List[PaperRecord], Dict[str, Any]]:
        """
        获取指定期刊最近几年内被引频次较好/最新的带摘要论文，同时返回期刊的元数据属性。
        返回格式：(papers_list, journal_metadata_dict)
        """
        source_info = self.resolve_journal_source(journal_name)
        if not source_info:
            raise ValueError(f"无法定位期刊 Source: {journal_name}")

        source_id = source_info["id"]
        source_display_name = source_info["display_name"]
        
        # 提取期刊关键学术指标与属性
        summary_stats = source_info.get("summary_stats", {})
        x_concepts = source_info.get("x_concepts", [])
        
        journal_metadata = {
            "display_name": source_display_name,
            "issn": source_info.get("issn", ["Unknown"])[0] if source_info.get("issn") else "Unknown",
            "h_index": summary_stats.get("h_index", "N/A"),
            "estimated_impact_factor": summary_stats.get("2yr_mean_citedness", "N/A"),
            "works_count": source_info.get("works_count", "N/A"),
            "cited_by_count": source_info.get("cited_by_count", "N/A"),
            "categories": [c.get("name") for c in x_concepts[:4] if c.get("name")]
        }

        current_year = datetime.now().year
        min_year = current_year - years

        # 构建检索条件
        filter_str = f"primary_location.source.id:{source_id},publication_year:>{min_year},has_abstract:true"
        url = f"{self.BASE_URL}/works"
        
        params = {
            "filter": filter_str,
            "sort": "cited_by_count:desc",
            "per-page": min(max_papers * 2, 100),
        }

        papers: List[PaperRecord] = []
        try:
            logger.info(f"开始抓取期刊 '{source_display_name}' 近 {years} 年论文，上限 {max_papers} 篇...")
            resp = requests.get(url, params=params, headers=self.headers, proxies={"http": None, "https": None}, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])

            for item in results:
                abstract_text = self._reconstruct_abstract(item.get("abstract_inverted_index"))
                if len(abstract_text.split()) < 60:
                    continue

                paper = PaperRecord(
                    id=item.get("id", ""),
                    doi=item.get("doi", "") or "",
                    title=item.get("title", "Untitled"),
                    abstract=abstract_text,
                    publication_year=item.get("publication_year", current_year),
                    cited_by_count=item.get("cited_by_count", 0),
                    source_title=source_display_name,
                )
                papers.append(paper)
                if len(papers) >= max_papers:
                    break

            logger.info(f"抓取完成，获得有效摘要论文 {len(papers)} 篇。")
            return papers, journal_metadata

        except Exception as e:
            logger.error(f"获取 OpenAlex 论文列表失败: {str(e)}")
            raise
