import os
import requests
import logging
from typing import List, Dict, Any, Optional
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
    """

    BASE_URL = "https://api.openalex.org"

    def __init__(self, email: Optional[str] = None):
        self.email = email or os.getenv("OPENALEX_EMAIL")
        self.headers = {
            "User-Agent": f"JournalProfileSkill/1.0 ({self.email or 'mailto:researcher@example.com'})"
        }

    def resolve_journal_source(self, journal_name: str) -> Optional[Dict[str, Any]]:
        """
        精确匹配或检索 OpenAlex 中的 Journal Source ID
        """
        url = f"{self.BASE_URL}/sources"
        params = {"search": journal_name, "per-page": 5}
        try:
            resp = requests.get(url, params=params, headers=self.headers, proxies={"http": None, "https": None}, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if not results:
                logger.error(f"未在 OpenAlex 中找到期刊: '{journal_name}'")
                return None
            
            # 优先完全匹配名称
            for res in results:
                if res.get("display_name", "").lower() == journal_name.lower():
                    logger.info(f"精确匹配到期刊: {res.get('display_name')} (ID: {res.get('id')})")
                    return res
            
            # 否则取第一个最相关的结果
            first_match = results[0]
            logger.info(f"采用相似匹配期刊: {first_match.get('display_name')} (ID: {first_match.get('id')})")
            return first_match
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
    ) -> List[PaperRecord]:
        """
        获取指定期刊最近几年内被引频次较好/最新的带摘要论文
        """
        source_info = self.resolve_journal_source(journal_name)
        if not source_info:
            raise ValueError(f"无法定位期刊 Source: {journal_name}")

        source_id = source_info["id"]
        source_display_name = source_info["display_name"]
        current_year = datetime.now().year
        min_year = current_year - years

        # 构建检索条件：属于该期刊 + 发表年份在时间范围内 + 必须带有摘要
        filter_str = f"primary_location.source.id:{source_id},publication_year:>{min_year},has_abstract:true"
        url = f"{self.BASE_URL}/works"
        
        # 为了保证代表性，优先按被引次数降序，确保抓取到优质范例论文
        params = {
            "filter": filter_str,
            "sort": "cited_by_count:desc",
            "per-page": min(max_papers * 2, 100),  # 多抓点备过滤
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
                # 过滤摘要过短（极可能不是研究性论文，如编辑社论/书评）
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
            return papers

        except Exception as e:
            logger.error(f"获取 OpenAlex 论文列表失败: {str(e)}")
            raise
