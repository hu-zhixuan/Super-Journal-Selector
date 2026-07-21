import unittest
from unittest.mock import patch, MagicMock
from fetch_papers import OpenAlexFetcher, PaperRecord, deduplicate_papers


class TestFetchPapers(unittest.TestCase):

    def test_deduplicate_papers(self):
        # 准备去重测试数据
        # 两个标题足够长，使得前 60 个字符完全一致，测试前缀去重
        p1 = PaperRecord(
            id="https://openalex.org/W1",
            title="A very long academic title about artificial intelligence and neural networks in 2024",
            abstract="Abstract 1",
            doi="https://doi.org/10.1000/1",
            cited_by_count=10,
            publication_year=2024,
            source_title="Computers in Human Behavior",
            concepts=[]
        )
        # DOI 相同
        p2 = PaperRecord(
            id="https://openalex.org/W2",
            title="A very long academic title about artificial intelligence and neural networks in 2024!",
            abstract="Abstract 2",
            doi="https://doi.org/10.1000/1",
            cited_by_count=15,
            publication_year=2024,
            source_title="Computers in Human Behavior",
            concepts=[]
        )
        # ID 相同
        p3 = PaperRecord(
            id="https://openalex.org/W1",
            title="Different Title",
            abstract="Abstract 3",
            doi="https://doi.org/10.1000/3",
            cited_by_count=5,
            publication_year=2024,
            source_title="Computers in Human Behavior",
            concepts=[]
        )
        # 归一化标题相同
        p4 = PaperRecord(
            id="https://openalex.org/W4",
            title="a very long academic title about artificial intelligence and neural networks in 2024",
            abstract="Abstract 4",
            doi="https://doi.org/10.1000/4",
            cited_by_count=50,
            publication_year=2024,
            source_title="Computers in Human Behavior",
            concepts=[]
        )
        # 年份 + 前缀相同
        p5 = PaperRecord(
            id="https://openalex.org/W5",
            title="A very long academic title about artificial intelligence and neural networks in 2024 with extra subtitle text",
            abstract="Abstract 5",
            doi="https://doi.org/10.1000/5",
            cited_by_count=2,
            publication_year=2024,
            source_title="Computers in Human Behavior",
            concepts=[]
        )

        raw_list = [p1, p2, p3, p4, p5]
        deduped = deduplicate_papers(raw_list)

        # 所有其它样本均应被过滤，仅留第一个
        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0].id, "https://openalex.org/W1")

    @patch("requests.get")
    def test_resolve_journal_source(self, mock_get):
        # 模拟 OpenAlex 期刊检索响应
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "id": "https://openalex.org/S1",
                    "display_name": "Computers in Human Behavior",
                    "counts_by_year": [{"year": 2024, "works_count": 100}],
                    "summary_stats": {"2yr_mean_citedness": 5.4}
                }
            ]
        }
        mock_get.return_value = mock_response

        fetcher = OpenAlexFetcher(email="test@example.com")
        res = fetcher.resolve_journal_source("Computers in Human Behavior")

        self.assertIsNotNone(res)
        self.assertEqual(res["id"], "https://openalex.org/S1")
        self.assertEqual(res["display_name"], "Computers in Human Behavior")


if __name__ == "__main__":
    unittest.main()
