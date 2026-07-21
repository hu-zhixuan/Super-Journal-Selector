import unittest
from unittest.mock import patch, MagicMock
from main import run_journal_profile_skill
from extract_features import FeatureExtractor
from fetch_papers import PaperRecord


class TestSkillWrapper(unittest.TestCase):

    def test_feature_extractor_failed_papers_logging(self):
        extractor = FeatureExtractor()
        
        # 准备一个特意导致抽取消耗异常的 PaperRecord Mock
        p = PaperRecord(
            id="W999",
            doi="10.1000/999",
            title="Broken Paper",
            abstract="Short abstract",
            publication_year=2024,
            cited_by_count=0,
            source_title="Test Journal"
        )

        with patch.object(extractor.llm, "call_json", side_effect=ValueError("Mocked LLM error")):
            results = extractor.extract_batch([p], max_workers=1)
            self.assertEqual(len(results), 0)
            self.assertEqual(len(extractor.failed_papers), 1)
            self.assertEqual(extractor.failed_papers[0]["paper_id"], "W999")
            self.assertEqual(extractor.failed_papers[0]["error_type"], "ValueError")

    @patch("main.OpenAlexFetcher.fetch_recent_papers")
    def test_run_journal_profile_skill_no_papers_error(self, mock_fetch):
        # 模拟无论文抓取到的场景
        mock_fetch.return_value = ([], {})
        
        res = run_journal_profile_skill(journal="Nonexistent Journal", years=3, max_papers=10)
        
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["error_code"], "NO_PAPERS_FETCHED")


if __name__ == "__main__":
    unittest.main()
