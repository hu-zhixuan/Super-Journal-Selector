import unittest
from aggregate import ProfileAggregator


class TestAggregate(unittest.TestCase):

    def test_cosine_similarity(self):
        # 简单相似度验证（包含停用词和标点）
        text1 = "The study uses Structural Equation Modeling (SEM) to analyze parenting."
        text2 = "Using Structural Equation Modeling (SEM) for adolescent analysis."
        
        sim = ProfileAggregator.calculate_cosine_similarity(text1, text2)
        self.assertGreater(sim, 0.4)
        self.assertLessEqual(sim, 1.0)

    def test_aggregate_stats(self):
        features = [
            {
                "paper_id": "W1",
                "title": "Paper One",
                "abstract": "We analyze child psychological wellbeing in this empirical study.",
                "method_category": "Quantitative_Empirical",
                "sample_size_approx": 100,
                "theoretical_frameworks": ["Self-Determination Theory", "SDT"],
                "analytical_tools": ["SEM"],
                "cited_by_count": 50,
                "publication_year": 2024,
                "open_science_practices": ["Open Data"],
                "statistical_reporting_style": "Reported mediation confidence intervals",
                "concepts": []
            },
            {
                "paper_id": "W2",
                "title": "Paper Two",
                "abstract": "A theoretical view on social media fatigue and need frustration.",
                "method_category": "Theoretical_Review",
                "sample_size_approx": -1,
                "theoretical_frameworks": ["Self-Determination Theory"],
                "analytical_tools": [],
                "cited_by_count": 20,
                "publication_year": 2023,
                "open_science_practices": ["None"],
                "statistical_reporting_style": "None",
                "concepts": []
            }
        ]

        agg = ProfileAggregator()
        stats = agg.aggregate(features, user_draft_text="social media fatigue parenting")

        self.assertEqual(stats["total_papers_analyzed"], 2)
        # 验证定量分布：1/2 = 50%
        method_stats = stats["method_distribution"]
        self.assertEqual(method_stats["Quantitative_Empirical"]["count"], 1)

        # 验证中位数样本量（定量数据中位数为 100）
        self.assertEqual(stats["sample_size_stats"]["median"], 100)


if __name__ == "__main__":
    unittest.main()
