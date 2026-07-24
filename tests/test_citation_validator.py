import unittest

from generate_profile import ProfileGenerator


class TestCitationValidator(unittest.TestCase):
    def test_citation_validator(self):
        aggregated_stats = {
            "most_similar_papers": [{"title": "The Impact of Social Media Fatigue on Well-being"}],
            "recommended_references": [
                {"title": "Self-Determination Theory in Mobile Media"},
                {"title": "Quantitative analysis of gaming addiction"},
            ],
            "representative_novelties": [],
        }

        # 模拟生成报告
        report = (
            "在本研究中，我们引用了文献《Self-Determination Theory in Mobile Media》作为理论基础。\n"
            "另外，还有一篇未校验文献《A Fake Paper Title about Social Pressure》。"
        )

        gen = ProfileGenerator()
        validated_report = gen.validate_citations(report, aggregated_stats)

        # 验证通过的文献应该保持原样
        self.assertIn("《Self-Determination Theory in Mobile Media》", validated_report)
        self.assertNotIn("《Self-Determination Theory in Mobile Media》`[⚠️ Unverified Reference]`", validated_report)

        # 未验证的文献后面应该被标记
        self.assertIn("《A Fake Paper Title about Social Pressure》`[⚠️ Unverified Reference]`", validated_report)


if __name__ == "__main__":
    unittest.main()
