import unittest
from unittest.mock import patch

from journal_router import JournalRouter


class TestJournalRouter(unittest.TestCase):
    def test_journal_router_mock_output(self):
        router = JournalRouter()

        mock_response = {
            "draft_summary_note": "A study on social media fatigue and psychological well-being using structural equation modeling.",
            "recommended_tiers": [
                {
                    "tier": "Reaching (冲刺)",
                    "journal_name": "MIS Quarterly",
                    "fit_score": 68,
                    "estimated_acceptance_rate": "15%-20%",
                    "reason": "High reputation but requires multi-wave panel dataset.",
                },
                {
                    "tier": "Target (主投)",
                    "journal_name": "Computers in Human Behavior",
                    "fit_score": 88,
                    "estimated_acceptance_rate": "25%-35%",
                    "reason": "Perfect match for adolescent digital media fatigue topic.",
                },
                {
                    "tier": "Safe (保底)",
                    "journal_name": "Computers in Human Behavior Reports",
                    "fit_score": 95,
                    "estimated_acceptance_rate": "45%-60%",
                    "reason": "High acceptance rate with strong open science support.",
                },
            ],
        }

        with patch.object(router.llm, "call_json", return_value=mock_response):
            res = router.route_journals("Adolescent social media fatigue and SEM analysis.")
            self.assertEqual(len(res["recommended_tiers"]), 3)
            self.assertEqual(res["recommended_tiers"][1]["journal_name"], "Computers in Human Behavior")
            self.assertEqual(res["recommended_tiers"][1]["fit_score"], 88)


if __name__ == "__main__":
    unittest.main()
