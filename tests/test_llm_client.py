import unittest

from llm_client import LLMClient, get_prompt_fingerprint


class TestLLMClient(unittest.TestCase):
    def test_prompt_fingerprint(self):
        # 验证相同参数生成相同指纹，不同参数生成不同指纹
        fp1 = get_prompt_fingerprint("v1.3", "deepseek-v4-flash", 0.1, "system prompt A")
        fp2 = get_prompt_fingerprint("v1.3", "deepseek-v4-flash", 0.1, "system prompt A")
        fp3 = get_prompt_fingerprint("v1.4", "deepseek-v4-flash", 0.1, "system prompt A")

        self.assertEqual(fp1, fp2)
        self.assertNotEqual(fp1, fp3)
        self.assertEqual(len(fp1), 10)

    def test_cost_statistics(self):
        client = LLMClient(model="deepseek-v4-flash")
        client.total_prompt_tokens = 1_000_000
        client.total_completion_tokens = 1_000_000
        client.total_api_calls = 5

        stats = client.get_cost_statistics()

        self.assertEqual(stats["total_api_calls"], 5)
        self.assertEqual(stats["total_prompt_tokens"], 1_000_000)
        self.assertEqual(stats["total_completion_tokens"], 1_000_000)
        # deepseek-v4-flash: 1M in = $0.1, 1M out = $0.2 => total $0.3
        self.assertEqual(stats["estimated_cost_usd"], 0.3)
        self.assertIsNotNone(stats["estimated_cost_cny"])

    def test_extract_json_from_text(self):
        client = LLMClient()
        raw_text = 'Here is the JSON:\n```json\n{"status": "ok", "value": 123}\n```\nHope it helps!'
        extracted = client.extract_json_from_text(raw_text)

        self.assertEqual(extracted.get("status"), "ok")
        self.assertEqual(extracted.get("value"), 123)


if __name__ == "__main__":
    unittest.main()
