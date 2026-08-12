import os
import shutil
import tempfile
import unittest

from main import _run_offline_demo, run_journal_profile_skill


class TestOfflineDemo(unittest.TestCase):
    def test_offline_demo_returns_report_without_api(self):
        """OFFLINE_DEMO 模式下无需 key/网络即可产出完整报告"""
        os.environ["OFFLINE_DEMO"] = "1"
        try:
            res = run_journal_profile_skill(journal="Computers in Human Behavior", years=3, max_papers=100)
        finally:
            os.environ.pop("OFFLINE_DEMO", None)
        self.assertEqual(res.get("status"), "success")
        self.assertTrue(res.get("offline_demo"))
        self.assertEqual(res["cost_statistics"]["total_api_calls"], 0)
        self.assertTrue(res["report_markdown"].startswith("# "))

    def test_offline_demo_missing_data_returns_error(self):
        """离线样例数据缺失时给出明确错误码 OFFline_DEMO_DATA_MISSING"""
        # 用临时目录模拟缺失样例数据（不含 papers.json / report.md）
        demo_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "offline_demo")
        if not os.path.isdir(demo_dir):
            self.skipTest("离线样例目录不存在，跳过")
        tmp = tempfile.mkdtemp()
        try:
            # 备份真实样例，用空目录顶替
            real_papers = os.path.join(demo_dir, "papers.json")
            real_report = os.path.join(demo_dir, "report.md")
            backup_dir = tempfile.mkdtemp()
            shutil.move(real_papers, os.path.join(backup_dir, "papers.json"))
            shutil.move(real_report, os.path.join(backup_dir, "report.md"))
            try:
                res = _run_offline_demo(journal="Demo Journal")
                self.assertEqual(res.get("status"), "error")
                self.assertEqual(res.get("error_code"), "OFFLINE_DEMO_DATA_MISSING")
            finally:
                shutil.move(os.path.join(backup_dir, "papers.json"), real_papers)
                shutil.move(os.path.join(backup_dir, "report.md"), real_report)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
