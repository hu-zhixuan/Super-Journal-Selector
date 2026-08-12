import os
import shutil
import tempfile
import unittest

from plot_charts import (
    HAS_MPL,
    generate_all_charts,
    plot_method_distribution,
    plot_radar,
)


class TestPlotCharts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.stats = {
            "method_distribution": {
                "Quantitative_Empirical": {"count": 3, "percentage": 50.0},
                "Theoretical_Review": {"count": 3, "percentage": 50.0},
            },
            "sample_size_stats": {"min": 91, "median": 494, "max": 898},
            "top_theories": [
                {"name": "Job Demands-Resources Model", "count": 3},
                {"name": "Self-Determination Theory", "count": 2},
                {"name": "Trust in AI", "count": 1},
            ],
            "open_science_stats": {"None": {"count": 6, "percentage": 100.0}},
            "top_reporting_styles": [
                {"style": "None / Qualitative Description", "count": 5},
                {"style": "Significance Testing (P-values)", "count": 1},
            ],
        }

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    @unittest.skipUnless(HAS_MPL, "matplotlib 不可用")
    def test_all_charts_generate_png_files(self):
        paths = generate_all_charts(self.stats, out_dir=self.tmp)
        self.assertIn("method_distribution", paths)
        for k, v in paths.items():
            self.assertIsNotNone(v, f"{k} 图表未生成")
            self.assertTrue(os.path.isfile(v), f"{k} 文件不存在")
            self.assertGreater(os.path.getsize(v), 0)

    @unittest.skipUnless(HAS_MPL, "matplotlib 不可用")
    def test_radar_generates(self):
        journals = [
            {"name": "A", "录用难度": 80, "范式契合": 60},
            {"name": "B", "录用难度": 50, "范式契合": 90},
        ]
        p = plot_radar(journals, os.path.join(self.tmp, "radar.png"))
        self.assertIsNotNone(p)
        self.assertTrue(os.path.isfile(p))

    @unittest.skipUnless(HAS_MPL, "matplotlib 不可用")
    def test_charts_skip_when_empty_data(self):
        p1 = plot_method_distribution({}, os.path.join(self.tmp, "x.png"))
        self.assertIsNone(p1)


if __name__ == "__main__":
    unittest.main()
