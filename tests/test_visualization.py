import tempfile
import unittest
from pathlib import Path

from trust_aware import generate_all_figures


class VisualizationTests(unittest.TestCase):
    def test_generates_publication_svg_figures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = generate_all_figures(directory, seed=13)

            self.assertEqual(len(paths), 6)
            for path in paths:
                self.assertTrue(path.exists(), path)
                content = path.read_text(encoding="utf-8")
                self.assertIn("<svg", content)
                self.assertIn("Trust-Aware", content)
                self.assertIn('role="img"', content)
                self.assertIn("<title", content)
                self.assertGreater(len(content), 3500)

            names = {Path(path).name for path in paths}
            self.assertIn("figure_01_policy_atlas.svg", names)
            self.assertIn("figure_04_ai_query_pipeline.svg", names)
            self.assertIn("figure_06_real_dataset_statistics.svg", names)


if __name__ == "__main__":
    unittest.main()
