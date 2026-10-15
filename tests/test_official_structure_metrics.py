import json
import tempfile
import unittest
from pathlib import Path

from src.evaluate_official_structure_metrics import aggregate_predictions, load_jsonl


class OfficialStructureMetricsTest(unittest.TestCase):
    def test_aggregates_overall_and_multiple_dimensions(self):
        rows = [
            {"id": "1", "type": "financial", "level": 1, "selected_structure": "table", "coverage_status": "full", "gold_leaf_recall": 1.0, "exact_canonical_match": True},
            {"id": "2", "type": "financial", "level": 2, "selected_structure": "chunk", "coverage_status": "partial", "gold_leaf_recall": 0.0, "exact_canonical_match": False},
            {"id": "3", "type": "paper", "level": 2, "selected_structure": "graph", "coverage_status": "partial", "gold_leaf_recall": 0.5, "exact_canonical_match": False},
        ]
        metrics = aggregate_predictions(rows, [{"id": "4", "stage": "preparation"}])
        self.assertEqual(metrics["overall"]["examples"], 3)
        self.assertEqual(metrics["overall"]["exact_canonical_match"], 1 / 3)
        self.assertEqual(metrics["by_structure"]["table"]["examples"], 1)
        self.assertEqual(metrics["by_type"]["paper"]["mean_gold_leaf_recall"], 0.5)
        self.assertEqual(metrics["by_level"]["2"]["examples"], 2)
        self.assertEqual(metrics["by_coverage_status"]["partial"]["examples"], 2)
        self.assertEqual(metrics["preparation_exclusions"], 1)

    def test_load_jsonl_skips_blank_lines_and_rejects_bad_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text('{"id":"1"}\n\n{"id":"2"}\n', encoding="utf-8")
            self.assertEqual([row["id"] for row in load_jsonl(path)], ["1", "2"])


if __name__ == "__main__":
    unittest.main()
