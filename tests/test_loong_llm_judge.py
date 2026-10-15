import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.evaluate_loong_llm_judge import aggregate_judgments, evaluate, parse_judge_response


class LoongJudgeTest(unittest.TestCase):
    def test_parse_valid_json_and_fenced_content(self):
        self.assertEqual(parse_judge_response('{"score": 82, "reason": "Correct."}')["score"], 82.0)
        self.assertEqual(parse_judge_response("```json\n{\"score\": 10, \"reason\": \"Wrong\"}\n```")["score"], 10.0)

    def test_parse_rejects_malformed_or_unbounded(self):
        for value in ('{"score": 101, "reason": "bad"}', '{"score": 2}', 'not json'):
            with self.assertRaises(ValueError):
                parse_judge_response(value)

    def test_dry_run_and_cache_reuse_without_network(self):
        rows = [{"id": "a", "type": "legal", "level": 2, "selected_structure": "graph", "judge_response": {"score": 75, "reason": "Mostly correct"}}]
        with tempfile.TemporaryDirectory() as directory, patch("src.evaluate_loong_llm_judge.request.urlopen", side_effect=AssertionError("network")):
            output = Path(directory)
            first = evaluate(rows, output, model="test-model", dry_run=True)
            rows[0]["judge_response"] = {"score": 1, "reason": "changed"}
            second = evaluate(rows, output, model="test-model", dry_run=True)
        self.assertEqual(first["overall"]["mean_score"], 75.0)
        self.assertEqual(second["overall"]["mean_score"], 75.0)
        self.assertEqual(second["by_type"]["legal"]["examples"], 1)

    def test_aggregate_dimensions(self):
        metrics = aggregate_judgments([{"score": 20, "type": "a", "level": 1, "selected_structure": "chunk"}, {"score": 80, "type": "a", "level": 2, "selected_structure": "table"}])
        self.assertEqual(metrics["overall"]["mean_score"], 50.0)
        self.assertEqual(metrics["by_level"]["2"]["mean_score"], 80.0)


if __name__ == "__main__":
    unittest.main()
