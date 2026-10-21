from __future__ import annotations

import unittest

from src.build_llm_five_type_preferences import preference_row, split_rows, validate_candidate
from src.five_type_router_data_v1 import FIVE_LABELS, validate_rows


class LLMPreferenceTests(unittest.TestCase):
    def candidate(self):
        return {
            "question": "Which person has the required skill?",
            "answer": "Avery",
            "evidence_quote": "Avery has skill: network operations",
            "task_family": "single_hop",
            "overview": "People profiles with skills and department fields.",
            "plans": {label: f"Use {label} to organize the relevant evidence." for label in FIVE_LABELS},
            "ranking": ["table", "chunk", "graph", "algorithm", "catalogue"],
        }

    def source(self, source_id="row-1", unit_id="unit-1"):
        return {
            "id": source_id,
            "source_root": unit_id,
            "documents": [{"title": "Profiles", "content": "Avery has skill: network operations"}],
        }

    def test_candidate_requires_all_structures_and_a_complete_ranking(self):
        candidate = self.candidate()
        self.assertEqual(validate_candidate(candidate)["ranking"][0], "table")
        candidate["ranking"][-1] = "table"
        with self.assertRaises(ValueError):
            validate_candidate(candidate)

    def test_preference_pair_uses_top_two_and_keeps_answer_out_of_router_prompt(self):
        row = preference_row(self.source(), self.candidate())
        validate_rows([row], "generated")
        self.assertEqual(row["chosen"][-1]["content"], "table")
        self.assertEqual(row["rejected"][-1]["content"], "chunk")
        prompt = row["chosen"][0]["content"]
        self.assertNotIn("Avery", prompt)
        self.assertNotIn("network operations", prompt)

    def test_supporting_quote_must_match_source(self):
        candidate = self.candidate()
        candidate["evidence_quote"] = "Avery is a network expert"
        with self.assertRaises(ValueError):
            preference_row(self.source(), candidate)

    def test_splits_keep_source_units_disjoint(self):
        rows = [
            {"unit_id": f"unit-{index % 10}", "id": str(index)}
            for index in range(100)
        ]
        splits = split_rows(rows, seed=17)
        units = {name: {row["unit_id"] for row in group} for name, group in splits.items()}
        self.assertFalse(units["train"] & units["validation"])
        self.assertFalse(units["train"] & units["test"])
        self.assertFalse(units["validation"] & units["test"])


if __name__ == "__main__":
    unittest.main()
