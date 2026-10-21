from __future__ import annotations

import unittest

from src.evaluate_external_five_stage_datasets import personwork_proxy_structure, run_metrics


class ExternalEvalMetricsTests(unittest.TestCase):
    def test_personwork_task_proxy_mapping(self):
        self.assertEqual(personwork_proxy_structure("hard_multi_hop_kinship"), "graph")
        self.assertEqual(personwork_proxy_structure("hard_attribute_filter"), "table")
        self.assertEqual(personwork_proxy_structure("hard_event_exception"), "algorithm")
        self.assertEqual(personwork_proxy_structure("hard_family_member"), "graph")

    def test_run_metrics_reports_answer_recall_and_token_cost(self):
        rows = [
            {
                "dataset": "VALUE-KAO-PersonWork-EN-v2",
                "task": "person",
                "task_type": "hard_attribute_filter",
                "gold": "Avery",
                "gold_names": ["Avery"],
                "answer": "Avery",
                "selected_structure": "table",
                "proxy_structure": "table",
                "structure_output": ["Avery has network operations skills"],
                "document_chars": 100,
                "latency_seconds": 2.0,
                "stage_latency_seconds": {"answer": 0.5},
                "input_tokens": 20,
                "output_tokens": 5,
                "generation_calls": 1,
            }
        ]
        result = run_metrics(rows)
        self.assertEqual(result["router_proxy_accuracy"], 1.0)
        self.assertEqual(result["answer_metrics"]["personwork"]["exact_match"], 1.0)
        self.assertEqual(result["structure_extraction"]["gold_fact_recall_proxy"], 1.0)
        self.assertEqual(result["cost_and_speed"]["total_input_tokens"], 20)


if __name__ == "__main__":
    unittest.main()
