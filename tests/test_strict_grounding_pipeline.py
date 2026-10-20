import unittest
from pathlib import Path
import sys
import tempfile
import types
from unittest.mock import patch

ROOT = Path(__file__).parents[1]

def import_runner_without_inference_dependencies():
    """Load prompt helpers without requiring CUDA packages in the test environment."""
    torch_stub = types.ModuleType("torch")
    peft_stub = types.ModuleType("peft")
    peft_stub.PeftModel = object
    transformers_stub = types.ModuleType("transformers")
    transformers_stub.AutoModelForCausalLM = object
    transformers_stub.AutoTokenizer = object
    with patch.dict(
        sys.modules,
        {"torch": torch_stub, "peft": peft_stub, "transformers": transformers_stub},
    ):
        from src import run_official_loong_structrag_qwen as module
    return module


runner = import_runner_without_inference_dependencies()


def import_resume_without_cuda():
    torch_stub = types.ModuleType("torch")
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    peft_stub = types.ModuleType("peft")
    peft_stub.PeftModel = object
    transformers_stub = types.ModuleType("transformers")
    transformers_stub.AutoModelForCausalLM = object
    transformers_stub.AutoTokenizer = object
    with patch.dict(
        sys.modules,
        {"torch": torch_stub, "peft": peft_stub, "transformers": transformers_stub},
    ):
        from src import run_official_loong_structrag_qwen_resume as module
    return module


resume_runner = import_resume_without_cuda()


class StrictGroundingPipelineTests(unittest.TestCase):
    def test_route_uses_upstream_label_priority_and_five_structures(self):
        with patch.object(runner, "generate", return_value="graph and table"):
            selected, _ = runner.route(None, None, "Q", [], 1024)
        self.assertEqual(selected, "table")

        with patch.object(runner, "generate", return_value="catalogue"):
            selected, _ = runner.route(None, None, "Q", [], 1024, "paper-five")
        self.assertEqual(selected, "catalogue")

    def test_router_prompts_keep_official_and_five_class_protocols_separate(self):
        released = runner.router_prompt("Q", [])
        paper_five = runner.router_prompt("Q", [], "paper-five")
        self.assertIn("one of the three words", released)
        self.assertNotIn("algorithm", released.lower())
        self.assertLess(released.index("Generally speaking"), released.index("Now, given"))
        self.assertLess(released.index("Now, given"), released.index("Simply output"))
        self.assertIn("one of the five words", paper_five)
        self.assertIn("catalogue", paper_five.lower())

    def test_chunk_extraction_preserves_source_title(self):
        with patch.object(runner, "generate", return_value="supported fact"):
            evidence = runner.extract(
                None, None, "chunk", "Q", ["Q"], ["Document A: source text"], 1024, 32
            )
        self.assertEqual(evidence, ["Retrieval result for Document A: supported fact"])

    def test_algorithm_and_catalogue_use_released_structure_prompts(self):
        prompts = {
            "construct_algorithm.txt": "ALGORITHM {requirement} {raw_content}",
            "construct_catalogue.txt": "CATALOGUE {requirement} {raw_content}",
        }

        def fake_read_prompt(_directory, filename):
            return prompts[filename]

        def fake_generate(_tokenizer, _model, messages, _tokens, _context):
            return messages[0]["content"]

        document = {"title": "Doc A", "document": "Source content"}
        with patch.object(runner, "read_prompt", side_effect=fake_read_prompt), patch.object(
            runner, "generate", side_effect=fake_generate
        ):
            _, algorithm = runner.construct(
                None, None, Path("prompts"), "Q", "algorithm", [document], 1024, 32
            )
            _, catalogue = runner.construct(
                None, None, Path("prompts"), "Q", "catalogue", [document], 1024, 32
            )
        self.assertIn("ALGORITHM", algorithm[0])
        self.assertIn("CATALOGUE", catalogue[0])

    def test_algorithm_and_catalogue_extraction_use_structure_specific_prompts(self):
        captured = []

        def fake_generate(_tokenizer, _model, messages, _tokens, _context):
            captured.append(messages[0]["content"])
            return "evidence"

        with patch.object(runner, "generate", side_effect=fake_generate):
            runner.extract(None, None, "algorithm", "Q", ["Q"], ["Rules"], 1024, 32)
            runner.extract(None, None, "catalogue", "Q", ["Q"], ["Outline"], 1024, 32)
        self.assertIn("Algorithms:", captured[0])
        self.assertIn("algorithm descriptions", captured[0])
        self.assertIn("Catalogues:", captured[1])
        self.assertIn("catalogue", captured[1])

    def test_query_only_decomposition_does_not_generate(self):
        with patch.object(runner, "generate", side_effect=AssertionError("generation should not run")):
            raw, questions = runner.decompose(
                None,
                None,
                ROOT / "prompts",
                "Return the requested JSON.",
                "unused",
                1024,
                4,
                32,
                query_only=True,
            )
        self.assertEqual(raw, "Return the requested JSON.")
        self.assertEqual(questions, ["Return the requested JSON."])

    def test_strict_merge_prompt_has_grounding_and_format_constraints(self):
        captured = {}

        def fake_generate(tokenizer, model, messages, max_new_tokens, context_limit):
            captured["content"] = messages[0]["content"]
            return "{}"

        with patch.object(runner, "generate", side_effect=fake_generate):
            answer = runner.merge(
                None,
                None,
                "Determine citation relationships.",
                ["Determine citation relationships."],
                ["Evidence: Paper A references Paper B."],
                "graph",
                1024,
                32,
                strict_grounding=True,
                task_type="paper",
            )
        self.assertEqual(answer, "{}")
        prompt = captured["content"]
        self.assertIn("only the supplied retrieval results", prompt)
        self.assertIn("Never invent or infer", prompt)
        self.assertIn("valid JSON", prompt)
        self.assertIn("Paper A references Paper B", prompt)

    def test_strict_extraction_prompt_limits_content_to_evidence(self):
        captured = []

        def fake_generate(tokenizer, model, messages, max_new_tokens, context_limit):
            captured.append(messages[0]["content"])
            return "evidence"

        with patch.object(runner, "generate", side_effect=fake_generate):
            runner.extract(None, None, "chunk", "Q", ["Q"], ["Document A"], 1024, 32, True)
            runner.extract(None, None, "table", "Q", ["Q"], ["Table A"], 1024, 32, True)
            runner.extract(None, None, "graph", "Q", ["Q"], ["Triple A"], 1024, 32, True)
        self.assertEqual(len(captured), 3)
        for prompt in captured:
            self.assertIn("only entities, numbers, titles, and relationships", prompt)
            self.assertIn("Do not infer, invent, or complete missing facts", prompt)

    def test_default_merge_keeps_released_length_instruction(self):
        captured = {}

        def fake_generate(tokenizer, model, messages, max_new_tokens, context_limit):
            captured["content"] = messages[0]["content"]
            return "answer"

        with patch.object(runner, "generate", side_effect=fake_generate):
            runner.merge(None, None, "Q", ["Q"], ["E"], "chunk", 1024, 32)
        self.assertIn("detailed, specific, and lengthy as possible", captured["content"])
        self.assertNotIn("Never invent or infer", captured["content"])

    def test_resume_config_allows_identical_run_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompts = root / "prompts"
            prompts.mkdir()
            for name in (
                "construct_graph.txt",
                "construct_table.txt",
                "construct_algorithm.txt",
                "construct_catalogue.txt",
                "decompose.txt",
            ):
                (prompts / name).write_text(name, encoding="utf-8")
            args = types.SimpleNamespace(
                dataset=root / "dataset.jsonl",
                base_model=root / "model",
                router_adapter=root / "adapter",
                prompts_dir=prompts,
                context_limit=32768,
                max_samples=20,
                stratified=True,
                seed=17,
                router_classes="released-three",
                max_subqueries=4,
                max_new_tokens_structure=4096,
                max_new_tokens_answer=4096,
                query_only_decomposition=True,
                strict_grounding=True,
            )
            run_dir = root / "run"
            run_dir.mkdir()
            config_path = run_dir / "run_config.json"
            config = resume_runner.run_config(args)
            resume_runner.ensure_run_config(config_path, config)
            resume_runner.ensure_run_config(config_path, config)
            changed = {**config, "strict_grounding": False}
            with self.assertRaises(RuntimeError):
                resume_runner.ensure_run_config(config_path, changed)

    def test_resume_refuses_legacy_checkpoint_without_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                resume_runner.ensure_run_config(
                    root / "run_config.json", {"pipeline_revision": "test"}
                )


if __name__ == "__main__":
    unittest.main()
