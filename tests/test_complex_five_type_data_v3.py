import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("complex_v3", ROOT / "scripts/build_complex_five_type_data_v3.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def profile(unit, index, **extra):
    row = {"unit_id": unit, "person_id": f"{unit}_P{index}", "name": f"N{unit}{index}", "family_id": f"{unit}_F",
           "age": 30 + index, "city": f"C{index % 2}", "company": f"Co{index % 2}", "department": f"D{index % 2}",
           "team": f"T{index % 2}", "position": "Engineer", "work_years": index, "skills": ["s1", "s2"],
           "father_id": None, "mother_id": None, "spouse_id": None, "children_ids": []}
    row.update(extra)
    return row


class ComplexFiveTypeDataV3Tests(unittest.TestCase):
    def test_grounded_graph_and_oracle(self):
        people = [profile("U", i) for i in range(3)]
        people[2]["father_id"] = people[1]["person_id"]
        people[1]["spouse_id"] = people[0]["person_id"]
        rows = MODULE.build_candidates("U_W000", people, 1, 3)
        graph = next(row for row in rows if row["label"] == "graph" and row["difficulty_level"] == 2)
        self.assertEqual(graph["oracle"]["hops"], 2)
        self.assertEqual(graph["oracle"]["answer"], people[0]["name"])
        self.assertEqual(graph["task_spec"]["steps"][-1]["to_id"], people[0]["person_id"])

    def test_impossible_graph_is_skipped(self):
        rows = MODULE.build_candidates("U_W000", [profile("U", 0)], 1, 3)
        self.assertFalse(any(row["label"] == "graph" and row["difficulty_level"] > 0 for row in rows))

    def test_exact_metadata_and_same_seed_reproducibility(self):
        people = [profile(f"U{i}", j) for i in range(4) for j in range(3)]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); source = base / "profiles.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in people), encoding="utf-8")
            first, second = base / "first", base / "second"
            MODULE.generate_dataset(source, first, train_count=20, validation_count=10, test_count=10, seed=8)
            MODULE.generate_dataset(source, second, train_count=20, validation_count=10, test_count=10, seed=8)
            for name in ("train", "validation", "test", "metadata"):
                suffix = ".jsonl" if name != "metadata" else ".json"
                self.assertEqual((first / (name + suffix)).read_bytes(), (second / (name + suffix)).read_bytes())
            rows = [json.loads(line) for line in (first / "train.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 20)
            self.assertEqual(len({row["id"] for row in rows}), 20)
            all_candidates = []
            for unit in ("U0_W000", "U1_W000", "U2_W000", "U3_W000"):
                fixture_people = [profile(unit, j) for j in range(3)]
                fixture_people[2]["father_id"] = fixture_people[1]["person_id"]
                fixture_people[1]["spouse_id"] = fixture_people[0]["person_id"]
                all_candidates.extend(MODULE.build_candidates(unit, fixture_people, 1, 8))
            self.assertEqual({row["label"] for row in all_candidates}, set(MODULE.FIVE_LABELS))
            for row in rows:
                self.assertEqual(row["difficulty_controls"]["document_count"], len(row["documents"]))
                self.assertEqual(row["difficulty_controls"]["char_length"], sum(len(doc["content"]) for doc in row["documents"]))

    def test_unsatisfiable_request_fails(self):
        with self.assertRaises(ValueError):
            MODULE.select_rows([], 1, 1)


if __name__ == "__main__":
    unittest.main()
