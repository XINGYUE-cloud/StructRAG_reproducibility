#!/usr/bin/env python3
"""Build a unit-disjoint synthetic five-type StructRAG router dataset.

This is an explicit extension for the two types absent from the public router
release. It uses facts emitted by the supplied person-profile generator and
records construction provenance, candidate structure simulations, and the
deterministic adjudication rule for every preference pair.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from src.five_type_router_data_v1 import FIVE_LABELS, label_counts, router_messages, write_jsonl


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def clean_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def build_unit_documents(people: list[dict[str, Any]]) -> list[dict[str, str]]:
    by_id = {str(person["person_id"]): person for person in people}
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for person in people:
        by_family[str(person["family_id"])] .append(person)

    documents: list[dict[str, str]] = []
    for family_id, members in sorted(by_family.items()):
        records: list[str] = []
        for person in sorted(members, key=lambda item: str(item["person_id"])):
            father = by_id.get(str(person.get("father_id", "")), {}).get("name", "无")
            mother = by_id.get(str(person.get("mother_id", "")), {}).get("name", "无")
            spouse = by_id.get(str(person.get("spouse_id", "")), {}).get("name", "无")
            children = [by_id.get(child_id, {}).get("name", child_id) for child_id in clean_list(person.get("children_ids"))]
            records.append(
                "\n".join(
                    [
                        f"姓名: {person['name']}",
                        f"性别: {person['gender']}",
                        f"年龄: {person['age']}",
                        f"公司: {person['company']}",
                        f"部门: {person['department']}",
                        f"小组: {person['team']}",
                        f"岗位: {person['position']}",
                        f"职级: {person['level']}",
                        f"工作年限: {person['work_years']}",
                        f"能力: {'、'.join(clean_list(person.get('skills')))}",
                        f"父亲: {father}",
                        f"母亲: {mother}",
                        f"配偶: {spouse}",
                        f"孩子: {'、'.join(children) if children else '无'}",
                    ]
                )
            )
        documents.append({"title": f"{family_id} 人物档案", "content": "\n\n".join(records)})
    return documents


def simulations(query: str) -> dict[str, str]:
    return {
        "chunk": f"Split the documents into independent records and locate direct evidence for: {query}",
        "table": f"Normalize candidate records into rows and fields, then compare/filter for: {query}",
        "graph": f"Create entity-relation edges and traverse relevant paths for: {query}",
        "algorithm": f"Create conditional decision steps and execute them for: {query}",
        "catalogue": f"Create a numbered hierarchical summary and navigate it for: {query}",
    }


def make_row(
    *,
    row_id: str,
    unit_id: str,
    label: str,
    query: str,
    answer: str,
    answer_keywords: list[str],
    documents: list[dict[str, str]],
    rng: random.Random,
) -> dict[str, Any]:
    if label not in FIVE_LABELS:
        raise ValueError(label)
    rejected = rng.choice([candidate for candidate in FIVE_LABELS if candidate != label])
    titles = [document["title"] for document in documents]
    messages = router_messages(query, titles)
    return {
        "id": row_id,
        "unit_id": unit_id,
        "label": label,
        "query": query,
        "answer": answer,
        "answer_keywords": answer_keywords,
        "documents": documents,
        "candidate_solutions": simulations(query),
        "adjudication": {
            "method": "construction-provenance rule",
            "target_structure": label,
            "reason": {
                "chunk": "A direct single-record attribute lookup was constructed.",
                "table": "A cross-record comparison/filtering task was constructed.",
                "graph": "A relationship-chain traversal task was constructed.",
                "algorithm": "A multi-condition configuration/decision task was constructed.",
                "catalogue": "A broad hierarchical-summary task was constructed.",
            }[label],
        },
        "chosen": messages + [{"role": "assistant", "content": label}],
        "rejected": messages + [{"role": "assistant", "content": rejected}],
    }


def unique_people(people: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(people, key=lambda item: str(item["person_id"]))


def family_graph_candidate(people: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    by_id = {str(person["person_id"]): person for person in people}
    candidates = []
    for person in people:
        father = by_id.get(str(person.get("father_id", "")))
        if father and father.get("spouse_id") and str(father["spouse_id"]) in by_id:
            candidates.append((person, by_id[str(father["spouse_id"])]))
    if not candidates:
        raise ValueError("No two-hop family relation available in unit")
    return candidates[0]


def records_for_unit(unit_id: str, people: list[dict[str, Any]], per_type: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    people = unique_people(people)
    by_id = {str(person["person_id"]): person for person in people}
    documents = build_unit_documents(people)
    output: list[dict[str, Any]] = []
    fields = [("公司", "company"), ("部门", "department"), ("岗位", "position"), ("所在城市", "city")]

    for index in range(per_type):
        person = people[index % len(people)]
        field_name, field_key = fields[index % len(fields)]
        output.append(
            make_row(
                row_id=f"{unit_id}-chunk-{index:03d}",
                unit_id=unit_id,
                label="chunk",
                query=f"在给定人物档案中，{person['name']}的{field_name}是什么？",
                answer=str(person[field_key]),
                answer_keywords=[str(person[field_key])],
                documents=documents,
                rng=rng,
            )
        )

        filter_person = people[(index * 3 + 1) % len(people)]
        matched = [
            item["name"]
            for item in people
            if item["company"] == filter_person["company"] and item["department"] == filter_person["department"]
        ]
        output.append(
            make_row(
                row_id=f"{unit_id}-table-{index:03d}",
                unit_id=unit_id,
                label="table",
                query=(
                    f"比较全部人物记录，筛选出同时在{filter_person['company']}的"
                    f"{filter_person['department']}工作的所有人，并给出姓名列表。"
                ),
                answer="、".join(sorted(matched)),
                answer_keywords=sorted(matched),
                documents=documents,
                rng=rng,
            )
        )

        source, target = family_graph_candidate(people)
        output.append(
            make_row(
                row_id=f"{unit_id}-graph-{index:03d}",
                unit_id=unit_id,
                label="graph",
                query=f"根据人物之间的关系链，{source['name']}的父亲的配偶是谁？",
                answer=str(target["name"]),
                answer_keywords=[str(target["name"])],
                documents=documents,
                rng=rng,
            )
        )

        candidate = people[(index * 5 + 2) % len(people)]
        required_skills = clean_list(candidate.get("skills"))[:2]
        selected = [
            item["name"]
            for item in people
            if item["company"] == candidate["company"]
            and item["department"] == candidate["department"]
            and all(skill in clean_list(item.get("skills")) for skill in required_skills)
        ]
        output.append(
            make_row(
                row_id=f"{unit_id}-algorithm-{index:03d}",
                unit_id=unit_id,
                label="algorithm",
                query=(
                    f"现需为{candidate['company']}的{candidate['department']}选择候选人：候选人必须"
                    f"同时具备{'和'.join(required_skills)}能力。请制定可执行的筛选决策步骤，并按步骤给出符合条件的姓名。"
                ),
                answer="、".join(sorted(selected)),
                answer_keywords=sorted(selected),
                documents=documents,
                rng=rng,
            )
        )

        catalogue_person = people[(index * 7 + 3) % len(people)]
        path = f"{catalogue_person['company']} > {catalogue_person['department']} > {catalogue_person['team']} > {catalogue_person['name']}"
        output.append(
            make_row(
                row_id=f"{unit_id}-catalogue-{index:03d}",
                unit_id=unit_id,
                label="catalogue",
                query=(
                    "请把全部人物档案按公司、部门、小组、成员组织为有编号的层级摘要，"
                    f"并标明{catalogue_person['name']}在该层级中的完整归属路径。"
                ),
                answer=path,
                answer_keywords=[
                    str(catalogue_person["company"]),
                    str(catalogue_person["department"]),
                    str(catalogue_person["team"]),
                    str(catalogue_person["name"]),
                ],
                documents=documents,
                rng=rng,
            )
        )
    return output


def split_units(unit_ids: list[str], seed: int, train_units: int, validation_units: int) -> dict[str, set[str]]:
    if train_units <= 0 or validation_units <= 0 or train_units + validation_units >= len(unit_ids):
        raise ValueError("Need positive train/validation counts and at least one held-out test unit")
    shuffled = list(unit_ids)
    random.Random(seed).shuffle(shuffled)
    return {
        "train": set(shuffled[:train_units]),
        "validation": set(shuffled[train_units : train_units + validation_units]),
        "test": set(shuffled[train_units + validation_units :]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--samples-per-type-per-unit", type=int, default=10)
    parser.add_argument("--train-units", type=int, default=18)
    parser.add_argument("--validation-units", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args()
    if args.samples_per_type_per_unit <= 0:
        raise ValueError("--samples-per-type-per-unit must be positive")
    if args.out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {args.out_dir}")

    people = read_jsonl(args.profiles)
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for person in people:
        by_unit[str(person["unit_id"])].append(person)
    unit_ids = sorted(by_unit)
    splits = split_units(unit_ids, args.seed, args.train_units, args.validation_units)

    all_rows: list[dict[str, Any]] = []
    for offset, unit_id in enumerate(unit_ids):
        all_rows.extend(
            records_for_unit(
                unit_id,
                by_unit[unit_id],
                args.samples_per_type_per_unit,
                args.seed + offset,
            )
        )
    by_split = {
        split_name: [row for row in all_rows if row["unit_id"] in split_unit_ids]
        for split_name, split_unit_ids in splits.items()
    }
    args.out_dir.mkdir(parents=True)
    for split_name, rows in by_split.items():
        write_jsonl(args.out_dir / f"{split_name}.jsonl", rows)
    write_jsonl(args.out_dir / "all.jsonl", all_rows)

    metadata = {
        "version": "synthetic_five_type_router_v1",
        "source": "Supplied person-profile dataset generator",
        "purpose": "Synthetic extension for algorithm and catalogue labels absent from public StructRAG router data",
        "seed": args.seed,
        "candidate_structures": list(FIVE_LABELS),
        "samples_per_type_per_unit": args.samples_per_type_per_unit,
        "split_unit_ids": {name: sorted(ids) for name, ids in splits.items()},
        "split_counts": {name: len(rows) for name, rows in by_split.items()},
        "split_label_counts": {name: label_counts(rows) for name, rows in by_split.items()},
        "leakage_policy": "All examples from a unit occur in exactly one split.",
        "construction_pipeline": {
            "synthesize": "Template-generated document QA tasks derived from generated person profiles.",
            "simulate": "Five candidate structure-use plans recorded per example.",
            "judge": "Construction-provenance adjudication assigns the type intentionally required by the generated task. This is deterministic, not an author-provided LLM judge.",
        },
        "limitation": "This is not the unreleased official five-type router training set and cannot support a paper-faithful five-type numeric comparison.",
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"split_counts": metadata["split_counts"], "split_label_counts": metadata["split_label_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
