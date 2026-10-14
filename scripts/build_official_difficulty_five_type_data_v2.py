#!/usr/bin/env python3
"""Build a five-type router/StructRAG dataset calibrated to Loong difficulty.

The public StructRAG router release has only three labels.  This generator keeps
the five paper structures separate and makes the difficulty controls explicit:
the four levels follow the released Loong level proportions (250/300/641/409
out of 1,600).  All examples from one information unit stay in one split.

The generated rows are educational synthetic data.  Their difficulty metadata is
an experimental control, not a claim that they are the unreleased official
five-type router examples.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from src.five_type_router_data_v1 import FIVE_LABELS, router_messages, write_jsonl


DIFFICULTY_LEVELS = (1, 2, 3, 4)
# Official Loong level counts reported in the paper/reproduction notes.
OFFICIAL_LEVEL_COUNTS = {1: 250, 2: 300, 3: 641, 4: 409}
OFFICIAL_LEVEL_WEIGHTS = {
    level: count / sum(OFFICIAL_LEVEL_COUNTS.values())
    for level, count in OFFICIAL_LEVEL_COUNTS.items()
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def clean_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def build_unit_documents(people: list[dict[str, Any]]) -> list[dict[str, str]]:
    by_id = {str(person["person_id"]): person for person in people}
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for person in people:
        by_family[str(person["family_id"])].append(person)

    documents: list[dict[str, str]] = []
    for family_id, members in sorted(by_family.items()):
        records: list[str] = []
        for person in sorted(members, key=lambda item: str(item["person_id"])):
            def name(pid: Any) -> str:
                return str(by_id.get(str(pid), {}).get("name", "无")) if pid else "无"

            children = [name(child_id) for child_id in clean_list(person.get("children_ids"))]
            records.append("\n".join([
                f"姓名: {person['name']}",
                f"性别: {person.get('gender', '未知')}",
                f"年龄: {person.get('age', '未知')}",
                f"城市: {person.get('city', '未知')}",
                f"公司: {person.get('company', '未知')}",
                f"部门: {person.get('department', '未知')}",
                f"小组: {person.get('team', '未知')}",
                f"岗位: {person.get('position', '未知')}",
                f"职级: {person.get('level', '未知')}",
                f"工作年限: {person.get('work_years', '未知')}",
                f"工作状态: {person.get('work_status', '未知')}",
                f"能力: {'、'.join(clean_list(person.get('skills'))) or '无'}",
                f"父亲: {name(person.get('father_id'))}",
                f"母亲: {name(person.get('mother_id'))}",
                f"配偶: {name(person.get('spouse_id'))}",
                f"孩子: {'、'.join(children) if children else '无'}",
            ]))
        documents.append({"title": f"{family_id} 人物档案", "content": "\n\n".join(records)})
    return documents


def graph_path(people: list[dict[str, Any]], start: dict[str, Any], level: int) -> tuple[list[str], list[str]]:
    by_id = {str(person["person_id"]): person for person in people}
    chains = [
        ["father_id"],
        ["mother_id", "spouse_id"],
        ["father_id", "father_id", "spouse_id"],
        ["spouse_id", "father_id", "father_id", "spouse_id"],
    ]
    chain = chains[level - 1]
    current = start
    ids = [str(start["person_id"])]
    relations: list[str] = []
    for field in chain:
        target_id = str(current.get(field, ""))
        if not target_id or target_id not in by_id:
            return [], []
        current = by_id[target_id]
        ids.append(target_id)
        relations.append({
            "father_id": "父亲",
            "mother_id": "母亲",
            "spouse_id": "配偶",
        }[field])
    return ids, relations


def choose_people(people: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    return sorted(people, key=lambda item: (str(item.get("person_id")), seed))


def level_controls(level: int) -> dict[str, int]:
    return {
        1: {"hops": 1, "conditions": 1, "distractors": 0, "hierarchy_depth": 2},
        2: {"hops": 2, "conditions": 2, "distractors": 1, "hierarchy_depth": 3},
        3: {"hops": 3, "conditions": 3, "distractors": 2, "hierarchy_depth": 4},
        4: {"hops": 4, "conditions": 4, "distractors": 3, "hierarchy_depth": 5},
    }[level]


def make_row(
    *, row_id: str, unit_id: str, label: str, level: int, query: str,
    answer: str, keywords: list[str], documents: list[dict[str, str]],
    rng: random.Random, task_type: str,
) -> dict[str, Any]:
    controls = level_controls(level)
    rejected = rng.choice([candidate for candidate in FIVE_LABELS if candidate != label])
    titles = [document["title"] for document in documents]
    messages = router_messages(query, titles)
    return {
        "id": row_id,
        "unit_id": unit_id,
        "label": label,
        "task_type": task_type,
        "difficulty_level": level,
        "difficulty_controls": controls,
        "query": query,
        "answer": answer,
        "answer_keywords": keywords,
        "documents": documents,
        "chosen": messages + [{"role": "assistant", "content": label}],
        "rejected": messages + [{"role": "assistant", "content": rejected}],
        "provenance": {
            "generator": "build_official_difficulty_five_type_data_v2",
            "difficulty_reference": "official Loong level distribution 250/300/641/409",
            "adjudication": "structure is selected from the operation required by the query",
        },
    }


def build_candidates(unit_id: str, people: list[dict[str, Any]], per_level: int, seed: int) -> list[dict[str, Any]]:
    people = choose_people(people, seed)
    by_id = {str(person["person_id"]): person for person in people}
    documents = build_unit_documents(people)
    rng = random.Random(seed)
    output: list[dict[str, Any]] = []
    if not people:
        return output

    for level in DIFFICULTY_LEVELS:
        for index in range(per_level):
            person = people[(index + level * 3) % len(people)]
            field_key, field_name = [("company", "公司"), ("department", "部门"), ("position", "岗位"), ("city", "城市")][index % 4]
            output.append(make_row(
                row_id=f"{unit_id}-L{level}-chunk-{index:03d}", unit_id=unit_id,
                label="chunk", level=level, task_type="direct_lookup",
                query=f"请在人物档案中查找{person['name']}的{field_name}。" + ("只依据直接记录回答。" if level == 1 else "同时忽略其他人物的相似字段。"),
                answer=str(person.get(field_key, "未知")), keywords=[str(person.get(field_key, "未知"))],
                documents=documents, rng=rng,
            ))

            target = people[(index * 3 + level) % len(people)]
            matched = [p for p in people if p.get("company") == target.get("company")]
            if level >= 2:
                matched = [p for p in matched if p.get("department") == target.get("department")]
            if level >= 3:
                matched = [p for p in matched if p.get("work_status") == target.get("work_status")]
            if level >= 4:
                matched = sorted(matched, key=lambda p: int(p.get("age", 0)), reverse=True)[:3]
            names = sorted(str(p["name"]) for p in matched)
            conditions = [f"公司={target.get('company')}"]
            if level >= 2:
                conditions.append(f"部门={target.get('department')}")
            if level >= 3:
                conditions.append(f"工作状态={target.get('work_status')}")
            if level >= 4:
                conditions.append("按年龄从大到小取前三名")
            output.append(make_row(
                row_id=f"{unit_id}-L{level}-table-{index:03d}", unit_id=unit_id,
                label="table", level=level, task_type="comparison_filter",
                query="请把人物记录整理成表格，筛选并比较：" + "；".join(conditions) + "，输出符合条件的姓名。",
                answer="、".join(names), keywords=names or ["无"], documents=documents, rng=rng,
            ))

            path_ids, relations = graph_path(people, person, level)
            if not path_ids:
                # Keep the row valid on worlds without a long enough family chain.
                path_ids, relations = [str(person["person_id"])], ["本人"]
            target_person = by_id[path_ids[-1]]
            query = f"沿关系链回答：{person['name']}" + "的".join([""] + relations) + "是谁？"
            output.append(make_row(
                row_id=f"{unit_id}-L{level}-graph-{index:03d}", unit_id=unit_id,
                label="graph", level=level, task_type="relation_chain",
                query=query, answer=str(target_person["name"]), keywords=[str(target_person["name"])],
                documents=documents, rng=rng,
            ))

            skills = clean_list(person.get("skills")) or [str(person.get("position", "岗位"))]
            required = skills[: min(level, len(skills))]
            selected = [
                p for p in people
                if all(skill in clean_list(p.get("skills")) for skill in required)
                and (level < 2 or p.get("company") == person.get("company"))
                and (level < 3 or p.get("department") == person.get("department"))
                and (level < 4 or int(p.get("work_years", 0)) >= int(person.get("work_years", 0)))
            ]
            selected_names = sorted(str(p["name"]) for p in selected)
            steps = "；".join([
                f"检查能力={','.join(required)}",
                "限定公司" if level >= 2 else "保留全部公司",
                "限定部门" if level >= 3 else "保留全部部门",
                "比较工作年限并保留满足者" if level >= 4 else "输出满足条件者",
            ])
            output.append(make_row(
                row_id=f"{unit_id}-L{level}-algorithm-{index:03d}", unit_id=unit_id,
                label="algorithm", level=level, task_type="conditional_decision",
                query=f"请按以下决策步骤选择候选人：{steps}。最终输出姓名列表。",
                answer="、".join(selected_names), keywords=selected_names or ["无"], documents=documents, rng=rng,
            ))

            cat = people[(index * 5 + level) % len(people)]
            path = [str(cat.get("company", "未知")), str(cat.get("department", "未知")), str(cat.get("team", "未知")), str(cat.get("name", "未知"))]
            if level >= 4:
                path.insert(0, str(cat.get("city", "未知")))
            hierarchy = "、".join(["城市", "公司", "部门", "小组", "成员"][-level - 1:])
            output.append(make_row(
                row_id=f"{unit_id}-L{level}-catalogue-{index:03d}", unit_id=unit_id,
                label="catalogue", level=level, task_type="hierarchical_summary",
                query=f"请按{hierarchy}建立编号层级摘要，并给出{cat['name']}的完整归属路径。",
                answer=" > ".join(path), keywords=path, documents=documents, rng=rng,
            ))
    return output


def split_units(unit_ids: list[str], seed: int, train_units: int, validation_units: int) -> dict[str, set[str]]:
    if train_units + validation_units >= len(unit_ids):
        raise ValueError("train/validation units must leave at least one test unit")
    shuffled = list(unit_ids)
    random.Random(seed).shuffle(shuffled)
    return {
        "train": set(shuffled[:train_units]),
        "validation": set(shuffled[train_units:train_units + validation_units]),
        "test": set(shuffled[train_units + validation_units:]),
    }


def target_counts(total: int) -> dict[int, int]:
    raw = {level: total * weight for level, weight in OFFICIAL_LEVEL_WEIGHTS.items()}
    counts = {level: int(raw[level]) for level in DIFFICULTY_LEVELS}
    remainder = total - sum(counts.values())
    for level in sorted(DIFFICULTY_LEVELS, key=lambda item: raw[item] - counts[item], reverse=True)[:remainder]:
        counts[level] += 1
    return counts


def select_split(rows: list[dict[str, Any]], total: int, seed: int) -> list[dict[str, Any]]:
    if total <= 0 or total > len(rows):
        raise ValueError(f"Requested {total} rows from {len(rows)} candidates")
    by_level: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_level[int(row["difficulty_level"])].append(row)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for level, count in target_counts(total).items():
        candidates = list(by_level[level])
        rng.shuffle(candidates)
        if len(candidates) < count:
            raise ValueError(f"Level {level} has {len(candidates)} candidates, needs {count}")
        selected.extend(candidates[:count])
    rng.shuffle(selected)
    return selected


def validate_splits(splits: dict[str, list[dict[str, Any]]]) -> None:
    seen: dict[str, str] = {}
    for split, rows in splits.items():
        for row in rows:
            prior = seen.setdefault(str(row["unit_id"]), split)
            if prior != split:
                raise ValueError(f"unit leakage: {row['unit_id']} in {prior} and {split}")
    for split, rows in splits.items():
        if len(rows) != len({row["id"] for row in rows}):
            raise ValueError(f"duplicate row ids in {split}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-units", type=int, default=18)
    parser.add_argument("--validation-units", type=int, default=6)
    parser.add_argument("--train-count", type=int, default=3200)
    parser.add_argument("--validation-count", type=int, default=800)
    parser.add_argument("--test-count", type=int, default=1600)
    parser.add_argument("--candidates-per-level", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {args.out_dir}")
    people = read_jsonl(args.profiles)
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for person in people:
        by_unit[str(person["unit_id"])].append(person)
    unit_ids = sorted(by_unit)
    splits_units = split_units(unit_ids, args.seed, args.train_units, args.validation_units)
    candidates_by_split: dict[str, list[dict[str, Any]]] = {name: [] for name in splits_units}
    for offset, unit_id in enumerate(unit_ids):
        split = next(name for name, ids in splits_units.items() if unit_id in ids)
        candidates_by_split[split].extend(build_candidates(unit_id, by_unit[unit_id], args.candidates_per_level, args.seed + offset))
    requested = {"train": args.train_count, "validation": args.validation_count, "test": args.test_count}
    selected = {split: select_split(rows, requested[split], args.seed + index) for index, (split, rows) in enumerate(candidates_by_split.items())}
    validate_splits(selected)
    args.out_dir.mkdir(parents=True)
    for split, rows in selected.items():
        write_jsonl(args.out_dir / f"{split}.jsonl", rows)

    metadata = {
        "version": "official_difficulty_five_type_v2",
        "source": "supplied person-profile generator",
        "seed": args.seed,
        "labels": list(FIVE_LABELS),
        "official_reference": {
            "benchmark": "Loong",
            "level_counts": OFFICIAL_LEVEL_COUNTS,
            "level_weights": OFFICIAL_LEVEL_WEIGHTS,
            "note": "Only the difficulty distribution is matched; this is not official Loong data.",
        },
        "split_units": {split: sorted(ids) for split, ids in splits_units.items()},
        "split_counts": {split: len(rows) for split, rows in selected.items()},
        "split_label_counts": {split: dict(sorted(Counter(row["label"] for row in rows).items())) for split, rows in selected.items()},
        "split_level_counts": {split: dict(sorted(Counter(str(row["difficulty_level"]) for row in rows).items())) for split, rows in selected.items()},
        "difficulty_controls": {str(level): level_controls(level) for level in DIFFICULTY_LEVELS},
        "leakage_policy": "All examples from one unit_id occur in exactly one split.",
        "candidate_count_by_split": {split: len(rows) for split, rows in candidates_by_split.items()},
    }
    write_json(args.out_dir / "metadata.json", metadata)
    print(json.dumps({"split_counts": metadata["split_counts"], "split_level_counts": metadata["split_level_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
