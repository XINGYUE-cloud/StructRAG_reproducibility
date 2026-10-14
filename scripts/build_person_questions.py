#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate person-profile QA files from basic person profiles.

Input:
  people_profiles.jsonl

Output:
  person_attribute.jsonl
  family_member.jsonl
  derived_kinship.jsonl
  multi_hop_kinship.jsonl
  attribute_filter.jsonl
  all_questions.jsonl
  person_question_metadata.json

Design principles:
  1. This script only generates Q/A metadata. It does NOT render contexts.
  2. QA is representation-agnostic: it stores gold_entity_ids instead of
     view-specific chunk IDs such as "U0001_P001::自然语言文本".
  3. The underlying knowledge uses only basic person-profile relations:
     father_id, mother_id, spouse_id, children_ids.
  4. Derived relations such as 哥哥、妹妹、伯父 are used only in questions and
     reasoning paths; they are NOT added to the knowledge base.
  5. Internal generation fields such as generation/family_id are not written
     into the QA output.

Example:
  python scripts/build_person_questions.py ^
    --profiles data/demo_profiles/people_profiles.jsonl ^
    --out data/demo_person_questions ^
    --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ATTRIBUTE_FIELDS = [
    ("配偶", "spouse_id", "人物"),
    ("父亲", "father_id", "人物"),
    ("母亲", "mother_id", "人物"),
    ("公司", "company", "文本"),
    ("总部/分部", "branch", "文本"),
    ("部门", "department", "文本"),
    ("小组", "team", "文本"),
    ("岗位", "position", "文本"),
    ("所在城市", "city", "文本"),
]

CHAIN_PATTERNS = [
    (["配偶", "父亲"], "中等-2跳"),
    (["配偶", "母亲"], "中等-2跳"),
    (["妹妹", "配偶"], "中等-2跳"),
    (["哥哥", "配偶"], "中等-2跳"),
    (["哥哥", "妹妹"], "中等-2跳"),
    (["妹妹", "配偶", "父亲"], "困难-3跳"),
    (["哥哥", "妹妹", "配偶"], "困难-3跳"),
    (["妹妹", "配偶", "伯父"], "困难-3跳"),
    (["哥哥", "妹妹", "配偶", "伯父"], "困难-4跳"),
    (["父亲", "哥哥", "配偶"], "困难-3跳"),
    (["母亲", "姐姐", "配偶"], "困难-3跳"),
]


@dataclass
class ResolvedStep:
    to_id: str
    evidence_ids: list[str]
    evidence: str


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}") from e
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, obj: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def group_by_unit(people: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for p in people:
        grouped.setdefault(str(p["unit_id"]), []).append(p)
    return grouped


def person_name(pid: str | None, people_by_id: dict[str, dict[str, Any]]) -> str:
    if not pid:
        return "无"
    p = people_by_id.get(pid)
    return str(p.get("name", pid)) if p else str(pid)


def ids_to_names(ids: list[str], people_by_id: dict[str, dict[str, Any]]) -> str:
    return "、".join(person_name(pid, people_by_id) for pid in ids) if ids else "无"


def same_parents(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return bool(a.get("father_id") and a.get("mother_id") and a.get("father_id") == b.get("father_id") and a.get("mother_id") == b.get("mother_id"))


def siblings(pid: str, people_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    p = people_by_id[pid]
    out = []
    for other in people_by_id.values():
        if other["person_id"] == pid:
            continue
        if same_parents(p, other):
            out.append(other)
    return out


def older_brothers(pid: str, people_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    p = people_by_id[pid]
    return [s for s in siblings(pid, people_by_id) if s.get("gender") == "男" and int(s.get("age", 0)) > int(p.get("age", 0))]


def younger_brothers(pid: str, people_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    p = people_by_id[pid]
    return [s for s in siblings(pid, people_by_id) if s.get("gender") == "男" and int(s.get("age", 0)) < int(p.get("age", 0))]


def older_sisters(pid: str, people_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    p = people_by_id[pid]
    return [s for s in siblings(pid, people_by_id) if s.get("gender") == "女" and int(s.get("age", 0)) > int(p.get("age", 0))]


def younger_sisters(pid: str, people_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    p = people_by_id[pid]
    return [s for s in siblings(pid, people_by_id) if s.get("gender") == "女" and int(s.get("age", 0)) < int(p.get("age", 0))]


def resolve_basic_relation(pid: str, relation: str, people_by_id: dict[str, dict[str, Any]]) -> list[ResolvedStep]:
    p = people_by_id[pid]
    src = person_name(pid, people_by_id)

    def one(target_id: str | None, evidence: str) -> list[ResolvedStep]:
        if not target_id or target_id not in people_by_id:
            return []
        return [ResolvedStep(to_id=target_id, evidence_ids=[pid, target_id], evidence=evidence)]

    if relation == "父亲":
        return one(p.get("father_id"), f"{src}的人物档案中记录了父亲字段。")
    if relation == "母亲":
        return one(p.get("mother_id"), f"{src}的人物档案中记录了母亲字段。")
    if relation in ("配偶", "伴侣"):
        return one(p.get("spouse_id"), f"{src}的人物档案中记录了配偶字段。")
    if relation == "孩子":
        out = []
        for cid in p.get("children_ids") or []:
            if cid in people_by_id:
                out.append(ResolvedStep(to_id=cid, evidence_ids=[pid, cid], evidence=f"{src}的人物档案中记录了孩子列表。"))
        return out
    if relation == "儿子":
        out = []
        for cid in p.get("children_ids") or []:
            c = people_by_id.get(cid)
            if c and c.get("gender") == "男":
                out.append(ResolvedStep(to_id=cid, evidence_ids=[pid, cid], evidence=f"{src}的孩子列表中包含{c['name']}，且{c['name']}性别为男。"))
        return out
    if relation == "女儿":
        out = []
        for cid in p.get("children_ids") or []:
            c = people_by_id.get(cid)
            if c and c.get("gender") == "女":
                out.append(ResolvedStep(to_id=cid, evidence_ids=[pid, cid], evidence=f"{src}的孩子列表中包含{c['name']}，且{c['name']}性别为女。"))
        return out
    return []


def resolve_derived_relation(pid: str, relation: str, people_by_id: dict[str, dict[str, Any]]) -> list[ResolvedStep]:
    p = people_by_id[pid]
    src = p["name"]

    def sibling_steps(candidates: list[dict[str, Any]], relation_name: str) -> list[ResolvedStep]:
        out = []
        for c in candidates:
            cname = c["name"]
            evidence = (
                f"{src}和{cname}父母相同；"
                f"{cname}性别为{c.get('gender')}，年龄为{c.get('age')}，"
                f"{src}年龄为{p.get('age')}，因此{cname}是{src}的{relation_name}。"
            )
            out.append(ResolvedStep(to_id=c["person_id"], evidence_ids=[pid, c["person_id"]], evidence=evidence))
        return out

    if relation == "哥哥":
        return sibling_steps(older_brothers(pid, people_by_id), "哥哥")
    if relation == "弟弟":
        return sibling_steps(younger_brothers(pid, people_by_id), "弟弟")
    if relation == "姐姐":
        return sibling_steps(older_sisters(pid, people_by_id), "姐姐")
    if relation == "妹妹":
        return sibling_steps(younger_sisters(pid, people_by_id), "妹妹")

    # 伯父 = 父亲的哥哥. 叔叔 = 父亲的弟弟.
    # 姑姑 = 父亲的姐妹. 舅舅/姨妈 from mother side.
    parent_relation_map = {
        "伯父": ("父亲", "哥哥"),
        "叔叔": ("父亲", "弟弟"),
        "姑姑": ("父亲", "姐姐或妹妹"),
        "舅舅": ("母亲", "哥哥或弟弟"),
        "姨妈": ("母亲", "姐姐或妹妹"),
    }
    if relation in parent_relation_map:
        parent_rel, sub_rel = parent_relation_map[relation]
        parent_steps = resolve_basic_relation(pid, parent_rel, people_by_id)
        if len(parent_steps) != 1:
            return []
        parent_id = parent_steps[0].to_id
        if relation == "伯父":
            candidates = older_brothers(parent_id, people_by_id)
        elif relation == "叔叔":
            candidates = younger_brothers(parent_id, people_by_id)
        elif relation == "姑姑":
            candidates = older_sisters(parent_id, people_by_id) + younger_sisters(parent_id, people_by_id)
        elif relation == "舅舅":
            candidates = older_brothers(parent_id, people_by_id) + younger_brothers(parent_id, people_by_id)
        elif relation == "姨妈":
            candidates = older_sisters(parent_id, people_by_id) + younger_sisters(parent_id, people_by_id)
        else:
            candidates = []
        out = []
        parent_name = person_name(parent_id, people_by_id)
        for c in candidates:
            cname = c["name"]
            evidence = f"{src}的{parent_rel}是{parent_name}；{cname}是{parent_name}的{sub_rel}，因此{cname}是{src}的{relation}。"
            out.append(ResolvedStep(to_id=c["person_id"], evidence_ids=[pid, parent_id, c["person_id"]], evidence=evidence))
        return out

    return []


def resolve_relation(pid: str, relation: str, people_by_id: dict[str, dict[str, Any]]) -> list[ResolvedStep]:
    basic = resolve_basic_relation(pid, relation, people_by_id)
    if basic:
        return basic
    return resolve_derived_relation(pid, relation, people_by_id)


def resolve_chain(
    start_id: str,
    chain: list[str],
    people_by_id: dict[str, dict[str, Any]],
    require_unique_each_step: bool = True,
) -> tuple[str, list[dict[str, Any]], list[str]] | None:
    current_id = start_id
    reasoning_path: list[dict[str, Any]] = []
    gold_ids: list[str] = [start_id]

    for idx, rel in enumerate(chain, start=1):
        options = resolve_relation(current_id, rel, people_by_id)
        if not options:
            return None
        if require_unique_each_step and len(options) != 1:
            return None
        step = options[0]
        from_name = person_name(current_id, people_by_id)
        to_name = person_name(step.to_id, people_by_id)
        reasoning_path.append({
            "step": idx,
            "from_id": current_id,
            "from": from_name,
            "relation": rel,
            "to_id": step.to_id,
            "to": to_name,
            "evidence_entity_ids": step.evidence_ids,
            "evidence": step.evidence,
        })
        for eid in step.evidence_ids:
            if eid not in gold_ids:
                gold_ids.append(eid)
        current_id = step.to_id

    return current_id, reasoning_path, gold_ids


def make_question_id(unit_id: str, counter: int) -> str:
    return f"{unit_id}_Q{counter:04d}"


def make_candidate_ids(unit_people: list[dict[str, Any]]) -> list[str]:
    return [p["person_id"] for p in unit_people]


def make_question_record(
    question_id: str,
    unit_id: str,
    task_type: str,
    difficulty: str,
    question: str,
    answer: str,
    answer_ids: list[str],
    gold_entity_ids: list[str],
    candidate_entity_ids: list[str],
    reasoning_path: list[dict[str, Any]],
    required_operations: list[str],
) -> dict[str, Any]:
    # Deduplicate while preserving order.
    gold_unique: list[str] = []
    for eid in gold_entity_ids:
        if eid and eid not in gold_unique:
            gold_unique.append(eid)
    ans_unique: list[str] = []
    for aid in answer_ids:
        if aid and aid not in ans_unique:
            ans_unique.append(aid)
    return {
        "question_id": question_id,
        "unit_id": unit_id,
        "task_type": task_type,
        "difficulty": difficulty,
        "question": question,
        "answer": answer,
        "answer_ids": ans_unique,
        "answer_type": "人物" if ans_unique else "文本",
        "gold_entity_ids": gold_unique,
        "candidate_entity_ids": candidate_entity_ids,
        "reasoning_path": reasoning_path,
        "required_operations": required_operations,
    }


def generate_attribute_questions(
    unit_id: str,
    unit_people: list[dict[str, Any]],
    people_by_id: dict[str, dict[str, Any]],
    rng: random.Random,
    counter_start: int,
    max_questions: int,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    counter = counter_start
    candidates = make_candidate_ids(unit_people)
    shuffled_people = list(unit_people)
    rng.shuffle(shuffled_people)

    for person in shuffled_people:
        field_options = list(ATTRIBUTE_FIELDS)
        rng.shuffle(field_options)
        for label, field, answer_kind in field_options:
            if len(rows) >= max_questions:
                return rows, counter
            value = person.get(field)
            if value in (None, "", [], "无"):
                continue
            if field.endswith("_id"):
                if value not in people_by_id:
                    continue
                answer = person_name(value, people_by_id)
                answer_ids = [value]
            else:
                answer = str(value)
                answer_ids = []
            q = f"{person['name']}的{label}是谁？" if answer_kind == "人物" else f"{person['name']}的{label}是什么？"
            rows.append(make_question_record(
                question_id=make_question_id(unit_id, counter),
                unit_id=unit_id,
                task_type="单人物档案查询",
                difficulty="简单-单块",
                question=q,
                answer=answer,
                answer_ids=answer_ids,
                gold_entity_ids=[person["person_id"]] + answer_ids,
                candidate_entity_ids=candidates,
                reasoning_path=[{
                    "step": 1,
                    "from_id": person["person_id"],
                    "from": person["name"],
                    "relation": label,
                    "to_id": answer_ids[0] if answer_ids else None,
                    "to": answer,
                    "evidence_entity_ids": [person["person_id"]] + answer_ids,
                    "evidence": f"查看{person['name']}的人物档案中的{label}字段。",
                }],
                required_operations=["人物档案字段查询"],
            ))
            counter += 1
            break
    return rows, counter


def generate_family_member_questions(
    unit_id: str,
    unit_people: list[dict[str, Any]],
    people_by_id: dict[str, dict[str, Any]],
    rng: random.Random,
    counter_start: int,
    max_questions: int,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    counter = counter_start
    candidates = make_candidate_ids(unit_people)
    people_with_children = [p for p in unit_people if p.get("children_ids")]
    rng.shuffle(people_with_children)

    for person in people_with_children:
        if len(rows) >= max_questions:
            break
        rel_options = ["孩子", "儿子", "女儿"]
        rng.shuffle(rel_options)
        for rel in rel_options:
            steps = resolve_relation(person["person_id"], rel, people_by_id)
            if not steps:
                continue
            answer_ids = [s.to_id for s in steps]
            answer = ids_to_names(answer_ids, people_by_id)
            q = f"{person['name']}的{rel}是谁？"
            path = []
            for i, s in enumerate(steps, start=1):
                path.append({
                    "step": i,
                    "from_id": person["person_id"],
                    "from": person["name"],
                    "relation": rel,
                    "to_id": s.to_id,
                    "to": person_name(s.to_id, people_by_id),
                    "evidence_entity_ids": s.evidence_ids,
                    "evidence": s.evidence,
                })
            gold = [person["person_id"]]
            for s in steps:
                gold.extend(s.evidence_ids)
            rows.append(make_question_record(
                question_id=make_question_id(unit_id, counter),
                unit_id=unit_id,
                task_type="家庭成员查询",
                difficulty="简单-列表筛选",
                question=q,
                answer=answer,
                answer_ids=answer_ids,
                gold_entity_ids=gold,
                candidate_entity_ids=candidates,
                reasoning_path=path,
                required_operations=["孩子列表查询", "性别筛选"] if rel in ("儿子", "女儿") else ["孩子列表查询"],
            ))
            counter += 1
            break
    return rows, counter


def generate_derived_relation_questions(
    unit_id: str,
    unit_people: list[dict[str, Any]],
    people_by_id: dict[str, dict[str, Any]],
    rng: random.Random,
    counter_start: int,
    max_questions: int,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    counter = counter_start
    candidates = make_candidate_ids(unit_people)
    relation_pool = ["哥哥", "姐姐", "弟弟", "妹妹", "伯父", "叔叔", "姑姑", "舅舅", "姨妈"]
    shuffled_people = list(unit_people)
    rng.shuffle(shuffled_people)

    for person in shuffled_people:
        if len(rows) >= max_questions:
            break
        rels = list(relation_pool)
        rng.shuffle(rels)
        for rel in rels:
            steps = resolve_relation(person["person_id"], rel, people_by_id)
            if len(steps) != 1:
                continue
            s = steps[0]
            answer = person_name(s.to_id, people_by_id)
            q = f"{person['name']}的{rel}是谁？"
            rows.append(make_question_record(
                question_id=make_question_id(unit_id, counter),
                unit_id=unit_id,
                task_type="派生亲属关系推理",
                difficulty="中等-派生关系",
                question=q,
                answer=answer,
                answer_ids=[s.to_id],
                gold_entity_ids=s.evidence_ids,
                candidate_entity_ids=candidates,
                reasoning_path=[{
                    "step": 1,
                    "from_id": person["person_id"],
                    "from": person["name"],
                    "relation": rel,
                    "to_id": s.to_id,
                    "to": answer,
                    "evidence_entity_ids": s.evidence_ids,
                    "evidence": s.evidence,
                }],
                required_operations=["基础亲属关系读取", "年龄/性别条件判断", "派生称谓推导"],
            ))
            counter += 1
            break
    return rows, counter


def generate_chain_questions(
    unit_id: str,
    unit_people: list[dict[str, Any]],
    people_by_id: dict[str, dict[str, Any]],
    rng: random.Random,
    counter_start: int,
    max_questions: int,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    counter = counter_start
    candidates = make_candidate_ids(unit_people)
    shuffled_people = list(unit_people)
    rng.shuffle(shuffled_people)
    # Prefer harder chains first so this task does not collapse into simple one-hop questions.
    patterns = list(CHAIN_PATTERNS)
    rng.shuffle(patterns)
    patterns.sort(key=lambda item: len(item[0]), reverse=True)

    for person in shuffled_people:
        if len(rows) >= max_questions:
            break
        for chain, difficulty in patterns:
            result = resolve_chain(person["person_id"], chain, people_by_id, require_unique_each_step=True)
            if result is None:
                continue
            answer_id, path, gold_ids = result
            # Avoid trivial self-answer unless the chain explicitly creates it; self-answer is often confusing.
            if answer_id == person["person_id"]:
                continue
            answer = person_name(answer_id, people_by_id)
            chain_text = "的".join(chain)
            q = f"{person['name']}的{chain_text}是谁？"
            rows.append(make_question_record(
                question_id=make_question_id(unit_id, counter),
                unit_id=unit_id,
                task_type="多步亲属链路推理",
                difficulty=difficulty,
                question=q,
                answer=answer,
                answer_ids=[answer_id],
                gold_entity_ids=gold_ids,
                candidate_entity_ids=candidates,
                reasoning_path=path,
                required_operations=["基础亲属关系读取", "派生称谓推导", "多步链路追踪"],
            ))
            counter += 1
            break
    return rows, counter


def generate_attribute_filter_questions(
    unit_id: str,
    unit_people: list[dict[str, Any]],
    people_by_id: dict[str, dict[str, Any]],
    rng: random.Random,
    counter_start: int,
    max_questions: int,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    counter = counter_start
    candidates = make_candidate_ids(unit_people)
    people_with_children = [p for p in unit_people if len(p.get("children_ids") or []) >= 2]
    rng.shuffle(people_with_children)

    for parent in people_with_children:
        if len(rows) >= max_questions:
            break
        child_ids = [cid for cid in parent.get("children_ids") or [] if cid in people_by_id]
        if len(child_ids) < 2:
            continue
        children = [people_by_id[cid] for cid in child_ids]
        query_types = ["年龄最大", "年龄最小", "岗位", "能力"]
        rng.shuffle(query_types)
        made = False
        for qt in query_types:
            if qt == "年龄最大":
                sorted_children = sorted(children, key=lambda x: int(x.get("age", 0)), reverse=True)
                if len(sorted_children) < 2 or sorted_children[0].get("age") == sorted_children[1].get("age"):
                    continue
                answer_ids = [sorted_children[0]["person_id"]]
                q = f"{parent['name']}的孩子中，谁年龄最大？"
                ops = ["孩子列表查询", "年龄比较"]
            elif qt == "年龄最小":
                sorted_children = sorted(children, key=lambda x: int(x.get("age", 0)))
                if len(sorted_children) < 2 or sorted_children[0].get("age") == sorted_children[1].get("age"):
                    continue
                answer_ids = [sorted_children[0]["person_id"]]
                q = f"{parent['name']}的孩子中，谁年龄最小？"
                ops = ["孩子列表查询", "年龄比较"]
            elif qt == "岗位":
                # Choose a position that uniquely identifies one child.
                counts: dict[str, list[str]] = {}
                for c in children:
                    counts.setdefault(str(c.get("position")), []).append(c["person_id"])
                unique = [(pos, ids) for pos, ids in counts.items() if pos and len(ids) == 1]
                if not unique:
                    continue
                pos, ids = rng.choice(unique)
                answer_ids = ids
                q = f"{parent['name']}的孩子中，谁的岗位是{pos}？"
                ops = ["孩子列表查询", "岗位筛选"]
            else:  # 能力
                counts: dict[str, list[str]] = {}
                for c in children:
                    for skill in c.get("skills") or []:
                        counts.setdefault(str(skill), []).append(c["person_id"])
                unique = [(skill, ids) for skill, ids in counts.items() if len(ids) == 1]
                if not unique:
                    continue
                skill, ids = rng.choice(unique)
                answer_ids = ids
                q = f"{parent['name']}的孩子中，谁具备{skill}能力？"
                ops = ["孩子列表查询", "能力筛选"]

            answer = ids_to_names(answer_ids, people_by_id)
            gold = [parent["person_id"]] + child_ids
            path = [{
                "step": 1,
                "from_id": parent["person_id"],
                "from": parent["name"],
                "relation": "孩子",
                "to_ids": child_ids,
                "to": ids_to_names(child_ids, people_by_id),
                "evidence_entity_ids": [parent["person_id"]] + child_ids,
                "evidence": f"先读取{parent['name']}的人物档案，得到其孩子列表。",
            }, {
                "step": 2,
                "operation": "筛选/比较",
                "answer_ids": answer_ids,
                "answer": answer,
                "evidence_entity_ids": child_ids,
                "evidence": "再读取这些孩子的人物档案，根据问题要求进行比较或筛选。",
            }]
            rows.append(make_question_record(
                question_id=make_question_id(unit_id, counter),
                unit_id=unit_id,
                task_type="属性筛选型关系问题",
                difficulty="中等-多块筛选",
                question=q,
                answer=answer,
                answer_ids=answer_ids,
                gold_entity_ids=gold,
                candidate_entity_ids=candidates,
                reasoning_path=path,
                required_operations=ops,
            ))
            counter += 1
            made = True
            break
        if not made:
            continue
    return rows, counter


def generate_questions_for_unit(
    unit_id: str,
    unit_people: list[dict[str, Any]],
    rng: random.Random,
    per_type_limit: int,
) -> list[dict[str, Any]]:
    people_by_id = {p["person_id"]: p for p in unit_people}
    rows: list[dict[str, Any]] = []
    counter = 1

    generators = [
        generate_attribute_questions,
        generate_family_member_questions,
        generate_derived_relation_questions,
        generate_chain_questions,
        generate_attribute_filter_questions,
    ]
    for gen in generators:
        new_rows, counter = gen(unit_id, unit_people, people_by_id, rng, counter, per_type_limit)
        rows.extend(new_rows)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate representation-agnostic person QA from person profiles.")
    parser.add_argument("--profiles", required=True, help="Path to people_profiles.jsonl")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-type-limit", type=int, default=20, help="Max questions per unit for each task type")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    profiles = read_jsonl(args.profiles)
    grouped = group_by_unit(profiles)

    all_questions: list[dict[str, Any]] = []
    per_unit_counts: dict[str, int] = {}
    for unit_id in sorted(grouped):
        unit_people = grouped[unit_id]
        qs = generate_questions_for_unit(unit_id, unit_people, rng, per_type_limit=args.per_type_limit)
        all_questions.extend(qs)
        per_unit_counts[unit_id] = len(qs)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Store each task family in an independent file. This keeps the person-QA
    # generator extensible: future project/organization/time QA generators can
    # write their own task files without mixing with person-profile tasks.
    task_file_map = {
        "单人物档案查询": "person_attribute.jsonl",
        "家庭成员查询": "family_member.jsonl",
        "派生亲属关系推理": "derived_kinship.jsonl",
        "多步亲属链路推理": "multi_hop_kinship.jsonl",
        "属性筛选型关系问题": "attribute_filter.jsonl",
    }

    task_groups: dict[str, list[dict[str, Any]]] = {task: [] for task in task_file_map}
    for q in all_questions:
        task_groups.setdefault(q["task_type"], []).append(q)

    written_files: dict[str, str] = {}
    for task_type, filename in task_file_map.items():
        rows = task_groups.get(task_type, [])
        write_jsonl(out_dir / filename, rows)
        written_files[task_type] = filename

    # Optional but convenient: one combined file for running a quick full-person-QA pass.
    write_jsonl(out_dir / "all_questions.jsonl", all_questions)

    task_counts: dict[str, int] = {}
    difficulty_counts: dict[str, int] = {}
    for q in all_questions:
        task_counts[q["task_type"]] = task_counts.get(q["task_type"], 0) + 1
        difficulty_counts[q["difficulty"]] = difficulty_counts.get(q["difficulty"], 0) + 1

    write_json(out_dir / "person_question_metadata.json", {
        "generator": "build_person_questions.py",
        "profiles_path": str(args.profiles),
        "num_units": len(grouped),
        "num_profiles": len(profiles),
        "num_questions": len(all_questions),
        "per_unit_counts": per_unit_counts,
        "task_counts": task_counts,
        "difficulty_counts": difficulty_counts,
        "task_files": written_files,
        "combined_file": "all_questions.jsonl",
        "seed": args.seed,
        "per_type_limit": args.per_type_limit,
        "notes": [
            "This script only generates person-profile QA, not project/organization/time QA.",
            "Each task family is written to a separate JSONL file.",
            "all_questions.jsonl is only a convenience combined copy.",
            "QA is representation-agnostic and does not contain rendered context.",
            "gold_entity_ids should be mapped to text/table/triples/graph views in a later sampling script.",
            "Derived kinship relations such as 哥哥、妹妹、伯父 appear only in QA and reasoning paths, not in the base KB.",
            "Internal profile fields such as generation/family_id are not emitted into questions.",
        ],
    })

    print(f"Generated {len(all_questions)} person questions from {len(profiles)} profiles across {len(grouped)} units.")
    print(f"Saved task files to: {out_dir}")
    print("Task counts:", task_counts)


if __name__ == "__main__":
    main()
