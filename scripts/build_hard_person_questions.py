#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_hard_person_questions.py

Generate a hard person-centric QA benchmark from people_profiles.jsonl.

This script is intentionally separated from build_person_questions.py.
It focuses on harder person tasks and increases the proportion of:
  1) multi-answer family member questions,
  2) derived kinship reasoning,
  3) multi-hop kinship chain reasoning,
  4) attribute filtering over family members.

It does NOT generate contexts and does NOT bind questions to a specific
knowledge representation. Each QA item only stores gold_entity_ids, answer(s),
reasoning path, task type, and difficulty. The mapping to text/table/triple/graph
views should be done later by the RAG/evaluation script.

Expected input profile fields include:
  unit_id, person_id, name, gender, age, personality, skills, city, company,
  branch, department, team, position, father_id, mother_id, spouse_id, children_ids.
Internal generation fields such as generation and family_id are ignored and are
never written to the output QA files.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

Person = Dict[str, Any]

TASK_HARD_FAMILY = "困难家庭成员查询"
TASK_DERIVED = "困难派生亲属关系推理"
TASK_MULTI_HOP = "困难多步亲属链路推理"
TASK_FILTER = "困难属性筛选型关系问题"

TASK_TO_FILE = {
    TASK_HARD_FAMILY: "hard_family_member.jsonl",
    TASK_DERIVED: "hard_derived_kinship.jsonl",
    TASK_MULTI_HOP: "hard_multi_hop_kinship.jsonl",
    TASK_FILTER: "hard_attribute_filter.jsonl",
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def safe_name(p: Optional[Person]) -> str:
    return p.get("name", "未知") if p else "未知"


def get_age(p: Person) -> int:
    try:
        return int(p.get("age", 0))
    except Exception:
        return 0


class PersonWorld:
    def __init__(self, persons: Sequence[Person]):
        self.persons = list(persons)
        self.by_id: Dict[str, Person] = {p["person_id"]: p for p in self.persons}
        self.by_unit: Dict[str, List[Person]] = defaultdict(list)
        for p in self.persons:
            self.by_unit[p["unit_id"]].append(p)

    def person(self, pid: Optional[str]) -> Optional[Person]:
        if not pid:
            return None
        return self.by_id.get(pid)

    def same_unit_people(self, p: Person) -> List[Person]:
        return self.by_unit.get(p["unit_id"], [])

    def children(self, pid: str) -> List[Person]:
        p = self.person(pid)
        if not p:
            return []
        ids = normalize_list(p.get("children_ids"))
        return [self.by_id[cid] for cid in ids if cid in self.by_id]

    def siblings(self, pid: str) -> List[Person]:
        p = self.person(pid)
        if not p:
            return []
        father = p.get("father_id")
        mother = p.get("mother_id")
        # Require at least one parent; prefer strict same father+mother when both known.
        out: List[Person] = []
        for q in self.same_unit_people(p):
            if q["person_id"] == pid:
                continue
            if father and mother:
                ok = q.get("father_id") == father and q.get("mother_id") == mother
            elif father:
                ok = q.get("father_id") == father
            elif mother:
                ok = q.get("mother_id") == mother
            else:
                ok = False
            if ok:
                out.append(q)
        return sorted(out, key=lambda x: (get_age(x), x.get("name", "")))

    def targets(self, pid: str, relation: str) -> List[Person]:
        p = self.person(pid)
        if not p:
            return []
        age = get_age(p)
        gender = p.get("gender")

        if relation == "父亲":
            q = self.person(p.get("father_id"))
            return [q] if q else []
        if relation == "母亲":
            q = self.person(p.get("mother_id"))
            return [q] if q else []
        if relation == "配偶":
            q = self.person(p.get("spouse_id"))
            return [q] if q else []
        if relation == "孩子":
            return self.children(pid)
        if relation == "儿子":
            return [c for c in self.children(pid) if c.get("gender") == "男"]
        if relation == "女儿":
            return [c for c in self.children(pid) if c.get("gender") == "女"]

        sibs = self.siblings(pid)
        if relation == "哥哥":
            return [s for s in sibs if s.get("gender") == "男" and get_age(s) > age]
        if relation == "弟弟":
            return [s for s in sibs if s.get("gender") == "男" and get_age(s) < age]
        if relation == "姐姐":
            return [s for s in sibs if s.get("gender") == "女" and get_age(s) > age]
        if relation == "妹妹":
            return [s for s in sibs if s.get("gender") == "女" and get_age(s) < age]

        father = self.person(p.get("father_id"))
        mother = self.person(p.get("mother_id"))
        if relation in {"伯父", "叔叔", "姑姑"} and father:
            fsibs = self.siblings(father["person_id"])
            if relation == "伯父":
                return [s for s in fsibs if s.get("gender") == "男" and get_age(s) > get_age(father)]
            if relation == "叔叔":
                return [s for s in fsibs if s.get("gender") == "男" and get_age(s) < get_age(father)]
            if relation == "姑姑":
                return [s for s in fsibs if s.get("gender") == "女"]

        if relation in {"舅舅", "姨妈"} and mother:
            msibs = self.siblings(mother["person_id"])
            if relation == "舅舅":
                return [s for s in msibs if s.get("gender") == "男"]
            if relation == "姨妈":
                return [s for s in msibs if s.get("gender") == "女"]

        return []

    def evidence_for_relation(self, source_id: str, relation: str, target_ids: Sequence[str]) -> List[str]:
        """Return person ids likely needed to justify a relation step."""
        ids = {source_id, *target_ids}
        src = self.person(source_id)
        if not src:
            return list(ids)
        # For direct relations, source profile plus target profile is enough.
        if relation in {"父亲", "母亲", "配偶", "孩子", "儿子", "女儿"}:
            return sorted(ids)
        # Sibling relations require source and target profiles; parent profiles help the model.
        if relation in {"哥哥", "姐姐", "弟弟", "妹妹"}:
            if src.get("father_id"):
                ids.add(src["father_id"])
            if src.get("mother_id"):
                ids.add(src["mother_id"])
            return sorted(ids)
        # Paternal/maternal aunt/uncle relations require parent and parent's siblings.
        if relation in {"伯父", "叔叔", "姑姑"}:
            if src.get("father_id"):
                ids.add(src["father_id"])
                father = self.person(src["father_id"])
                if father:
                    if father.get("father_id"):
                        ids.add(father["father_id"])
                    if father.get("mother_id"):
                        ids.add(father["mother_id"])
            return sorted(ids)
        if relation in {"舅舅", "姨妈"}:
            if src.get("mother_id"):
                ids.add(src["mother_id"])
                mother = self.person(src["mother_id"])
                if mother:
                    if mother.get("father_id"):
                        ids.add(mother["father_id"])
                    if mother.get("mother_id"):
                        ids.add(mother["mother_id"])
            return sorted(ids)
        return sorted(ids)


def names(world: PersonWorld, ids: Sequence[str]) -> List[str]:
    return [world.by_id[pid]["name"] for pid in ids if pid in world.by_id]


def answer_string(world: PersonWorld, ids: Sequence[str]) -> str:
    unique = []
    seen = set()
    for pid in ids:
        if pid not in seen and pid in world.by_id:
            seen.add(pid)
            unique.append(pid)
    return "、".join(names(world, unique)) if unique else "无"


def make_item(
    *,
    question_id: str,
    unit_id: str,
    task_type: str,
    difficulty: str,
    question: str,
    answer_ids: Sequence[str],
    gold_entity_ids: Sequence[str],
    reasoning_path: Sequence[Dict[str, Any]],
    required_operations: Sequence[str],
    notes: str = "",
    world: PersonWorld,
) -> Dict[str, Any]:
    answer_ids_unique: List[str] = []
    seen = set()
    for aid in answer_ids:
        if aid in world.by_id and aid not in seen:
            seen.add(aid)
            answer_ids_unique.append(aid)
    gold_unique: List[str] = []
    seen_gold = set()
    for gid in gold_entity_ids:
        if gid in world.by_id and gid not in seen_gold:
            seen_gold.add(gid)
            gold_unique.append(gid)
    return {
        "question_id": question_id,
        "unit_id": unit_id,
        "task_type": task_type,
        "difficulty": difficulty,
        "question": question,
        "answer": answer_string(world, answer_ids_unique),
        "answer_ids": answer_ids_unique,
        "answer_id": answer_ids_unique[0] if len(answer_ids_unique) == 1 else None,
        "answer_type": "人物" if answer_ids_unique else "无",
        "gold_entity_ids": gold_unique,
        "reasoning_path": list(reasoning_path),
        "required_operations": list(required_operations),
        "notes": notes,
    }


def relation_step(world: PersonWorld, source_id: str, relation: str, target_ids: Sequence[str]) -> Dict[str, Any]:
    src = world.person(source_id)
    return {
        "from_id": source_id,
        "from": safe_name(src),
        "relation": relation,
        "to_ids": list(target_ids),
        "to": names(world, target_ids),
        "evidence_entity_ids": world.evidence_for_relation(source_id, relation, target_ids),
    }


def gen_hard_family(world: PersonWorld, rng: random.Random) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    qidx = 0
    for unit_id, people in world.by_unit.items():
        for p in people:
            pid = p["person_id"]
            children = world.children(pid)
            if len(children) < 2:
                continue
            # Multi-answer children queries.
            rels = [("孩子", children)]
            sons = [c for c in children if c.get("gender") == "男"]
            daughters = [c for c in children if c.get("gender") == "女"]
            if len(sons) >= 1 and len(children) >= 2:
                rels.append(("儿子", sons))
            if len(daughters) >= 1 and len(children) >= 2:
                rels.append(("女儿", daughters))
            for rel, targets in rels:
                target_ids = [x["person_id"] for x in targets]
                gold = world.evidence_for_relation(pid, rel, target_ids)
                qidx += 1
                items.append(make_item(
                    question_id=f"{unit_id}_HF{qidx:05d}",
                    unit_id=unit_id,
                    task_type=TASK_HARD_FAMILY,
                    difficulty="中等-多答案家庭成员",
                    question=f"{p['name']}的{rel}是谁？",
                    answer_ids=target_ids,
                    gold_entity_ids=gold,
                    reasoning_path=[relation_step(world, pid, rel, target_ids)],
                    required_operations=["家庭成员查询", "多答案生成"],
                    notes="该问题通常需要输出一个或多个家庭成员。",
                    world=world,
                ))
    rng.shuffle(items)
    return items


def gen_derived(world: PersonWorld, rng: random.Random) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    qidx = 0
    relations = ["哥哥", "姐姐", "弟弟", "妹妹", "伯父", "叔叔", "姑姑", "舅舅", "姨妈"]
    for unit_id, people in world.by_unit.items():
        for p in people:
            pid = p["person_id"]
            for rel in relations:
                targets = world.targets(pid, rel)
                if not targets:
                    continue
                target_ids = [t["person_id"] for t in targets]
                gold = world.evidence_for_relation(pid, rel, target_ids)
                qidx += 1
                hop_note = "需要由父母、年龄、性别等基础字段推导，知识库中不显式存储该关系。"
                items.append(make_item(
                    question_id=f"{unit_id}_HD{qidx:05d}",
                    unit_id=unit_id,
                    task_type=TASK_DERIVED,
                    difficulty="困难-派生亲属关系",
                    question=f"{p['name']}的{rel}是谁？",
                    answer_ids=target_ids,
                    gold_entity_ids=gold,
                    reasoning_path=[relation_step(world, pid, rel, target_ids)],
                    required_operations=["派生亲属关系推导"],
                    notes=hop_note,
                    world=world,
                ))
    rng.shuffle(items)
    return items


def execute_chain(world: PersonWorld, start_id: str, chain: Sequence[str], unique_intermediate: bool = True) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    current_ids = [start_id]
    reasoning: List[Dict[str, Any]] = []
    gold: List[str] = [start_id]
    for rel in chain:
        next_ids: List[str] = []
        # To keep question answer unambiguous, require a single active branch before the last step.
        if unique_intermediate and len(current_ids) != 1:
            return [], [], []
        for sid in current_ids:
            targets = world.targets(sid, rel)
            tids = [t["person_id"] for t in targets]
            if not tids:
                return [], [], []
            # For intermediate steps require uniqueness; final step can have multiple answers.
            if unique_intermediate and rel != chain[-1] and len(tids) != 1:
                return [], [], []
            reasoning.append(relation_step(world, sid, rel, tids))
            gold.extend(world.evidence_for_relation(sid, rel, tids))
            next_ids.extend(tids)
        # Remove duplicates while preserving order.
        dedup: List[str] = []
        seen = set()
        for nid in next_ids:
            if nid not in seen:
                dedup.append(nid)
                seen.add(nid)
        current_ids = dedup
    return current_ids, reasoning, gold


def gen_multi_hop(world: PersonWorld, rng: random.Random) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    qidx = 0
    # Chains are intentionally harder than ordinary single-hop profile lookup.
    chains = [
        ["妹妹", "配偶"],
        ["姐姐", "配偶"],
        ["哥哥", "配偶"],
        ["弟弟", "配偶"],
        ["配偶", "父亲"],
        ["配偶", "母亲"],
        ["妹妹", "配偶", "父亲"],
        ["姐姐", "配偶", "母亲"],
        ["哥哥", "配偶", "父亲"],
        ["弟弟", "配偶", "母亲"],
        ["父亲", "哥哥"],
        ["父亲", "姐姐"],
        ["父亲", "妹妹"],
        ["母亲", "哥哥"],
        ["母亲", "姐姐"],
        ["配偶", "父亲", "哥哥"],
        ["配偶", "母亲", "姐姐"],
    ]
    for unit_id, people in world.by_unit.items():
        for p in people:
            pid = p["person_id"]
            for chain in chains:
                answer_ids, reasoning, gold = execute_chain(world, pid, chain, unique_intermediate=True)
                if not answer_ids:
                    continue
                qidx += 1
                chain_text = "的".join(chain)
                items.append(make_item(
                    question_id=f"{unit_id}_HM{qidx:05d}",
                    unit_id=unit_id,
                    task_type=TASK_MULTI_HOP,
                    difficulty=f"困难-{len(chain)}跳亲属链路",
                    question=f"{p['name']}的{chain_text}是谁？",
                    answer_ids=answer_ids,
                    gold_entity_ids=gold,
                    reasoning_path=reasoning,
                    required_operations=["多步链路追踪", "派生亲属关系推导"],
                    notes="链路中的哥哥/姐姐/弟弟/妹妹等关系不在知识库中显式存储，需要通过基础人物档案推导。",
                    world=world,
                ))
    rng.shuffle(items)
    return items


def gen_attribute_filter(world: PersonWorld, rng: random.Random) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    qidx = 0
    for unit_id, people in world.by_unit.items():
        for p in people:
            pid = p["person_id"]
            children = world.children(pid)
            if len(children) < 2:
                continue
            child_ids = [c["person_id"] for c in children]
            gold_base = [pid] + child_ids

            # Age max/min.
            for op, label, keyfunc in [
                ("age_max", "年龄最大", lambda c: get_age(c)),
                ("age_min", "年龄最小", lambda c: -get_age(c)),
            ]:
                # keyfunc for min returns negative, easier to use max.
                best_val = max(keyfunc(c) for c in children)
                ans = [c for c in children if keyfunc(c) == best_val]
                if not ans:
                    continue
                qidx += 1
                items.append(make_item(
                    question_id=f"{unit_id}_HA{qidx:05d}",
                    unit_id=unit_id,
                    task_type=TASK_FILTER,
                    difficulty="困难-家庭成员属性比较",
                    question=f"{p['name']}的孩子中，谁{label}？",
                    answer_ids=[a["person_id"] for a in ans],
                    gold_entity_ids=gold_base,
                    reasoning_path=[
                        relation_step(world, pid, "孩子", child_ids),
                        {"operation": f"比较孩子的年龄，选择{label}者", "candidate_entity_ids": child_ids},
                    ],
                    required_operations=["家庭成员检索", "属性比较"],
                    notes="需要先找到全部孩子，再比较年龄。",
                    world=world,
                ))

            # Position filter among children.
            pos_counter = Counter(c.get("position") for c in children if c.get("position"))
            for pos, _ in pos_counter.items():
                ans = [c for c in children if c.get("position") == pos]
                if not ans:
                    continue
                qidx += 1
                items.append(make_item(
                    question_id=f"{unit_id}_HA{qidx:05d}",
                    unit_id=unit_id,
                    task_type=TASK_FILTER,
                    difficulty="困难-家庭成员岗位筛选",
                    question=f"{p['name']}的孩子中，谁的岗位是{pos}？",
                    answer_ids=[a["person_id"] for a in ans],
                    gold_entity_ids=gold_base,
                    reasoning_path=[
                        relation_step(world, pid, "孩子", child_ids),
                        {"operation": f"筛选岗位为{pos}的孩子", "candidate_entity_ids": child_ids},
                    ],
                    required_operations=["家庭成员检索", "属性筛选"],
                    notes="需要先找到全部孩子，再按岗位字段筛选。",
                    world=world,
                ))

            # Skills/personality filters among children; sample a few to avoid too many similar questions.
            skill_to_children: Dict[str, List[Person]] = defaultdict(list)
            personality_to_children: Dict[str, List[Person]] = defaultdict(list)
            for c in children:
                for sk in normalize_list(c.get("skills")):
                    skill_to_children[str(sk)].append(c)
                for pe in normalize_list(c.get("personality")):
                    personality_to_children[str(pe)].append(c)
            skill_items = list(skill_to_children.items())
            personality_items = list(personality_to_children.items())
            rng.shuffle(skill_items)
            rng.shuffle(personality_items)
            for sk, ans in skill_items[:2]:
                qidx += 1
                items.append(make_item(
                    question_id=f"{unit_id}_HA{qidx:05d}",
                    unit_id=unit_id,
                    task_type=TASK_FILTER,
                    difficulty="困难-家庭成员能力筛选",
                    question=f"{p['name']}的孩子中，谁具备{sk}能力？",
                    answer_ids=[a["person_id"] for a in ans],
                    gold_entity_ids=gold_base,
                    reasoning_path=[
                        relation_step(world, pid, "孩子", child_ids),
                        {"operation": f"筛选具备{sk}能力的孩子", "candidate_entity_ids": child_ids},
                    ],
                    required_operations=["家庭成员检索", "能力字段筛选"],
                    notes="需要先找到全部孩子，再按能力字段筛选。",
                    world=world,
                ))
            for pe, ans in personality_items[:2]:
                qidx += 1
                items.append(make_item(
                    question_id=f"{unit_id}_HA{qidx:05d}",
                    unit_id=unit_id,
                    task_type=TASK_FILTER,
                    difficulty="困难-家庭成员性格筛选",
                    question=f"{p['name']}的孩子中，谁的性格包含{pe}？",
                    answer_ids=[a["person_id"] for a in ans],
                    gold_entity_ids=gold_base,
                    reasoning_path=[
                        relation_step(world, pid, "孩子", child_ids),
                        {"operation": f"筛选性格包含{pe}的孩子", "candidate_entity_ids": child_ids},
                    ],
                    required_operations=["家庭成员检索", "性格字段筛选"],
                    notes="需要先找到全部孩子，再按性格字段筛选。",
                    world=world,
                ))
    rng.shuffle(items)
    return items


def limit_items(items: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if limit and limit > 0:
        return items[:limit]
    return items


def reassign_question_ids(rows_by_task: Dict[str, List[Dict[str, Any]]]) -> None:
    """Assign compact deterministic ids after shuffling/limiting."""
    prefix = {
        TASK_HARD_FAMILY: "HF",
        TASK_DERIVED: "HD",
        TASK_MULTI_HOP: "HM",
        TASK_FILTER: "HA",
    }
    counters: Dict[str, int] = defaultdict(int)
    for task, rows in rows_by_task.items():
        for row in rows:
            counters[task] += 1
            row["question_id"] = f"{row['unit_id']}_{prefix.get(task, 'HQ')}{counters[task]:05d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate hard person-centric QA from people profiles.")
    parser.add_argument("--profiles", required=True, help="Path to people_profiles.jsonl")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-type-limit", type=int, default=50, help="Max questions per hard task type; 0 means no limit")
    parser.add_argument("--no-all", action="store_true", help="Do not write all_hard_questions.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    profiles_path = Path(args.profiles)
    out_dir = Path(args.out)
    people = read_jsonl(profiles_path)
    world = PersonWorld(people)

    rows_by_task: Dict[str, List[Dict[str, Any]]] = {
        TASK_HARD_FAMILY: limit_items(gen_hard_family(world, rng), args.per_type_limit),
        TASK_DERIVED: limit_items(gen_derived(world, rng), args.per_type_limit),
        TASK_MULTI_HOP: limit_items(gen_multi_hop(world, rng), args.per_type_limit),
        TASK_FILTER: limit_items(gen_attribute_filter(world, rng), args.per_type_limit),
    }
    reassign_question_ids(rows_by_task)

    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, Any]] = []
    for task, filename in TASK_TO_FILE.items():
        rows = rows_by_task.get(task, [])
        write_jsonl(out_dir / filename, rows)
        all_rows.extend(rows)

    if not args.no_all:
        # Shuffle the global file so --limit does not only take the first task type.
        all_for_write = list(all_rows)
        rng.shuffle(all_for_write)
        write_jsonl(out_dir / "all_hard_questions.jsonl", all_for_write)

    counts = {task: len(rows) for task, rows in rows_by_task.items()}
    metadata = {
        "script": "build_hard_person_questions.py",
        "source_profiles": str(profiles_path),
        "seed": args.seed,
        "per_type_limit": args.per_type_limit,
        "num_people": len(people),
        "num_units": len(world.by_unit),
        "task_files": TASK_TO_FILE,
        "task_counts": counts,
        "total_questions": sum(counts.values()),
        "design_notes": [
            "This is a hard QA set focused on complex person-centric reasoning.",
            "It intentionally reduces simple one-person attribute lookup questions.",
            "Only basic profile facts are assumed: parents, spouse, children, gender, age, and attributes.",
            "Brother/sister/uncle/aunt relations are derived during QA generation and are not stored as base facts.",
            "QA items store gold_entity_ids only and do not bind to a specific knowledge representation.",
        ],
        "internal_fields_not_required": ["generation", "family_id"],
    }
    write_json(out_dir / "hard_question_metadata.json", metadata)

    print(f"Saved hard questions to {out_dir}")
    for task, n in counts.items():
        print(f"{task}: {n}")
    print(f"Total: {sum(counts.values())}")


if __name__ == "__main__":
    main()
