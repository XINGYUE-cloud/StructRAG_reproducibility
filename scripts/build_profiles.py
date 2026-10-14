#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate basic person profiles from a lexicon.

This script builds only the basic person-profile world. It does NOT generate:
  - derived relations such as 哥哥/妹妹/伯父/叔叔
  - four representation views
  - QA samples
  - project profiles

Basic relation fields are limited to:
  father_id, mother_id, spouse_id, children_ids

Outputs:
  people_profiles.jsonl
  relations_basic.jsonl
  metadata.json
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable


@dataclass
class PersonProfile:
    unit_id: str
    person_id: str
    name: str
    gender: str
    age: int
    generation: str
    family_id: str
    personality: list[str]
    skills: list[str]
    city: str
    company: str
    branch: str
    department: str
    team: str
    position: str
    level: str
    work_years: int
    work_status: str
    strength_scenarios: list[str]
    appearance: list[str]
    father_id: str | None = None
    mother_id: str | None = None
    spouse_id: str | None = None
    children_ids: list[str] | None = None

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        if d["children_ids"] is None:
            d["children_ids"] = []
        return d


@dataclass
class BasicRelation:
    relation_id: str
    unit_id: str
    subject_id: str
    subject_name: str
    relation: str
    object_id: str
    object_name: str


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def choose_name(rng: random.Random, lexicon: dict[str, Any], gender: str, used_names: set[str], surname: str | None = None) -> str:
    surnames = lexicon["surnames"]
    given_names = lexicon["given_names"][gender]
    for _ in range(500):
        s = surname if surname is not None else rng.choice(surnames)
        g = rng.choice(given_names)
        name = s + g
        if name not in used_names:
            used_names.add(name)
            return name
    idx = len(used_names) + 1
    name = f"{surname or rng.choice(surnames)}{rng.choice(given_names)}{idx}"
    used_names.add(name)
    return name


def sample_age(rng: random.Random, lexicon: dict[str, Any], generation: str) -> int:
    conf = lexicon["age_groups"][generation]
    return rng.randint(conf["min"], conf["max"])


def sample_personality(rng: random.Random, lexicon: dict[str, Any], k: int = 4) -> list[str]:
    dims = list(lexicon["personality_dimensions"].values())
    rng.shuffle(dims)
    tags = [rng.choice(dim) for dim in dims[:k]]
    return tags


def sample_level(rng: random.Random, lexicon: dict[str, Any], generation: str, age: int) -> str:
    # Prefer generation-aware candidates, then lightly constrain by age.
    candidates = list(lexicon.get("generation_level_candidates", {}).get(generation, lexicon.get("levels", [])))
    if not candidates:
        return "中级工程师"
    if age < 24:
        candidates = [x for x in candidates if x in {"实习生", "初级工程师", "中级工程师"}] or candidates
    elif age < 30:
        candidates = [x for x in candidates if x in {"初级工程师", "中级工程师", "高级工程师"}] or candidates
    elif age > 58:
        candidates = [x for x in candidates if x in {"专家", "主管", "经理", "总监"}] or candidates
    return rng.choice(candidates)


def sample_work_years(rng: random.Random, lexicon: dict[str, Any], level: str, age: int) -> int:
    ranges = lexicon.get("level_work_year_range", {})
    lo, hi = ranges.get(level, [1, max(1, age - 22)])
    # Work years should not exceed a rough age-based upper bound.
    hi = min(int(hi), max(0, age - 18))
    lo = min(int(lo), hi)
    return rng.randint(lo, hi) if hi >= lo else 0


def sample_appearance(rng: random.Random, lexicon: dict[str, Any], k: int = 3) -> list[str]:
    dims = list(lexicon.get("appearance_dimensions", {}).values())
    if not dims:
        return []
    rng.shuffle(dims)
    return [rng.choice(dim) for dim in dims[:k]]


def sample_strength_scenarios(rng: random.Random, lexicon: dict[str, Any], position: str, k: int = 3) -> list[str]:
    base = list(lexicon.get("position_scenario_map", {}).get(position, []))
    pool = [x for x in lexicon.get("strength_scenarios", []) if x not in base]
    rng.shuffle(base)
    rng.shuffle(pool)
    scenarios = base[: min(k, len(base))]
    while len(scenarios) < k and pool:
        scenarios.append(pool.pop())
    return scenarios


def sample_work_profile(rng: random.Random, lexicon: dict[str, Any], generation: str, age: int) -> dict[str, str | int | list[str]]:
    position = rng.choice(lexicon["positions"])
    base_skills = list(lexicon.get("position_skill_map", {}).get(position, []))
    extra_pool = [s for s in lexicon.get("skills", []) if s not in base_skills]
    rng.shuffle(base_skills)
    skills = base_skills[: rng.randint(2, min(4, max(2, len(base_skills))))]
    if extra_pool and rng.random() < 0.75:
        skills.append(rng.choice(extra_pool))

    company = rng.choice(lexicon["companies"])
    branch = rng.choice(lexicon["branches"][company])
    city = branch.replace("总部", "").replace("分部", "")
    department = rng.choice(lexicon["departments"])
    team_candidates = lexicon.get("department_team_map", {}).get(department, lexicon["teams"])
    team = rng.choice(team_candidates)

    level = sample_level(rng, lexicon, generation, age)
    work_years = sample_work_years(rng, lexicon, level, age)
    work_status = rng.choice(lexicon.get("work_statuses", ["任务适中"]))
    strength_scenarios = sample_strength_scenarios(rng, lexicon, position, k=3)
    appearance = sample_appearance(rng, lexicon, k=3)

    return {
        "city": city,
        "company": company,
        "branch": branch,
        "department": department,
        "team": team,
        "position": position,
        "skills": skills,
        "level": level,
        "work_years": work_years,
        "work_status": work_status,
        "strength_scenarios": strength_scenarios,
        "appearance": appearance,
    }


def make_person(
    rng: random.Random,
    lexicon: dict[str, Any],
    unit_id: str,
    person_id: str,
    family_id: str,
    generation: str,
    gender: str,
    used_names: set[str],
    surname: str | None = None,
) -> PersonProfile:
    age = sample_age(rng, lexicon, generation)
    work = sample_work_profile(rng, lexicon, generation=generation, age=age)
    return PersonProfile(
        unit_id=unit_id,
        person_id=person_id,
        name=choose_name(rng, lexicon, gender, used_names, surname=surname),
        gender=gender,
        age=age,
        generation=generation,
        family_id=family_id,
        personality=sample_personality(rng, lexicon, k=4),
        skills=list(work["skills"]),
        city=str(work["city"]),
        company=str(work["company"]),
        branch=str(work["branch"]),
        department=str(work["department"]),
        team=str(work["team"]),
        position=str(work["position"]),
        level=str(work["level"]),
        work_years=int(work["work_years"]),
        work_status=str(work["work_status"]),
        strength_scenarios=list(work["strength_scenarios"]),
        appearance=list(work["appearance"]),
        children_ids=[],
    )


def link_spouses(a: PersonProfile, b: PersonProfile) -> None:
    a.spouse_id = b.person_id
    b.spouse_id = a.person_id


def link_child(father: PersonProfile, mother: PersonProfile, child: PersonProfile) -> None:
    child.father_id = father.person_id
    child.mother_id = mother.person_id
    if father.children_ids is None:
        father.children_ids = []
    if mother.children_ids is None:
        mother.children_ids = []
    father.children_ids.append(child.person_id)
    mother.children_ids.append(child.person_id)


def next_person_id(unit_idx: int, counter: int) -> str:
    return f"U{unit_idx:04d}_P{counter:03d}"


def generate_unit(
    rng: random.Random,
    lexicon: dict[str, Any],
    unit_idx: int,
    n_families: int,
    min_children_g2: int,
    max_children_g2: int,
    min_children_g3: int,
    max_children_g3: int,
) -> tuple[list[PersonProfile], list[BasicRelation]]:
    """Generate one world unit consisting of several family trees."""

    unit_id = f"U{unit_idx:04d}"
    used_names: set[str] = set()
    people: list[PersonProfile] = []
    pid_counter = 1

    def new_person(family_id: str, generation: str, gender: str, surname: str | None = None) -> PersonProfile:
        nonlocal pid_counter
        p = make_person(
            rng=rng,
            lexicon=lexicon,
            unit_id=unit_id,
            person_id=next_person_id(unit_idx, pid_counter),
            family_id=family_id,
            generation=generation,
            gender=gender,
            used_names=used_names,
            surname=surname,
        )
        pid_counter += 1
        people.append(p)
        return p

    for fam_idx in range(1, n_families + 1):
        family_id = f"{unit_id}_F{fam_idx:02d}"
        family_surname = rng.choice(lexicon["surnames"])

        grandpa = new_person(family_id, "祖辈", "男", surname=family_surname)
        grandma = new_person(family_id, "祖辈", "女", surname=rng.choice(lexicon["surnames"]))
        link_spouses(grandpa, grandma)

        n_g2_children = rng.randint(min_children_g2, max_children_g2)
        g2_children: list[PersonProfile] = []
        for i in range(n_g2_children):
            gender = "男" if i == 0 else rng.choice(["男", "女"])
            child = new_person(family_id, "父母辈", gender, surname=family_surname)
            link_child(grandpa, grandma, child)
            g2_children.append(child)

        marriage_candidates = list(g2_children)
        rng.shuffle(marriage_candidates)
        n_marriages = min(len(marriage_candidates), rng.randint(1, 2))
        for adult in marriage_candidates[:n_marriages]:
            spouse_gender = "女" if adult.gender == "男" else "男"
            spouse_surname = rng.choice([s for s in lexicon["surnames"] if s != family_surname])
            spouse = new_person(family_id, "父母辈", spouse_gender, surname=spouse_surname)
            spouse.age = max(34, min(58, adult.age + rng.randint(-5, 5)))
            link_spouses(adult, spouse)

            father, mother = (adult, spouse) if adult.gender == "男" else (spouse, adult)
            n_g3_children = rng.randint(min_children_g3, max_children_g3)
            child_genders = ["男", "女"] + [rng.choice(["男", "女"]) for _ in range(max(0, n_g3_children - 2))]
            rng.shuffle(child_genders)
            for gender in child_genders:
                surname = father.name[0]
                g3 = new_person(family_id, "子女辈", gender, surname=surname)
                link_child(father, mother, g3)

    people_by_id = {p.person_id: p for p in people}
    relations: list[BasicRelation] = []
    rid = 1

    def add_rel(subj: PersonProfile, rel: str, obj_id: str) -> None:
        nonlocal rid
        obj = people_by_id[obj_id]
        relations.append(
            BasicRelation(
                relation_id=f"{unit_id}_R{rid:05d}",
                unit_id=unit_id,
                subject_id=subj.person_id,
                subject_name=subj.name,
                relation=rel,
                object_id=obj.person_id,
                object_name=obj.name,
            )
        )
        rid += 1

    for p in people:
        if p.father_id:
            add_rel(p, "父亲", p.father_id)
        if p.mother_id:
            add_rel(p, "母亲", p.mother_id)
        if p.spouse_id:
            add_rel(p, "配偶", p.spouse_id)
        for child_id in p.children_ids or []:
            add_rel(p, "孩子", child_id)

    return people, relations


def main() -> None:
    parser = argparse.ArgumentParser(description="基于词库生成基础人物档案世界")
    parser.add_argument("--lexicon", type=str, default="data/lexicon.json", help="词库 JSON 路径")
    parser.add_argument("--out", type=str, default="data/profiles", help="输出目录")
    parser.add_argument("--n-units", type=int, default=1, help="生成多少个独立信息单元")
    parser.add_argument("--families-per-unit", type=int, default=3, help="每个信息单元包含多少个家族")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--min-children-g2", type=int, default=2, help="每个祖辈家庭的父母辈孩子最少数量")
    parser.add_argument("--max-children-g2", type=int, default=4, help="每个祖辈家庭的父母辈孩子最多数量")
    parser.add_argument("--min-children-g3", type=int, default=2, help="每对父母辈夫妻的子女辈孩子最少数量")
    parser.add_argument("--max-children-g3", type=int, default=3, help="每对父母辈夫妻的子女辈孩子最多数量")
    args = parser.parse_args()

    lexicon = load_json(args.lexicon)
    rng = random.Random(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_people: list[dict[str, Any]] = []
    all_relations: list[dict[str, Any]] = []
    unit_summaries: list[dict[str, Any]] = []

    for unit_idx in range(1, args.n_units + 1):
        people, relations = generate_unit(
            rng=rng,
            lexicon=lexicon,
            unit_idx=unit_idx,
            n_families=args.families_per_unit,
            min_children_g2=args.min_children_g2,
            max_children_g2=args.max_children_g2,
            min_children_g3=args.min_children_g3,
            max_children_g3=args.max_children_g3,
        )
        all_people.extend([p.to_json() for p in people])
        all_relations.extend([asdict(r) for r in relations])
        unit_summaries.append({
            "unit_id": f"U{unit_idx:04d}",
            "n_people": len(people),
            "n_basic_relations": len(relations),
            "n_families": args.families_per_unit,
        })

    write_jsonl(out_dir / "people_profiles.jsonl", all_people)
    write_jsonl(out_dir / "relations_basic.jsonl", all_relations)

    metadata = {
        "version": "person_profile_world_v0.2",
        "seed": args.seed,
        "n_units": args.n_units,
        "families_per_unit": args.families_per_unit,
        "profile_fields_added_in_v0.2": ["level", "work_years", "work_status", "strength_scenarios", "appearance"],
        "relation_policy": "底层只保存父亲、母亲、配偶、孩子；哥哥、妹妹、伯父等关系后续由规则推导。",
        "outputs": {
            "people_profiles": "people_profiles.jsonl",
            "relations_basic": "relations_basic.jsonl",
        },
        "unit_summaries": unit_summaries,
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved profiles to {out_dir}")
    print(f"people: {len(all_people)}")
    print(f"basic relations: {len(all_relations)}")
    print("新增人物字段：level, work_years, work_status, strength_scenarios, appearance")


if __name__ == "__main__":
    main()
