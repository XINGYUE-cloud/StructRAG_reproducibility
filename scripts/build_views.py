#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render basic person profiles into four knowledge-base views.

Input:
  people_profiles.jsonl
  relations_basic.jsonl

Output:
  kb_text.jsonl
  kb_table.jsonl
  kb_triples.jsonl
  kb_graph.jsonl
  kb_all_views.jsonl
  view_metadata.json

Important:
  This script treats fields such as `generation` and `family_id` as internal
  generation-only fields. They are NOT rendered into any knowledge-base view.

The four views contain the same person-level facts:
  1. 自然语言文本
  2. 表格形式
  3. 三元组形式
  4. 图结构形式

This script does NOT generate QA, projects, derived relations such as 哥哥/妹妹/伯父,
or retrieval samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


VIEW_TYPES = ["自然语言文本", "表格形式", "三元组形式", "图结构形式"]

# Only these profile fields are allowed to be exposed in knowledge-base content.
# `generation` and `family_id` are intentionally excluded.
PUBLIC_PROFILE_FIELDS = [
    "name",
    "gender",
    "age",
    "personality",
    "skills",
    "city",
    "company",
    "branch",
    "department",
    "team",
    "position",
    "level",
    "work_years",
    "work_status",
    "strength_scenarios",
    "appearance",
    "father_id",
    "mother_id",
    "spouse_id",
    "children_ids",
]

FIELD_LABELS = {
    "name": "姓名",
    "gender": "性别",
    "age": "年龄",
    "personality": "性格",
    "skills": "能力",
    "city": "城市",
    "company": "公司",
    "branch": "总部/分部",
    "department": "部门",
    "team": "小组",
    "position": "岗位",
    "level": "职级",
    "work_years": "工作年限",
    "work_status": "工作状态",
    "strength_scenarios": "擅长场景",
    "appearance": "外在描述",
    "father": "父亲",
    "mother": "母亲",
    "spouse": "配偶",
    "children": "孩子",
}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
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


def clean_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, list):
        return [str(v) for v in values if v not in (None, "")]
    return [str(values)]


def join_values(values: Any, empty: str = "无") -> str:
    items = clean_list(values)
    return "、".join(items) if items else empty


def name_of(person_id: str | None, people_by_id: dict[str, dict[str, Any]], empty: str = "无") -> str:
    if not person_id:
        return empty
    person = people_by_id.get(person_id)
    return str(person.get("name", person_id)) if person else str(person_id)


def names_of(person_ids: list[str] | None, people_by_id: dict[str, dict[str, Any]], empty: str = "无") -> str:
    ids = person_ids or []
    if not ids:
        return empty
    return "、".join(name_of(pid, people_by_id, empty=str(pid)) for pid in ids)


def public_person_view(person: dict[str, Any], people_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Return only public fields for rendering. Internal generation fields are removed."""
    return {
        "name": person.get("name"),
        "gender": person.get("gender"),
        "age": person.get("age"),
        "personality": clean_list(person.get("personality")),
        "skills": clean_list(person.get("skills")),
        "city": person.get("city"),
        "company": person.get("company"),
        "branch": person.get("branch"),
        "department": person.get("department"),
        "team": person.get("team"),
        "position": person.get("position"),
        "level": person.get("level", "无"),
        "work_years": person.get("work_years", "无"),
        "work_status": person.get("work_status", "无"),
        "strength_scenarios": clean_list(person.get("strength_scenarios")),
        "appearance": clean_list(person.get("appearance")),
        "father": name_of(person.get("father_id"), people_by_id),
        "mother": name_of(person.get("mother_id"), people_by_id),
        "spouse": name_of(person.get("spouse_id"), people_by_id),
        "children": names_of(person.get("children_ids") or [], people_by_id),
    }


def render_text(person: dict[str, Any], people_by_id: dict[str, dict[str, Any]]) -> str:
    v = public_person_view(person, people_by_id)
    name = v["name"]
    return (
        f"{name}，{v['gender']}，{v['age']}岁，"
        f"性格包括{join_values(v['personality'])}，能力包括{join_values(v['skills'])}。"
        f"{name}的职级是{v['level']}，已有{v['work_years']}年工作经验，当前工作状态是{v['work_status']}，"
        f"擅长场景包括{join_values(v['strength_scenarios'])}，外在描述包括{join_values(v['appearance'])}。"
        f"{name}的父亲是{v['father']}，母亲是{v['mother']}，配偶是{v['spouse']}，孩子包括{v['children']}。"
        f"{name}任职于{v['company']}，位于{v['branch']}，所属部门是{v['department']}，"
        f"所属小组是{v['team']}，岗位是{v['position']}，所在城市是{v['city']}。"
    )


def render_table(person: dict[str, Any], people_by_id: dict[str, dict[str, Any]]) -> str:
    v = public_person_view(person, people_by_id)
    columns = [
        "人物", "性别", "年龄", "性格", "能力", "职级", "工作年限", "工作状态", "擅长场景", "外在描述",
        "父亲", "母亲", "配偶", "孩子", "公司", "总部/分部", "部门", "小组", "岗位", "城市",
    ]
    values = [
        v["name"], v["gender"], str(v["age"]), join_values(v["personality"]), join_values(v["skills"]),
        str(v["level"]), str(v["work_years"]), str(v["work_status"]), join_values(v["strength_scenarios"]), join_values(v["appearance"]),
        v["father"], v["mother"], v["spouse"], v["children"], v["company"], v["branch"],
        v["department"], v["team"], v["position"], v["city"],
    ]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    row = "| " + " | ".join(values) + " |"
    return "\n".join([header, sep, row])


def render_triples(person: dict[str, Any], people_by_id: dict[str, dict[str, Any]]) -> str:
    v = public_person_view(person, people_by_id)
    name = v["name"]
    triples: list[tuple[str, str, str]] = [
        (name, "性别", str(v["gender"])),
        (name, "年龄", str(v["age"])),
        (name, "所在城市", str(v["city"])),
        (name, "任职于", str(v["company"])),
        (name, "位于", str(v["branch"])),
        (name, "所属部门", str(v["department"])),
        (name, "所属小组", str(v["team"])),
        (name, "岗位", str(v["position"])),
        (name, "职级", str(v["level"])),
        (name, "工作年限", f"{v['work_years']}年"),
        (name, "工作状态", str(v["work_status"])),
    ]
    for tag in v["personality"]:
        triples.append((name, "性格", tag))
    for skill in v["skills"]:
        triples.append((name, "能力", skill))
    for scene in v["strength_scenarios"]:
        triples.append((name, "擅长场景", scene))
    for desc in v["appearance"]:
        triples.append((name, "外在描述", desc))
    if v["father"] != "无":
        triples.append((name, "父亲", v["father"]))
    if v["mother"] != "无":
        triples.append((name, "母亲", v["mother"]))
    if v["spouse"] != "无":
        triples.append((name, "配偶", v["spouse"]))
    if v["children"] != "无":
        for child_name in v["children"].split("、"):
            triples.append((name, "孩子", child_name))
    return "\n".join(f"({s}, {r}, {o})" for s, r, o in triples)


def render_graph(person: dict[str, Any], people_by_id: dict[str, dict[str, Any]]) -> str:
    v = public_person_view(person, people_by_id)
    name = v["name"]
    lines = [f"{name}:"]
    simple_edges = [
        ("性别", v["gender"]),
        ("年龄", f"{v['age']}岁"),
        ("所在城市", v["city"]),
        ("任职于", v["company"]),
        ("位于", v["branch"]),
        ("所属部门", v["department"]),
        ("所属小组", v["team"]),
        ("岗位", v["position"]),
        ("职级", v["level"]),
        ("工作年限", f"{v['work_years']}年"),
        ("工作状态", v["work_status"]),
    ]
    for rel, obj in simple_edges:
        lines.append(f"  {rel} -> {obj}")
    for tag in v["personality"]:
        lines.append(f"  性格 -> {tag}")
    for skill in v["skills"]:
        lines.append(f"  能力 -> {skill}")
    for scene in v["strength_scenarios"]:
        lines.append(f"  擅长场景 -> {scene}")
    for desc in v["appearance"]:
        lines.append(f"  外在描述 -> {desc}")
    if v["father"] != "无":
        lines.append(f"  父亲 -> {v['father']}")
    if v["mother"] != "无":
        lines.append(f"  母亲 -> {v['mother']}")
    if v["spouse"] != "无":
        lines.append(f"  配偶 -> {v['spouse']}")
    if v["children"] != "无":
        for child_name in v["children"].split("、"):
            lines.append(f"  孩子 -> {child_name}")
    return "\n".join(lines)


def make_record(person: dict[str, Any], representation_type: str, content: str) -> dict[str, Any]:
    """Create one retrievable KB chunk record."""
    return {
        "chunk_id": f"{person['person_id']}::{representation_type}",
        "unit_id": person["unit_id"],
        "chunk_type": "人物档案",
        "entity_type": "人物",
        "entity_id": person["person_id"],
        "entity_name": person["name"],
        "representation_type": representation_type,
        "content": content,
        "content_chars": len(content),
    }


def validate_no_internal_fields(records: list[dict[str, Any]]) -> None:
    """Fail fast if generation-only fields leak into view records."""
    forbidden_keys = {"generation", "family_id"}
    forbidden_terms = ["generation", "父母辈", "祖辈", "子女辈", "family_id"]
    for r in records:
        leaked_keys = forbidden_keys.intersection(r.keys())
        if leaked_keys:
            raise ValueError(f"Internal keys leaked into record {r.get('chunk_id')}: {leaked_keys}")
        content = json.dumps(r, ensure_ascii=False)
        for term in forbidden_terms:
            if term in content:
                raise ValueError(f"Internal term {term!r} leaked into record {r.get('chunk_id')}")


def build_views(people: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    people_by_id = {p["person_id"]: p for p in people}
    views: dict[str, list[dict[str, Any]]] = {view_type: [] for view_type in VIEW_TYPES}
    for person in people:
        rendered = {
            "自然语言文本": render_text(person, people_by_id),
            "表格形式": render_table(person, people_by_id),
            "三元组形式": render_triples(person, people_by_id),
            "图结构形式": render_graph(person, people_by_id),
        }
        for view_type, content in rendered.items():
            views[view_type].append(make_record(person, view_type, content))
    return views


def main() -> None:
    parser = argparse.ArgumentParser(description="把基础人物档案渲染成四种知识库结构")
    parser.add_argument("--profiles", type=str, required=True, help="people_profiles.jsonl 路径")
    parser.add_argument(
        "--relations",
        type=str,
        default=None,
        help="relations_basic.jsonl 路径。当前脚本主要从 profiles 字段渲染，保留该参数用于一致性检查。",
    )
    parser.add_argument("--out", type=str, required=True, help="输出目录")
    parser.add_argument("--skip-validation", action="store_true", help="跳过内部字段泄漏检查")
    args = parser.parse_args()

    profiles_path = Path(args.profiles)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    people = read_jsonl(profiles_path)
    if not people:
        raise ValueError(f"No profiles found in {profiles_path}")

    n_relations = None
    if args.relations:
        n_relations = len(read_jsonl(args.relations))

    views = build_views(people)
    all_records: list[dict[str, Any]] = []
    file_map = {
        "自然语言文本": "kb_text.jsonl",
        "表格形式": "kb_table.jsonl",
        "三元组形式": "kb_triples.jsonl",
        "图结构形式": "kb_graph.jsonl",
    }

    for view_type, records in views.items():
        if not args.skip_validation:
            validate_no_internal_fields(records)
        write_jsonl(out_dir / file_map[view_type], records)
        all_records.extend(records)

    if not args.skip_validation:
        validate_no_internal_fields(all_records)
    write_jsonl(out_dir / "kb_all_views.jsonl", all_records)

    metadata = {
        "version": "person_profile_views_v0.2",
        "source_profiles": str(profiles_path),
        "source_relations": str(args.relations) if args.relations else None,
        "n_people": len(people),
        "n_basic_relations": n_relations,
        "view_types": VIEW_TYPES,
        "public_profile_fields": PUBLIC_PROFILE_FIELDS,
        "profile_fields_added_in_v0.2": ["level", "work_years", "work_status", "strength_scenarios", "appearance"],
        "outputs": {
            "自然语言文本": "kb_text.jsonl",
            "表格形式": "kb_table.jsonl",
            "三元组形式": "kb_triples.jsonl",
            "图结构形式": "kb_graph.jsonl",
            "all_views": "kb_all_views.jsonl",
        },
        "internal_fields_removed": ["generation", "family_id"],
        "note": "本脚本只生成知识库视图，不生成 QA、项目、派生亲属关系或检索样本。",
    }
    write_json(out_dir / "view_metadata.json", metadata)

    print(f"Saved KB views to {out_dir}")
    print(f"people: {len(people)}")
    if n_relations is not None:
        print(f"basic relations: {n_relations}")
    for view_type, filename in file_map.items():
        print(f"{filename}: {len(views[view_type])} chunks ({view_type})")
    print("已过滤内部生成字段：generation, family_id")
    print("新增公开字段：level, work_years, work_status, strength_scenarios, appearance")


if __name__ == "__main__":
    main()
