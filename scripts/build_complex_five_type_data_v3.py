#!/usr/bin/env python3
"""Build grounded, deterministic complex five-type synthetic data."""
from __future__ import annotations
import argparse, json, random, sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.five_type_router_data_v1 import FIVE_LABELS, router_messages, write_jsonl

LEVELS = (1, 2, 3, 4)
WEIGHTS = {1: .15625, 2: .1875, 3: .400625, 4: .255625}

def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]

def clone_world(source_root, people, variant):
    prefix = f'{source_root}_W{variant:03d}'
    ordered = sorted(people, key=lambda row: str(row['person_id']))
    mapping = {str(row['person_id']): f'{prefix}_P{i:03d}' for i, row in enumerate(ordered)}
    cloned = []
    for index, person in enumerate(ordered):
        row = dict(person); row.update(source_root=source_root, unit_id=prefix,
                                       person_id=mapping[str(person['person_id'])],
                                       name=f"{person.get('name', 'Person')}_{variant:03d}_{index:03d}",
                                       family_id=f"{prefix}_{person.get('family_id', 'F')}")
        for field in ('father_id', 'mother_id', 'spouse_id'):
            row[field] = mapping.get(str(person.get(field))) if person.get(field) else None
        row['children_ids'] = [mapping[str(value)] for value in person.get('children_ids', []) if str(value) in mapping]
        cloned.append(row)
    return cloned

def build_documents(people):
    names = {str(row['person_id']): row['name'] for row in people}; families = defaultdict(list)
    for row in people: families[str(row['family_id'])].append(row)
    docs = []
    for family_id, members in sorted(families.items()):
        records = []
        for person in sorted(members, key=lambda row: row['person_id']):
            def name(value): return names.get(str(value), '无') if value else '无'
            records.append('\n'.join((
                f"person_id: {person['person_id']}", f"姓名: {person['name']}",
                f"年龄: {person.get('age', '')}", f"城市: {person.get('city', '')}",
                f"公司: {person.get('company', '')}", f"部门: {person.get('department', '')}",
                f"小组: {person.get('team', '')}", f"岗位: {person.get('position', '')}",
                f"工作年限: {person.get('work_years', '')}",
                f"能力: {'、'.join(map(str, person.get('skills', [])))}",
                f"父亲: {name(person.get('father_id'))}", f"母亲: {name(person.get('mother_id'))}",
                f"配偶: {name(person.get('spouse_id'))}",
                f"孩子: {'、'.join(name(value) for value in person.get('children_ids', [])) or '无'}")))
        docs.append({'doc_id': family_id, 'title': f'{family_id} 人物档案', 'content': '\n\n'.join(records)})
    return docs

def graph_path(people, start, hops):
    by_id = {str(row['person_id']): row for row in people}; current = start; seen = {str(start['person_id'])}; steps = []
    for _ in range(hops):
        found = None
        for field in ('father_id', 'mother_id', 'spouse_id', 'children_ids'):
            values = current.get(field) if field == 'children_ids' else [current.get(field)]
            for value in values or []:
                if str(value) in by_id and str(value) not in seen:
                    found = (field, str(value)); break
            if found: break
        if found is None: return None
        field, target_id = found; steps.append({'relation': field, 'from_id': str(current['person_id']), 'to_id': target_id})
        current = by_id[target_id]; seen.add(target_id)
    return steps, current

def make_row(world, people, docs, label, level, variant, index, rng):
    person = people[(index + variant + level) % len(people)]; pid = str(person['person_id']); evidence = []
    spec = {'level': level}; oracle = {}
    if label == 'chunk':
        field = ('company', 'department', 'position', 'city')[(index + variant) % 4]
        field_name = {'company': '公司', 'department': '部门', 'position': '岗位', 'city': '城市'}[field]
        query = f'记录 {pid} 的{field_name}是什么？'; answer = str(person.get(field, '')); evidence = [pid]
        spec.update(operation='single_hop_lookup', field=field); oracle = {'type': 'field', 'person_id': pid, 'field': field, 'value': answer}
    elif label == 'graph':
        result = graph_path(people, person, level)
        if result is None: return None
        steps, target = result; query = f"从记录 {pid} 出发，依次沿 {'、'.join(step['relation'] for step in steps)} 关系，目标人物的姓名是什么？"
        answer = target['name']; evidence = [steps[0]['from_id']] + [step['to_id'] for step in steps]
        spec.update(operation='explicit_relation_path', steps=steps); oracle = {'type': 'path_target', 'target_id': target['person_id'], 'hops': len(steps), 'answer': answer}
    elif label == 'table':
        field = ('company', 'department', 'city')[index % 3]; value = person.get(field); threshold = int(person.get('age', 0)) - (variant % 3)
        matches = sorted((row for row in people if row.get(field) == value and int(row.get('age', 0)) >= threshold), key=lambda row: row['person_id'])
        query = f'筛选{field}为 {value} 且年龄不低于 {threshold} 的全部记录，按姓名升序输出。'; answer = '、'.join(row['name'] for row in matches); evidence = [row['person_id'] for row in matches]
        spec.update(operation='filter_sort', conditions=[{'field': field, 'equals': value}, {'field': 'age', 'gte': threshold}], sort='person_id'); oracle = {'type': 'names', 'person_ids': evidence, 'answer': answer}
    elif label == 'algorithm':
        skills = person.get('skills') or ['']; skill = str(skills[variant % len(skills)]); company = person.get('company'); minimum = int(person.get('work_years', 0))
        matches = sorted((row for row in people if skill in row.get('skills', []) and row.get('company') == company and int(row.get('work_years', 0)) >= minimum), key=lambda row: row['person_id'])
        query = f'执行规则：先保留公司为 {company}，再保留具备能力 {skill}，最后保留工作年限至少 {minimum}；输出姓名。'; answer = '、'.join(row['name'] for row in matches); evidence = [row['person_id'] for row in matches]
        spec.update(operation='sequential_filter', rules=[{'field': 'company', 'equals': company}, {'field': 'skills', 'contains': skill}, {'field': 'work_years', 'gte': minimum}], order=['company', 'skills', 'work_years']); oracle = {'type': 'names', 'person_ids': evidence, 'answer': answer}
    else:
        depth = min(4, level + 1); fields = ('company', 'department', 'team', 'name')[:depth]; query = f'在层级深度 {depth} 下，给出记录 {pid} 的完整归属路径。'; answer = ' > '.join(str(person.get(field, '')) for field in fields); evidence = [pid]
        spec.update(operation='hierarchy_path', depth=depth, target_id=pid, fields=list(fields)); oracle = {'type': 'hierarchy', 'target_id': pid, 'depth': depth, 'answer': answer}
    grounded_docs = [doc for doc in docs if any(str(item) in doc['content'] for item in evidence)]
    chosen = grounded_docs + [doc for doc in docs if doc not in grounded_docs][:min(level - 1, len(docs) - len(grounded_docs))]
    if not chosen: chosen = docs[:1]
    controls = {'hops': len(spec.get('steps', [])) if label == 'graph' else 1 if label == 'chunk' else 0,
                'conditions': len(spec.get('conditions', spec.get('rules', []))), 'distractors': len(chosen) - len(grounded_docs),
                'document_count': len(chosen), 'char_length': sum(len(doc['content']) for doc in chosen), 'hierarchy_depth': spec.get('depth', 0)}
    titles = [doc['title'] for doc in chosen]; messages = router_messages(query, titles); rejected = rng.choice([item for item in FIVE_LABELS if item != label])
    return {'id': f'{world}-L{level}-{label}-V{variant:03d}-{index:04d}', 'unit_id': world, 'source_root': world.split('_W')[0], 'label': label,
            'difficulty_level': level, 'difficulty_controls': controls, 'query': query, 'answer': answer,
            'answer_keywords': answer.split('、') if answer else ['无'], 'documents': chosen, 'source_evidence_ids': evidence,
            'task_spec': spec, 'oracle': oracle, 'chosen': messages + [{'role': 'assistant', 'content': label}],
            'rejected': messages + [{'role': 'assistant', 'content': rejected}], 'provenance': {'generator': 'build_complex_five_type_data_v3', 'source_root': world.split('_W')[0], 'synthetic_world': world}}

def build_candidates(world, people, per_level, seed):
    docs = build_documents(people); rng = random.Random(seed); rows = []
    for level in LEVELS:
        for variant in range(per_level):
            for label in FIVE_LABELS:
                row = make_row(world, people, docs, label, level, variant, variant, rng)
                if row: rows.append(row)
    return rows

def level_counts(total):
    raw = {level: total * WEIGHTS[level] for level in LEVELS}; counts = {level: int(raw[level]) for level in LEVELS}
    for level in sorted(LEVELS, key=lambda item: raw[item] - counts[item], reverse=True)[:total - sum(counts.values())]: counts[level] += 1
    return counts

def select_rows(rows, count, seed):
    if count > len(rows): raise ValueError(f'requested {count} rows from {len(rows)} candidates')
    groups = defaultdict(list)
    for row in rows: groups[int(row['difficulty_level'])].append(row)
    rng = random.Random(seed); selected = []
    for level, needed in level_counts(count).items(): rng.shuffle(groups[level]); selected.extend(groups[level][:needed])
    rng.shuffle(selected); return selected

def generate_dataset(profiles, out_dir, train_count=16000, validation_count=2000, test_count=3200, seed=20261015, **_):
    out_dir = Path(out_dir)
    if out_dir.exists(): raise FileExistsError(f'Refusing to overwrite existing output directory: {out_dir}')
    grouped = defaultdict(list)
    for row in read_jsonl(profiles): grouped[str(row['unit_id'])].append(row)
    total = train_count + validation_count + test_count
    variants = max(1, (total + len(grouped) * 19 - 1) // (len(grouped) * 19))
    # Assign source roots before cloning. Every clone of one supplied unit stays
    # in the same split, so synthetic variants cannot leak the original world.
    roots = sorted(grouped)
    shuffled_roots = list(roots)
    random.Random(seed).shuffle(shuffled_roots)
    train_root_count = max(1, int(len(roots) * .6))
    validation_root_count = max(1, int(len(roots) * .2))
    root_split = {
        'train': set(shuffled_roots[:train_root_count]),
        'validation': set(shuffled_roots[train_root_count:train_root_count + validation_root_count]),
        'test': set(shuffled_roots[train_root_count + validation_root_count:]),
    }
    split_units = {name: [] for name in root_split}
    for root, people in sorted(grouped.items()):
        split = next(name for name, assigned in root_split.items() if root in assigned)
        split_units[split].extend((f'{root}_W{variant:03d}', clone_world(root, people, variant)) for variant in range(variants))
    requests = {'train': train_count, 'validation': validation_count, 'test': test_count}; selected = {}; candidates = {}
    per_level = max(1, (max(requests.values()) + len(split_units['train']) * 20 - 1) // (len(split_units['train']) * 20))
    for name, items in split_units.items():
        rows = []
        for offset, (world, people) in enumerate(items): rows.extend(build_candidates(world, people, per_level, seed + offset))
        candidates[name] = rows; selected[name] = select_rows(rows, requests[name], seed + len(name))
    out_dir.mkdir(parents=True)
    for name, rows in selected.items(): write_jsonl(out_dir / f'{name}.jsonl', rows)
    metadata = {'version': 'complex_five_type_v3_verified', 'seed': seed, 'labels': list(FIVE_LABELS), 'split_counts': {name: len(rows) for name, rows in selected.items()},
                'split_units': {name: [world for world, _ in items] for name, items in split_units.items()}, 'split_source_roots': {name: sorted(assigned) for name, assigned in root_split.items()},
                'split_label_counts': {name: dict(Counter(row['label'] for row in rows)) for name, rows in selected.items()}, 'split_level_counts': {name: dict(Counter(str(row['difficulty_level']) for row in rows)) for name, rows in selected.items()},
                'candidate_count_by_split': {name: len(rows) for name, rows in candidates.items()}, 'source_profile_rows': sum(map(len, grouped.values())), 'leakage_policy': 'world unit_id and source_root assigned to exactly one split'}
    (out_dir / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8'); return metadata

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--profiles', type=Path, required=True); parser.add_argument('--out-dir', type=Path, required=True); parser.add_argument('--train-count', type=int, default=16000); parser.add_argument('--validation-count', type=int, default=2000); parser.add_argument('--test-count', type=int, default=3200); parser.add_argument('--seed', type=int, default=20261015)
    args = parser.parse_args(); print(json.dumps(generate_dataset(**vars(args))['split_counts'], ensure_ascii=False))
if __name__ == '__main__': main()
