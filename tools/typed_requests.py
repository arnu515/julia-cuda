"""Converts typed-decisions test.parquet to julia-rs requests (rows.jsonl) and labels (labels.jsonl),
the same way Julia-1's scripts/reproduce_typed.py builds them. Usage:
    uv run --no-project --with pyarrow python tools/typed_requests.py test.parquet OUTDIR"""
import json, sys, pyarrow.parquet as pq
rows, labels = [], []
for case in pq.read_table(sys.argv[1]).to_pylist():
    state, gold = json.loads(case['state']), json.loads(case['gold'])
    for qid, q in json.loads(case['questions']).items():
        kind, crit = q['type'], q.get('criteria')
        if crit is None and kind == 'noul':
            crit = {'false': 'false', 'true': 'true'}
        if isinstance(crit, list):
            crit = {str(i): v for i, v in enumerate(crit)}
        keys = ['false', 'true'] if kind == 'noul' else list(crit)
        rows.append(dict(state=state, question=q['instructions'], type=kind, options=[crit[k] for k in keys]))
        labels.append(dict(type=kind, keys=keys, gold=str(gold[qid]['label'])))
assert len(rows) == 2000
for name, items in [('rows.jsonl', rows), ('labels.jsonl', labels)]:
    with open(f'{sys.argv[2]}/{name}', 'w') as f:
        f.writelines(json.dumps(x) + '\n' for x in items)
