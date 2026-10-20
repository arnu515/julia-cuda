"""score.py labels.jsonl results.jsonl [baseline.jsonl]: accuracy, and agreement / logit change vs a baseline."""
import json, sys
labels = [json.loads(l) for l in open(sys.argv[1])]
res = [json.loads(l) for l in open(sys.argv[2])]
pred = [max(range(len(r['probabilities'])), key=r['probabilities'].__getitem__) for r in res]
correct = sum(l['keys'][p] == l['gold'] for l, p in zip(labels, pred))
print(f"{sys.argv[2]}: {correct}/{len(labels)} correct", end='')
if len(sys.argv) > 3:
    base = [json.loads(l) for l in open(sys.argv[3])]
    bp = [max(range(len(r['probabilities'])), key=r['probabilities'].__getitem__) for r in base]
    d = [abs(a - b) for r, s in zip(res, base) for a, b in zip(r['logits'], s['logits'])]
    print(f" | same answer as baseline {sum(a == b for a, b in zip(pred, bp))}/{len(bp)} | mean |Δlogit| {sum(d)/len(d):.4f} max {max(d):.3f}", end='')
print()
