"""Fallback behaviour cache (contracts C2, C3). CPU is fine, ~10-20 min.
python fallback/build_cache_min.py   -> fallback/behaviour_cache.pt, fallback/traces.jsonl
"""
import ast, json, os, subprocess, sys, tempfile

def calls(tests):
    out = []
    for t in tests:
        try:
            n = ast.parse(t).body[0]
        except SyntaxError:
            continue
        if not isinstance(n, ast.Assert):
            continue
        e = n.test.left if isinstance(n.test, ast.Compare) else n.test
        if isinstance(e, ast.Call) and isinstance(e.func, ast.Name):
            out.append(e)
    return out

def mutate(v):
    if isinstance(v, bool): return [not v]
    if isinstance(v, int): return [0, 1, -1, -v, v + 1, v * 100 + 7]
    if isinstance(v, float): return [0.0, -v]
    if isinstance(v, str): return ["", v[:1], v[::-1], v + v, v * 5]
    if isinstance(v, list): return [[], v[:1], v[::-1], v + v[:1], v * 3]
    if isinstance(v, tuple): return [(), v[:1], tuple(reversed(v))]
    return []

def probes(cs, cap=20):
    seen, res, parsed = set(), [], []
    for c in cs:
        try:
            parsed.append((c.func.id, [ast.literal_eval(a) for a in c.args]))
        except Exception:
            pass
    def add(name, args):
        s = f"{name}({', '.join(repr(x) for x in args)})"
        if s not in seen:
            seen.add(s); res.append(s)
    for name, args in parsed:
        add(name, args)
    for name, args in parsed:
        for i, a in enumerate(args):
            for m in mutate(a):
                add(name, args[:i] + [m] + args[i + 1:])
    return res[:cap]

RUNNER = r'''
import json, signal, sys
def _alarm(*_): raise TimeoutError
signal.signal(signal.SIGALRM, _alarm)
{setup}
{code}
_res = []
for _e in json.loads(sys.stdin.read()):
    signal.alarm(1)                      # per-probe limit: one slow probe can't kill the whole task
    try:
        _r = repr(eval(_e))
    except BaseException as _x:
        _r = "Timeout" if isinstance(_x, TimeoutError) else type(_x).__name__
    finally:
        signal.alarm(0)
    _res.append([_e, _r[:200]])
print("@@" + json.dumps(_res))
'''

def run(code, setup, pr, timeout=30):
    try:
        p = subprocess.run([sys.executable, "-c", RUNNER.format(setup=setup or "", code=code)],
                           input=json.dumps(pr), capture_output=True, text=True, timeout=timeout,
                           cwd=tempfile.gettempdir(), env={**os.environ, "PYTHONHASHSEED": "0"})
    except subprocess.TimeoutExpired:
        return []
    line = next((l for l in p.stdout.splitlines() if l.startswith("@@")), None)
    return [] if line is None else [f"{e} -> {r}" for e, r in json.loads(line[2:])]

def embed(enc, lines):
    import torch
    v = enc.encode(lines, convert_to_tensor=True, normalize_embeddings=True).mean(0).float().cpu()
    return torch.nn.functional.normalize(v, dim=0)

if __name__ == "__main__":
    import torch
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    ds = load_dataset("google-research-datasets/mbpp", "full")
    enc = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    cache, traces = {}, []
    for split in ["train", "validation", "prompt"]:
        for ex in ds[split]:
            lines = run(ex["code"], ex["test_setup_code"], probes(calls(ex["test_list"])))
            if lines:
                cache[int(ex["task_id"])] = embed(enc, lines)
                traces.append({"task_id": int(ex["task_id"]), "trace": lines})
    assert not any(11 <= k <= 510 for k in cache)
    torch.save(cache, "fallback/behaviour_cache.pt")
    with open("fallback/traces.jsonl", "w") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")
    print("cached", len(cache), "of 474")
