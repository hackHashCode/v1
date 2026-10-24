"""Probe generation, sandboxed execution, and behaviour encoders for PreCode-JEPA.

Pipeline per task: MBPP test asserts -> probe calls (originals + type-aware edge cases)
-> run reference solution in a subprocess (1 s per probe, PYTHONHASHSEED=0, run twice, keep
deterministic lines) -> trace lines "f(args) -> repr(output)" -> 768-d vector z.

Encoders (all return L2-normalised float32 tensors of shape (768,)):
  features : signed feature hashing of behaviour features (no downloads, deterministic)
  mpnet    : all-mpnet-base-v2 per-line embeddings, mean-pooled (needs Hugging Face access)
  hybrid   : normalize(0.5 * mpnet + 0.5 * features)
"""
import ast, hashlib, json, os, re, subprocess, sys, tempfile

DIM = 768

# ---------------------------------------------------------------- probes
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
    if isinstance(v, float): return [0.0, -v, v * 2.5]
    if isinstance(v, str): return ["", v[:1], v[::-1], v + v, v * 5]
    if isinstance(v, list):
        num = bool(v) and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)
        return [[], v[:1], v[::-1], v + v[:1], v * 3] + ([[-x for x in v]] if num else [])
    if isinstance(v, tuple): return [(), v[:1], tuple(reversed(v)), v + v[:1]]
    if isinstance(v, dict): return [{}, dict(list(v.items())[:1])]
    return []

def make_probes(test_list, cap=20):
    seen, res, parsed = set(), [], []
    for c in calls(test_list):
        try:
            parsed.append((c.func.id, [ast.literal_eval(a) for a in c.args]))
        except Exception:
            pass
    def add(name, args):
        s = f"{name}({', '.join(repr(x) for x in args)})"
        if s not in seen:
            seen.add(s); res.append(s)
    for name, args in parsed:                      # originals first
        add(name, args)
    for name, args in parsed:                      # then one argument mutated at a time
        for i, a in enumerate(args):
            for m in mutate(a):
                add(name, args[:i] + [m] + args[i + 1:])
    return res[:cap]

# ---------------------------------------------------------------- sandbox
RUNNER = r'''
import json, signal, sys
def _alarm(*_): raise TimeoutError
signal.signal(signal.SIGALRM, _alarm)
{setup}
{code}
_res = []
for _e in json.loads(sys.stdin.read()):
    signal.alarm(1)
    try:
        _r = repr(eval(_e))
    except BaseException as _x:
        _r = "Timeout" if isinstance(_x, TimeoutError) else type(_x).__name__
    finally:
        signal.alarm(0)
    _res.append([_e, _r[:200]])
print("@@" + json.dumps(_res))
'''

def _limits():
    try:
        import resource, platform
        if platform.system() == "Linux":
            resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    except Exception:
        pass

def run_probes(code, setup, probes, timeout=30):
    try:
        p = subprocess.run([sys.executable, "-c", RUNNER.format(setup=setup or "", code=code)],
                           input=json.dumps(probes), capture_output=True, text=True, timeout=timeout,
                           cwd=tempfile.gettempdir(), env={**os.environ, "PYTHONHASHSEED": "0"},
                           preexec_fn=_limits)
    except subprocess.TimeoutExpired:
        return []
    line = next((l for l in p.stdout.splitlines() if l.startswith("@@")), None)
    return [] if line is None else [f"{e} -> {r}" for e, r in json.loads(line[2:])]

def trace(code, setup, probes):
    """Deterministic trace: run twice, keep lines that agree."""
    a, b = run_probes(code, setup, probes), run_probes(code, setup, probes)
    return [x for x, y in zip(a, b) if x == y]

# ---------------------------------------------------------------- encoders
def _h(s):
    d = hashlib.sha256(s.encode()).digest()
    return int.from_bytes(d[:4], "little") % DIM, (1.0 if d[4] & 1 else -1.0)

_EXC = re.compile(r"^[A-Z][A-Za-z]*(Error|Exception|Timeout)$|^Timeout$|^StopIteration$")

def _out_type(r):
    if _EXC.match(r): return "EXC:" + r
    if r in ("True", "False"): return "bool"
    if r == "None": return "None"
    if re.fullmatch(r"-?\d+", r): return "int"
    if re.fullmatch(r"-?\d+\.\d*(e-?\d+)?|-?inf|nan", r): return "float"
    return {"'": "str", '"': "str", "[": "list", "(": "tuple", "{": "dict"}.get(r[:1], "other")

def _in_kind(call):
    inner = call[call.find("(") + 1:-1]
    kinds = []
    if re.search(r"(\(|, )(\[\]|''|\"\"|\(\)|\{\})(,|\)|$)", "(" + inner + ")"): kinds.append("empty")
    if re.search(r"(^|[\[(, ])-\d", inner): kinds.append("neg")
    if len(inner) > 60: kinds.append("large")
    return kinds or ["plain"]

def behaviour_features(lines):
    """Weighted behaviour features of a trace. Exact lines dominate (they separate near-miss
    bugs); coarser features (output types, exceptions on edge inputs) give cross-task structure."""
    feats = []
    for l in lines:
        call, _, r = l.partition(" -> ")
        t = _out_type(r)
        feats.append(("line:" + l, 2.0))                 # exact behaviour on this input
        feats.append(("out:" + r, 1.0))                  # output value
        feats.append(("type:" + t, 0.5))                 # output type
        for k in _in_kind(call):
            feats.append((f"edge:{k}->{t}", 0.7))        # e.g. edge:empty->EXC:IndexError
    return feats

_CALL_NAME = re.compile(r"^[A-Za-z_]\w*\(")


def anonymize(lines):
    """Rename the called function to f in every trace line. Function names are almost unique per MBPP task (447 of
    470) and the prompt shows the name, so a name-bearing target can be "predicted" by reading the name; about half
    of a name-bearing vector depends on the name alone (cos 0.52 to its anonymised version)."""
    return [_CALL_NAME.sub("f(", l, count=1) for l in lines]


def encode_features(lines, anonymize_names=False):
    """Signed feature hashing with sqrt damping of repeated features: a feature seen on every probe
    (e.g. 'returns int') must not drown out the few probes where a bug changes the output.
    Chosen on the mutant test: plain sums gave ref-vs-mutant cos 0.90 (code text 0.95); sqrt gives
    0.86 while keeping cross-task structure (mean cos ~0.06); counting once gives 0.79 but ~0 structure."""
    import math, torch
    if anonymize_names:
        lines = anonymize(lines)
    acc = {}
    for f, w in behaviour_features(lines):
        acc[f] = acc.get(f, 0.0) + w
    v = torch.zeros(DIM)
    for f, w in acc.items():
        i, s = _h(f); v[i] += s * math.sqrt(w)
    return torch.nn.functional.normalize(v, dim=0)

_MPNET = None
def encode_mpnet(lines):
    import torch
    global _MPNET
    if _MPNET is None:
        from sentence_transformers import SentenceTransformer
        _MPNET = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    v = _MPNET.encode(lines, convert_to_tensor=True, normalize_embeddings=True).mean(0).float().cpu()
    return torch.nn.functional.normalize(v, dim=0)

def encode(lines, encoder="features", anonymize_names=False):
    import torch
    if anonymize_names: lines = anonymize(lines)
    if encoder == "features": return encode_features(lines)
    if encoder == "mpnet": return encode_mpnet(lines)
    if encoder == "hybrid":
        return torch.nn.functional.normalize(0.5 * encode_mpnet(lines) + 0.5 * encode_features(lines), dim=0)
    raise ValueError(encoder)

def encode_code_text(code, encoder="features"):
    """Code-TEXT baseline (what an LLM-JEPA-style target sees): same machinery on code tokens."""
    import torch
    if encoder == "features":
        v = torch.zeros(DIM)
        toks = re.findall(r"[A-Za-z_]\w*|\d+|\S", code)
        for t in toks + [a + " " + b for a, b in zip(toks, toks[1:])]:
            i, s = _h("tok:" + t); v[i] += s
        return torch.nn.functional.normalize(v, dim=0)
    return encode_mpnet([code])

# ---------------------------------------------------------------- data
MBPP_RAW = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"

def load_mbpp_all():
    try:
        from datasets import load_dataset
        ds = load_dataset("google-research-datasets/mbpp", "full")
        return [ex for s in ds for ex in ds[s]]
    except Exception:
        import urllib.request
        return [json.loads(l) for l in urllib.request.urlopen(MBPP_RAW, timeout=60).read().decode().splitlines()]

def load_mbpp_train():
    return [ex for ex in load_mbpp_all() if not 11 <= ex["task_id"] <= 510]
