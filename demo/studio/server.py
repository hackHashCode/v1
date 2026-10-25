"""PreCode Studio: live side-by-side demo backend.

Run from the repo root:
    uvicorn demo.studio.server:app --host 0.0.0.0 --port 8010
    MOCK=1 uvicorn demo.studio.server:app --port 8010      # no GPU / no checkpoints needed

What it does for one prompt:
  1. /api/generate (SSE)  streams code from every method, one after another, and sends PreCode's
                          "imagined behaviour": pred_head(h_prompt) -> nearest cached training tasks.
  2. /api/execute         runs every program in a sandbox on the user's asserts (if any) and on a
                          shared probe set, encodes each program's actual behaviour with the same
                          encoder as the cache, and places everything on the behaviour map.

Environment:
  MODELS_DIR   default /data/precode/models     (contract C4: merged HF model dirs)
  HF_REPO      default MoistPotato/precode-jepa-qwen2.5-coder-1.5b   runs missing from MODELS_DIR are downloaded
               from <HF_REPO>/<run>_s<seed>/ (set HF_REPO= to disable)
  DEMO_SEED    default 42                       picks <run>_s<seed>
  RUNS         default sft,jepa,precodefull_aux  (add base for zero-shot, sft_loop for the SFT control)
  BASE_MODEL   default Qwen/Qwen2.5-Coder-1.5B (zero-shot column; use the same base the runs were trained from)
  GPU_MODE     auto | all | swap                swap keeps models in CPU RAM, moves one to GPU at a time
  PARALLEL     1 (default) = all methods generate at the same time when every model fits on the GPU
  H_POOL       last | mean   fallback only; each run's precode_config.json wins (must match training)
  MAX_NEW      default 384
  BEHAVIOUR_CACHE  default behaviour_cache_anon.pt (use behaviour_cache.pt for the original Llama runs)
  DTYPE        bf16 | fp16 | fp32 to override the automatic choice
  MOCK         1 = canned generations, no torch models (UI development)
"""
import ast, csv, json, math, os, re, subprocess, sys, tempfile, threading, time
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "cache"))
import behaviour as B  # noqa: E402  (probes, sandbox, encoders: same code that built the cache)

SYSTEM = "You are an expert Python programmer. Write a correct Python function for the task. Output only code."
MOCK = os.environ.get("MOCK", "0") == "1"
SEED = os.environ.get("DEMO_SEED", "42")
MODELS_DIR = os.environ.get("MODELS_DIR", "/data/precode/models")
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-Coder-1.5B")
H_POOL = os.environ.get("H_POOL", "last")
MAX_NEW = int(os.environ.get("MAX_NEW", "384"))
HF_REPO = os.environ.get("HF_REPO", "MoistPotato/precode-jepa-qwen2.5-coder-1.5b")
PARALLEL = os.environ.get("PARALLEL", "1") == "1"

LABELS = {
    "base": ("Zero-shot", f"{BASE_MODEL.split('/')[-1]}, no fine-tuning"),
    "sft": ("SFT", "next-token loss only"),
    "sft_loop": ("SFT control", "PreCode's training loop, behaviour loss off"),
    "jepa": ("LLM-JEPA", "predicts its own code-text embedding"),
    "precode": ("PreCode", "predicts execution behaviour (ours)"),
    "precodence": ("PreCode-NCE", "behaviour target, InfoNCE over all tasks"),
    "precodefull_aux": ("PreCode", "behaviour head (InfoNCE), trained predictor (ours)"),
    "precodefull_prefix": ("PreCode-full +prefix", "writes while reading its own behaviour guess"),
    "precodefull": ("PreCode-full", "prefix + hard-task weighting (ours)"),
}

# ------------------------------------------------------------------ data: cache, traces, train tasks
CACHE_FILE = os.environ.get("BEHAVIOUR_CACHE", "behaviour_cache_anon.pt")    # must match the cache the runs trained on
CACHE = {int(k): F.normalize(v.float(), dim=0) for k, v in torch.load(os.path.join(ROOT, "cache", CACHE_FILE)).items()}
IDS = sorted(CACHE)
Z = torch.stack([CACHE[t] for t in IDS])                     # [N, 768]
TRACES = {}
for line in open(os.path.join(ROOT, "cache", "traces.jsonl"), encoding="utf-8"):
    r = json.loads(line); TRACES[int(r["task_id"])] = r["trace"]
DESC = {}
for line in open(os.path.join(ROOT, "datasets", "mbpp_train.jsonl"), encoding="utf-8"):
    r = json.loads(line)
    user = next(m["content"] for m in r["messages"] if m["role"] == "user")
    DESC[int(r["task_id"])] = user.split("\n")[0]


def build_map():
    """2-D layout of the behaviour space. t-SNE on cosine distance; new points are placed by a
    cosine-weighted average of their nearest cached tasks (t-SNE can't embed new points)."""
    path = os.path.join(HERE, f"map_cache_{CACHE_FILE.replace('.pt', '')}.json")
    if os.path.exists(path):
        d = json.load(open(path, encoding="utf-8"))
        if d.get("ids") == IDS:
            return np.array(d["xy"])
    from sklearn.manifold import TSNE
    xy = TSNE(n_components=2, metric="cosine", init="random", perplexity=30, random_state=0).fit_transform(Z.numpy())
    xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))
    json.dump({"ids": IDS, "xy": xy.tolist()}, open(path, "w", encoding="utf-8"))
    return xy


XY = build_map()


def place(z, k=6):
    sims = Z @ F.normalize(z.float(), dim=0)
    top = torch.topk(sims, k)
    w = torch.clamp(top.values, min=1e-3) ** 4
    xy = (XY[top.indices.numpy()] * (w / w.sum()).numpy()[:, None]).sum(0)
    return [float(xy[0]), float(xy[1])]


def nearest(z, k=3):
    sims = Z @ F.normalize(z.float(), dim=0)
    top = torch.topk(sims, k)
    return [{"task_id": IDS[i], "cos": round(float(s), 3), "desc": DESC.get(IDS[i], ""), "trace": TRACES.get(IDS[i], [])[:5]}
            for s, i in zip(top.values.tolist(), top.indices.tolist())]


# ------------------------------------------------------------------ models
def fetch_from_hub(run):
    """Download <HF_REPO>/<run>/ (merged weights + heads) into MODELS_DIR, or the HF cache if that isn't writable."""
    from huggingface_hub import snapshot_download
    try:
        os.makedirs(MODELS_DIR, exist_ok=True); root = MODELS_DIR
    except OSError:
        root = os.path.join(tempfile.gettempdir(), "precode_models")
    snapshot_download(HF_REPO, allow_patterns=[f"{run}/*"], local_dir=root)
    return os.path.join(root, run)


def run_dirs():
    names = [r.strip() for r in os.environ.get("RUNS", "sft,jepa,precodefull_aux").split(",") if r.strip()]
    out = {}
    for n in names:
        if n == "base":
            out[n] = BASE_MODEL; continue
        path = os.path.join(MODELS_DIR, f"{n}_s{SEED}")
        if not os.path.isdir(path) and HF_REPO and not MOCK:
            print(f"[studio] {path} not found, downloading {HF_REPO}/{n}_s{SEED}", flush=True)
            path = fetch_from_hub(f"{n}_s{SEED}")
        out[n] = path
    return out


RUNS = run_dirs()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" and torch.cuda.is_bf16_supported() else (torch.float16 if DEVICE == "cuda" else torch.float32)
DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(os.environ.get("DTYPE", ""), DTYPE)   # override, e.g. bf16 on a small-RAM CPU box
GPU_LOCK = threading.Lock()
MODELS, HEADS, PREFIX, POOL, TOKS, TOK = {}, {}, {}, {}, {}, None
sys.path.insert(0, os.path.join(ROOT, "train"))
from precode_runtime import ensure_chat_template, extract_code, load_precode_parts, stop_ids  # noqa: E402


def load_head(path, dim_in):
    sd = torch.load(path, map_location="cpu")
    if "weight" in sd:
        head = torch.nn.Linear(dim_in, sd["weight"].shape[0], bias=False)
    else:  # --pred_mlp: keys 0.weight / 2.weight
        head = torch.nn.Sequential(torch.nn.Linear(dim_in, dim_in // 2, bias=False), torch.nn.GELU(),
                                   torch.nn.Linear(dim_in // 2, sd["2.weight"].shape[0], bias=False))
    head.load_state_dict(sd)
    return head.float().eval()


if not MOCK:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    gpu_mode = os.environ.get("GPU_MODE", "auto")
    if gpu_mode == "auto":
        vram = torch.cuda.get_device_properties(0).total_memory / 2**30 if DEVICE == "cuda" else 0
        gpu_mode = "all" if vram >= 2.8 * len(RUNS) + 1.5 else "swap"
    GPU_MODE = gpu_mode
    for name, path in list(RUNS.items()):
        if name != "base" and not os.path.isdir(path):
            print(f"[studio] skipping {name}: {path} not found", flush=True)
            del RUNS[name]; continue
        m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=DTYPE).eval()
        if GPU_MODE == "all":
            m = m.to(DEVICE)
        MODELS[name] = m
        TOKS[name] = ensure_chat_template(AutoTokenizer.from_pretrained(path))   # one per model: threads can't share a fast tokenizer
        if TOK is None:
            TOK = TOKS[name]
        if name != "base":
            head, inj, _cfg = load_precode_parts(path, m.config.hidden_size)
            POOL[name] = _cfg.get("h_pool", H_POOL)
            if head is not None and name != "sft_loop":   # the control's head is never trained (lambda 0)
                HEADS[name] = head
            if inj is not None:
                PREFIX[name] = inj
    print(f"[studio] device={DEVICE} dtype={DTYPE} gpu_mode={GPU_MODE} runs={list(MODELS)} heads={list(HEADS)} prefix={list(PREFIX)}", flush=True)
else:
    GPU_MODE = "mock"
    HEADS = {k: None for k in RUNS if k.startswith("precode")}


def user_message(task, tests):
    return task.strip() + ("\n" + tests[0].strip() if tests else "")


def extract(t):
    if "```" in t:
        t = t.split("```")[1]
        t = t[len("python"):] if t.startswith("python") else t
    return t.strip()


def prompt_ids(user, tok=None):
    tok = tok or TOK
    text = tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False, return_tensors="pt").input_ids   # same tokenisation as training


def on_gpu(name):
    m = MODELS[name]
    if GPU_MODE == "swap" and DEVICE == "cuda":
        m.to(DEVICE)
    return m


def off_gpu(name):
    if GPU_MODE == "swap" and DEVICE == "cuda":
        MODELS[name].to("cpu"); torch.cuda.empty_cache()


def predict_z(name, m, ids):
    """pred_head(h_prompt): h at the last prompt token (or mean over prompt), final layer, as in PreCodeTrainer."""
    with torch.no_grad():
        out = m(ids, output_hidden_states=True)
    h_all = out.hidden_states[-1][0].float()
    h = h_all.mean(0) if POOL.get(name, H_POOL) == "mean" else h_all[-1]
    with torch.no_grad():
        return F.normalize(HEADS[name](h.cpu()), dim=0)


# ------------------------------------------------------------------ mock generation (UI development)
SPECS = json.load(open(os.path.join(ROOT, "demo", "specs.json"), encoding="utf-8"))


@lru_cache(maxsize=1)
def tfidf():
    from sklearn.feature_extraction.text import TfidfVectorizer
    v = TfidfVectorizer().fit([DESC.get(t, "") for t in IDS])
    return v, v.transform([DESC.get(t, "") for t in IDS])


def mock_z(task):
    v, M = tfidf()
    s = (M @ v.transform([task]).T).toarray()[:, 0]
    idx = np.argsort(-s)[:3]
    z = sum(float(s[i] + 0.05) * Z[i] for i in idx) + 0.15 * torch.randn(Z.shape[1], generator=torch.Generator().manual_seed(len(task)))
    return F.normalize(z, dim=0)


def mock_code(name, task, tests):
    show = next((t for t in SHOWCASE.get("tasks", []) if t["task"].strip() == task.strip()), None) if "SHOWCASE" in globals() else None
    if show:   # UI preview only: PreCode gets the reference, the others a stub with the right name
        if name.startswith("precode"):
            return show["reference"]
        fn = re.search(r"assert\s+([A-Za-z_]\w*)\s*\(", show["tests"][0]).group(1)
        return f"def {fn}(*args):\n    # mock output ({name})\n    return args[0] if args else None"
    spec = next((s for s in SPECS if s["task"].strip() == task.strip()), None)
    if spec:
        return spec["reference"] if name.startswith("precode") or (name == "jepa" and len(task) % 2) else spec["plausible_bug"]
    fn = "solve"
    if tests:
        m = re.search(r"assert\s+([A-Za-z_]\w*)\s*\(", tests[0]); fn = m.group(1) if m else fn
    return f"def {fn}(xs):\n    # mock output ({name})\n    return sorted(xs)[-2] if len(xs) > 1 else None"


# ------------------------------------------------------------------ app
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
def stage():   # the minimal stage demo: one claim, one task, SFT vs PreCode
    return FileResponse(os.path.join(HERE, "stage.html"), headers={"Cache-Control": "no-cache"})   # always serve the latest page


@app.get("/studio")
def index():   # the full studio: every method, predicted behaviour, probe grid, behaviour map, results
    return FileResponse(os.path.join(HERE, "index.html"), headers={"Cache-Control": "no-cache"})   # always serve the latest page


RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results", "qc15"))
OURS = os.environ.get("OURS", "precodefull_aux")          # the method the demo calls "PreCode"
SHOWN = ["sft", "sft_loop", "jepa", OURS]   # zero-shot left out: its chat-format eval is not comparable
NAMES = {"base": "Zero-shot", "sft": "SFT", "sft_loop": "SFT control (λ = 0)", "jepa": "LLM-JEPA", OURS: "PreCode"}


def read_results():
    """Final multi-seed results for the Results tab: results/qc15/comparison.json (written by eval/compare.py)."""
    p = os.path.join(RESULTS_DIR, "comparison.json")
    if not os.path.exists(p):
        return {}
    d = json.load(open(p, encoding="utf-8"))
    table = []
    for m in SHOWN:
        row = d.get("table", {}).get(m)
        if not row:
            continue
        cell = {ds: {"mean": row[ds]["plus"]["mean"], "std": row[ds]["plus"]["std"], "n": row[ds]["plus"]["n"],
                     "ci": row[ds].get("plus_ci95")} for ds in ("mbppplus224", "humanevalplus") if ds in row}
        table.append({"method": m, "name": NAMES.get(m, m), **cell})
    want = {(OURS, "sft"), (OURS, "sft_loop"), (OURS, "jepa"), ("sft_loop", "sft")}
    tests = [{**t, "a_name": NAMES.get(t["a"], t["a"]), "b_name": NAMES.get(t["b"], t["b"])}
             for t in d.get("tests", []) if (t["a"], t["b"]) in want]
    beh = {}
    for run, mt in d.get("metrics", {}).items():
        m = run.rsplit("_s", 1)[0]
        if m in SHOWN:
            for k, v in mt.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "n_tasks":
                    beh.setdefault(m, {}).setdefault(k, []).append(v)
    behaviour = [{"method": m, "name": NAMES[m], **{k: round(sum(v) / len(v), 4) for k, v in beh[m].items()}} for m in SHOWN if m in beh]
    mutant = {}
    mp = os.path.join(ROOT, "results", "plots", "mutant_numbers_features_anon.json")
    if not os.path.exists(mp):
        mp = os.path.join(ROOT, "results", "plots", "mutant_numbers_features.json")
    if os.path.exists(mp):
        md = json.load(open(mp, encoding="utf-8"))
        mutant = {k: md[k] for k in ("n_mutants", "behaviour_ref_vs_mutant", "codetext_ref_vs_mutant",
                                     "behaviour_ref_vs_other_task", "frac_mutants_where_behaviour_separates_more", "mean_differing_probes") if k in md}
        mutant["pairs"] = list(zip(md.get("pairs_codetext", []), md.get("pairs_behaviour", [])))
    return {"table": table, "tests": tests, "behaviour": behaviour, "mutant": mutant, "ours": OURS}


SHOWCASE = {}
_sp = os.path.join(HERE, "showcase.json")
if os.path.exists(_sp):
    SHOWCASE = json.load(open(_sp, encoding="utf-8"))
REC_DIR = os.path.join(HERE, "recordings")


@app.get("/api/meta")
def meta():
    show = [{"label": t["label"], "task": t["task"], "tests": t["tests"], "setup": t.get("setup", ""), "reference": t["reference"],
             "task_id": t["task_id"], "record": t["record"], "group": "showcase"} for t in SHOWCASE.get("tasks", [])]
    specs = [{"label": s["label"], "task": s["task"], "tests": s["tests"], "reference": s.get("reference"), "group": "specs"} for s in SPECS]
    recs = sorted(f[:-5] for f in os.listdir(REC_DIR) if f.endswith(".json")) if os.path.isdir(REC_DIR) else []
    return {
        "mock": MOCK, "seed": SEED, "gpu_mode": GPU_MODE, "device": DEVICE, "parallel": PARALLEL and (MOCK or GPU_MODE == "all"),
        "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None, "base_model": BASE_MODEL,
        "runs": [{"id": k, "name": "PreCode" if k == OURS else LABELS.get(k, (k, ""))[0], "blurb": LABELS.get(k, ("", k))[1],
                  "imagines": k in HEADS, "ours": k == OURS}
                 for k in RUNS if MOCK or k in MODELS],
        "examples": show + specs,
        "showcase": {k: SHOWCASE.get(k) for k in ("rule", "n_picked", "n_reverse", "n_tasks", "source", "stage")},
        "recordings": recs,
        "map": [{"id": t, "x": round(float(XY[i][0]), 4), "y": round(float(XY[i][1]), 4), "d": DESC.get(t, "")[:110]} for i, t in enumerate(IDS)],
        "results": read_results(),
    }


@app.get("/api/recording/{name}")
def recording(name: str):
    path = os.path.join(REC_DIR, os.path.basename(name) + ".json")
    if not os.path.exists(path):
        return JSONResponse({"error": "no such recording"}, status_code=404)
    return FileResponse(path, media_type="application/json")


def sse(obj):
    return "data: " + json.dumps(obj) + "\n\n"


@app.post("/api/generate")
def generate(req: dict):
    task = (req.get("task") or "").strip()
    tests = [t for t in req.get("tests", []) if t.strip()]
    wanted = [r for r in req.get("runs", list(RUNS)) if r in RUNS]
    user = user_message(task, tests)
    rid = os.urandom(3).hex()
    where = {"type": "server", "rid": rid, "container": os.environ.get("MODAL_TASK_ID", os.uname().nodename),
             "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else DEVICE, "time": time.strftime("%H:%M:%S"),
             "models": {n: (RUNS[n] if n == "base" else f"{HF_REPO}/{n}_s{SEED}") for n in wanted}, "mock": MOCK}
    print(f"[generate] req={rid} runs={wanted} task={task[:70]!r}", flush=True)

    def one(name, emit):
        """Generate for one method, pushing events through emit(). Runs in its own thread when PARALLEL."""
        t0 = time.time()
        emit({"type": "start", "run": name})
        if MOCK:
            if name in HEADS:
                z = mock_z(task)
                emit({"type": "imagined", "run": name, "z": z.tolist(), "nearest": nearest(z), "xy": place(z)})
            code = mock_code(name, task, tests)
            for i in range(0, len(code), 6):
                time.sleep(0.02 + 0.01 * (hash(name) % 3)); emit({"type": "token", "run": name, "text": code[i:i + 6]})
            emit({"type": "done", "run": name, "code": code, "ms": int((time.time() - t0) * 1000)})
            return
        from transformers import TextIteratorStreamer
        lock = threading.Lock() if parallel else GPU_LOCK   # sequential mode shares the GPU one model at a time
        with lock:
            m = on_gpu(name)
            text = ""
            try:
                tok = TOKS.get(name, TOK)
                ids = prompt_ids(user, tok).to(m.device)
                z = None
                if name in HEADS:
                    z = predict_z(name, m, ids)
                    emit({"type": "imagined", "run": name, "z": z.tolist(), "nearest": nearest(z), "xy": place(z)})
                streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
                gen = dict(max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id, eos_token_id=stop_ids(tok), streamer=streamer)
                if name in PREFIX:   # PreCode-full: [BOS, 4 soft tokens from predicted z, rest], as in training
                    with torch.no_grad():
                        pre = PREFIX[name](z.unsqueeze(0)).to(m.device)
                        e = m.get_input_embeddings()(ids)
                        emb = torch.cat([e[:, :1], pre.to(e.dtype), e[:, 1:]], 1)
                    gen.update(inputs_embeds=emb, attention_mask=torch.ones(emb.shape[:2], dtype=torch.long, device=m.device))
                else:
                    gen.update(input_ids=ids, attention_mask=torch.ones_like(ids))
                th = threading.Thread(target=m.generate, kwargs=gen)
                th.start()
                for piece in streamer:
                    text += piece
                    emit({"type": "token", "run": name, "text": piece})
                th.join()
            except Exception as ex:   # one broken model must not kill the other columns
                emit({"type": "error", "run": name, "message": f"{type(ex).__name__}: {ex}"[:300]})
            finally:
                off_gpu(name)
        secs = time.time() - t0
        ntok = len(TOKS.get(name, TOK)(text, add_special_tokens=False).input_ids) if text else 0
        print(f"[generate] req={rid} {name}: {ntok} tokens in {secs:.2f}s ({ntok / max(secs, 1e-6):.0f} tok/s)", flush=True)
        emit({"type": "done", "run": name, "code": extract_code(text), "ms": int(secs * 1000), "tokens": ntok})

    parallel = PARALLEL and (MOCK or GPU_MODE == "all")

    def stream():   # all methods at once: one thread per model, events merged as they arrive
        import queue
        q = queue.Queue()
        yield sse(where)
        threads = [threading.Thread(target=one, args=(n, q.put), daemon=True) for n in wanted]
        for th in threads:
            th.start()
        finished = 0
        while finished < len(wanted):
            ev = q.get()
            if ev["type"] == "done":
                finished += 1
            yield sse(ev)
        yield sse({"type": "end"})

    def stream_sequential():
        import queue
        yield sse(where)
        for name in wanted:
            q = queue.Queue()
            th = threading.Thread(target=one, args=(name, q.put), daemon=True); th.start()
            while True:
                ev = q.get(); yield sse(ev)
                if ev["type"] == "done":
                    break
        yield sse({"type": "end"})

    body = stream() if parallel else stream_sequential()
    return StreamingResponse(body, media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ------------------------------------------------------------------ execution
TEST_RUNNER = r'''
import json, signal, sys
_HAS_ALARM = hasattr(signal, "SIGALRM")
def _alarm(*_): raise TimeoutError
if _HAS_ALARM: signal.signal(signal.SIGALRM, _alarm)
def _set(n):
    if _HAS_ALARM: signal.alarm(n)
_res = []
try:
    _set(3)
    exec(compile({code!r}, "<model>", "exec"), globals())
    _set(0)
    _load = None
except BaseException as _x:
    _load = type(_x).__name__ + ": " + str(_x)[:120]
for _t in json.loads(sys.stdin.read()):
    if _load:
        _res.append([False, "does not run (" + _load + ")"]); continue
    _set(2)
    try:
        exec(_t, globals()); _res.append([True, ""])
    except AssertionError:
        _res.append([False, "wrong answer"])
    except BaseException as _x:
        _res.append([False, "Timeout" if isinstance(_x, TimeoutError) else type(_x).__name__ + ": " + str(_x)[:120]])
    finally:
        _set(0)
print("@@" + json.dumps(_res))
'''

POSIX = os.name != "nt"
SANDBOX_KW = {"preexec_fn": B._limits} if POSIX else {}


def run_tests(code, tests):
    if not tests:
        return []
    try:
        p = subprocess.run([sys.executable, "-c", TEST_RUNNER.format(code=code)], input=json.dumps(tests), capture_output=True,
                           text=True, timeout=10 + 2 * len(tests), cwd=tempfile.gettempdir(), **SANDBOX_KW)
        line = next((l for l in p.stdout.splitlines() if l.startswith("@@")), None)
        if line:
            return [{"test": t, "ok": ok, "why": why} for t, (ok, why) in zip(tests, json.loads(line[2:]))]
    except subprocess.TimeoutExpired:
        pass
    return [{"test": t, "ok": False, "why": "Timeout"} for t in tests]


GENERIC = {
    1: [[], [1], [5, 5, 3], [3, 1, 2], [-1, -5, -3], [1, 1, 2, 2], [2, 9, 4, 9, 1], list(range(1, 11)), [0, 0, 1],
        "", "a", "racecar", "Hello World", "aabbcc", 0, 1, 7, -3, 10, 2.5],
    2: [([1, 2, 3], 2), ([], 1), ([5, 5, 3], 0), ([1, 2, 3, 4, 5], -1), ("hello", "l"), ("", "a"), ("abc", 2),
        (3, 5), (0, 0), (-2, 7), (10, 3), ([1, 3, 5], [2, 4, 6]), ([], []), ([1, 1], [1])],
    3: [(1, 2, 3), (3, 2, 1), (0, 0, 0), (-1, 5, 2), ([1, 2, 3], 0, 2), ("hello", 1, 3), (5, 5, 5)],
}


def signature(codes):
    """(function name, arity) from the first program that parses; last top-level def wins."""
    for c in codes:
        try:
            defs = [n for n in ast.parse(c).body if isinstance(n, ast.FunctionDef)]
        except SyntaxError:
            continue
        if defs:
            f = defs[-1]
            return f.name, len(f.args.args) - len(f.args.defaults)
    return None, None


def probes_for(codes, tests):
    if tests:
        p = B.make_probes(tests, cap=16)
        if p:
            return p, "derived from your tests: originals, then one argument changed at a time"
    name, arity = signature(codes)
    if not name:
        return [], "no runnable function found"
    bank = GENERIC.get(arity) or GENERIC[1]
    calls = [f"{name}({v!r})" if arity == 1 else f"{name}({', '.join(repr(x) for x in v)})" for v in bank]
    return calls, f"generic inputs for a {arity}-argument function (no tests given)"


PROBE_RUNNER = r'''
import json, signal, sys
_HAS_ALARM = hasattr(signal, "SIGALRM")
def _alarm(*_): raise TimeoutError
if _HAS_ALARM: signal.signal(signal.SIGALRM, _alarm)
def _set(n):
    if _HAS_ALARM: signal.alarm(n)
_res = []
try:
    _set(3); exec(compile({code!r}, "<model>", "exec"), globals()); _set(0); _ok = True
except BaseException:
    _ok = False
for _e in json.loads(sys.stdin.read()):
    if not _ok:
        _res.append([_e, "DoesNotRun"]); continue
    _set(1)
    try:
        _r = repr(eval(_e))
    except BaseException as _x:
        _r = "Timeout" if isinstance(_x, TimeoutError) else type(_x).__name__
    finally:
        _set(0)
    _res.append([_e, _r[:200]])
print("@@" + json.dumps(_res))
'''


def run_probes(code, probes):
    """Same contract as cache/behaviour.py run_probes, but runs on Windows too. On Windows there is
    no per-probe alarm, so one hanging probe costs the whole program (whole-run timeout instead)."""
    if not probes:
        return []
    try:
        p = subprocess.run([sys.executable, "-c", PROBE_RUNNER.format(code=code)], input=json.dumps(probes),
                           capture_output=True, text=True, timeout=5 + len(probes), cwd=tempfile.gettempdir(),
                           env={**os.environ, "PYTHONHASHSEED": "0"}, **SANDBOX_KW)
    except subprocess.TimeoutExpired:
        return [f"{e} -> Timeout" for e in probes]
    line = next((l for l in p.stdout.splitlines() if l.startswith("@@")), None)
    return [] if line is None else [f"{e} -> {r}" for e, r in json.loads(line[2:])]


def trace(code, probes):
    a, b = run_probes(code, probes), run_probes(code, probes)
    return [x for x, y in zip(a, b) if x == y]


def rename(code, name):
    """Point every program's main function at the shared probe name so calls line up."""
    n2, _ = signature([code])
    if not n2 or n2 == name:
        return code
    return code + f"\n{name} = {n2}\n"


@app.post("/api/execute")
def execute(req: dict):
    setup = (req.get("setup") or "").strip()
    codes = {k: (setup + "\n" + (v or "")) if setup else (v or "") for k, v in req.get("codes", {}).items()}
    tests = [t for t in req.get("tests", []) if t.strip()]
    reference = req.get("reference")
    z_pred = {k: torch.tensor(v) for k, v in (req.get("imagined") or {}).items()}

    calls, how = probes_for(list(codes.values()), tests)
    pname = calls[0].split("(")[0] if calls else None
    programs = dict(codes)
    if reference:
        programs = {"reference": reference, **programs}

    traces = {}
    for k, c in programs.items():
        lines = trace(rename(c, pname), calls) if calls else []
        traces[k] = {l.partition(" -> ")[0]: l.partition(" -> ")[2] for l in lines}

    # keep probes the programs actually accept (drop rows where every program raised TypeError)
    rows = []
    for call in calls:
        outs = {k: traces[k].get(call, "—") for k in programs}
        model_outs = [v for k, v in outs.items() if k != "reference"]
        if model_outs and all(v.startswith("TypeError") or v == "—" for v in model_outs):
            continue
        vals = [v for v in model_outs if v != "—"]
        majority = max(set(vals), key=vals.count) if vals else None
        rows.append({"call": call, "outs": outs, "truth": outs.get("reference"), "majority": majority,
                     "split": len(set(vals)) > 1})

    results = {}
    for k, c in codes.items():
        lines = [f"{r['call']} -> {r['outs'][k]}" for r in rows if r["outs"].get(k, "—") != "—"]
        z = B.encode_features(lines, anonymize_names=CACHE_FILE.endswith("_anon.pt")) if lines else None
        item = {"tests": run_tests(c, tests),
                "xy": place(z) if z is not None else None,
                "agree_ref": (sum(1 for r in rows if r["truth"] is not None and r["outs"][k] == r["truth"]), sum(1 for r in rows if r["truth"] is not None))}
        if z is not None:
            item["nearest"] = nearest(z, 1)
            if k in z_pred:
                item["imagined_vs_actual"] = round(float(F.cosine_similarity(z, z_pred[k], dim=0)), 3)
        results[k] = item
    print("[execute] " + ", ".join(f"{k}: {sum(t['ok'] for t in v['tests'])}/{len(v['tests'])} asserts" for k, v in results.items())
          + f", {len(rows)} probe inputs", flush=True)
    return JSONResponse({"probes": rows, "probe_note": how, "results": results, "n_programs": len(codes),
                         "split_count": sum(1 for r in rows if r["split"]), "has_reference": bool(reference)})


@app.get("/health")
def health():
    return {"ok": True, "mock": MOCK, "runs": list(RUNS), "heads": list(HEADS), "gpu_mode": GPU_MODE}
