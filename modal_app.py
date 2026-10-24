"""PreCode-JEPA on Modal: tune every method on validation data, train each with 3 seeds, evaluate, compute behaviour
metrics, build the comparison. All jobs run in parallel on Modal GPUs.

One-time setup (laptop, repo root):
    pip install modal && python -m modal setup
    python -m modal secret create huggingface HF_TOKEN=hf_xxx

Everything for the default model (Qwen2.5-Coder-1.5B base):
    python -m modal run --detach modal_app.py
Stages (each can be run on its own; finished work is skipped):
    --stage sweep     lr x epochs grid per method, trained on mbpp_train_noval (384), scored on val+valext (365)
    --stage select    best config per method from the sweep -> results/best_config.json
    --stage train     every method x seed on the full 474 tasks with its selected config
    --stage eval      MBPP+ (224) and HumanEval+ (164) pass@1 for every run
    --stage analyze   behaviour metrics on MBPP+ test tasks + t-SNE figure
    --stage compare   COMPARISON.md        --stage download   results -> ./results_modal_<model>/
    --stage debug     print what base + the given methods actually generate on 12 validation tasks
Other flags: --model qc15|llama1b, --methods a,b, --seeds 42,7,1234, --force (redo everything), --force-methods a,b
(redo the sweep for these), --extra "<trainer flags>". Train/eval redo automatically when the selected config changes.

Methods:
    base                zero-shot (eval only)
    sft, jepa, precode, precodence      train/finetune.py via train/run_all.sh (Gagan's trainer)
    sft_loop            train/precode_full.py with lambda 0: SFT through the PreCode-full loop (control)
    precodefull_aux, precodefull_prefix, precodefull   train/precode_full.py --mode aux | prefix | full
Storage: Modal volume "precode-data", mounted at /precode-vol and linked as /data/precode.
    llama1b keeps the original paths /data/precode/{models,samples,results}; other models use /data/precode/<model>/...
"""
import json, os, shutil, subprocess, sys, time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/root/repo"
VOL = "/precode-vol"      # volume mount point; nothing in the image may touch this path before mount
GPU = os.environ.get("PRECODE_GPU", "L40S")

# model tag -> (HF id, finetune.py --last_token for LLM-JEPA: the end-of-turn token of the chat format)
MODELS = {"qc15": ("Qwen/Qwen2.5-Coder-1.5B", -2),               # base model; ChatML ends "<|im_end|>\n" -> -2
          "llama1b": ("unsloth/Llama-3.2-1B-Instruct", -1)}      # original runs; ends "<|eot_id|>" -> -1
FINETUNE_METHODS = ["sft", "jepa", "precode", "precodence"]
FULL_METHODS = {"sft_loop": "aux --lbd 0 --head_warmup_epochs 0",
                "precodefull_aux": "aux", "precodefull_prefix": "prefix", "precodefull": "full"}
ALL_METHODS = ["base"] + FINETUNE_METHODS + list(FULL_METHODS)
SWEEP_LR = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4]     # widened: several methods picked an edge of the old 3x2 grid
SWEEP_EPOCHS = [1, 2, 3]
USES_CACHE = {"precode", "precodence", "precodefull_aux", "precodefull_prefix", "precodefull"}
DEFAULT_CFG = {"lr": 5e-5, "epochs": 2}
CACHE = {"qc15": "cache/behaviour_cache_anon.pt",        # name-free target (function renamed to f before encoding)
         "llama1b": "cache/behaviour_cache.pt"}          # original runs used the name-bearing cache

app = modal.App("precode-jepa")
vol = modal.Volume.from_name("precode-data", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.6.0", "transformers==4.55.0", "peft==0.17.0", "accelerate>=1.0", "datasets>=3.0",
                 "evalplus>=0.3.1", "scipy", "pandas", "tabulate", "huggingface_hub>=0.30", "scikit-learn",
                 "numpy<2.3", "matplotlib")
    .env({"TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(HERE, REPO, ignore=[".git", "**/__pycache__", "results_modal*", "**/*.pyc", "demo/studio/map_cache.json"])
)
SECRETS = [modal.Secret.from_name("huggingface")]


def link_data():
    """Point caches at the volume and link /data/precode -> <VOL>/precode (done at runtime: anything under the mount
    path at image-build time makes Modal refuse the mount)."""
    os.environ["HF_HOME"] = f"{VOL}/hf"
    os.environ["XDG_CACHE_HOME"] = f"{VOL}/cache"
    os.makedirs(f"{VOL}/precode", exist_ok=True)
    os.makedirs("/data", exist_ok=True)
    if not os.path.islink("/data/precode"):
        if os.path.isdir("/data/precode") and not os.listdir("/data/precode"):
            os.rmdir("/data/precode")
        os.symlink(f"{VOL}/precode", "/data/precode")


def paths(tag):
    base = "/data/precode" if tag == "llama1b" else f"/data/precode/{tag}"
    return {k: f"{base}/{k}" for k in ("models", "samples", "results", "sweep_models")}


def sh(cmd, env=None, cwd=REPO):
    print("$", cmd, flush=True)
    p = subprocess.run(cmd, shell=True, cwd=cwd, env={**os.environ, **(env or {})})
    if p.returncode:
        raise RuntimeError(f"command failed ({p.returncode}): {cmd}")


def run_name(method, seed):
    return "base_s0" if method == "base" else f"{method}_s{seed}"


def _train(tag, method, seed, lr, epochs, out_root, run, train_file, extra):
    """Train one run into out_root/run with either trainer. Returns the output dir."""
    hf_id, last_token = MODELS[tag]
    out = f"{out_root}/{run}"
    os.makedirs(out_root, exist_ok=True)
    if method in FINETUNE_METHODS:
        sh("tr -d '\\r' < train/run_all.sh > /tmp/run_all.sh")      # Windows checkouts upload CRLF; bash dies on it
        ep = str(int(epochs)) if float(epochs).is_integer() else str(epochs)   # finetune.py --num_epochs is an int
        sh(f"bash /tmp/run_all.sh {method} {seed}",
           env={"EPOCHS": ep, "LR": str(lr), "MODEL": hf_id, "MODELS_ROOT": out_root, "NAME": run, "TRAIN": train_file,
                "CACHE": CACHE[tag],
                "EXTRA": f"--last_token {last_token} {extra}".strip()})
    elif method in FULL_METHODS:
        sh(f"python train/precode_full.py --mode {FULL_METHODS[method]} --seed {seed} --epochs {epochs} --lr {lr} "
           f"--model {hf_id} --train_file {train_file} --cache {CACHE[tag]} --out {out} {extra} 2>&1 | tee {out_root}/{run}.log",
           env={"HF_HUB_OFFLINE": "1"})
    else:
        raise ValueError(f"unknown method {method}")
    if not os.path.exists(f"{out}/DONE"):
        raise RuntimeError(f"{run}: training finished without writing DONE; see {out_root}/{run}.log")
    return out


@app.function(image=image, volumes={VOL: vol}, secrets=SECRETS, timeout=1800)
def prefetch(tag: str):
    link_data()
    from huggingface_hub import snapshot_download
    snapshot_download(MODELS[tag][0])
    sh("python -c \"from evalplus.data import get_mbpp_plus, get_human_eval_plus; get_mbpp_plus(); get_human_eval_plus()\"")
    for p in paths(tag).values():
        os.makedirs(p, exist_ok=True)
    vol.commit()
    return f"prefetched {MODELS[tag][0]}"


@app.function(image=image, gpu=GPU, volumes={VOL: vol}, secrets=SECRETS, timeout=6 * 3600)
def sweep_job(tag: str, method: str, lr: float, epochs: int, extra: str = "", force: bool = False):
    """Train on the 384 non-validation tasks (seed 42), score on the 365 validation tasks, delete the weights."""
    link_data(); vol.reload()
    P = paths(tag)
    name = f"{method}_lr{lr:g}_e{epochs}"
    res_path = f"{P['results']}/sweep/{name}.json"
    cache_used = CACHE[tag] if method in USES_CACHE else "none"
    if os.path.exists(res_path) and not force:
        old = json.load(open(res_path))
        if old.get("cache", CACHE["llama1b"] if method in USES_CACHE else "none") == cache_used:
            return old
    out = _train(tag, method, 42, lr, epochs, P["sweep_models"], name, "datasets/mbpp_train_noval.jsonl", extra)
    sh(f"python eval/val_hf.py --model {out} --out /tmp/val.json", env={"HF_HUB_OFFLINE": "1"})
    res = {**json.load(open("/tmp/val.json")), "method": method, "lr": lr, "epochs": epochs, "cache": cache_used}
    res.pop("samples", None)
    res.pop("per_task", None)
    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    json.dump(res, open(res_path, "w"))
    shutil.rmtree(out, ignore_errors=True)                       # ~3 GB each; only the score is needed
    vol.commit()
    return res


@app.function(image=image, gpu=GPU, volumes={VOL: vol}, secrets=SECRETS, timeout=3 * 3600)
def debug_job(tag: str, kind: str, lr: float = 2e-5, epochs: int = 1, n: int = 12):
    """Show what a model actually writes on the first n validation tasks: raw completion, extracted code, error.
    kind = base (zero-shot) or a method name (trained on the 384 non-val tasks into /tmp, nothing saved)."""
    link_data(); vol.reload()
    rows = open(f"{REPO}/datasets/mbpp_val.jsonl").read().splitlines()[:n]
    open("/tmp/val_n.jsonl", "w").write("\n".join(rows) + "\n")
    model = MODELS[tag][0]
    if kind != "base":
        model = _train(tag, kind, 42, lr, epochs, "/tmp/dbg_models", f"dbg_{kind}", "datasets/mbpp_train_noval.jsonl", "")
    sh(f"python eval/val_hf.py --model {model} --out /tmp/dbg.json --sets /tmp/val_n.jsonl", env={"HF_HUB_OFFLINE": "1"})
    res = json.load(open("/tmp/dbg.json"))
    lines = [f"===== {kind} (lr {lr:g}, ep {epochs}) : {res['passed']}/{res['n']} passed ====="]
    for s in res["samples"][:6]:
        lines += [f"--- task {s['task']}  ok={s['ok']}  error={s['error']!r}", f"PROMPT: {s['prompt'][:160]!r}",
                  f"RAW:    {s['raw'][:600]!r}", f"CODE:   {s['code'][:400]!r}"]
    return "\n".join(lines)


@app.function(image=image, volumes={VOL: vol}, timeout=600)
def select(tag: str):
    """Pick the best (lr, epochs) per method by validation pass@1; ties -> fewer epochs, then lower lr."""
    link_data(); vol.reload()
    P = paths(tag)
    best, table = {}, []
    d = f"{P['results']}/sweep"
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        r = json.load(open(f"{d}/{f}"))
        if r["lr"] not in SWEEP_LR or r["epochs"] not in SWEEP_EPOCHS:
            continue
        if r.get("cache", CACHE["llama1b"] if r["method"] in USES_CACHE else "none") != (CACHE[tag] if r["method"] in USES_CACHE else "none"):
            continue                                               # made with an outdated behaviour cache
        table.append(r)
        key = (r["pass1"], -r["epochs"], -r["lr"])
        if r["method"] not in best or key > best[r["method"]][0]:
            best[r["method"]] = (key, {"lr": r["lr"], "epochs": r["epochs"], "val_pass1": r["pass1"]})
    cfg = {m: v[1] for m, v in best.items()}
    json.dump(cfg, open(f"{P['results']}/best_config.json", "w"), indent=1)
    lines = ["| method | lr | epochs | val pass@1 (365) |", "|---|---|---|---|"]
    for r in sorted(table, key=lambda r: (r["method"], r["lr"], r["epochs"])):
        star = " **(selected)**" if cfg.get(r["method"], {}).get("lr") == r["lr"] and cfg[r["method"]]["epochs"] == r["epochs"] else ""
        lines.append(f"| {r['method']} | {r['lr']:g} | {r['epochs']} | {100 * r['pass1']:.1f}{star} |")
    open(f"{P['results']}/SWEEP.md", "w").write("# Validation sweep (seed 42, train 384 / val 365)\n\n" + "\n".join(lines) + "\n")
    vol.commit()
    return cfg, "\n".join(lines)


@app.function(image=image, gpu=GPU, volumes={VOL: vol}, secrets=SECRETS, timeout=6 * 3600)
def train(tag: str, method: str, seed: int, force: bool = False, extra: str = ""):
    link_data(); vol.reload()
    P = paths(tag)
    run = run_name(method, seed)
    bc = f"{P['results']}/best_config.json"
    cfg = (json.load(open(bc)).get(method) if os.path.exists(bc) else None) or DEFAULT_CFG
    want = {"method": method, "seed": seed, "lr": cfg["lr"], "epochs": cfg["epochs"],
            "cache": CACHE[tag] if method in USES_CACHE else "none", "extra": extra}
    tc = f"{P['models']}/{run}/train_config.json"
    have = json.load(open(tc)) if os.path.exists(tc) else None
    if os.path.exists(f"{P['models']}/{run}/DONE") and not force and have == want:
        return f"{run}: already trained with the selected config, skipped"
    t0 = time.time()
    _train(tag, method, seed, cfg["lr"], cfg["epochs"], P["models"], run, "datasets/mbpp_train.jsonl", extra)
    json.dump(want, open(tc, "w"))
    vol.commit()
    return f"{run}: trained (lr {cfg['lr']:g}, {cfg['epochs']} ep) in {int(time.time() - t0)} s"


def _score(eval_json, run, dataset, ids, results, train_cfg=None):
    """Same rule as eval/score.py (pass = base AND plus), one file per run so parallel evals never race."""
    res = json.load(open(eval_json))["eval"]
    per, nb, npl = {}, 0, 0
    for t in ids:
        r = res[t][0]
        b = r["base_status"] == "pass"; p = b and r["plus_status"] == "pass"
        nb += b; npl += p; per[t] = bool(p)
    os.makedirs(f"{results}/per_task", exist_ok=True); os.makedirs(f"{results}/rows", exist_ok=True)
    json.dump(per, open(f"{results}/per_task/{run}_{dataset}.json", "w"))
    method, seed = run.rsplit("_s", 1)
    row = {"run": run, "method": method, "seed": int(seed), "dataset": dataset, "pass1_base": round(nb / len(ids), 4),
           "pass1_plus": round(npl / len(ids), 4), "n_tasks": len(ids), "timestamp": int(time.time()), "train_cfg": train_cfg}
    json.dump(row, open(f"{results}/rows/{run}_{dataset}.json", "w"))
    return row


@app.function(image=image, gpu=GPU, volumes={VOL: vol}, secrets=SECRETS, timeout=3 * 3600)
def evaluate(tag: str, run: str, force: bool = False):
    link_data(); vol.reload()
    P = paths(tag)
    model = MODELS[tag][0] if run == "base_s0" else f"{P['models']}/{run}"
    if run != "base_s0" and not os.path.exists(f"{model}/DONE"):
        return {"run": run, "error": "not trained (no DONE)"}
    tc = f"{model}/train_config.json"
    trained_with = json.load(open(tc)) if run != "base_s0" and os.path.exists(tc) else {"base": MODELS[tag][0]}
    rp = [f"{P['results']}/rows/{run}_{d}.json" for d in ("mbppplus224", "humanevalplus")]
    if not force and all(os.path.exists(p) for p in rp):
        old = [json.load(open(p)) for p in rp]
        if all(r.get("train_cfg") == trained_with for r in old):
            return {"run": run, "rows": old}
    out = f"{P['samples']}/{run}"
    os.makedirs(out, exist_ok=True)
    rows = []
    for ds, name in (("mbpp", "mbppplus224"), ("humaneval", "humanevalplus")):
        samples = f"{out}/{ds}.jsonl"
        for f in (f"{out}/{ds}_eval_results.json", f"{out}/{ds}.eval_results.json"):   # EvalPlus silently reuses old results
            if os.path.exists(f):
                os.remove(f)
        sh(f"python eval/generate_hf.py --model {model} --dataset {ds} --out {samples} --zout {out}/{ds}_z.json",
           env={"HF_HUB_OFFLINE": "1"})
        sh(f"evalplus.evaluate --dataset {ds} --samples {samples}")
        res = next(f for f in (f"{out}/{ds}_eval_results.json", f"{out}/{ds}.eval_results.json") if os.path.exists(f))
        ids = json.load(open(f"{REPO}/eval/mbppplus_test_ids.json")) if ds == "mbpp" else list(json.load(open(res))["eval"])
        rows.append(_score(res, run, name, ids, P["results"], trained_with))
    vol.commit()
    return {"run": run, "rows": rows}


@app.function(image=image, volumes={VOL: vol}, cpu=8, timeout=3600)
def analyze_build(tag: str):
    link_data(); vol.reload()
    P = paths(tag)
    if not os.path.exists(f"{P['results']}/test_behaviour_anon.json"):   # name-free test vectors (old name-bearing file is ignored)
        sh(f"python eval/analyze.py build-test --results {P['results']}")
        vol.commit()
    return "test behaviour ready"


@app.function(image=image, volumes={VOL: vol}, cpu=8, timeout=3600)
def analyze_run(tag: str, run: str):
    link_data(); vol.reload()
    P = paths(tag)
    if not os.path.exists(f"{P['samples']}/{run}/mbpp.jsonl"):
        return f"{run}: no samples"
    sh(f"python eval/analyze.py run --run {run} --samples {P['samples']} --results {P['results']}")
    if os.path.exists(f"{P['samples']}/{run}/mbpp_z.json") and run.endswith("_s42") and not run.startswith("sft_loop"):
        sh(f"python eval/analyze.py figure --run {run} --samples {P['samples']} --results {P['results']}")
    vol.commit()
    return f"{run}: analysed"


@app.function(image=image, volumes={VOL: vol}, timeout=900)
def compare(tag: str):
    link_data(); vol.reload()
    P = paths(tag)
    sh(f"python eval/compare.py --results {P['results']} --models {P['models']}")
    vol.commit()
    sweep = f"{P['results']}/SWEEP.md"
    return open(f"{P['results']}/COMPARISON.md").read() + ("\n" + open(sweep).read() if os.path.exists(sweep) else "")


@app.function(image=image, volumes={VOL: vol}, timeout=600)
def fetch_results(tag: str):
    """Every small results file (text and the t-SNE PNGs) so the laptop can save them; models stay on the volume."""
    import base64
    link_data(); vol.reload()
    P = paths(tag)
    files = {}
    for d, _, fs in os.walk(P["results"]):
        for f in fs:
            p = os.path.join(d, f)
            data = open(p, "rb").read()
            files[os.path.relpath(p, P["results"])] = ("b64", base64.b64encode(data).decode()) if f.endswith(".png") else ("txt", data.decode())
    for d in sorted(os.listdir(P["models"])) if os.path.isdir(P["models"]) else []:
        p = f"{P['models']}/{d}/train_log.jsonl"
        if os.path.exists(p):
            files[f"train_logs/{d}.jsonl"] = ("txt", open(p).read())
    return files


@app.function(image=image, volumes={VOL: vol}, secrets=SECRETS, timeout=3 * 3600)
def upload(tag: str, run: str, repo: str):
    link_data(); vol.reload()
    sh(f"python train/save_to_hub.py {run} --repo {repo} --root {paths(tag)['models']}")
    return f"{run} -> {repo}"


@app.local_entrypoint()
def main(stage: str = "all", model: str = "qc15", methods: str = ",".join(ALL_METHODS), seeds: str = "42,7,1234",
         force: bool = False, force_methods: str = "", extra: str = "", upload_repo: str = ""):
    import base64
    if model not in MODELS:
        sys.exit(f"unknown model {model}; choose from {list(MODELS)}")
    ms = [m.strip() for m in methods.split(",") if m.strip()]
    bad = [m for m in ms if m not in ALL_METHODS]
    if bad:
        sys.exit(f"unknown methods {bad}; choose from {ALL_METHODS}")
    ss = [int(s) for s in seeds.split(",") if s.strip()]
    trainable = [m for m in ms if m != "base"]
    runs = (["base_s0"] if "base" in ms else []) + [run_name(m, s) for m in trainable for s in ss]
    print(f"model {model} = {MODELS[model][0]} | GPU {GPU}")
    show = lambda r: print(r if not isinstance(r, Exception) else f"FAILED: {r}")

    if stage in ("all", "sweep", "train", "eval", "debug"):
        print(prefetch.remote(model))
    if stage == "debug":
        kinds = ["base"] + trainable
        for out in debug_job.starmap([(model, k) for k in kinds], return_exceptions=True):
            print(out if not isinstance(out, Exception) else f"FAILED: {out}")
        return
    if stage in ("all", "sweep"):
        fm = {x.strip() for x in force_methods.split(",") if x.strip()}
        jobs = [(model, m, lr, ep, extra, force or m in fm) for m in trainable for lr in SWEEP_LR for ep in SWEEP_EPOCHS]
        print(f"sweep: {len(jobs)} jobs ({len(trainable)} methods x {len(SWEEP_LR)} lr x {len(SWEEP_EPOCHS)} epochs)")
        for r in sweep_job.starmap(jobs, return_exceptions=True):
            show(r if isinstance(r, Exception) else f"{r['method']:20s} lr {r['lr']:g} ep {r['epochs']}  val {r['pass1']:.3f}")
    if stage in ("all", "sweep", "select"):
        cfg, table = select.remote(model)
        print(table); print("selected:", json.dumps(cfg))
    if stage in ("all", "train"):
        print(f"training {len(trainable) * len(ss)} runs")
        for r in train.starmap([(model, m, s, force, extra) for m in trainable for s in ss], return_exceptions=True):
            show(r)
    if stage in ("all", "eval"):
        print(f"evaluating {len(runs)} runs")
        for r in evaluate.starmap([(model, r, force) for r in runs], return_exceptions=True):
            if isinstance(r, Exception):
                show(r); continue
            for row in r.get("rows", []):
                print(f"{row['run']:28s} {row['dataset']:14s} base {row['pass1_base']:.3f}  plus {row['pass1_plus']:.3f}")
            if "error" in r:
                print(f"{r['run']}: {r['error']}")
    if stage in ("all", "analyze"):
        print(analyze_build.remote(model))
        for r in analyze_run.starmap([(model, r) for r in runs], return_exceptions=True):
            show(r)
    if stage in ("all", "eval", "analyze", "compare"):
        print(compare.remote(model))
    if stage in ("all", "analyze", "compare", "download"):
        files = fetch_results.remote(model)
        dest = os.path.join(HERE, f"results_modal_{model}")
        for rel, (kind, data) in files.items():
            p = os.path.join(dest, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            if kind == "b64":
                open(p, "wb").write(base64.b64decode(data))
            else:
                open(p, "w", encoding="utf-8").write(data)
        print(f"saved {len(files)} files to {dest} (see COMPARISON.md and SWEEP.md there)")
    if upload_repo:
        for r in upload.starmap([(model, run, upload_repo) for run in runs if run != "base_s0"], return_exceptions=True):
            show(r)
