"""PreCode-JEPA on Modal: train every method x seed in parallel, evaluate each run, build the comparison.

One-time setup (laptop, repo root):
    pip install modal && modal setup
    modal secret create huggingface HF_TOKEN=hf_xxx            # read access is enough; write access for --upload

Full pipeline (train -> eval -> compare), all methods, 3 seeds:
    modal run modal_app.py
Subsets / single steps:
    modal run modal_app.py --methods sft,precodefull --seeds 42
    modal run modal_app.py --stage train --methods precodefull_prefix --seeds 42,7
    modal run modal_app.py --stage eval  --methods sft,precodefull --seeds 42,7,1234
    modal run modal_app.py --stage compare
    modal run modal_app.py --stage download                     # COMPARISON.md + results -> ./results_modal/
Background (laptop can sleep):  modal run --detach modal_app.py

Methods:
    base                zero-shot Llama-3.2-1B-Instruct (eval only)
    sft, jepa, precode, precodence          train/finetune.py via train/run_all.sh (Gagan's recipe, unchanged)
    precodefull_aux, precodefull_prefix, precodefull   train/precode_full.py --mode aux | prefix | full
Everything lands on the Modal volume "precode-data" (mounted at /vol, linked as /data/precode):
    /data/precode/models/<run>/   (contract C4)   /data/precode/samples/<run>/   /data/precode/results/
"""
import json, os, subprocess, sys, time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/root/repo"
GPU = os.environ.get("PRECODE_GPU", "L40S")          # L40S (48 GB, bf16) is plenty for a 1B model; A100-40GB also fine
BASE_MODEL = "unsloth/Llama-3.2-1B-Instruct"
FINETUNE_METHODS = ["sft", "jepa", "precode", "precodence"]
FULL_METHODS = {"precodefull_aux": "aux", "precodefull_prefix": "prefix", "precodefull": "full"}
ALL_METHODS = ["base"] + FINETUNE_METHODS + list(FULL_METHODS)

app = modal.App("precode-jepa")
vol = modal.Volume.from_name("precode-data", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.6.0", "transformers==4.55.0", "peft==0.17.0", "accelerate>=1.0", "datasets>=3.0",
                 "evalplus>=0.3.1", "scipy", "pandas", "tabulate", "huggingface_hub>=0.30", "scikit-learn", "numpy<2.3")
    .env({"HF_HOME": "/vol/hf", "XDG_CACHE_HOME": "/vol/cache", "TOKENIZERS_PARALLELISM": "false",
          "PYTHONUNBUFFERED": "1"})
    .add_local_dir(HERE, REPO, ignore=[".git", "**/__pycache__", "results_modal", "**/*.pyc", "demo/studio/map_cache.json"])
)
SECRETS = [modal.Secret.from_name("huggingface")]


def link_data():
    """The volume is mounted at /vol (Modal refuses /data: it already exists in the base image). Link
    /data/precode -> /vol/precode so every path in CONTRACTS.md and train/run_all.sh keeps working."""
    os.makedirs("/vol/precode", exist_ok=True)
    os.makedirs("/data", exist_ok=True)
    if not os.path.islink("/data/precode"):
        if os.path.isdir("/data/precode") and not os.listdir("/data/precode"):
            os.rmdir("/data/precode")
        os.symlink("/vol/precode", "/data/precode")


def sh(cmd, env=None, cwd=REPO):
    print("$", cmd, flush=True)
    p = subprocess.run(cmd, shell=True, cwd=cwd, env={**os.environ, **(env or {})})
    if p.returncode:
        raise RuntimeError(f"command failed ({p.returncode}): {cmd}")


def run_name(method, seed):
    return "base_s0" if method == "base" else f"{method}_s{seed}"


@app.function(image=image, volumes={"/vol": vol}, secrets=SECRETS, timeout=1800)
def prefetch():
    """Download the base model into the volume's HF cache once, so training can run with HF_HUB_OFFLINE=1
    exactly as train/run_all.sh expects, and EvalPlus datasets so eval doesn't race on the download."""
    link_data()
    from huggingface_hub import snapshot_download
    snapshot_download(BASE_MODEL)
    sh("python -c \"from evalplus.data import get_mbpp_plus, get_human_eval_plus; get_mbpp_plus(); get_human_eval_plus()\"")
    os.makedirs("/data/precode/models", exist_ok=True)
    vol.commit()
    return "ok"


@app.function(image=image, gpu=GPU, volumes={"/vol": vol}, secrets=SECRETS, timeout=6 * 3600)
def train(method: str, seed: int, epochs: float = 2, force: bool = False, extra: str = ""):
    link_data()
    run = run_name(method, seed)
    out = f"/data/precode/models/{run}"
    vol.reload()
    if os.path.exists(f"{out}/DONE") and not force:
        return f"{run}: already trained (DONE), skipped"
    t0 = time.time()
    if method in FINETUNE_METHODS:
        sh(f"bash train/run_all.sh {method} {seed}", env={"EPOCHS": str(epochs), "EXTRA": extra})
    elif method in FULL_METHODS:
        sh(f"python train/precode_full.py --mode {FULL_METHODS[method]} --seed {seed} --epochs {epochs} "
           f"--model {BASE_MODEL} --out {out} {extra} 2>&1 | tee /data/precode/models/{run}.log",
           env={"HF_HUB_OFFLINE": "1"})
    else:
        raise ValueError(f"unknown method {method}")
    if not os.path.exists(f"{out}/DONE"):
        raise RuntimeError(f"{run}: training finished without writing DONE; see /data/precode/models/{run}.log")
    vol.commit()
    return f"{run}: trained in {int(time.time() - t0)} s"


def _score(eval_json, run, dataset, ids):
    """Same rule as eval/score.py (pass = base AND plus), but one file per run so parallel evals never race."""
    res = json.load(open(eval_json))["eval"]
    per, nb, npl = {}, 0, 0
    for t in ids:
        r = res[t][0]
        b = r["base_status"] == "pass"; p = b and r["plus_status"] == "pass"
        nb += b; npl += p; per[t] = bool(p)
    root = "/data/precode/results"
    os.makedirs(f"{root}/per_task", exist_ok=True); os.makedirs(f"{root}/rows", exist_ok=True)
    json.dump(per, open(f"{root}/per_task/{run}_{dataset}.json", "w"))
    method, seed = run.rsplit("_s", 1)
    row = {"run": run, "method": method, "seed": int(seed), "dataset": dataset, "pass1_base": round(nb / len(ids), 4),
           "pass1_plus": round(npl / len(ids), 4), "n_tasks": len(ids), "timestamp": int(time.time())}
    json.dump(row, open(f"{root}/rows/{run}_{dataset}.json", "w"))
    return row


@app.function(image=image, gpu=GPU, volumes={"/vol": vol}, secrets=SECRETS, timeout=3 * 3600)
def evaluate(run: str):
    link_data()
    vol.reload()
    model = BASE_MODEL if run == "base_s0" else f"/data/precode/models/{run}"
    if run != "base_s0" and not os.path.exists(f"{model}/DONE"):
        return {"run": run, "error": "not trained (no DONE)"}
    out = f"/data/precode/samples/{run}"
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
        rows.append(_score(res, run, name, ids))
    vol.commit()
    return {"run": run, "rows": rows}


@app.function(image=image, volumes={"/vol": vol}, timeout=900)
def compare():
    link_data()
    vol.reload()
    sh("python eval/compare.py --results /data/precode/results --models /data/precode/models")
    vol.commit()
    return open("/data/precode/results/COMPARISON.md").read()


@app.function(image=image, volumes={"/vol": vol}, timeout=600)
def fetch_results():
    """Return every small results file so the laptop can save them (models stay on the volume)."""
    link_data()
    vol.reload()
    files = {}
    root = "/data/precode/results"
    for d, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(d, f)
            files[os.path.relpath(p, root)] = open(p).read()
    for d in sorted(os.listdir("/data/precode/models")) if os.path.isdir("/data/precode/models") else []:
        p = f"/data/precode/models/{d}/train_log.jsonl"
        if os.path.exists(p):
            files[f"train_logs/{d}.jsonl"] = open(p).read()
    return files


@app.function(image=image, volumes={"/vol": vol}, secrets=SECRETS, timeout=3 * 3600)
def upload(run: str, repo: str):
    """Back up one run to a private HF repo (needs a write token in the 'huggingface' secret)."""
    link_data()
    vol.reload()
    sh(f"python train/save_to_hub.py {run} --repo {repo}")
    return f"{run} -> {repo}"


@app.local_entrypoint()
def main(stage: str = "all", methods: str = ",".join(ALL_METHODS), seeds: str = "42,7,1234", epochs: float = 2,
         force: bool = False, extra: str = "", upload_repo: str = ""):
    ms = [m.strip() for m in methods.split(",") if m.strip()]
    bad = [m for m in ms if m not in ALL_METHODS]
    if bad:
        sys.exit(f"unknown methods {bad}; choose from {ALL_METHODS}")
    ss = [int(s) for s in seeds.split(",") if s.strip()]
    trainable = [(m, s) for m in ms if m != "base" for s in ss]
    runs = (["base_s0"] if "base" in ms else []) + [run_name(m, s) for m, s in trainable]

    if stage in ("all", "train"):
        print(prefetch.remote())
        print(f"training {len(trainable)} runs in parallel on {GPU}: {[run_name(m, s) for m, s in trainable]}")
        for r in train.starmap([(m, s, epochs, force, extra) for m, s in trainable], return_exceptions=True):
            print(r if not isinstance(r, Exception) else f"FAILED: {r}")
    if stage in ("all", "eval"):
        if stage == "eval":
            print(prefetch.remote())
        print(f"evaluating {len(runs)} runs: {runs}")
        for r in evaluate.map(runs, return_exceptions=True):
            if isinstance(r, Exception):
                print(f"FAILED: {r}"); continue
            for row in r.get("rows", []):
                print(f"{row['run']:28s} {row['dataset']:14s} base {row['pass1_base']:.3f}  plus {row['pass1_plus']:.3f}")
            if "error" in r:
                print(f"{r['run']}: {r['error']}")
    if stage in ("all", "eval", "compare"):
        print(compare.remote())
    if stage in ("all", "download", "compare"):
        files = fetch_results.remote()
        dest = os.path.join(HERE, "results_modal")
        for rel, text in files.items():
            p = os.path.join(dest, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "w").write(text)
        print(f"saved {len(files)} files to {dest} (see results_modal/COMPARISON.md)")
    if upload_repo:
        for r in upload.starmap([(run, upload_repo) for run in runs if run != "base_s0"], return_exceptions=True):
            print(r)
