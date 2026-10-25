"""PreCode Studio on Modal: the live demo (UI + API) on one GPU, with the trained models baked into the image.

Deploy (from the repo root, with a Modal account set up: `pip install modal && python -m modal setup`):
    python -m modal deploy demo/studio/modal_studio.py
Modal prints a URL like https://<workspace>--precode-studio-web.modal.run. Open it: that is the demo.

Before going on stage, keep one container warm so the first click doesn't wait for a cold start (~1 min):
    STUDIO_KEEP_WARM=1 python -m modal deploy demo/studio/modal_studio.py
and redeploy without it afterwards (a warm L40S bills while idle).

Models come from the public Hugging Face repo below (seed-42 runs), downloaded once at image build time.
Env at deploy time: STUDIO_GPU (default L40S), STUDIO_RUNS (default sft,jepa,precodefull_aux), STUDIO_HF_REPO,
STUDIO_KEEP_WARM (0/1). Add `base` to STUDIO_RUNS to show the zero-shot column, `sft_loop` for the SFT control.
"""
import os

import modal

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = "/root/repo"
GPU = os.environ.get("STUDIO_GPU", "L40S")
RUNS = os.environ.get("STUDIO_RUNS", "sft,jepa,precodefull_aux")
HF_REPO = os.environ.get("STUDIO_HF_REPO", "MoistPotato/precode-jepa-qwen2.5-coder-1.5b")
BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B"
SEED = "42"
KEEP_WARM = int(os.environ.get("STUDIO_KEEP_WARM", "0"))


def download_models(runs: str, repo: str):
    """Runs once while building the image: fetch every run folder (and the base model if shown).
    Arguments are passed explicitly: the build container doesn't see the deploying machine's env vars."""
    from huggingface_hub import snapshot_download
    runs = [r.strip() for r in runs.split(",") if r.strip()]
    snapshot_download(repo, allow_patterns=[f"{r}_s{SEED}/*" for r in runs if r != "base"], local_dir="/models")
    if "base" in runs:
        snapshot_download(BASE_MODEL)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.6.0", "transformers==4.55.0", "accelerate>=1.0", "huggingface_hub>=0.30",
                 "fastapi[standard]", "scikit-learn", "numpy<2.3")
    .env({"HF_HOME": "/hf", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1", "MODELS_DIR": "/models",
          "RUNS": RUNS, "HF_REPO": HF_REPO, "BASE_MODEL": BASE_MODEL, "DEMO_SEED": SEED, "GPU_MODE": "all", "PARALLEL": "1"})
    .run_function(download_models, kwargs={"runs": RUNS, "repo": HF_REPO})
    .add_local_dir(ROOT, REPO, ignore=[".git", "**/__pycache__", "**/*.pyc", "results_modal*", "fallback", "*.safetensors"])
)
app = modal.App("precode-studio", image=image)


@app.function(gpu=GPU, timeout=3600, scaledown_window=20 * 60, min_containers=KEEP_WARM)
@modal.concurrent(max_inputs=16)        # /api/meta and /api/execute must not wait behind a running generation
@modal.asgi_app()
def web():
    import sys
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    from demo.studio.server import app as studio
    return studio
