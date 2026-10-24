"""Back up a trained model folder to a PRIVATE Hugging Face model repo (one subfolder per run).
python train/save_to_hub.py <run> [--repo <hf_user>/precode-jepa-models]   |   python train/save_to_hub.py --all"""
import argparse, glob, os, sys
from huggingface_hub import HfApi
ap = argparse.ArgumentParser()
ap.add_argument("run", nargs="?")
ap.add_argument("--repo", default=os.environ.get("HF_MODEL_REPO"))
ap.add_argument("--all", action="store_true")
ap.add_argument("--root", default="/data/precode/models")
a = ap.parse_args()
if not a.repo:
    sys.exit("set --repo or HF_MODEL_REPO")
api = HfApi()
api.create_repo(a.repo, repo_type="model", private=True, exist_ok=True)
runs = [os.path.basename(d.rstrip("/")) for d in glob.glob(f"{a.root}/*/") if os.path.exists(f"{d}/DONE")] if a.all else [a.run]
for run in runs:
    src = f"{a.root}/{run}"
    if not os.path.exists(f"{src}/DONE"):
        print(f"[hub] skip {run}: no DONE file"); continue
    api.upload_folder(repo_id=a.repo, folder_path=src, path_in_repo=run,
                      ignore_patterns=["checkpoint-*", "checkpoint-*/**", "logs/**"], commit_message=run)
    if os.path.exists(f"{a.root}/{run}.log"):
        api.upload_file(path_or_fileobj=f"{a.root}/{run}.log", path_in_repo=f"logs/{run}.log", repo_id=a.repo, commit_message=f"{run} log")
    print(f"[hub] uploaded {run} -> https://huggingface.co/{a.repo}/tree/main/{run}")
