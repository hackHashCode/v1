# usage: HF_HUB_OFFLINE=1 /venv/evalv/bin/python eval/val_mbpp.py <model_dir>      [VAL=datasets/mbpp_val.jsonl,datasets/mbpp_valext.jsonl]
# Greedy pass@1 on held-out MBPP (all asserts). Appends "<run>  <set> k/n" to results/val_sweep.txt, per-task to results/val_per_task/.
import json, os, subprocess, sys, tempfile
from vllm import LLM, SamplingParams
def extract(t):
    if "```" in t:
        t = t.split("```")[1]; t = t[len("python"):] if t.startswith("python") else t
    return t.strip()
run = os.path.basename(sys.argv[1].rstrip('/'))
llm = LLM(sys.argv[1], gpu_memory_utilization=0.28, max_model_len=1024); tok = llm.get_tokenizer()
os.makedirs("results/val_per_task", exist_ok=True)
for vf in os.environ.get("VAL", "datasets/mbpp_val.jsonl,datasets/mbpp_valext.jsonl").split(","):
    name = os.path.basename(vf).removeprefix("mbpp_").removesuffix(".jsonl")
    rows = [json.loads(l) for l in open(vf)]
    texts = [tok.apply_chat_template(r["messages"][:2], tokenize=False, add_generation_prompt=True) for r in rows]
    outs = llm.generate(texts, SamplingParams(temperature=0.0, max_tokens=384))
    per = {}
    for r, o in zip(rows, outs):
        src = (r.get("setup") or "") + "\n" + extract(o.outputs[0].text) + "\n" + "\n".join(r["tests"])
        try: per[r["task_id"]] = subprocess.run([sys.executable, "-c", src], capture_output=True, timeout=10, cwd=tempfile.gettempdir()).returncode == 0
        except subprocess.TimeoutExpired: per[r["task_id"]] = False
    ok = sum(per.values())
    line = f"{run}  {name} {ok}/{len(rows)} = {ok/len(rows):.3f}"
    print(line); open("results/val_sweep.txt", "a").write(line + "\n")
    json.dump(per, open(f"results/val_per_task/{run}__{name}.json", "w"))
