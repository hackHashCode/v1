# usage: python eval/generate.py --model DIR --dataset {mbpp,humaneval} --out FILE [--n 1] [--temperature 0.0] [--prompts FILE]
# [WS, env evalv] -- contracts C6, C7, C13. The prompt must match training exactly.
import argparse, json
from vllm import LLM, SamplingParams
from evalplus.data import get_mbpp_plus, get_human_eval_plus

SYSTEM = "You are an expert Python programmer. Write a correct Python function for the task. Output only code."

def user_msg(ds, p):
    if ds == "mbpp":
        return p["prompt"].strip().strip('"').strip()        # description + first assert
    return "Complete this Python function. Return the whole function.\n\n" + p["prompt"]

def extract(t):
    if "```" in t:
        t = t.split("```")[1]
        if t.startswith("python"):
            t = t[len("python"):]
    return t.strip()

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--dataset", choices=["mbpp", "humaneval"], required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=1)
ap.add_argument("--temperature", type=float, default=0.0)
ap.add_argument("--prompts")                                   # optional {"task_id", "prompt"} overrides
a = ap.parse_args()

probs = get_mbpp_plus() if a.dataset == "mbpp" else get_human_eval_plus()
over = {}
if a.prompts:
    for l in open(a.prompts):
        r = json.loads(l); over[r["task_id"]] = r["prompt"]
    assert all(t in probs for t in over), "prompt override for an unknown task_id"

llm = LLM(a.model, gpu_memory_utilization=0.28, max_model_len=1024)
tok = llm.get_tokenizer()
ids = list(probs)                                              # all tasks: EvalPlus requires them
texts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                  {"role": "user", "content": over.get(i, user_msg(a.dataset, probs[i]))}],
                                 tokenize=False, add_generation_prompt=True) for i in ids]
outs = llm.generate(texts, SamplingParams(n=a.n, temperature=a.temperature, max_tokens=384, seed=0))
with open(a.out, "w") as f:
    for i, o in zip(ids, outs):
        for c in o.outputs:
            f.write(json.dumps({"task_id": i, "solution": extract(c.text)}) + "\n")
print("wrote", a.out)
