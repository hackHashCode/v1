# usage: python eval/generate_hf.py --model DIR_OR_HF_ID --dataset {mbpp,humaneval} --out FILE [--batch 32] [--zout FILE]
# Same contract as eval/generate.py (C6, C7, C13: identical prompt, greedy, 384 tokens, first ``` fence) but with plain
# Hugging Face generate, so it also handles PreCode-full prefix runs and base checkpoints (end-of-turn stop tokens,
# ChatML fallback). Every method in the Modal pipeline is generated with this script, so decoding is identical.
import argparse, json, os, sys

import torch
from evalplus.data import get_human_eval_plus, get_mbpp_plus
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train"))
from precode_runtime import ensure_chat_template, extract_code, generate_batch, load_precode_parts  # noqa: E402

SYSTEM = "You are an expert Python programmer. Write a correct Python function for the task. Output only code."


def user_msg(ds, p):
    if ds == "mbpp":
        return p["prompt"].strip().strip('"').strip()          # description + first assert
    return "Complete this Python function. Return the whole function.\n\n" + p["prompt"]


def load(model_path):
    dev = "cuda" if torch.cuda.is_available() else "cpu"      # cpu only for smoke tests
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if dev == "cuda" else torch.float32
    tok = ensure_chat_template(AutoTokenizer.from_pretrained(model_path))
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype).to(dev).eval()
    head, inj, cfg = load_precode_parts(model_path, model.config.hidden_size) if os.path.isdir(model_path) else (None, None, {})
    head = head.to(dev) if head is not None else None
    inj = inj.to(dev) if inj is not None else None
    return tok, model, head, inj, cfg


def run(tok, model, head, inj, cfg, user_msgs, batch=32, max_new=384):
    texts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": u}],
                                     tokenize=False, add_generation_prompt=True) for u in user_msgs]
    codes, zs = [], []
    for s in range(0, len(texts), batch):
        outs, z = generate_batch(model, tok, texts[s:s + batch], head, inj, cfg.get("h_pool", "last"), max_new)
        codes += [sys.modules[__name__].extract_code(o) for o in outs]   # looked up at call time (val_hf records raw text)
        if z is not None:
            zs += z.float().cpu().tolist()
        print(f"[gen] {min(s + batch, len(texts))}/{len(texts)}", flush=True)
    return codes, zs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", choices=["mbpp", "humaneval"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_new", type=int, default=384)
    ap.add_argument("--prompts", help="optional {task_id, prompt} overrides (paraphrase eval)")
    ap.add_argument("--zout", help="optional: save predicted behaviour vectors {task_id: [768]} for PreCode runs")
    a = ap.parse_args()

    probs = get_mbpp_plus() if a.dataset == "mbpp" else get_human_eval_plus()
    over = {}
    if a.prompts:
        for l in open(a.prompts):
            r = json.loads(l); over[r["task_id"]] = r["prompt"]
    tok, model, head, inj, cfg = load(a.model)
    print(f"[gen] {a.model} | pred_head={'yes' if head is not None else 'no'} | prefix={'yes' if inj is not None else 'no'}", flush=True)
    ids = list(probs)                                          # all tasks: EvalPlus requires them
    codes, zs = run(tok, model, head, inj, cfg, [over.get(i, user_msg(a.dataset, probs[i])) for i in ids], a.batch, a.max_new)
    with open(a.out, "w") as f:
        for t, c in zip(ids, codes):
            f.write(json.dumps({"task_id": t, "solution": c}) + "\n")
    if a.zout and zs:
        json.dump(dict(zip(ids, zs)), open(a.zout, "w"))
    print("wrote", a.out)
