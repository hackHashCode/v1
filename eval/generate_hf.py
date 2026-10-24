# usage: python eval/generate_hf.py --model DIR --dataset {mbpp,humaneval} --out FILE [--batch 32] [--zout FILE]
# Same contract as eval/generate.py (C6, C7, C13: identical prompt, greedy, 384 tokens, first ``` fence), but with
# plain Hugging Face generate so it also handles PreCode-full runs (prefix_injector.pt): those generate with
# [BOS, prefix(pred_head(h_prompt)), rest]. Every method in the Modal pipeline is generated with this script so
# decoding is identical across methods.
import argparse, json, os, sys

import torch
from evalplus.data import get_human_eval_plus, get_mbpp_plus
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train"))
from precode_runtime import load_precode_parts, predict_z, prefixed_inputs, prompt_state  # noqa: E402

SYSTEM = "You are an expert Python programmer. Write a correct Python function for the task. Output only code."


def user_msg(ds, p):
    if ds == "mbpp":
        return p["prompt"].strip().strip('"').strip()          # description + first assert
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

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
tok = AutoTokenizer.from_pretrained(a.model)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype).cuda().eval()
head, inj, cfg = load_precode_parts(a.model, model.config.hidden_size) if os.path.isdir(a.model) else (None, None, {})
if head is not None: head = head.cuda()
if inj is not None: inj = inj.cuda()
print(f"[gen] {a.model} | pred_head={'yes' if head is not None else 'no'} | prefix={'yes' if inj is not None else 'no'}", flush=True)

ids_all = list(probs)                      # all tasks: EvalPlus requires them
texts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                  {"role": "user", "content": over.get(i, user_msg(a.dataset, probs[i]))}],
                                 tokenize=False, add_generation_prompt=True) for i in ids_all]
zs = {}
with open(a.out, "w") as f:
    for s in range(0, len(texts), a.batch):
        enc = tok(texts[s:s + a.batch], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        gen = dict(max_new_tokens=a.max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        with torch.no_grad():
            if head is not None:
                z = predict_z(head, prompt_state(model, enc.input_ids, enc.attention_mask, cfg.get("h_pool", "last")))
                for j, t in enumerate(ids_all[s:s + a.batch]):
                    zs[t] = z[j].tolist()
            if inj is not None:
                emb, mask = prefixed_inputs(model, enc.input_ids, enc.attention_mask, inj(z))
                out = model.generate(inputs_embeds=emb, attention_mask=mask, **gen)        # returns new tokens only
                new = out
            else:
                out = model.generate(**enc, **gen)
                new = out[:, enc.input_ids.shape[1]:]
        for t, o in zip(ids_all[s:s + a.batch], new):
            f.write(json.dumps({"task_id": t, "solution": extract(tok.decode(o, skip_special_tokens=True))}) + "\n")
        print(f"[gen] {min(s + a.batch, len(texts))}/{len(texts)}", flush=True)
if a.zout and zs:
    json.dump(zs, open(a.zout, "w"))
print("wrote", a.out)
