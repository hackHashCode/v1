"""PreCode-full trainer: behaviour prediction + prefix conditioning + hard-task weighting, staged.

This adds the parts that train/finetune.py never had. sft / jepa / precode / precodence still train with
train/finetune.py (via train/run_all.sh) so earlier results stay comparable; this script trains the
PreCode-full family:

  --mode aux     NTP + lambda * L_beh                                  (behaviour prediction only)
  --mode prefix  + 4 soft tokens from predicted_z, prepended after BOS (the model reads its own guess)
  --mode full    + per-task NTP weights w = 1 - cos(predicted_z, z), detached, mean 1

Loss (per batch of B tasks):
  h_i      = final-layer hidden state at the last prompt token (pass 1, prompt only)
  zhat_i   = normalize(pred_head(h_i))                                 768-d
  L_beh    = InfoNCE(zhat_i vs ALL cached z, tau)   or   1 - cos(zhat_i, z_i)      (--beh_loss)
  prefix_i = prefix_injector(zhat_i)  -> [n_prefix, hidden]           (detached input by default)
  L_ntp,i  = mean CE over code tokens of pass 2 (prefix + full sequence) or pass 1 if prefix is off
  w_i      = clamp(1 - cos(zhat_i, z_i), 0, 1), detached, normalised to mean 1  (only once weighting is on)
  loss     = mean_i(w_i * L_ntp,i) + lambda * L_beh

Stages (fractions of total optimiser steps, so they work for any epoch count):
  [0, prefix_start)      NTP + L_beh                 the head learns to predict behaviour first
  [prefix_start, end]    + prefix conditioning
  [weight_start, end]    + hard-task weighting        (mode full only)

Output (contract C4 compatible): merged HF model + tokenizer, pred_head.pt (same shapes as finetune.py),
prefix_injector.pt, precode_config.json, train_log.jsonl, DONE (written last).

Usage (repo root):
  python train/precode_full.py --mode full --seed 42 --out /data/precode/models/precodefull_s42
"""
import argparse, json, math, os, random, time

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["aux", "prefix", "full"], default="full")
ap.add_argument("--out", required=True)
ap.add_argument("--train_file", default="datasets/mbpp_train.jsonl")
ap.add_argument("--cache", default="cache/behaviour_cache_anon.pt", help="name-free behaviour cache (cache/anonymize_cache.py)")
ap.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--epochs", type=float, default=2)
ap.add_argument("--batch_size", type=int, default=4)
ap.add_argument("--grad_accum", type=int, default=4)
ap.add_argument("--max_length", type=int, default=512)
ap.add_argument("--lr", type=float, default=5e-5, help="LoRA lr (same recipe as run_all.sh)")
ap.add_argument("--head_lr", type=float, default=1e-3, help="pred_head lr (finetune.py sweeps: head barely learns at 5e-5)")
ap.add_argument("--prefix_lr", type=float, default=5e-4)
ap.add_argument("--warmup", type=float, default=0.05)
ap.add_argument("--lora_rank", type=int, default=16)
ap.add_argument("--beh_loss", choices=["infonce", "cosine"], default="infonce")
ap.add_argument("--tau", type=float, default=0.07)
ap.add_argument("--lbd", type=float, default=None, help="default 0.5 for infonce, 1.0 for cosine")
ap.add_argument("--head_warmup_epochs", type=int, default=40,
                help="before LoRA training, fit pred_head alone on frozen base-model prompt states (cheap). Without it the "
                     "head is still near-random when the prefix turns on (smoke run: top-1 0.9%%, cos 0.04 after 60 steps)")
ap.add_argument("--head", choices=["linear", "mlp"], default="mlp")
ap.add_argument("--n_prefix", type=int, default=4)
ap.add_argument("--prefix_start", type=float, default=0.34, help="fraction of steps before prefix turns on")
ap.add_argument("--weight_start", type=float, default=0.67, help="fraction of steps before weighting turns on")
ap.add_argument("--no_detach_prefix", action="store_true", help="let NTP gradients flow through the prefix into pred_head")
ap.add_argument("--log_every", type=int, default=10)
a = ap.parse_args()
if a.lbd is None:
    a.lbd = 0.5 if a.beh_loss == "infonce" else 1.0
USE_PREFIX = a.mode in ("prefix", "full")
USE_WEIGHT = a.mode == "full"

random.seed(a.seed); torch.manual_seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"     # cpu only for smoke tests
amp = torch.bfloat16 if dev == "cpu" or torch.cuda.is_bf16_supported() else torch.float16
os.makedirs(a.out, exist_ok=True)
if os.path.exists(os.path.join(a.out, "DONE")):
    os.remove(os.path.join(a.out, "DONE"))

# ------------------------------------------------------------------ data (same format and masking as finetune.py)
from precode_runtime import ensure_chat_template   # noqa: E402  (same folder)
tok = ensure_chat_template(AutoTokenizer.from_pretrained(a.model))   # base checkpoints may lack a chat template
tok.padding_side = "right"
rows = [json.loads(l) for l in open(a.train_file)]
examples = []
for r in rows:
    msgs = r["messages"]
    # tokenise prompt and completion separately and concatenate: the prompt tokens are then exactly what the
    # model sees at inference (apply_chat_template(..., add_generation_prompt=True)), with no merges across the seam
    prompt_text = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    full_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    assert full_text.startswith(prompt_text), "chat template: prompt text is not a prefix of the full chat"
    prompt = tok(prompt_text, add_special_tokens=False)["input_ids"]
    full = (prompt + tok(full_text[len(prompt_text):], add_special_tokens=False)["input_ids"])[:a.max_length]
    if len(prompt) >= len(full):
        continue  # prompt alone fills max_length; nothing to learn
    labels = [-100] * len(prompt) + full[len(prompt):]
    examples.append({"ids": full, "labels": labels, "plen": len(prompt), "task_id": int(r["task_id"])})
print(f"[data] {len(examples)} examples (dropped {len(rows) - len(examples)} over-long)", flush=True)

cache = torch.load(a.cache)
Z_ids = sorted(int(k) for k in cache)
Z = F.normalize(torch.stack([cache[k].float() for k in Z_ids]), dim=-1).to(dev)   # [N, 768]
Z_pos = {t: i for i, t in enumerate(Z_ids)}
print(f"[cache] {len(Z_ids)} behaviour vectors; {sum(e['task_id'] in Z_pos for e in examples)} train tasks covered", flush=True)


def collate(batch):
    L = max(len(e["ids"]) for e in batch)
    ids = torch.full((len(batch), L), tok.pad_token_id)
    lab = torch.full((len(batch), L), -100)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, e in enumerate(batch):
        n = len(e["ids"])
        ids[i, :n] = torch.tensor(e["ids"]); lab[i, :n] = torch.tensor(e["labels"]); att[i, :n] = 1
    return ids, lab, att, torch.tensor([e["plen"] for e in batch]), [e["task_id"] for e in batch]


# ------------------------------------------------------------------ model
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)
model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=a.lora_rank, lora_alpha=2 * a.lora_rank,
                                         lora_dropout=0.1, target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                                           "gate_proj", "up_proj", "down_proj"]))
model.to(dev)
H = model.config.hidden_size
emb = model.get_input_embeddings()

if a.head == "linear":   # identical shapes to finetune.py so demo/studio and existing tools load it
    pred_head = nn.Linear(H, Z.shape[1], bias=False)
else:
    pred_head = nn.Sequential(nn.Linear(H, H // 2, bias=False), nn.GELU(), nn.Linear(H // 2, Z.shape[1], bias=False))
pred_head = pred_head.to(dev, torch.float32)


class PrefixInjector(nn.Module):
    """predicted_z (768) -> n_prefix soft tokens (hidden). Initialised at the scale of real token embeddings
    (bias = mean embedding, small weights) so the prefix starts as near-neutral tokens, not noise."""
    def __init__(self, d_in, hidden, n, emb_weight):
        super().__init__()
        self.n, self.hidden = n, hidden
        self.proj = nn.Linear(d_in, n * hidden)
        with torch.no_grad():
            std = emb_weight.float().std().item()
            self.proj.weight.normal_(0, 0.5 * std)
            self.proj.bias.copy_(emb_weight.float().mean(0).repeat(n))

    def forward(self, z):
        return self.proj(z).view(-1, self.n, self.hidden)


prefix_inj = PrefixInjector(Z.shape[1], H, a.n_prefix, emb.weight).to(dev, torch.float32) if USE_PREFIX else None



def beh_loss_fn(zh, tgt_idx):
    if a.beh_loss == "infonce":
        return F.cross_entropy(zh @ Z.T / a.tau, tgt_idx)
    return 1 - (zh * Z[tgt_idx]).sum(-1).mean()


if a.head_warmup_epochs > 0:
    # Stage 0: prompt states from the frozen base model (LoRA adapters start at zero, so this IS the base model),
    # then train pred_head alone. ~474 forward passes + a few hundred tiny head steps: about a minute.
    model.eval()
    hs, idx = [], []
    with torch.no_grad():
        for s in range(0, len(examples), 16):
            chunk = [e for e in examples[s:s + 16] if e["task_id"] in Z_pos]
            if not chunk:
                continue
            ids_c, _, att_c, plen_c, tids_c = collate(chunk)
            Lp = int(plen_c.max())
            with torch.autocast(dev, dtype=amp):
                o = model(input_ids=ids_c[:, :Lp].to(dev), attention_mask=att_c[:, :Lp].to(dev), output_hidden_states=True)
            hs.append(o.hidden_states[-1][torch.arange(len(chunk), device=dev), plen_c.to(dev) - 1].float())
            idx += [Z_pos[t] for t in tids_c]
    Hc, Ic = torch.cat(hs), torch.tensor(idx, device=dev)
    hopt = torch.optim.AdamW(pred_head.parameters(), lr=a.head_lr, weight_decay=0.0)
    pred_head.train()
    for ep in range(a.head_warmup_epochs):
        perm = torch.randperm(len(Hc), device=dev)
        for s in range(0, len(Hc), 64):
            b = perm[s:s + 64]
            l = beh_loss_fn(F.normalize(pred_head(Hc[b]), dim=-1), Ic[b])
            hopt.zero_grad(); l.backward(); hopt.step()
    with torch.no_grad():
        zh = F.normalize(pred_head(Hc), dim=-1)
        print(json.dumps({"head_warmup": a.head_warmup_epochs, "n": len(Hc), "loss": round(beh_loss_fn(zh, Ic).item(), 4),
                          "cos_own": round((zh * Z[Ic]).sum(-1).mean().item(), 4),
                          "top1": round((zh @ Z.T).argmax(-1).eq(Ic).float().mean().item(), 4)}), flush=True)
    del hopt, Hc, hs
    model.train()

lora_params = [p for p in model.parameters() if p.requires_grad]
groups = [{"params": lora_params, "lr": a.lr}, {"params": list(pred_head.parameters()), "lr": a.head_lr}]
if prefix_inj is not None:
    groups.append({"params": list(prefix_inj.parameters()), "lr": a.prefix_lr})
opt = torch.optim.AdamW(groups, weight_decay=0.0)
base_lrs = [g["lr"] for g in groups]

steps_per_epoch = math.ceil(len(examples) / (a.batch_size * a.grad_accum))
total_steps = max(1, int(round(steps_per_epoch * a.epochs)))
warm = max(1, int(a.warmup * total_steps))
prefix_step = int(a.prefix_start * total_steps) if USE_PREFIX else None
weight_step = int(a.weight_start * total_steps) if USE_WEIGHT else None
print(f"[sched] {total_steps} optimiser steps; prefix on at step {prefix_step}; weighting on at step {weight_step}", flush=True)


def lr_scale(step):
    return step / warm if step < warm else max(0.0, (total_steps - step) / max(1, total_steps - warm))


def insert_after_bos(x, extra):
    """[B, L, ...] + [B, n, ...] -> [B, L+n, ...] with extra placed right after position 0 (BOS)."""
    return torch.cat([x[:, :1], extra, x[:, 1:]], dim=1)


def per_sample_ntp(logits, labels):
    lg, lb = logits[:, :-1].float(), labels[:, 1:]
    ce = F.cross_entropy(lg.reshape(-1, lg.size(-1)), lb.reshape(-1), ignore_index=-100, reduction="none").view(lb.shape)
    m = (lb != -100).float()
    return (ce * m).sum(1) / m.sum(1).clamp(min=1)          # mean over code tokens only


def step_loss(ids, lab, att, plen, tids, step):
    prefix_on = USE_PREFIX and step >= prefix_step
    weight_on = USE_WEIGHT and step >= weight_step
    B = ids.size(0)
    with torch.autocast(dev, dtype=amp):
        if prefix_on:   # pass 1: prompt only (causal, so h at plen-1 only sees the prompt)
            Lp = int(plen.max())
            o1 = model(input_ids=ids[:, :Lp], attention_mask=att[:, :Lp], output_hidden_states=True)
        else:           # single pass: full sequence gives both NTP and h
            o1 = model(input_ids=ids, attention_mask=att, output_hidden_states=True)
        h = o1.hidden_states[-1][torch.arange(B, device=dev), plen - 1].float()
    zhat = F.normalize(pred_head(h), dim=-1)

    keep = [i for i, t in enumerate(tids) if t in Z_pos]
    if keep:
        tgt = torch.tensor([Z_pos[tids[i]] for i in keep], device=dev)
        cos_own = (zhat[keep] * Z[tgt]).sum(-1)
        if a.beh_loss == "infonce":
            beh = F.cross_entropy(zhat[keep] @ Z.T / a.tau, tgt)
            top1 = (zhat[keep] @ Z.T).argmax(-1).eq(tgt).float().mean()
        else:
            beh = 1 - cos_own.mean(); top1 = torch.tensor(float("nan"))
    else:
        cos_own = torch.zeros(0, device=dev); beh = torch.zeros((), device=dev); top1 = torch.tensor(float("nan"))

    with torch.autocast(dev, dtype=amp):
        if prefix_on:   # pass 2: [BOS, prefix, rest] with matching mask and labels
            zin = zhat.detach() if not a.no_detach_prefix else zhat
            pre = prefix_inj(zin).to(amp)
            e = emb(ids).to(amp)
            o2 = model(inputs_embeds=insert_after_bos(e, pre),
                       attention_mask=insert_after_bos(att, torch.ones(B, a.n_prefix, dtype=att.dtype, device=dev)))
            lab2 = insert_after_bos(lab, torch.full((B, a.n_prefix), -100, dtype=lab.dtype, device=dev))
            ntp_i = per_sample_ntp(o2.logits, lab2)
        else:
            ntp_i = per_sample_ntp(o1.logits, lab)

    if weight_on and keep:
        w = torch.ones(B, device=dev)
        w[keep] = (1 - cos_own.detach()).clamp(0, 1)
        if len(keep) < B:   # tasks missing from the cache get the batch-average weight
            others = [i for i in range(B) if i not in keep]
            w[others] = w[keep].mean()
        w = w / w.mean().clamp(min=1e-6)
    else:
        w = torch.ones(B, device=dev)
    ntp = (w * ntp_i).mean()
    loss = ntp + a.lbd * beh
    return loss, {"ntp": ntp.item(), "beh": beh.item(), "cos_own": cos_own.mean().item() if len(cos_own) else float("nan"),
                  "top1": top1.item(), "w_max": w.max().item(), "prefix": int(prefix_on), "weight": int(weight_on)}


# ------------------------------------------------------------------ train
model.train(); pred_head.train()
if prefix_inj is not None:
    prefix_inj.train()
log = open(os.path.join(a.out, "train_log.jsonl"), "w")
step, micro, t0 = 0, 0, time.time()
order = list(range(len(examples)))
acc = {}
announced = set()
while step < total_steps:
    random.shuffle(order)
    for s in range(0, len(order), a.batch_size):
        batch = collate([examples[i] for i in order[s:s + a.batch_size]])
        ids, lab, att, plen = (x.to(dev) for x in batch[:4])
        loss, st = step_loss(ids, lab, att, plen, batch[4], step)
        (loss / a.grad_accum).backward()
        for k, v in st.items():
            acc.setdefault(k, []).append(v)
        micro += 1
        if micro % a.grad_accum:
            continue
        params = lora_params + list(pred_head.parameters()) + (list(prefix_inj.parameters()) if prefix_inj else [])
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        for g, base in zip(opt.param_groups, base_lrs):
            g["lr"] = base * lr_scale(step)
        opt.step(); opt.zero_grad(set_to_none=True)
        step += 1
        for flag, msg in (("prefix", "prefix conditioning ON"), ("weight", "hard-task weighting ON")):
            if acc.get(flag) and acc[flag][-1] and flag not in announced:
                announced.add(flag); print(f"[stage] step {step}: {msg}", flush=True)
        if step % a.log_every == 0 or step == total_steps:
            rec = {"step": step, "epoch": round(step / steps_per_epoch, 3), "grad_norm": round(gn, 3),
                   "sec": int(time.time() - t0), **{k: round(sum(v) / len(v), 4) for k, v in acc.items() if v and not math.isnan(sum(v))}}
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
            acc = {}
        if step >= total_steps:
            break
log.close()

# ------------------------------------------------------------------ save
model.eval()
torch.save(pred_head.state_dict(), os.path.join(a.out, "pred_head.pt"))
if prefix_inj is not None:
    torch.save(prefix_inj.state_dict(), os.path.join(a.out, "prefix_injector.pt"))
json.dump({"mode": a.mode, "n_prefix": a.n_prefix if USE_PREFIX else 0, "insert_after": 1, "h_pool": "last",
           "head": a.head, "beh_loss": a.beh_loss, "tau": a.tau, "lbd": a.lbd, "detach_prefix": not a.no_detach_prefix,
           "prefix_start": a.prefix_start, "weight_start": a.weight_start, "epochs": a.epochs, "lr": a.lr,
           "head_lr": a.head_lr, "prefix_lr": a.prefix_lr, "seed": a.seed, "base_model": a.model,
           "behaviour_dim": int(Z.shape[1]), "hidden": H}, open(os.path.join(a.out, "precode_config.json"), "w"), indent=1)
merged = model.merge_and_unload().to(torch.bfloat16)   # bf16 on disk: half the size, same as inference dtype
merged.save_pretrained(a.out)
tok.save_pretrained(a.out)
open(os.path.join(a.out, "DONE"), "w").close()   # contract C4: written last
print(f"[done] saved to {a.out} in {int(time.time() - t0)} s", flush=True)
