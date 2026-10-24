"""Inference helpers shared by eval/generate_hf.py and demo/studio: load pred_head / prefix_injector from a run
folder and generate with the same [BOS, prefix, rest] layout that train/precode_full.py trains with."""
import json, os, re

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_head(path, hidden):
    sd = torch.load(path, map_location="cpu")
    if "weight" in sd:
        head = nn.Linear(hidden, sd["weight"].shape[0], bias=False)
    else:   # mlp: 0.weight / 2.weight
        head = nn.Sequential(nn.Linear(hidden, hidden // 2, bias=False), nn.GELU(),
                             nn.Linear(hidden // 2, sd["2.weight"].shape[0], bias=False))
    head.load_state_dict(sd)
    return head.float().eval()


class PrefixInjector(nn.Module):
    def __init__(self, d_in, hidden, n):
        super().__init__()
        self.n, self.hidden = n, hidden
        self.proj = nn.Linear(d_in, n * hidden)

    def forward(self, z):
        return self.proj(z).view(-1, self.n, self.hidden)


def load_precode_parts(run_dir, hidden):
    """-> (pred_head or None, prefix_injector or None, config dict)."""
    cfg = {}
    cp = os.path.join(run_dir, "precode_config.json")
    if os.path.exists(cp):
        cfg = json.load(open(cp))
    hp = os.path.join(run_dir, "pred_head.pt")
    head = load_head(hp, hidden) if os.path.exists(hp) else None
    inj = None
    ip = os.path.join(run_dir, "prefix_injector.pt")
    if os.path.exists(ip) and cfg.get("n_prefix", 0) > 0:
        sd = torch.load(ip, map_location="cpu")
        inj = PrefixInjector(sd["proj.weight"].shape[1], hidden, cfg["n_prefix"])
        inj.load_state_dict(sd)
        inj = inj.float().eval()
    return head, inj, cfg


@torch.no_grad()
def prompt_state(model, ids, mask, h_pool="last"):
    """Final-layer hidden state at the last prompt token. Works with left padding (last position = last token)."""
    out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    hs = out.hidden_states[-1].float()
    if h_pool == "mean":
        m = mask.float().unsqueeze(-1)
        return (hs * m).sum(1) / m.sum(1).clamp(min=1)
    return hs[:, -1]


def predict_z(head, h):
    return F.normalize(head.to(h.device)(h), dim=-1)


def prefixed_inputs(model, ids, mask, prefix):
    """Left-padded ids [B, L] + prefix [B, n, H] -> inputs_embeds / attention_mask with the prefix right after
    each row's BOS (BOS sits at the first unmasked position under left padding)."""
    emb = model.get_input_embeddings()(ids)
    prefix = prefix.to(emb.dtype)
    rows_e, rows_m = [], []
    n = prefix.size(1)
    for i in range(ids.size(0)):
        b = int((mask[i] == 0).sum())          # number of left pads; BOS at index b
        rows_e.append(torch.cat([emb[i, :b + 1], prefix[i], emb[i, b + 1:]], 0))
        rows_m.append(torch.cat([mask[i, :b + 1], torch.ones(n, dtype=mask.dtype, device=mask.device), mask[i, b + 1:]], 0))
    return torch.stack(rows_e), torch.stack(rows_m)


# ------------------------------------------------------------------ tokenizer / generation helpers (any model)
CHATML = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
          "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


def ensure_chat_template(tok):
    """Base (non-instruct) checkpoints may ship without a chat template. Use ChatML (Qwen's own format) so training,
    eval and the demo all build identical prompts."""
    if tok.chat_template is None:
        tok.chat_template = CHATML
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def stop_ids(tok):
    """Stop at EOS and at every control token of the form <|...|>. Chat-tuned models end with <|im_end|> (Qwen) or
    <|eot_id|> (Llama), but a code BASE model like Qwen2.5-Coder was pretrained on whole repositories and, after a
    function, emits <|repo_name|> / <|file_sep|> / <|fim_prefix|> and keeps going. None of these ever belongs in an
    answer, so all of them end generation (same rule for every method)."""
    ids = {tok.eos_token_id} if tok.eos_token_id is not None else set()
    for i, at in getattr(tok, "added_tokens_decoder", {}).items():
        s = str(getattr(at, "content", at))
        if re.fullmatch(r"<\|[^|<>]+\|>", s) and i != tok.bos_token_id:
            ids.add(int(i))
    for s in ("<|im_end|>", "<|eot_id|>", "<|endoftext|>"):
        i = tok.convert_tokens_to_ids(s)
        if isinstance(i, int) and i != tok.unk_token_id and i >= 0:
            ids.add(i)
    return sorted(ids)


def extract_code(t):
    """First ``` fenced block if any (contract C13); then cut at any leftover control marker (<|...) or a fake next
    turn ("user" / ".user" / "assistant" line) that a base model may invent."""
    if "```" in t:
        t = t.split("```")[1]
        if t.startswith("python"):
            t = t[len("python"):]
    t = t.split("<|")[0]
    t = re.split(r"\n\s*\.?\s*(?:user|assistant|Human:|### )\s*(?:\n|$)", t)[0]
    return t.strip()


@torch.no_grad()
def generate_batch(model, tok, texts, head=None, inj=None, h_pool="last", max_new=384):
    """Greedy generation for a list of chat-formatted prompt strings. Returns (decoded completions, z or None)."""
    tok.padding_side = "left"
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    gen = dict(max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id, eos_token_id=stop_ids(tok))
    z = None
    if head is not None:
        z = predict_z(head, prompt_state(model, enc.input_ids, enc.attention_mask, h_pool))
    if inj is not None:
        emb, mask = prefixed_inputs(model, enc.input_ids, enc.attention_mask, inj.to(z.device)(z))
        new = model.generate(inputs_embeds=emb, attention_mask=mask, **gen)       # new tokens only
    else:
        out = model.generate(**enc, **gen)
        new = out[:, enc.input_ids.shape[1]:]
    return [tok.decode(o, skip_special_tokens=True) for o in new], z
