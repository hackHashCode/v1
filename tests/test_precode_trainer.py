"""CPU unit test for PreCodeTrainer (tiny random Llama, no downloads).
python tests/test_precode_trainer.py
Checks: h is taken at the last prompt token, loss is finite, pred_head and LoRA both get gradients,
JEPA loss falls over a few optimizer steps, and missing cache entries are skipped."""
import os, sys, tempfile
import torch, torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM, TrainingArguments, default_data_collator
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
from finetune import PreCodeTrainer

torch.manual_seed(0)
H, D, V, L = 64, 16, 100, 24
cfg = LlamaConfig(vocab_size=V, hidden_size=H, intermediate_size=128, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
model = get_peft_model(LlamaForCausalLM(cfg), LoraConfig(task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8,
                       target_modules=["q_proj", "v_proj"]))
model.pred_head = torch.nn.Linear(H, D, bias=False)

# 8 examples: prompt length varies, then code, then padding (labels -100 on prompt + pad)
rows = []
for i in range(8):
    p = 5 + i % 4; c = 6
    ids = torch.randint(3, V, (L,)).tolist()
    labels = [-100] * p + ids[p:p + c] + [-100] * (L - p - c)
    mask = [1] * (p + c) + [0] * (L - p - c)
    rows.append({"input_ids": ids, "labels": labels, "attention_mask": mask, "task_id": 600 + i})
ds = Dataset.from_list(rows)
tmp = tempfile.mkdtemp()
cache = {600 + i: F.normalize(torch.randn(D), dim=0) for i in range(7)}   # task 607 deliberately missing
torch.save(cache, f"{tmp}/cache.pt")

args = TrainingArguments(output_dir=tmp, per_device_train_batch_size=4, num_train_epochs=30, learning_rate=5e-3,
                         report_to="none", remove_unused_columns=False, logging_steps=1000, save_strategy="no",
                         use_cpu=True)
tr = PreCodeTrainer(model=model, args=args, train_dataset=ds, data_collator=default_data_collator,
                    lbd=1.0, cache_path=f"{tmp}/cache.pt")

# 1) index check: h must be at position (first label != -100) - 1 == prompt_len - 1
batch = default_data_collator(rows[:4])
first = (batch["labels"] != -100).int().argmax(1) - 1
assert first.tolist() == [4, 5, 6, 7], first.tolist()

# 2) one loss + backward: finite, gradients reach pred_head and LoRA
b = {k: v.clone() for k, v in batch.items()}
loss = tr.compute_loss(model, b)
assert torch.isfinite(loss), loss
loss.backward()
assert model.pred_head.weight.grad is not None and model.pred_head.weight.grad.abs().sum() > 0
lora_g = [p.grad for n, p in model.named_parameters() if "lora_" in n and p.grad is not None]
assert lora_g and sum(g.abs().sum() for g in lora_g) > 0
model.zero_grad()

# 3) cos(z_hat, z) should rise after training
def mean_cos():
    with torch.no_grad():
        bb = default_data_collator(rows[:7])
        out = model(input_ids=bb["input_ids"], attention_mask=bb["attention_mask"], output_hidden_states=True)
        idx = ((bb["labels"] != -100).int().argmax(1) - 1)
        h = out.hidden_states[-1][torch.arange(7), idx].float()
        z = torch.stack([cache[int(t)] for t in bb["task_id"]])
        return F.cosine_similarity(model.pred_head(h), z, dim=-1).mean().item()
before = mean_cos()
tr.train()
after = mean_cos()
print(f"cos before {before:.3f} after {after:.3f}")
assert after > before + 0.3, (before, after)

# 4) pred_head saved separately survives merge
torch.save(model.pred_head.state_dict(), f"{tmp}/pred_head.pt")
merged = model.merge_and_unload()
assert not hasattr(merged, "pred_head")
assert torch.load(f"{tmp}/pred_head.pt")["weight"].shape == (D, H)

# 5) InfoNCE variant: finite loss, and z_hat retrieves its own task among all cached z after training
torch.manual_seed(1)
model2 = get_peft_model(LlamaForCausalLM(cfg), LoraConfig(task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8,
                        target_modules=["q_proj", "v_proj"]))
model2.pred_head = torch.nn.Linear(H, D, bias=False)
tr2 = PreCodeTrainer(model=model2, args=args, train_dataset=ds, data_collator=default_data_collator,
                     lbd=1.0, cache_path=f"{tmp}/cache.pt", precode_loss="infonce", infonce_temp=0.07)
assert tr2.Z.shape == (7, D)
l2 = tr2.compute_loss(model2, {k: v.clone() for k, v in batch.items()})
assert torch.isfinite(l2)
def top1(m, trn):
    with torch.no_grad():
        bb = default_data_collator(rows[:7])
        out = m(input_ids=bb["input_ids"], attention_mask=bb["attention_mask"], output_hidden_states=True)
        idx = ((bb["labels"] != -100).int().argmax(1) - 1)
        zh = F.normalize(m.pred_head(out.hidden_states[-1][torch.arange(7), idx].float()), dim=-1)
        pred = (zh @ trn.Z.T).argmax(1).tolist()
        return sum(trn.z_ids[p] == int(t) for p, t in zip(pred, bb["task_id"])) / 7
b1 = top1(model2, tr2); tr2.train(); a1 = top1(model2, tr2)
print(f"infonce top-1 retrieval before {b1:.2f} after {a1:.2f}")
assert a1 >= 0.85, a1

# 6) new knobs: head_lr param group, lambda schedules, mean pool / middle layer, MLP head, GA fix
torch.manual_seed(2)
model3 = get_peft_model(LlamaForCausalLM(cfg), LoraConfig(task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8,
                        target_modules=["q_proj", "v_proj"]))
model3.pred_head = torch.nn.Sequential(torch.nn.Linear(H, H // 2, bias=False), torch.nn.GELU(), torch.nn.Linear(H // 2, D, bias=False))
args3 = TrainingArguments(output_dir=tmp, per_device_train_batch_size=4, num_train_epochs=2, learning_rate=1e-4,
                          report_to="none", remove_unused_columns=False, logging_steps=1000, save_strategy="no", use_cpu=True)
tr3 = PreCodeTrainer(model=model3, args=args3, train_dataset=ds, data_collator=default_data_collator, lbd=1.0,
                     cache_path=f"{tmp}/cache.pt", head_lr=1e-2, lbd_schedule="decay", h_pool="mean", h_layer=1)
assert tr3.model_accepts_loss_kwargs is False
opt = tr3.create_optimizer()
lrs = sorted(g["lr"] for g in opt.param_groups)
assert lrs == [1e-4, 1e-2], lrs
assert sum(p.numel() for p in opt.param_groups[1]["params"]) == sum(p.numel() for p in model3.pred_head.parameters())
assert torch.isfinite(tr3.compute_loss(model3, {k: v.clone() for k, v in batch.items()}))
tr3.state.max_steps = 100
for t, want in [(0, 1.0), (49, 1.0), (75, 0.5), (100, 0.0)]:
    tr3.state.global_step = t; assert abs(tr3._lbd_now() - want) < 1e-6, (t, tr3._lbd_now())
tr3.lbd_schedule = "linear"; tr3.state.global_step = 25; assert abs(tr3._lbd_now() - 0.75) < 1e-6
tr3.state.global_step = 0; tr3.lbd_schedule = "decay"
tr3.train()
print("ALL PRECODE TRAINER TESTS PASSED")
