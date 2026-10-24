import json, os, subprocess, sys, tempfile
from vllm import LLM, SamplingParams
def extract(t):
    if "```" in t:
        t = t.split("```")[1]; t = t[len("python"):] if t.startswith("python") else t
    return t.strip()
rows = [json.loads(l) for l in open("datasets/mbpp_val.jsonl")]
llm = LLM(sys.argv[1], gpu_memory_utilization=0.28, max_model_len=1024); tok = llm.get_tokenizer()
texts = [tok.apply_chat_template(r["messages"][:2], tokenize=False, add_generation_prompt=True) for r in rows]
outs = llm.generate(texts, SamplingParams(temperature=0.0, max_tokens=384))
ok = 0
for r, o in zip(rows, outs):
    src = (r.get("setup") or "") + "\n" + extract(o.outputs[0].text) + "\n" + "\n".join(r["tests"])
    try: ok += subprocess.run([sys.executable, "-c", src], capture_output=True, timeout=10, cwd=tempfile.gettempdir()).returncode == 0
    except subprocess.TimeoutExpired: pass
line = f"{os.path.basename(sys.argv[1].rstrip('/'))}  val {ok}/{len(rows)} = {ok/len(rows):.3f}"
print(line); open("results/val_sweep.txt", "a").write(line + "\n")
