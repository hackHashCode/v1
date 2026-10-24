"""Demo backend (contract C12). [WS, env train] from repo root:
    uvicorn demo.server:app --host 0.0.0.0 --port 8000
Laptop: ssh -N -L 8000:localhost:8000 <ws-host>
"""
import os, subprocess, sys, tempfile
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM

SYSTEM = "You are an expert Python programmer. Write a correct Python function for the task. Output only code."
SEED = os.environ.get("DEMO_SEED", "42")
RUNS = {k: f"/data/precode/models/{k}_s{SEED}" for k in ("sft", "jepa", "precode")}
M = {k: (AutoTokenizer.from_pretrained(p),
         AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.bfloat16).cuda().eval()) for k, p in RUNS.items()}
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def extract(t):
    if "```" in t:
        t = t.split("```")[1]
        t = t[len("python"):] if t.startswith("python") else t
    return t.strip()

def score(code, tests):
    body = "\n".join(f"try:\n    {t}\n    print('@P')\nexcept BaseException:\n    print('@F')" for t in tests)
    try:
        out = subprocess.run([sys.executable, "-c", code + "\n" + body], capture_output=True, text=True,
                             timeout=10, cwd=tempfile.gettempdir()).stdout
    except subprocess.TimeoutExpired:
        out = ""
    return sum(l == "@P" for l in out.splitlines())

@app.get("/health")
def health():
    return {"ok": True, "runs": RUNS}

@app.post("/solve")
def solve(req: dict):
    task, tests, n = req["task"], req.get("tests", []), int(req.get("n", 5))
    user = task.strip() + ("\n" + tests[0] if tests else "")  # first test shown, all tests scored
    res = {}
    for k, (tok, m) in M.items():
        ids = tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                                      add_generation_prompt=True, return_tensors="pt").cuda()
        with torch.no_grad():
            out = m.generate(ids, do_sample=n > 1, temperature=0.8 if n > 1 else None, num_return_sequences=n,
                             max_new_tokens=384, pad_token_id=tok.eos_token_id)
        codes = [extract(tok.decode(o[ids.shape[1]:], skip_special_tokens=True)) for o in out]
        res[k] = [{"code": c, "passed": score(c, tests), "total": len(tests)} for c in codes]
    return res
