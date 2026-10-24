"""[Laptop] pip install anthropic evalplus; export ANTHROPIC_API_KEY=...  -> fallback/mbppplus_paraphrased.jsonl"""
import json, random
import anthropic
from evalplus.data import get_mbpp_plus
P = get_mbpp_plus()
ids = [k for k in P if 11 <= int(k.split("/")[1]) <= 510]
random.seed(0); pick = random.sample(ids, 60)
cl = anthropic.Anthropic()
with open("fallback/mbppplus_paraphrased.jsonl", "w") as f:
    for t in pick:
        desc, test = P[t]["prompt"].strip().strip('"').strip().rsplit("\n", 1)
        r = cl.messages.create(model="claude-sonnet-5-5", max_tokens=300, messages=[{"role": "user", "content":
            "Reword this programming task with different wording but exactly the same meaning. "
            "Add no hints or examples. Reply with the reworded task only.\n\n" + desc}])
        f.write(json.dumps({"task_id": t, "prompt": r.content[0].text.strip() + "\n" + test}) + "\n")
print("done")
