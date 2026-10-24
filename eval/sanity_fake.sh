#!/usr/bin/env bash
# usage: bash eval/sanity_fake.sh   [WS, env evalv, repo root]
# Scores the canonical MBPP+ solutions with the real score.py: plus should be ~1.0, and Mbpp/11
# (replaced by `pass`) must come out false. Runs in /tmp/sanity so results/ is never touched.
set -euo pipefail
REPO=$(pwd); S=/tmp/sanity; rm -rf $S; mkdir -p $S/eval
cp eval/mbppplus_test_ids.json $S/eval/
python - <<'EOF'
import json
from evalplus.data import get_mbpp_plus
with open("/tmp/sanity/fake.jsonl", "w") as f:
    for t, p in get_mbpp_plus().items():
        sol = "pass" if t == "Mbpp/11" else p["prompt"] + p["canonical_solution"]
        f.write(json.dumps({"task_id": t, "solution": sol}) + "\n")
EOF
evalplus.evaluate --dataset mbpp --samples $S/fake.jsonl
R=$S/fake_eval_results.json; [ -f $R ] || R=$S/fake.eval_results.json
echo "evaluator wrote: $R"
cd $S && python $REPO/eval/score.py $R fake_s0 mbppplus224
python -c "import json; d = json.load(open('results/per_task/fake_s0_mbppplus224.json')); print('Mbpp/11 ->', d['Mbpp/11'], '(want False)'); print('other failures:', [k for k, v in d.items() if not v and k != 'Mbpp/11'])"
