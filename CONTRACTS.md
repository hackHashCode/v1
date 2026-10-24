# Contracts (the only connection points between tracks)

| # | Artifact | Producer -> consumer | Format |
|---|---|---|---|
| C1 | `datasets/mbpp_train.jsonl` | Gagan | `{"task_id": int, "messages": [system, user, assistant]}`, 474 rows, no task_id in 11-510 |
| C2 | `cache/behaviour_cache.pt` | Shryes -> Gagan | `torch.save({int task_id: float32 tensor (768,)})`, L2-normalised |
| C3 | `cache/traces.jsonl` | Shryes | `{"task_id": int, "trace": ["f([5, 5, 4]) -> 4", ...]}` |
| C4 | `/data/precode/models/<run>/` | Gagan -> Nikhita, Shryes | merged HF model + tokenizer; PreCode runs add `pred_head.pt`; `DONE` written last |
| C5 | run name | all | `{sft,jepa,precode,precodence,precodecond}_s{42,7,1234}` (precodence = PreCode with InfoNCE over all cached z); zero-shot = `base_s0` |
| C6 | `eval/generate.py` | Nikhita | `--model DIR --dataset {mbpp,humaneval} --out FILE [--n 1] [--temperature 0.0] [--prompts FILE]` |
| C7 | samples file | generate -> score | EvalPlus jsonl `{"task_id": "Mbpp/11", "solution": "..."}`, all 378 MBPP+ tasks |
| C8 | `eval/mbppplus_test_ids.json` | Nikhita | list of 224 `"Mbpp/<id>"`, 11 <= id <= 510 |
| C9 | `results/runs.csv` | Nikhita -> Gagan | `run,method,seed,dataset,pass1_base,pass1_plus,n_tasks,timestamp`; datasets `mbppplus224, humanevalplus, para60_orig, para60_para` |
| C10 | `results/per_task/<run>_<dataset>.json` | Nikhita | `{"Mbpp/11": true, ...}` (base AND plus pass) |
| C11 | `results/attempts/<run>.json` | Nikhita | `{"Mbpp/11": 1..6}` (6 = not solved in 5) |
| C12 | demo API | Gagan -> Nikhita | `POST /solve {"task", "tests": [...], "n"}` -> `{"sft": [{"code","passed","total"}], "jepa": [...], "precode": [...]}` |
| C13 | prompt | all | system = `SYSTEM` below; MBPP train user = `text + "\n" + test_list[0]`; MBPP+ eval user = `prompt.strip().strip('"').strip()`; HumanEval+ user = `"Complete this Python function. Return the whole function.\n\n" + prompt`; chat template with `add_generation_prompt=True`; take code from first ``` fence if present |

`SYSTEM = "You are an expert Python programmer. Write a correct Python function for the task. Output only code."`

Paraphrase file (`eval/paraphrase/mbppplus_paraphrased.jsonl`): `{"task_id": "Mbpp/11", "prompt": "<reworded description>\n<original first assert>"}`.
