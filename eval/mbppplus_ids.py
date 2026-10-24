# usage: python eval/mbppplus_ids.py   [WS, env evalv, repo root] -- contract C8, run once
import json
from evalplus.data import get_mbpp_plus

ids = [k for k in get_mbpp_plus() if 11 <= int(k.split("/")[1]) <= 510]
assert len(ids) == 224, len(ids)
json.dump(ids, open("eval/mbppplus_test_ids.json", "w"))
print("wrote", len(ids), "ids")
