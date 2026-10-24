import json
from evalplus.data import get_mbpp_plus
ids = [k for k in get_mbpp_plus() if 11 <= int(k.split("/")[1]) <= 510]
assert len(ids) == 224, len(ids)
json.dump(ids, open("fallback/mbppplus_test_ids.json", "w"))
print(len(ids))
