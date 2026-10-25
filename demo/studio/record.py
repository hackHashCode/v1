"""Record live studio runs so the demo can replay them with no GPU (and as a fallback if the live server fails).

    python demo/studio/record.py --api http://localhost:8010            # every example in /api/meta
    python demo/studio/record.py --api https://...modal.run --only Mbpp_17,Mbpp_161

Writes demo/studio/recordings/<slug>.json: the exact event stream (with timings) and the execution results.
Standard library only, so it runs anywhere."""
import argparse, json, os, re, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def slug(e):
    return re.sub(r"[^A-Za-z0-9]+", "_", e["task_id"]) if e.get("task_id") else "spec_" + re.sub(r"[^a-z0-9]+", "_", e["label"].lower())


def post(api, path, body, stream=False):
    req = urllib.request.Request(api + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=600)


def record(api, meta, e):
    runs = [r["id"] for r in meta["runs"]]
    t0 = time.time(); events = []
    with post(api, "/api/generate", {"task": e["task"], "tests": e["tests"], "runs": runs}) as r:
        for raw in r:
            line = raw.decode().strip()
            if line.startswith("data: "):
                ev = json.loads(line[6:]); ev["t"] = int((time.time() - t0) * 1000); events.append(ev)
    codes = {ev["run"]: ev["code"] for ev in events if ev["type"] == "done"}
    imagined = {ev["run"]: ev["z"] for ev in events if ev["type"] == "imagined"}
    with post(api, "/api/execute", {"codes": codes, "tests": e["tests"], "reference": e.get("reference"),
                                    "imagined": imagined, "setup": e.get("setup", "")}) as r:
        ex = json.loads(r.read())
    for ev in events:   # the predicted vector is only needed live; keep recordings small
        ev.pop("z", None)
    return {"task": e["task"], "tests": e["tests"], "reference": e.get("reference"), "task_id": e.get("task_id"),
            "runs": runs, "runs_meta": [{k: r[k] for k in ("id", "name", "blurb", "ours")} for r in meta["runs"]],
            "events": events, "exec": ex, "gpu": meta.get("gpu"), "recorded_at": time.strftime("%Y-%m-%d %H:%M")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True)
    ap.add_argument("--only", default="", help="comma-separated slugs, e.g. Mbpp_17,spec_palindrome")
    a = ap.parse_args()
    api = a.api.rstrip("/")
    meta = json.loads(urllib.request.urlopen(api + "/api/meta", timeout=120).read())
    if meta.get("mock"):
        raise SystemExit("server is in MOCK mode; record from the real models")
    only = {s for s in a.only.split(",") if s}
    os.makedirs(os.path.join(HERE, "recordings"), exist_ok=True)
    for e in meta["examples"]:
        s = slug(e)
        if only and s not in only:
            continue
        rec = record(api, meta, e)
        json.dump(rec, open(os.path.join(HERE, "recordings", s + ".json"), "w"), indent=0)
        verdicts = {k: f"{sum(t['ok'] for t in v['tests'])}/{len(v['tests'])}" for k, v in rec["exec"]["results"].items()}
        print(f"{s:40s} {verdicts}", flush=True)


if __name__ == "__main__":
    main()
