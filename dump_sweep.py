"""Pull all sweep-* runs from wandb and dump eval history + config to JSON."""
import json
import wandb

ENTITY_PROJECT = "matthewgorbett/cme-grpo"
EVAL_KEYS = [
    "_step",
    "eval/math500_pass_at_1",
    "eval/amc23_pass_at_1",
    "eval/aime24_pass_at_1",
    "eval/math500/pass_at_1",
    "eval/amc23/pass_at_1",
    "eval/aime24/pass_at_1",
]

api = wandb.Api()
runs = api.runs(ENTITY_PROJECT, filters={"display_name": {"$regex": "^sweep-"}})

out = []
for r in runs:
    if r.state != "finished":
        print(f"skip {r.name} ({r.state})")
        continue
    summary = {k: v for k, v in r.summary.items() if not k.startswith("_")}
    eval_keys_in_run = [k for k in summary.keys() if k.startswith("eval/")]
    # scan_history(keys=...) breaks on '@' in key names — pull rows that contain any eval key
    history_full = list(r.scan_history())
    history = [
        {k: row.get(k) for k in (["_step", "_runtime"] + eval_keys_in_run) if k in row}
        for row in history_full
        if any(k in row for k in eval_keys_in_run)
    ]
    cfg = {k: v for k, v in r.config.items() if not k.startswith("_")}
    out.append({
        "id": r.id,
        "name": r.name,
        "state": r.state,
        "verifier": cfg.get("model", {}).get("verifier") if isinstance(cfg.get("model"), dict) else cfg.get("verifier"),
        "generator": cfg.get("model", {}).get("generator") if isinstance(cfg.get("model"), dict) else cfg.get("generator"),
        "eval_keys": eval_keys_in_run,
        "summary": summary,
        "history": history,
    })
    print(f"ok   {r.name}  steps={len(history)}  eval_keys={eval_keys_in_run}")

with open("sweep_dump.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"\nwrote sweep_dump.json with {len(out)} runs")
