"""Plan log: every planning run is appended as one JSON line to a gzip file per local day
(<config>/heat_planner/plans/<entry_id>/plans-YYYY-MM-DD.jsonl.gz); files older than the retention are removed.
Read with e.g. `zcat plans-2026-11-25.jsonl.gz | jq .`"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import os

import numpy as np


def _default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (dt.datetime, dt.date)):
        return o.isoformat()
    return str(o)


def write(folder: str, now: dt.datetime, record: dict, keep_days: int) -> None:
    """append `record` to today's file and prune old files (blocking: run in the executor)"""
    os.makedirs(folder, exist_ok=True)
    line = json.dumps(record, default=_default, ensure_ascii=False, separators=(",", ":")) + "\n"
    with gzip.open(os.path.join(folder, f"plans-{now.date().isoformat()}.jsonl.gz"), "at", encoding="utf-8") as f:
        f.write(line)
    oldest = (now.date() - dt.timedelta(days=keep_days)).isoformat()
    for name in os.listdir(folder):
        if name.startswith("plans-") and name.endswith(".jsonl.gz") and name[6:16] < oldest:
            try:
                os.remove(os.path.join(folder, name))
            except OSError:
                pass
