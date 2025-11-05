"""Minimal JSONL logger (stub)."""

from datetime import datetime
from pathlib import Path
import json

def get_log_path(session_id: str) -> Path:
    Path("function_explorer/data").mkdir(parents=True, exist_ok=True)
    return Path(f"function_explorer/data/session_{session_id}.jsonl")

def log_event(session_id: str, payload: dict):
    rec = {"timestamp": datetime.utcnow().isoformat()+"Z", **payload}
    path = get_log_path(session_id)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
