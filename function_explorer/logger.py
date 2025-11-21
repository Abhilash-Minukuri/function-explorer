from __future__ import annotations

import csv
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import config

_SESSION_LOG_STATE: Dict[str, Dict[str, Any]] = {}


def normalize_param_value(param: str, value: Any) -> float:
    cfg = config.PARAM_BOUNDS.get(param, {"min": -10.0, "max": 10.0, "step": 0.1})
    try:
        num = float(value)
    except (TypeError, ValueError):
        num = float(config.DEFAULT_PARAMS.get(param, 0.0))
    num = max(cfg["min"], min(cfg["max"], num))
    step = cfg.get("step", 0.1) or 0.1
    quantized = round(num / step) * step
    if quantized == -0.0:
        quantized = 0.0
    return float(f"{quantized:.12g}")


def normalize_params(raw: Optional[Dict[str, Any]]) -> Dict[str, float]:
    params = {}
    for key, default_val in config.DEFAULT_PARAMS.items():
        params[key] = normalize_param_value(key, (raw or {}).get(key, default_val))
    return params


def normalized_equal(param: str, old_value: Any, new_value: Any) -> bool:
    old_norm = normalize_param_value(param, old_value)
    new_norm = normalize_param_value(param, new_value)
    return abs(old_norm - new_norm) < 1e-9


def next_seq_and_elapsed(session_id: str, t_client_ms: Optional[int]) -> Dict[str, Any]:
    state = _SESSION_LOG_STATE.setdefault(
        session_id, {"seq": 0, "last_t_client": None, "last_t_server_ms": None}
    )
    now_ms = int(time.time() * 1000)
    seq = state["seq"] + 1
    state["seq"] = seq
    elapsed = 0
    if t_client_ms is not None and state["last_t_client"] is not None:
        elapsed = max(int(t_client_ms - state["last_t_client"]), 0)
    elif state["last_t_server_ms"] is not None:
        elapsed = max(now_ms - state["last_t_server_ms"], 0)
    state["last_t_client"] = t_client_ms
    state["last_t_server_ms"] = now_ms
    ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    return {"seq": seq, "elapsed_time_ms": elapsed, "t_server_iso": ts, "now_ms": now_ms}


def session_log_path(session_id: str) -> Path:
    return config.DATA_DIR / f"session_{session_id}.jsonl"


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()


def flatten_record_for_csv(record: Dict[str, Any]) -> Dict[str, Any]:
    flat = dict.fromkeys(config.SCHEMA_COLUMNS, None)
    for key, value in record.items():
        if key in flat:
            flat[key] = value
    return flat


def build_csv_content(records: List[Dict[str, Any]]) -> Optional[str]:
    if not records:
        return None
    rows = [flatten_record_for_csv(rec) for rec in records]
    columns = list(config.SCHEMA_COLUMNS)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        if row.get("t_client_ms") is not None:
            try:
                row["t_client_ms"] = str(int(row["t_client_ms"]))
            except (TypeError, ValueError):
                pass
        writer.writerow({col: row.get(col) for col in columns})
    return buffer.getvalue()


def build_excel_csv_content(records: List[Dict[str, Any]]) -> Optional[str]:
    return build_csv_content(records)
