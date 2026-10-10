"""SQLite and append-only JSONL persistence for simulator runs."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import HardwareDesign, SimulationResult


class Storage:
    def __init__(self, path: str | os.PathLike[str] | None = None, jsonl_path: str | os.PathLike[str] | None = None):
        self.path = Path(path or os.getenv("ACCELTWIN_DB_PATH", "data/acceltwin.db"))
        self.jsonl_path = Path(jsonl_path or os.getenv("ACCELTWIN_EVENT_LOG", "data/events.jsonl"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def _initialize(self) -> None:
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                trace_name TEXT NOT NULL,
                design_json TEXT NOT NULL,
                result_json TEXT NOT NULL
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at DESC)")

    @staticmethod
    def _dump(model: Any) -> str:
        if hasattr(model, "model_dump_json"):
            return model.model_dump_json()
        if hasattr(model, "json"):
            return model.json()
        return json.dumps(model, sort_keys=True)

    def create(self, result: SimulationResult | Dict[str, Any], design: HardwareDesign | Dict[str, Any] | None = None) -> str:
        result_data = json.loads(self._dump(result)) if isinstance(result, SimulationResult) else dict(result)
        design_data = json.loads(self._dump(design)) if isinstance(design, HardwareDesign) else dict(design or {})
        if not design_data:
            design_data = json.loads(self._dump(HardwareDesign.baseline()))
        run_id = uuid.uuid4().hex
        created_at = datetime.now(timezone.utc).isoformat()
        record = {
            "id": run_id, "created_at": created_at,
            "trace_name": str(result_data.get("trace_name", "unknown")),
            "design": design_data,
            "result": result_data,
        }
        with self._lock:
            with self._connect() as db:
                db.execute(
                    "INSERT INTO runs(id, created_at, trace_name, design_json, result_json) VALUES (?, ?, ?, ?, ?)",
                    (run_id, created_at, record["trace_name"], json.dumps(design_data), json.dumps(result_data)),
                )
            # Append, never rewrite or compact this audit stream.
            with self.jsonl_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
                stream.flush()
        return run_id

    def get(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return self._row_to_dict(row) if row else None

    def list(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        limit = max(0, min(int(limit), 500))
        offset = max(0, int(offset))
        with self._connect() as db:
            rows = db.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [self._row_to_dict(row) for row in rows]

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"], "created_at": row["created_at"], "trace_name": row["trace_name"],
            "design": json.loads(row["design_json"]), "result": json.loads(row["result_json"]),
        }

