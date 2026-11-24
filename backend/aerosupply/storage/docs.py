"""Document store (MongoDB).

Holds what does not fit a fixed relational schema:
  signals          unstructured supplier events (news, audits, quality notifications, surveys)
  risk_snapshots   one document per supplier and run: score, drivers, context (nested, evolving)
  simulations      what-if scenarios saved from the dashboard
  pipeline_runs    run log: step timings, data-quality report, model metrics

With MONGO_URL set (Docker) a real MongoDB is used; otherwise an embedded mongomock instance is
persisted to data/docstore.json so the project runs with zero infrastructure.
"""
from __future__ import annotations

import threading
from pathlib import Path

from bson import json_util

from ..config import DATA_DIR, MONGO_DB, MONGO_URL

_lock = threading.Lock()
_instance = None


class DocStore:
    def __init__(self, url: str | None = MONGO_URL, path: Path = DATA_DIR / "docstore.json"):
        self.path = Path(path)
        if url:
            from pymongo import MongoClient
            self.client = MongoClient(url, serverSelectionTimeoutMS=5000, tz_aware=False)
            self.embedded = False
        else:
            import mongomock
            self.client = mongomock.MongoClient()
            self.embedded = True
        self.db = self.client[MONGO_DB]
        if self.embedded:
            self._load()

    def __getitem__(self, name):
        return self.db[name]

    def _load(self):
        if self.path.exists():
            data = json_util.loads(self.path.read_text(encoding="utf-8"))
            for name, docs in data.items():
                if docs:
                    self.db[name].insert_many(docs)

    def reload(self):
        """Re-read the embedded store (another process — the pipeline — may have written it)."""
        if self.embedded:
            for name in self.db.list_collection_names():
                self.db.drop_collection(name)
            self._load()

    def flush(self):
        if not self.embedded:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {n: list(self.db[n].find({}, {"_id": 0})) for n in self.db.list_collection_names()}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json_util.dumps(data), encoding="utf-8")
        tmp.replace(self.path)

    def replace_collection(self, name: str, docs: list[dict], indexes=()):
        self.db.drop_collection(name)
        if docs:
            self.db[name].insert_many([dict(d) for d in docs])
        for idx in indexes:
            self.db[name].create_index(idx)


def get_store() -> DocStore:
    global _instance
    with _lock:
        if _instance is None:
            _instance = DocStore()
        return _instance
