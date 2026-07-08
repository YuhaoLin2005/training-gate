"""Gate 3: Audit Logger — JSONL experiment tracking."""
import json
from datetime import datetime

class AuditLogger:
    """Structured JSONL logging. Same pattern as log-regeneration.py."""
    def __init__(self, path="./training-audit.jsonl"):
        self.path = path
        self.entries = []

    def log(self, **kwargs):
        entry = {"timestamp": datetime.now().isoformat(), **kwargs}
        self.entries.append(entry)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def flush(self):
        """Rewrite audit file from memory (dedup/reset)."""
        with open(self.path, "w", encoding="utf-8") as f:
            for e in self.entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")


def write_summary(path, **kwargs):
    """Write final experiment summary as JSON."""
    summary = {"timestamp": datetime.now().isoformat(), **kwargs}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary
