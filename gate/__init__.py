"""training-gate: Structured quality gates for model fine-tuning.

Gate 0 (health):  Pre-training environment check
Gate 1 (monitor): In-training loss plateau detection
Gate 2 (verify):  Post-training checkpoint + behavioral comparison
Gate 3 (audit):   JSONL experiment tracking
"""

from .health import check as health_check, GateBlockedError
from .monitor import PlateauMonitor, PlateauCallback
from .verify import verify_checkpoint, compare_behavior
from .audit import AuditLogger, write_summary
