"""Gate 1: In-Training Monitor — loss plateau detection."""
from transformers import TrainerCallback

class PlateauMonitor:
    """Tracks loss, detects plateaus. No HF dependency."""
    def __init__(self, patience=5, threshold=0.01):
        self.patience = patience
        self.threshold = threshold
        self.best_loss = float("inf")
        self.stall_count = 0
        self.history = []

    def update(self, loss, step, epoch=0):
        self.history.append({"step": step, "loss": loss, "epoch": epoch})
        if loss < self.best_loss * (1 - self.threshold):
            self.best_loss = loss
            self.stall_count = 0
            return {"status": "improving", "best_loss": round(self.best_loss, 4)}
        self.stall_count += 1
        return {"status": "plateau" if self.stall_count >= self.patience else "stalling",
                "best_loss": round(self.best_loss, 4), "stall_steps": self.stall_count}

    def summary(self):
        if not self.history: return {}
        first, last = self.history[0]["loss"], self.history[-1]["loss"]
        return {"first": round(first,4), "last": round(last,4),
                "best": round(self.best_loss,4),
                "improved": last < first * 0.95,
                "pct": round((1-last/first)*100,1) if first else 0}


class PlateauCallback(TrainerCallback):
    """HF Trainer callback wrapping PlateauMonitor."""
    def __init__(self, patience=5, threshold=0.01, verbose=True):
        self.monitor = PlateauMonitor(patience, threshold)
        self.verbose = verbose
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            r = self.monitor.update(logs["loss"], state.global_step, state.epoch)
            if self.verbose:
                s = r["status"]
                tag = "v" if s=="improving" else f"~{r.get('stall_steps','?')}"
                print(f"  step={state.global_step:3d} loss={logs['loss']:.4f} {tag}")
