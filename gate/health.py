"""Gate 0: Pre-Training Health Check."""
import torch

class GateBlockedError(RuntimeError):
    """Any gate failure. Equivalent to delivery-gate exit 2."""
    pass

def check(required_vram_gb=3.0):
    if not torch.cuda.is_available():
        raise GateBlockedError("GATE 0 FAILED: No CUDA GPU")
    gpu = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    free = total - torch.cuda.memory_allocated() / 1e9
    ok = free >= required_vram_gb
    result = {"gpu_name": gpu, "vram_total_gb": round(total,1),
              "vram_free_gb": round(free,1), "torch_version": torch.__version__,
              "cuda_version": torch.version.cuda, "all_ok": ok}
    if not ok:
        raise GateBlockedError(f"GATE 0 FAILED: {free:.1f}GB free, need {required_vram_gb:.1f}GB")
    return result
