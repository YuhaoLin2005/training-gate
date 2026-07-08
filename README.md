# training-gate

> **Loss curves lie.** Your loss drops from 9.2 to 8.8. But every output became `199999...` — and perplexity never flagged it.
> This tool adds behavioral quality gates that catch what loss curves miss.

## What

4 gates covering the training lifecycle:

```
Gate 0 (health) -> Gate 1 (monitor) -> Gate 2 (verify) -> Gate 3 (audit)
     |                   |                    |                  |
  GPU/VRAM            loss plateau        checkpoint         JSONL log
  check               detection           + behavior         experiment
                                           comparison         tracking
```

Core differentiator: Gate 2's behavioral comparison — compares base vs fine-tuned on unseen prompts. Most people only check loss.

## Quick Start

```python
from gate import health_check, PlateauCallback, verify_checkpoint, compare_behavior, AuditLogger

# Gate 0
health = health_check(required_vram_gb=3.0)

# Gate 1
monitor = PlateauCallback(patience=3)
trainer = Trainer(..., callbacks=[monitor])
trainer.train()

# Gate 2
ckpt = verify_checkpoint("./my-checkpoint")
behavior = compare_behavior(base_model, ft_model, tokenizer, unseen_prompts)

# Gate 3
audit = AuditLogger("./experiment.jsonl")
```

## Run

```bash
pip install torch transformers peft modelscope
python examples/train_with_gates.py
```

## Architecture Mapping

| training-gate | delivery-gate (Agent) | Production ML |
|--------------|----------------------|---------------|
| Gate 0: health | Pre-Task Calibration | Env validation |
| Gate 1: monitor | Stale detection | EarlyStopping |
| Gate 2: verify | Delivery Verification | Model validation |
| Gate 3: audit | Regeneration log | W&B / MLflow |

## License

MIT
