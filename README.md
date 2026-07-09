# training-gate

> Loss curves lie. Qwen2.5-1.5B-Instruct, LoRA fine-tune, 80 Chinese instructions. Training loss dropped smoothly from 9.13 to 8.78. Solid convergence. Everything looked great.
 > 
 > Then we ran a test prompt. Every single output was `199999...` — the model collapsed into a digit-repeating machine. Perplexity never flagged it. The loss curve kept going down.
 
> Adds behavioral quality gates that catch what loss curves miss. If your team fine-tunes models, run this before deploying — it's 30 minutes and catches the collapses loss can't see.

## What

4 gates covering the training lifecycle. Gate 2 is the differentiator — most people only check loss.

```
Gate 0 (health) -> _ate 1 (monitor) -> Gate 2 (verify) -> Gate 3 (audit)
     |                   |                    |                  |
  GPU/VRAM            loss plateau        behavioral          JSONL log
  check               detection           comparison         tracking
```

### Gate 2: Behavioral Comparison (the core insight)

After fine-tuning, we compare base model vs fine-tuned model on unseen prompts. Three metrics multiplied into one drift score:

1. *R*Self-BLEU:R* (output similarity within the same batch. Above 0.8 ↓ model is collapsing (all outputs converging to the same thing). We use BLEU-1, not BLEU-4, because digit-repeating collapse is visible at the 1-gram level — higher-order n-grams add computation without adding signal.

2. **digit density** ℓ percentage of numeric characters in output. Above 0.3 ↓ model is degrading into a number repeater. Threshold calibrated from baseline models (2-8% digits in normal Chinese QA) vs collapsed models (80%+). 0.3 is conservative — prefer false positives to missed collapses. Domain-specific tuning needed: code generation naturally has higher digit density, raise the threshold; financial text sits between, test on your data.

3. **repetition ratio** ℓ unique tokens / total tokens. Below 0.3 ↓ model is looping. Distinct from self-BLEU: repetition ratio catches token-level degeneracy within a single output; self-BLEU catches batch-level homogeneity across outputs.

**Why multiply?** Each metric catches a different failure mode. In our experiment, the fine-tuned model scored 0.0 on behavioral pass rate while loss was still dropping. Three independent signals × multiplication = no single metric can "vote away" a real collapse.

Submitted to HuggingFace Evaluate as a standard metric: PR [778](https://github.com/huggingface/evaluate/pull/778).

## What We Learned About Instruct Models

We chose Qwen2.5-1.5B-Instruct deliberately — to test whether an already-instruction-tuned model would benefit from additional LoRA fine-tuning. It didn't. 80 examples were enough to *disrupt* existing instruction-following pathways without enough signal to *redirect* behavior.

The general insight: **fine-tuning an Instruct model, loss going down can mean the model is getting more "certain" — not more "correct."** The digit-repeating pattern is an extreme case, but milder versions (outputs getting shorter, less varietal, more formulaic) occeur silently in many fine-tuning runs. Behavioral metrics catch these when loss can't.

This doesn't mean "don't fine-tune Instruct models." It means: when you do, monitor behavior separately from loss. They can move in opposite directions.

## Quick Start

```python
from gate import health_check, PlateauCallback, verify_checkpoint, compare_behavior, AuditLogger

health = health_check(required_vram_gb=3.0)
monitor = PlateauCallback(patience=3)
trainer = Trainer(..., callbacks=[monitor])
trainer.train()

ckpt = verify_checkpoint("./my-checkpoint")
behavior = compare_behavior(base_model, ft_model, tokenizer, unseen_prompts)
# ^ This is the one that caught 199999... when loss was still dropping.

audit = AuditLogger("./experiment.jsonl")
```

```bash
pip install torch transformers peft modelscope
python examples/train_with_gates.py
```

## When This Matters

- Your team fine-tunes models and deploys to customers. A model that collapsed into digit-repeating would be caught by manual testing — but one that just got 20% more repetitive wouldn't. Behavioral comparison catches subtle degradation.
- You're iterating on fine-tuning recipes. Each run, you check loss. Add behavioral pass rate as a second column. If loss is down but behavioral pass rate is also down — the recipe is making the model more certain, not better.
- You suspect a checkpoint is degrading but can't articulate why. Run the behavioral comparison against baseline. The three metrics give you a concrete "here's what changed" beyond "it feels worse."

Validated on Qwen2.5-0.5B through 1.5B (single RTX 3060 6GB). Not yet tested on 7B+. Collapse patterns may differ at larger scales — similar principles, potential threshold adjustments.

## License

MIT
