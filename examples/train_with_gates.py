"""Example: Fine-tune Qwen2.5-1.5B-Instruct with training-gate.

Demonstrates all 4 gates: health -> train(monitor) -> verify -> audit
"""
import sys, os, json, torch, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gate import health_check, PlateauCallback, verify_checkpoint, compare_behavior
from gate import AuditLogger, write_summary, GateBlockedError

from modelscope import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType

# ==== Gate 0: Health Check ====
print("="*60)
print("GATE 0: HEALTH CHECK")
print("="*60)
try:
    h = health_check(required_vram_gb=3.0)
    print(f"  GPU: {h['gpu_name']} | VRAM: {h['vram_total_gb']}GB, {h['vram_free_gb']}GB free")
    print("  [PASS]")
except GateBlockedError as e:
    print(f"  [FAIL] {e}")
    sys.exit(2)

# ==== Load model & data ====
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
print(f"\nLoading {MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True, use_cache=False)
lora = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","v_proj"],
                  lora_dropout=0.1, task_type=TaskType.CAUSAL_LM)
model = get_peft_model(model, lora)
total_p = sum(p.numel() for p in model.parameters())
train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  LoRA: {train_p/1e6:.2f}M / {total_p/1e9:.2f}B")

data_path = "C:/Users/86131/Desktop/training_data_clean.json"
with open(data_path, "r", encoding="utf-8") as f:
    raw = json.load(f)

def fmt(i, r):
    return f"<|im_start|>user\n{i}<|im_end|>\n<|im_start|>assistant\n{r}<|im_end|>"

samples = []
for item in raw[:80]:
    parts = item.split("\n回答:")
    if len(parts) == 2:
        samples.append(fmt(parts[0].replace("指令:","").strip(), parts[1].strip()))
print(f"  Data: {len(samples)} samples")

enc = tokenizer(samples, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
class CD(Dataset):
    def __init__(self, e): self.e = e
    def __len__(self): return len(self.e["input_ids"])
    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.e.items()}
        item["labels"] = item["input_ids"].clone()
        return item

# ==== Gate 1: Training with Monitor ====
print("\n" + "="*60)
print("GATE 1: TRAINING (plateau monitor)")
print("="*60)

audit = AuditLogger("./training-gate-audit.jsonl")
audit.log(event="training_start", samples=len(samples), model=MODEL)

monitor = PlateauCallback(patience=3, threshold=0.01)
t0 = time.time()
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./gate-demo-out", per_device_train_batch_size=1,
        gradient_accumulation_steps=8, num_train_epochs=1, learning_rate=5e-5,
        fp16=True, logging_steps=5, save_strategy="no",
        warmup_ratio=0.1, max_grad_norm=1.0, gradient_checkpointing=True,
    ),
    train_dataset=CD(enc), callbacks=[monitor],
)
trainer.train()
ckpt_dir = "./gate-demo-ckpt"
model.save_pretrained(ckpt_dir)
train_time = (time.time() - t0) / 60
ms = monitor.monitor.summary()
print(f"  Done: {train_time:.1f}min | Loss: {ms.get('first','?')} -> {ms.get('last','?')}")
audit.log(event="training_done", train_time_min=round(train_time,1), **ms)

# ==== Gate 2: Checkpoint Verification ====
print("\n" + "="*60)
print("GATE 2: VERIFY CHECKPOINT")
print("="*60)
ckpt = verify_checkpoint(ckpt_dir)
print(f"  {'PASS' if ckpt['exists'] else 'FAIL'} ({ckpt.get('size_mb','?')}MB)")
audit.log(event="verify_checkpoint", **ckpt)

# ==== Gate 3: Behavioral Comparison ====
print("\n" + "="*60)
print("GATE 3: BEHAVIORAL COMPARISON")
print("="*60)

tests = [
    "你的方案被客户拒绝了三次，每次反馈不同。怎么判断继续改还是换方向？",
    "刚写完复杂逻辑准备push，同事不在没人code review，怎么办？",
    "PM问这个需求能做吗。60%能做，40%需要学新技术。怎么回复？",
    "换了新电脑，配置文件没拷过来。开始工作前应该做什么？",
    "你是后端开发，前端同事提了PR请你review，但你不懂JavaScript。",
]

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
behavior = compare_behavior(base_model, model, tokenizer, tests)
print(f"  OK: {behavior['ok']}/{behavior['total']} ({behavior['rate']*100:.0f}%)")
for r in behavior["results"]:
    tag = "OK" if r["ok"] else ("DIGITS" if r.get("digit_density",0)>0.3 else "REPEAT" if r.get("repetition_ratio",1)<0.4 else "WEAK")
    print(f"  [{tag}] {r['prompt'][:50]}...")
    print(f"    ft: {r['ft_sample'][:100]}...")

del base_model; torch.cuda.empty_cache()
audit.log(event="behavioral_test", ok=behavior["ok"], rate=behavior["rate"])

# ==== Final ====
print("\n" + "="*60)
print("FINAL")
print("="*60)
all_ok = h["all_ok"] and ms.get("improved", False) and ckpt["exists"]
print(f"  Gate 0 (health):  {'PASS' if h['all_ok'] else 'FAIL'}")
print(f"  Gate 1 (monitor): {'PASS' if ms.get('improved') else 'FAIL'}")
print(f"  Gate 2 (verify):  {'PASS' if ckpt['exists'] else 'FAIL'}")
print(f"  Gate 3 (behavior):{'PASS' if behavior['passed'] else 'INFO'}")
print(f"  OVERALL: {'PASS' if all_ok else 'PARTIAL'}")

write_summary("./training-gate-summary.json",
    model=MODEL, samples=len(samples), train_time_min=round(train_time,1),
    trainable_params=train_p, trainable_pct=round(train_p/total_p*100,2),
    loss=ms, checkpoint=ckpt, behavioral=behavior["rate"], health=h["all_ok"])
audit.log(event="experiment_done", overall=all_ok)
print("\n  Audit: ./training-gate-audit.jsonl")
print("  Summary: ./training-gate-summary.json")
print("  training-gate demo complete.")
