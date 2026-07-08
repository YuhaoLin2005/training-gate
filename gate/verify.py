"""Gate 2: Behavioral comparison with multi-metric analysis.

Core differentiator: self-BLEU, digit density, repetition ratio, length ratio.
Catches collapses invisible to loss curves.
"""
import os, torch


def verify_checkpoint(path):
    adapter = os.path.join(path, "adapter_config.json")
    model_file = os.path.join(path, "adapter_model.safetensors")
    exists = os.path.exists(adapter) and os.path.getsize(adapter) > 0
    size_mb = os.path.getsize(model_file) / 1e6 if os.path.exists(model_file) else 0
    return {"exists": exists, "size_mb": round(size_mb, 1), "path": path}


def _bleu1(candidate, reference):
    """Unigram BLEU — fraction of tokens found in reference."""
    cand = candidate.split()
    ref = set(reference.split())
    if not cand: return 0.0
    return sum(1 for t in cand if t in ref) / len(cand)


def _self_bleu(outputs):
    """Pairwise BLEU-1 among all outputs. >0.8 = mode collapse."""
    if len(outputs) < 2: return 0.0
    scores = []
    for i in range(len(outputs)):
        for j in range(i+1, len(outputs)):
            scores.append(_bleu1(outputs[i], outputs[j]))
    return sum(scores) / len(scores) if scores else 0.0


def _digit_density(text):
    """Fraction of chars that are digits. >0.3 = numeric collapse."""
    if not text: return 0.0
    return sum(1 for c in text if c.isdigit()) / len(text)


def _repetition_ratio(text):
    """Unique/total token ratio. <0.3 = degenerate repetition."""
    tokens = text.split()
    if len(tokens) < 5: return 1.0
    return len(set(tokens)) / len(tokens)


def compare_behavior(base_model, ft_model, tokenizer, test_prompts, max_new=100):
    """Multi-metric behavioral comparison on unseen prompts.

    Metrics: digit_density, repetition_ratio, length vs base, self-BLEU.
    Returns per-prompt results + aggregate diagnostic.
    """
    results, ft_outputs = [], []
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(ft_model.device)
        with torch.no_grad():
            base = tokenizer.decode(
                base_model.generate(**inputs, max_new_tokens=max_new, do_sample=False)[0],
                skip_special_tokens=True)
            ft = tokenizer.decode(
                ft_model.generate(**inputs, max_new_tokens=max_new, do_sample=False)[0],
                skip_special_tokens=True)
        decoded = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        ft_only, base_only = ft[len(decoded):].strip(), base[len(decoded):].strip()
        ft_outputs.append(ft_only)

        dd, rr = _digit_density(ft_only), _repetition_ratio(ft_only)
        ok = dd < 0.5 and len(ft_only) > 10 and rr > 0.3

        results.append({
            "prompt": prompt[:60], "base_len": len(base_only), "ft_len": len(ft_only),
            "different": ft_only != base_only,
            "digit_density": round(dd, 3), "repetition_ratio": round(rr, 3),
            "ok": ok, "ft_sample": ft_only[:120], "base_sample": base_only[:80],
        })

    ok_count = sum(r["ok"] for r in results)
    sb = _self_bleu(ft_outputs)
    avg_dd = sum(r["digit_density"] for r in results) / len(results) if results else 0
    avg_rr = sum(r["repetition_ratio"] for r in results) / len(results) if results else 0

    issues = []
    if sb > 0.8: issues.append(f"self-BLEU={sb:.2f} (mode collapse)")
    if avg_dd > 0.3: issues.append(f"digit_density={avg_dd:.2f} (numeric degradation)")
    if avg_rr < 0.4: issues.append(f"repetition_ratio={avg_rr:.2f} (token degeneration)")
    diagnostic = "healthy" if not issues else "; ".join(issues)

    return {
        "total": len(results), "ok": ok_count,
        "rate": round(ok_count/len(results), 2) if results else 0,
        "passed": ok_count >= 2,
        "self_bleu": round(sb, 3), "avg_digit_density": round(avg_dd, 3),
        "avg_repetition_ratio": round(avg_rr, 3), "diagnostic": diagnostic,
        "results": results,
    }
