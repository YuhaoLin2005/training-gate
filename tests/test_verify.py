"""Tests for gate.verify metrics."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gate.verify import _digit_density, _repetition_ratio, _self_bleu

def test_digit_density():
    assert _digit_density("正常回答") == 0.0
    assert _digit_density("1999999999") > 0.9
    assert _digit_density("") == 0.0

def test_repetition():
    assert _repetition_ratio("正常的有变化的中文回答") > 0.5
    assert _repetition_ratio("重复 重复 重复 重复") < 0.5

def test_self_bleu():
    assert _self_bleu(["相同", "相同", "相同"]) > 0.9
    assert _self_bleu(["完全不同内容", "另一段文本", "新话题"]) < 0.8
    assert _self_bleu(["只有一个"]) == 0.0

print("All tests passed!")
