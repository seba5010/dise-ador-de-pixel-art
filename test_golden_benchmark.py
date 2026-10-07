import torch
import numpy as np
from pixel_ai_engine.golden_benchmark import GoldenBenchmark


def test_golden_benchmark_selects_deterministic_subset():
    samples = [{"char_id": f"char_{i % 5}", "frame_idx": i} for i in range(100)]
    benchmark1 = GoldenBenchmark(samples)
    benchmark2 = GoldenBenchmark(samples)
    assert benchmark1.benchmark_indices == benchmark2.benchmark_indices
    assert len(benchmark1.benchmark_indices) == GoldenBenchmark.DEFAULT_BENCHMARK_SIZE


def test_golden_benchmark_evaluates_healthy_tensors():
    benchmark = GoldenBenchmark()
    # Identical predictions and targets -> perfect score
    pred = torch.zeros((4, 32, 32))
    tgt = torch.zeros((4, 32, 32))
    result = benchmark.evaluate_tensors([pred], [tgt])
    assert result["is_healthy"] is True
    assert result["true_collapse_detected"] is False
    assert result["golden_score"] == 100.0


def test_golden_benchmark_detects_true_collapse():
    benchmark = GoldenBenchmark()
    pred_good = torch.zeros((4, 32, 32))
    tgt = torch.zeros((4, 32, 32))
    benchmark.evaluate_tensors([pred_good], [tgt])

    # Now simulate garbage prediction (complete noise / white vs black)
    pred_bad = torch.ones((4, 32, 32))
    result = benchmark.evaluate_tensors([pred_bad], [tgt])
    assert result["true_collapse_detected"] is True
    assert result["is_healthy"] is False
