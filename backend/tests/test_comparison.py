"""Characterization tests for generate_comparison (extracted from InferenceService).
Dependency-free — builds ModelPerformance via pydantic; no torch/GPU needed."""
from app.models import ModelPerformance, PrecisionType
from app.services.comparison import generate_comparison


def _mp(model_id, t_ms, throughput, mem_mb):
    return ModelPerformance(
        model_id=model_id, inference_time_ms=t_ms, throughput=throughput,
        memory_usage_mb=mem_mb, precision=PrecisionType.FP32, device_info={},
    )


def test_empty_results():
    c = generate_comparison([])
    assert c["models"] == []
    assert c["speedup"] == {}
    assert c["memory_reduction"] == {}
    assert c["best_performance"] is None
    assert c["best_memory_efficiency"] is None


def test_collects_basic_series_in_order():
    c = generate_comparison([_mp("base", 100.0, 10.0, 1000.0), _mp("fast", 50.0, 20.0, 500.0)])
    assert c["models"] == ["base", "fast"]
    assert c["inference_time_ms"] == [100.0, 50.0]
    assert c["throughput"] == [10.0, 20.0]
    assert c["memory_usage_mb"] == [1000.0, 500.0]


def test_speedup_and_memory_reduction_relative_to_first():
    c = generate_comparison([_mp("base", 100.0, 10.0, 1000.0), _mp("fast", 50.0, 20.0, 500.0)])
    # base (index 0) is the reference and is skipped
    assert "base" not in c["speedup"]
    assert c["speedup"]["fast"] == 2.0                 # 100 / 50
    assert c["memory_reduction"]["fast"] == 50.0       # (1000-500)/1000*100


def test_bests_pick_minimum_time_and_memory_independently():
    results = [_mp("a", 80.0, 12.0, 900.0), _mp("b", 120.0, 8.0, 400.0), _mp("c", 60.0, 15.0, 700.0)]
    c = generate_comparison(results)
    assert c["best_performance"]["model_id"] == "c"        # lowest inference_time (60)
    assert c["best_memory_efficiency"]["model_id"] == "b"  # lowest memory (400)
    assert c["speedup"]["b"] == round(80.0 / 120.0, 2)
    assert c["speedup"]["c"] == round(80.0 / 60.0, 2)
