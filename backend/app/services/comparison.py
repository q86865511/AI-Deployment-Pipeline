"""Pure helper: build a model-performance comparison (speedup / memory-reduction /
bests) from a list of ModelPerformance results.

Dependency-free (only app.models / pydantic + stdlib typing) so it is unit-testable
without torch/tensorrt/GPU. Extracted verbatim from InferenceService.generate_comparison;
the service now delegates here.
"""
from typing import Any, Dict, List

from app.models import ModelPerformance


def generate_comparison(results: List[ModelPerformance]) -> Dict[str, Any]:
    """生成模型性能比較結果"""
    comparison = {
        "models": [],
        "inference_time_ms": [],
        "throughput": [],
        "memory_usage_mb": [],
        "speedup": {},
        "memory_reduction": {},
        "best_performance": None,
        "best_memory_efficiency": None
    }

    # 收集基本數據
    for perf in results:
        comparison["models"].append(perf.model_id)
        comparison["inference_time_ms"].append(perf.inference_time_ms)
        comparison["throughput"].append(perf.throughput)
        comparison["memory_usage_mb"].append(perf.memory_usage_mb)

    # 找出基準模型（首個結果）並計算相對指標
    if results:
        base_perf = results[0]
        base_time = base_perf.inference_time_ms
        base_memory = base_perf.memory_usage_mb

        # 計算每個模型相對於基準的加速比和內存減少率
        for i, perf in enumerate(results):
            if i > 0:  # 跳過基準模型自身
                model_id = perf.model_id
                comparison["speedup"][model_id] = round(base_time / perf.inference_time_ms, 2)
                comparison["memory_reduction"][model_id] = round(
                    (base_memory - perf.memory_usage_mb) / base_memory * 100, 2
                )

        # 找出最佳性能和內存效率的模型
        best_perf_idx = comparison["inference_time_ms"].index(min(comparison["inference_time_ms"]))
        best_mem_idx = comparison["memory_usage_mb"].index(min(comparison["memory_usage_mb"]))

        comparison["best_performance"] = {
            "model_id": results[best_perf_idx].model_id,
            "inference_time_ms": results[best_perf_idx].inference_time_ms,
            "throughput": results[best_perf_idx].throughput
        }

        comparison["best_memory_efficiency"] = {
            "model_id": results[best_mem_idx].model_id,
            "memory_usage_mb": results[best_mem_idx].memory_usage_mb
        }

    return comparison
