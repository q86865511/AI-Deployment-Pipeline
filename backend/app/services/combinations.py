"""Pure helper: build the list of test combinations (original model + converted
models across precisions × batch sizes) for a benchmark task.

Dependency-free (only app.models / pydantic + stdlib typing) so it is unit-testable
without torch/tensorrt/GPU. Extracted verbatim from TestManager.create_test_task;
the service now delegates here.
"""
from typing import Any, Dict, List

from app.models import PrecisionType


def build_test_combinations(model_id: str, batch_sizes: List[int], precisions: List[str],
                            image_size: int) -> List[Dict[str, Any]]:
    """生成所有測試組合，包含原始模型 + 各精度×批次的轉換模型。"""
    combinations = []

    # 首先添加原始模型的測試組合
    for batch_size in batch_sizes:
        combinations.append({
            "batch_size": batch_size,
            "precision": "original",  # 標記為原始模型
            "image_size": image_size,
            "status": "completed",  # 原始模型不需要轉換
            "conversion_job_id": None,
            "target_model_id": model_id,  # 直接使用源模型ID
            "inference_results": None,
            "error": None,
            "is_original": True  # 新增標記以區分原始模型
        })

    # 然後添加轉換後模型的測試組合
    for precision in precisions:
        precision_enum = PrecisionType.FP16 if precision.lower() == 'fp16' else PrecisionType.FP32
        for batch_size in batch_sizes:
            combinations.append({
                "batch_size": batch_size,
                "precision": precision_enum.value,
                "image_size": image_size,
                "status": "pending",
                "conversion_job_id": None,
                "target_model_id": None,
                "inference_results": None,
                "error": None,
                "is_original": False  # 標記為轉換模型
            })

    return combinations
