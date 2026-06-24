"""Pure helper: map a ModelType to the YOLO `task` string.

Dependency-free (only app.models / pydantic) so it is unit-testable without
torch/tensorrt/GPU. Extracted from the duplicated `_get_task_from_model_type`
that previously lived identically in both inference_service and conversion_service.
"""
from app.models import ModelType


def task_from_model_type(model_type: ModelType) -> str:
    """根據模型類型確定YOLO task參數"""
    if model_type == ModelType.YOLOV8_POSE:
        return "pose"
    elif model_type == ModelType.YOLOV8_SEG:
        return "segment"
    elif model_type == ModelType.YOLOV8:
        return "detect"
    else:
        return "detect"  # 默認為檢測任務
