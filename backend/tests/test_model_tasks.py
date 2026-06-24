"""Characterization tests for task_from_model_type (deduped from inference/conversion services).
Dependency-free — exercises only app.models + the pure helper (no torch/GPU)."""
from app.models import ModelType
from app.services.model_tasks import task_from_model_type


def test_pose_maps_to_pose():
    assert task_from_model_type(ModelType.YOLOV8_POSE) == "pose"


def test_seg_maps_to_segment():
    assert task_from_model_type(ModelType.YOLOV8_SEG) == "segment"


def test_yolov8_maps_to_detect():
    assert task_from_model_type(ModelType.YOLOV8) == "detect"


def test_custom_defaults_to_detect():
    assert task_from_model_type(ModelType.CUSTOM) == "detect"


def test_every_model_type_returns_valid_task():
    valid = {"pose", "segment", "detect"}
    for mt in ModelType:
        assert task_from_model_type(mt) in valid
