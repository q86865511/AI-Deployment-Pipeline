"""Characterization tests for build_test_combinations (extracted from TestManager.create_test_task).
Dependency-free — exercises only app.models + the pure helper (no torch/GPU)."""
from app.services.combinations import build_test_combinations


def test_counts_and_original_flags():
    c = build_test_combinations("m1", [1, 8], ["fp16", "fp32"], 640)
    # 2 original (one per batch) + 2 precisions * 2 batches = 6
    assert len(c) == 6
    originals = [x for x in c if x["is_original"]]
    converted = [x for x in c if not x["is_original"]]
    assert len(originals) == 2
    assert len(converted) == 4


def test_original_combinations_shape():
    c = build_test_combinations("src-model", [4], ["fp16"], 512)
    o = next(x for x in c if x["is_original"])
    assert o["precision"] == "original"
    assert o["status"] == "completed"
    assert o["target_model_id"] == "src-model"
    assert o["conversion_job_id"] is None
    assert o["image_size"] == 512
    assert o["batch_size"] == 4


def test_converted_combinations_shape():
    c = build_test_combinations("src", [2], ["fp16", "fp32"], 640)
    converted = [x for x in c if not x["is_original"]]
    assert {x["precision"] for x in converted} == {"fp16", "fp32"}
    for cc in converted:
        assert cc["status"] == "pending"
        assert cc["target_model_id"] is None
        assert cc["image_size"] == 640


def test_precision_is_case_insensitive_for_fp16():
    c = build_test_combinations("m", [1], ["FP16"], 640)
    conv = [x for x in c if not x["is_original"]]
    assert conv[0]["precision"] == "fp16"


def test_unknown_precision_falls_back_to_fp32():
    c = build_test_combinations("m", [1], ["int8"], 640)
    conv = [x for x in c if not x["is_original"]]
    assert conv[0]["precision"] == "fp32"


def test_empty_inputs_yield_empty_list():
    assert build_test_combinations("m", [], [], 320) == []
