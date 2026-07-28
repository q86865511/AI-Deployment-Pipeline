"""app.utils.path_safety 的行為測試（純邏輯、免 GPU / 免 FastAPI）。

對應上傳端點的路徑穿越防護：
- benchmark.py 的資料集 ZIP 檔名
- models.py / model_service.py 的模型名稱
"""
import os

import pytest

from app.utils.path_safety import ensure_within, is_within, sanitize_filename


class TestSanitizeFilename:
    @pytest.mark.parametrize("evil", [
        "../../../evil.zip",
        "..\\..\\..\\evil.zip",
        "/etc/passwd",
        "C:\\Windows\\System32\\evil.zip",
        "subdir/evil.zip",
    ])
    def test_路徑成分被剝除(self, evil):
        cleaned = sanitize_filename(evil)
        assert os.sep not in cleaned
        assert "/" not in cleaned and "\\" not in cleaned
        assert ".." not in cleaned

    def test_保留合法檔名(self):
        assert sanitize_filename("coco_val2017.zip") == "coco_val2017.zip"
        assert sanitize_filename("yolov8n-pose.pt") == "yolov8n-pose.pt"

    def test_白名單外字元被替換(self):
        assert sanitize_filename("bad name;rm -rf.zip") == "bad_name_rm_-rf.zip"

    def test_保留中文名稱(self):
        # 專案為中文介面，合法的中文檔名不應被打亂
        assert sanitize_filename("驗證資料集.zip") == "驗證資料集.zip"

    def test_空值與保留名稱退回預設(self):
        assert sanitize_filename("") == "unnamed"
        assert sanitize_filename("..") == "unnamed"
        assert sanitize_filename("../..", default="fallback") == "fallback"

    def test_清洗結果可安全組成路徑(self):
        root = os.path.join(os.sep, "srv", "data", "datasets")
        target = os.path.join(root, sanitize_filename("../../../evil.zip"))
        assert is_within(root, target)


class TestIsWithin:
    def test_目錄內為真(self, tmp_path):
        root = str(tmp_path)
        assert is_within(root, os.path.join(root, "a", "b.txt"))

    def test_穿越到上層為假(self, tmp_path):
        root = str(tmp_path / "datasets")
        assert not is_within(root, os.path.join(root, "..", "..", "evil.zip"))

    def test_ensure_within_穿越時拋錯(self, tmp_path):
        root = str(tmp_path / "model_repository")
        with pytest.raises(ValueError):
            ensure_within(root, os.path.join(root, "..", "escaped"))

    def test_ensure_within_合法時回傳絕對路徑(self, tmp_path):
        root = str(tmp_path)
        result = ensure_within(root, os.path.join(root, "ok.txt"))
        assert result == os.path.abspath(os.path.join(root, "ok.txt"))
