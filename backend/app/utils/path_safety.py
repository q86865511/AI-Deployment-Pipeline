"""
使用者輸入轉檔案路徑時的清洗與複核工具

上傳端點（資料集 ZIP、模型檔）會拿使用者提供的名稱去組路徑，
未清洗時 "../../../evil.zip" 之類的名稱會讓檔案落在預期目錄之外。
本模組刻意不依賴 FastAPI 或任何第三方套件，方便在免 GPU 的 CI 中直接測試。
"""
import os
import re

# 名稱字元白名單：Unicode 文字/數字/底線（\w，含中日韓字元）、連字號、點；
# 其餘（路徑分隔符、空白、shell 特殊字元…）一律換成底線。
_ALLOWED_NAME_PATTERN = re.compile(r'[^\w.-]', re.UNICODE)

# 清洗後仍屬危險的名稱（純點號會被檔案系統解讀為上層目錄）。
_RESERVED_NAMES = {"", ".", ".."}


def sanitize_filename(name: str, default: str = "unnamed") -> str:
    """清洗使用者提供的檔名 / 名稱，使其只剩單一路徑元件且僅含白名單字元。

    Args:
        name: 使用者提供的原始名稱（可能含 ../ 、絕對路徑或跳脫字元）
        default: 清洗後為空或為保留名稱時使用的替代值

    Returns:
        安全的單一路徑元件名稱
    """
    if not name:
        return default

    # 先把 Windows 風格的分隔符正規化，再取 basename 去掉所有目錄成分
    base = os.path.basename(str(name).replace('\\', '/').rstrip('/'))
    cleaned = _ALLOWED_NAME_PATTERN.sub('_', base)

    # 去掉開頭的點，避免產生隱藏檔或 "..foo" 這類名稱
    cleaned = cleaned.lstrip('.')

    if cleaned in _RESERVED_NAMES:
        return default
    return cleaned


def is_within(root: str, path: str) -> bool:
    """檢查 path 落地後是否仍在 root 目錄內（含 root 本身）。"""
    root_abs = os.path.abspath(root)
    path_abs = os.path.abspath(path)
    try:
        # os.path.commonpath 會在跨磁碟機等情況丟出 ValueError，視為不安全
        return os.path.commonpath([root_abs, path_abs]) == root_abs
    except ValueError:
        return False


def ensure_within(root: str, path: str) -> str:
    """確認 path 仍在 root 內並回傳其絕對路徑，否則丟出 ValueError。"""
    if not is_within(root, path):
        raise ValueError(f"路徑逃逸出允許的目錄範圍: {path}")
    return os.path.abspath(path)
