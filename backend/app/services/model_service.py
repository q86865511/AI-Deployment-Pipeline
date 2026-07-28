import os
import json
import uuid
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone, timedelta
from app.models import ModelInfo, ModelType, ModelFormat
from app.utils.path_safety import sanitize_filename, is_within

class ModelService:
    """
    模型服務，負責模型管理和存儲庫掃描
    完全基於model_repository目錄中的metadata，廢除models.json
    """

    def __init__(self):
        """初始化模型服務"""
        self.models: Dict[str, ModelInfo] = {}
        self.model_repository_path = os.environ.get("MODEL_REPOSITORY_PATH", "model_repository")
        
        # 確保模型存儲庫目錄存在
        os.makedirs(self.model_repository_path, exist_ok=True)
        
        # 初始化時掃描模型存儲庫
        self._scan_model_repository()

    def get_models(self, model_type: Optional[ModelType] = None, 
                   format: Optional[ModelFormat] = None,
                   skip: int = 0, limit: int = 10) -> List[ModelInfo]:
        """獲取模型列表，支持過濾和分頁"""
        # 每次查詢前重新掃描，確保獲取最新狀態
        self._scan_model_repository()
        
        models = list(self.models.values())
        
        # 過濾
        if model_type:
            models = [m for m in models if m.type == model_type]
        if format:
            models = [m for m in models if m.format == format]
        
        # 按創建時間排序
        models.sort(key=lambda m: m.created_at, reverse=True)
        
        # 分頁
        return models[skip:skip + limit]

    def get_model_by_id(self, model_id: str) -> Optional[ModelInfo]:
        """根據ID獲取模型"""
        # 每次查詢前重新掃描，確保獲取最新狀態
        self._scan_model_repository()
        return self.models.get(model_id)

    def get_model_by_name(self, name: str) -> Optional[ModelInfo]:
        """根據名稱獲取模型"""
        # 每次查詢前重新掃描，確保獲取最新狀態
        self._scan_model_repository()
        for model in self.models.values():
            if model.name == name:
                return model
        return None

    def save_model(self, model: ModelInfo) -> ModelInfo:
        """保存模型信息到model_repository中的metadata"""
        self.models[model.id] = model
        
        # 根據模型路徑確定metadata文件位置
        model_dir = self._get_model_directory(model.path)
        if model_dir:
            self._write_metadata_file(model_dir, model)
        
        return model

    def delete_model(self, model_id: str) -> bool:
        """刪除模型"""
        model = self.models.get(model_id)
        if not model:
            return False
        
        try:
            # 獲取模型目錄
            model_dir = self._get_model_directory(model.path)
            
            if model_dir and os.path.exists(model_dir):
                # 刪除整個模型目錄
                import shutil
                shutil.rmtree(model_dir)
                print(f"已刪除模型目錄: {model_dir}")
            
            # 從內存中刪除
            del self.models[model_id]
            
            return True
        except Exception as e:
            print(f"刪除模型失敗: {str(e)}")
            return False

    def _scan_model_repository(self):
        """掃描模型存儲庫，重新載入所有模型"""
        print(f"掃描模型存儲庫: {self.model_repository_path}")
        
        # 清空現有模型列表
        self.models.clear()
        
        if not os.path.exists(self.model_repository_path):
            print(f"模型存儲庫目錄不存在: {self.model_repository_path}")
            return
        
        # 遍歷所有子目錄
        for item in os.listdir(self.model_repository_path):
            item_path = os.path.join(self.model_repository_path, item)
            
            if os.path.isdir(item_path):
                try:
                    self._scan_model_directory(item_path)
                except Exception as e:
                    print(f"掃描模型目錄 {item_path} 時出錯: {str(e)}")
        
        print(f"掃描完成，找到 {len(self.models)} 個模型")

    def _scan_model_directory(self, model_dir: str):
        """掃描單個模型目錄"""
        # 先檢查是否有metadata文件
        metadata_file = os.path.join(model_dir, "metadata.json")
        if os.path.exists(metadata_file):
            model = self._load_model_from_metadata(model_dir, metadata_file)
            if model:
                self.models[model.id] = model
        else:
            # 沒有metadata文件，嘗試自動檢測
            model = self._detect_and_create_model(model_dir)
            if model:
                self.models[model.id] = model

    def _load_model_from_metadata(self, model_dir: str, metadata_file: str):
        """從metadata文件載入模型"""
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # 獲取模型ID，如果沒有則生成新的
            model_id = metadata.get("model_id")
            if not model_id:
                model_id = str(uuid.uuid4())
                metadata["model_id"] = model_id
                # 更新metadata文件
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # 查找模型文件
            model_file_path = self._find_model_file(model_dir)
            if not model_file_path:
                print(f"在目錄 {model_dir} 中找不到模型文件")
                return
            
            # 確定模型格式
            model_format = self._determine_model_format(model_file_path)
            
            # 創建ModelInfo對象
            model = ModelInfo(
                id=model_id,
                name=metadata.get("display_name", os.path.basename(model_dir)),
                type=ModelType(metadata.get("type", "yolov8")),
                format=model_format,
                path=model_file_path,
                size_mb=os.path.getsize(model_file_path) / (1024 * 1024),
                created_at=self._parse_datetime(metadata.get("created_at")),
                description=metadata.get("description", f"模型: {metadata.get('display_name', os.path.basename(model_dir))}"),
                metadata=metadata
            )
            
            self.models[model_id] = model
            print(f"載入模型: {model.name} (ID: {model_id})")
            
        except Exception as e:
            print(f"載入metadata文件 {metadata_file} 時出錯: {str(e)}")

    def _detect_and_create_model(self, model_dir: str):
        """檢測模型目錄並創建模型對象（如果沒有metadata）"""
        try:
            # 查找模型文件
            model_file_path = self._find_model_file(model_dir)
            if not model_file_path:
                print(f"在目錄 {model_dir} 中未找到有效的模型文件")
                return None
            
            model_format = self._determine_model_format(model_file_path)
            
            # 特殊處理TensorRT Engine文件
            if model_format == ModelFormat.ENGINE:
                if not self._validate_tensorrt_engine(model_file_path):
                    print(f"TensorRT Engine文件驗證失敗: {model_file_path}")
                    return None
                
                # 確保同時存在.engine和.plan文件
                plan_path = os.path.join(os.path.dirname(model_file_path), "model.plan")
                
                # 如果原始文件是.engine，創建.plan副本用於Triton
                if model_file_path.endswith('.engine') and not os.path.exists(plan_path):
                    try:
                        import shutil
                        shutil.copy2(model_file_path, plan_path)
                        print(f"為Triton創建.plan文件: {plan_path}")
                    except Exception as e:
                        print(f"創建.plan文件失敗: {e}")
                
                # 如果原始文件是.plan，檢查是否有對應的.engine文件
                elif model_file_path.endswith('.plan'):
                    engine_path = model_file_path.replace('.plan', '.engine')
                    if not os.path.exists(engine_path):
                        try:
                            import shutil
                            shutil.copy2(model_file_path, engine_path)
                            print(f"為測試創建.engine文件: {engine_path}")
                            # 更新模型路徑指向.engine文件（用於測試）
                            model_file_path = engine_path
                        except Exception as e:
                            print(f"創建.engine文件失敗: {e}")
            
            # 特殊處理PyTorch模型文件
            elif model_format == ModelFormat.PT:
                # PyTorch模型處理 - 保留原始文件和TorchScript文件
                original_pt_path = model_file_path
                torchscript_path = os.path.join(os.path.dirname(model_file_path), "model.torchscript")
                
                # 檢查是否需要生成TorchScript文件
                if not os.path.exists(torchscript_path):
                    try:
                        self._convert_to_torchscript(model_file_path, torchscript_path)
                        print(f"為Triton創建TorchScript文件: {torchscript_path}")
                    except Exception as e:
                        print(f"TorchScript轉換失敗: {e}")
                        # 如果轉換失敗，嘗試檢查原始文件是否已經是TorchScript
                        if self._is_torchscript(model_file_path):
                            try:
                                import shutil
                                shutil.copy2(model_file_path, torchscript_path)
                                print(f"複製TorchScript文件: {torchscript_path}")
                            except Exception as e2:
                                print(f"複製TorchScript文件失敗: {e2}")
                
                # 保持模型路徑指向原始文件（用於YOLO測試）
            
            # 生成模型ID和名稱
            model_id = str(uuid.uuid4())
            model_name = os.path.basename(model_dir)
            
            # 創建模型對象
            model = ModelInfo(
                id=model_id,
                name=model_name,
                type=ModelType.YOLOV8,  # 默認類型
                format=model_format,
                path=model_file_path,
                size_mb=os.path.getsize(model_file_path) / (1024 * 1024),
                created_at=datetime.now(timezone.utc),
                description=f"檢測到的模型: {model_name}",
                metadata={
                    "model_id": model_id,
                    "display_name": model_name,
                    "type": "yolov8",
                    "format": model_format.value,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "is_trt_model": True,
                    "triton_model_name": model_name,
                    "triton_model_dir": model_dir,
                    "version": "1",
                    "platform": self._get_platform_from_format(model_format),
                    "auto_detected": True
                }
            )
            
            # 保存metadata
            self._write_metadata_file(model_dir, model)
            
            # 檢查或創建config.pbtxt
            config_path = os.path.join(model_dir, "config.pbtxt")
            if not os.path.exists(config_path) or model_format in [ModelFormat.ENGINE, ModelFormat.PT]:
                # 對於Engine模型、PT模型或缺少配置的模型，重新創建配置
                self._create_triton_config(model_dir, model)
            
            print(f"自動檢測並創建模型: {model_name} ({model_format.value})")
            return model
            
        except Exception as e:
            print(f"檢測模型目錄 {model_dir} 時出錯: {str(e)}")
            return None

    def _find_model_file(self, model_dir: str) -> Optional[str]:
        """在模型目錄中查找模型文件"""
        # 查找版本目錄（通常是1）
        version_dirs = ["1", "2", "3"]
        model_extensions = [".pt", ".onnx", ".engine", ".plan"]
        
        for version in version_dirs:
            version_dir = os.path.join(model_dir, version)
            if os.path.exists(version_dir):
                for file in os.listdir(version_dir):
                    file_path = os.path.join(version_dir, file)
                    if os.path.isfile(file_path):
                        for ext in model_extensions:
                            if file.endswith(ext):
                                return file_path
        
        # 如果版本目錄中沒找到，直接在模型目錄中查找
        for file in os.listdir(model_dir):
            file_path = os.path.join(model_dir, file)
            if os.path.isfile(file_path):
                for ext in model_extensions:
                    if file.endswith(ext):
                        return file_path
        
        return None

    def _determine_model_format(self, file_path: str) -> ModelFormat:
        """根據文件擴展名確定模型格式"""
        if file_path.endswith(".pt"):
            return ModelFormat.PT
        elif file_path.endswith(".onnx"):
            return ModelFormat.ONNX
        elif file_path.endswith(".engine") or file_path.endswith(".plan"):
            return ModelFormat.ENGINE
        else:
            return ModelFormat.PT  # 默認

    def _get_platform_from_format(self, model_format: ModelFormat) -> str:
        """根據模型格式獲取平台名稱"""
        if model_format == ModelFormat.PT:
            return "pytorch_libtorch"
        elif model_format == ModelFormat.ONNX:
            return "onnxruntime_onnx"
        elif model_format == ModelFormat.ENGINE:
            return "tensorrt_plan"
        else:
            return "unknown"

    def _get_model_directory(self, model_path: str) -> Optional[str]:
        """根據模型路徑獲取模型目錄"""
        # 模型路徑通常是 /path/to/model_repository/model_name/1/model.ext
        # 我們需要返回 /path/to/model_repository/model_name
        path_parts = model_path.replace("\\", "/").split("/")
        
        # 找到model_repository的位置
        try:
            repo_index = -1
            for i, part in enumerate(path_parts):
                if "model_repository" in part:
                    repo_index = i
                    break
            
            if repo_index >= 0 and repo_index + 1 < len(path_parts):
                # 模型目錄是model_repository後的第一個目錄
                model_dir_parts = path_parts[:repo_index + 2]
                return "/".join(model_dir_parts)
        except Exception as e:
            print(f"解析模型目錄路徑時出錯: {str(e)}")
        
        return None

    def _parse_datetime(self, datetime_str: Optional[str]) -> datetime:
        """解析日期時間字符串"""
        if not datetime_str:
            return datetime.now(timezone.utc)
        
        try:
            # 嘗試解析ISO格式
            return datetime.fromisoformat(datetime_str.replace("Z", "+00:00"))
        except:
            return datetime.now(timezone.utc)

    def _write_metadata_file(self, model_dir: str, model: ModelInfo):
        """寫入metadata文件"""
        metadata_file = os.path.join(model_dir, "metadata.json")
        
        # 構建metadata
        metadata = {
            "model_id": model.id,
            "display_name": model.name,
            "type": model.type.value,
            "format": model.format.value,
            "created_at": model.created_at.isoformat(),
            "description": model.description,
            "is_trt_model": True,
            "triton_model_name": model.name,
            "triton_model_dir": model_dir,
            "version": "1",
            "platform": self._get_platform_from_format(model.format)
        }
        
        # 合併現有metadata
        if model.metadata:
            metadata.update(model.metadata)
            # 確保關鍵字段不被覆蓋
            metadata["model_id"] = model.id
            metadata["display_name"] = model.name
            metadata["type"] = model.type.value
            metadata["format"] = model.format.value
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    async def refresh_models(self):
        """刷新模型列表"""
        self._scan_model_repository()
        return len(self.models)

    def create_model_from_upload(self, name: str, model_type: str, file_path: str, description: str = None) -> ModelInfo:
        """從上傳文件創建模型"""
        # 生成唯一ID
        model_id = str(uuid.uuid4())

        # 確定模型格式
        model_format = self._determine_model_format(file_path)

        # 模型名稱來自使用者輸入，先清洗再組目錄，落地前以 commonpath 複核（縱深防禦）
        name = sanitize_filename(name, default=f"model_{model_id.split('-')[0]}")

        # 創建模型目錄
        model_dir = os.path.join(self.model_repository_path, name)
        if not is_within(self.model_repository_path, model_dir):
            raise ValueError(f"模型名稱不安全，已拒絕建立目錄: {name}")
        version_dir = os.path.join(model_dir, "1")
        os.makedirs(version_dir, exist_ok=True)
        
        # 移動文件到版本目錄
        import shutil
        target_file = os.path.join(version_dir, f"model{os.path.splitext(file_path)[1]}")
        shutil.move(file_path, target_file)
        
        # 創建模型對象
        model = ModelInfo(
            id=model_id,
            name=name,
            type=ModelType(model_type),
            format=model_format,
            path=target_file,
            size_mb=os.path.getsize(target_file) / (1024 * 1024),
            created_at=datetime.now(timezone.utc),
            description=description or f"上傳的模型: {name}",
            metadata={
                "model_id": model_id,
                "display_name": name,
                "type": model_type,
                "format": model_format.value,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "is_trt_model": True,
                "triton_model_name": name,
                "triton_model_dir": model_dir,
                "version": "1",
                "platform": self._get_platform_from_format(model_format),
                "uploaded": True
            }
        )
        
        # 保存metadata
        self._write_metadata_file(model_dir, model)
        
        # 創建config.pbtxt
        self._create_triton_config(model_dir, model)
        
        # 添加到模型列表
        self.models[model_id] = model
        
        return model

    def _create_triton_config(self, model_dir: str, model: ModelInfo):
        """創建Triton配置文件"""
        config_path = os.path.join(model_dir, "config.pbtxt")
        
        # 獲取模型文件名
        model_filename = os.path.basename(model.path)
        
        # 根據格式設置平台
        platform = self._get_platform_from_format(model.format)
        
        # 為TensorRT Engine模型特殊處理
        if model.format == ModelFormat.ENGINE:
            # 確保同時存在.engine和.plan文件
            original_engine_path = model.path
            plan_filename = "model.plan"
            plan_path = os.path.join(os.path.dirname(model.path), plan_filename)
            
            # 如果原始文件是.engine，檢查是否有對應的.plan文件
            if model.path.endswith('.engine'):
                if not os.path.exists(plan_path):
                    print(f"警告：找不到對應的.plan文件: {plan_path}")
                    # 如果沒有.plan文件，嘗試創建一個副本
                    try:
                        import shutil
                        shutil.copy2(model.path, plan_path)
                        print(f"為Triton創建.plan文件: {plan_path}")
                    except Exception as e:
                        print(f"創建.plan文件失敗: {e}")
                        # 如果無法創建.plan文件，則使用.engine文件
                        plan_filename = "model.engine"
            
            # 如果原始文件是.plan，檢查是否有對應的.engine文件
            elif model.path.endswith('.plan'):
                engine_path = model.path.replace('.plan', '.engine')
                if not os.path.exists(engine_path):
                    try:
                        import shutil
                        shutil.copy2(model.path, engine_path)
                        print(f"為測試創建.engine文件: {engine_path}")
                        # 更新模型路徑指向.engine文件（用於測試）
                        model.path = engine_path
                    except Exception as e:
                        print(f"創建.engine文件失敗: {e}")
            
            # 從模型名稱中提取batch size信息
            batch_size = 1
            if "_batch" in model.name:
                try:
                    batch_part = model.name.split("_batch")[1].split("_")[0]
                    batch_size = int(batch_part)
                except (ValueError, IndexError):
                    batch_size = 1
            
            # TensorRT Engine模型的配置 - 使用.plan文件和原始batch size
            config_content = f"""name: "{model.name}"
platform: "tensorrt_plan"
max_batch_size: {batch_size}
input [
  {{
    name: "images"
    data_type: TYPE_FP32
    dims: [ 3, 640, 640 ]
  }}
]
output [
  {{
    name: "output0"
    data_type: TYPE_FP32
    dims: [ 84, 8400 ]
  }}
]
default_model_filename: "{plan_filename}"
instance_group [
  {{
    count: 1
    kind: KIND_GPU
  }}
]
"""
        elif model.format == ModelFormat.PT:
            # PyTorch模型處理 - 保留原始文件和TorchScript文件
            original_pt_path = model.path
            torchscript_filename = "model.torchscript"
            torchscript_path = os.path.join(os.path.dirname(model.path), torchscript_filename)
            
            # 檢查是否需要轉換為TorchScript
            if not os.path.exists(torchscript_path) and self._is_torchscript(model.path):
                # 原始文件已經是TorchScript格式，直接重命名
                try:
                    import shutil
                    shutil.copy2(model.path, torchscript_path)
                    print(f"為Triton創建TorchScript文件: {torchscript_path}")
                except Exception as e:
                    print(f"創建TorchScript文件失敗: {e}")
                    torchscript_filename = model_filename
            elif not os.path.exists(torchscript_path):
                # 需要轉換為TorchScript
                try:
                    self._convert_to_torchscript(model.path, torchscript_path)
                    print(f"轉換並創建TorchScript文件: {torchscript_path}")
                except Exception as e:
                    print(f"轉換TorchScript失敗: {e}")
                    torchscript_filename = model_filename
            
            # 從模型名稱中提取batch size信息
            batch_size = 1
            if "_batch" in model.name:
                try:
                    batch_part = model.name.split("_batch")[1].split("_")[0]
                    batch_size = int(batch_part)
                except (ValueError, IndexError):
                    batch_size = 1
            
            config_content = f"""name: "{model.name}"
platform: "pytorch_libtorch"
max_batch_size: {batch_size}
input [
  {{
    name: "images"
    data_type: TYPE_FP32
    dims: [ 3, 640, 640 ]
  }}
]
output [
  {{
    name: "output0"
    data_type: TYPE_FP32
    dims: [ 84, 8400 ]
  }}
]
default_model_filename: "{torchscript_filename}"
instance_group [
  {{
    count: 1
    kind: KIND_GPU
  }}
]
"""
        else:
            # 其他格式的配置 - 從模型名稱中提取batch size
            batch_size = 1
            if "_batch" in model.name:
                try:
                    batch_part = model.name.split("_batch")[1].split("_")[0]
                    batch_size = int(batch_part)
                except (ValueError, IndexError):
                    batch_size = 1
            
            # 動態檢測ONNX模型的輸出維度
            output_dims = "[ 84, 8400 ]"  # 默認為YOLOv8檢測模型
            if model.format == ModelFormat.ONNX:
                try:
                    # 嘗試檢測ONNX模型的實際輸出維度
                    detected_dims = self._detect_onnx_output_dims(model.path)
                    if detected_dims:
                        output_dims = detected_dims
                        print(f"檢測到ONNX模型輸出維度: {output_dims}")
                except Exception as e:
                    print(f"檢測ONNX輸出維度失敗，使用默認值: {e}")
            
            # 根據格式設置不同的輸入輸出維度
            if model.format == ModelFormat.ONNX:
                # ONNX模型：max_batch_size > 0，Triton會自動添加batch維度
                # 實際模型輸出是[batch, concat_dim, anchors]，配置中只需要[concat_dim, anchors]
                config_content = f"""name: "{model.name}"
platform: "{platform}"
max_batch_size: {batch_size}
input [
  {{
    name: "images"
    data_type: TYPE_FP32
    dims: [ 3, 640, 640 ]
  }}
]
output [
  {{
    name: "output0"
    data_type: TYPE_FP32
    dims: {output_dims}
  }}
]
default_model_filename: "{model_filename}"
instance_group [
  {{
    count: 1
    kind: KIND_GPU
  }}
]
"""
            else:
                # 其他格式模型的通用配置
                config_content = f"""name: "{model.name}"
platform: "{platform}"
max_batch_size: {batch_size}
input [
  {{
    name: "images"
    data_type: TYPE_FP32
    dims: [ 3, 640, 640 ]
  }}
]
output [
  {{
    name: "output0"
    data_type: TYPE_FP32
    dims: [ 84, 8400 ]
  }}
]
default_model_filename: "{model_filename}"
instance_group [
  {{
    count: 1
    kind: KIND_GPU
  }}
]
"""
        
        # 寫入配置文件
        try:
            with open(config_path, "w", encoding="utf-8") as config_file:
                config_file.write(config_content)
            print(f"創建Triton配置文件: {config_path}")
            print(f"使用的max_batch_size: {batch_size}")
        except Exception as e:
            print(f"寫入配置文件失敗: {e}")
            raise

    def _is_torchscript(self, model_path: str) -> bool:
        """檢查模型是否已經是TorchScript格式"""
        try:
            import torch
            # 嘗試載入為TorchScript模型
            torch.jit.load(model_path, map_location='cpu')
            return True
        except Exception:
            return False
    
    def _convert_to_torchscript(self, source_path: str, target_path: str):
        """將PyTorch模型轉換為TorchScript"""
        try:
            import torch
            from ultralytics import YOLO
            
            # 使用YOLO載入模型
            model = YOLO(source_path)
            
            # 轉換為TorchScript
            model.export(format='torchscript', dynamic=False, simplify=True)
            
            # 查找生成的TorchScript文件
            import os
            import glob
            base_name = os.path.splitext(source_path)[0]
            torchscript_files = glob.glob(f"{base_name}*.torchscript") + glob.glob(f"{base_name}*.torchscript.pt")
            
            if torchscript_files:
                # 移動到目標位置
                import shutil
                shutil.move(torchscript_files[0], target_path)
                print(f"TorchScript轉換成功: {target_path}")
            else:
                raise Exception("未找到轉換後的TorchScript文件")
                
        except Exception as e:
            print(f"TorchScript轉換失敗: {e}")
            raise

    def _validate_tensorrt_engine(self, engine_path: str) -> bool:
        """驗證TensorRT Engine文件是否有效"""
        try:
            # 檢查文件是否存在且不為空
            if not os.path.exists(engine_path) or os.path.getsize(engine_path) == 0:
                return False
            
            # 簡單的文件頭檢查
            with open(engine_path, 'rb') as f:
                header = f.read(8)
                # TensorRT engine文件應該有特定的文件頭
                if len(header) < 8:
                    return False
            
            return True
        except Exception as e:
            print(f"驗證TensorRT Engine失敗: {e}")
            return False

    def _detect_onnx_output_dims(self, model_path: str) -> Optional[str]:
        """檢測ONNX模型的實際輸出維度"""
        try:
            import onnx
            
            # 載入ONNX模型
            model = onnx.load(model_path)
            
            # 獲取輸出信息
            output_info = model.graph.output[0]  # 假設只有一個輸出
            output_shape = output_info.type.tensor_type.shape
            
            dims = []
            for i, dim in enumerate(output_shape.dim):
                if i == 0:  # 跳過batch維度
                    continue
                    
                if dim.dim_value:
                    dims.append(str(dim.dim_value))
                else:
                    dims.append("-1")  # 動態維度
            
            if len(dims) >= 2:
                result = f"[ {', '.join(dims)} ]"
                print(f"檢測到ONNX模型原始輸出維度: {[d.dim_value if d.dim_value else -1 for d in output_shape.dim]}")
                print(f"轉換為Triton配置維度: {result}")
                return result
            else:
                return None
                
        except ImportError:
            print("警告: 未安裝onnx，無法檢測模型維度")
            return None
        except Exception as e:
            print(f"檢測ONNX模型維度時出錯: {e}")
            return None 