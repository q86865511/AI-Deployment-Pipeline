import os
import time
import json
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import uuid
from datetime import datetime, timezone, timedelta
import cv2
import io
import asyncio
import torch
import gc
import subprocess
import shutil
from app.services.model_service import ModelService
import tensorrt as trt
import concurrent.futures
from functools import partial
import multiprocessing
from queue import Queue
import threading

from app.models import ModelInfo, ModelPerformance, PrecisionType, ModelType
from app.services.model_tasks import task_from_model_type
from app.services.comparison import generate_comparison as compute_comparison

# 單一組態的效能量測重複次數上限：避免 12 組矩陣中的單一組合長時間佔用 GPU。
# 使用者若填入更大的值會被截到此上限（會印出警告，結果檔記錄實際執行次數）。
# 可用環境變數 MAX_BENCHMARK_ITERATIONS 調整。
MAX_BENCHMARK_ITERATIONS = int(os.environ.get("MAX_BENCHMARK_ITERATIONS", "10"))


def _resolve_iterations(requested: int) -> int:
    """把使用者請求的迭代次數收斂到上限，並在被截斷時明確告知。"""
    effective = max(1, min(int(requested), MAX_BENCHMARK_ITERATIONS))
    if effective < requested:
        print(
            f"警告: 請求的迭代次數 {requested} 超過上限 "
            f"MAX_BENCHMARK_ITERATIONS={MAX_BENCHMARK_ITERATIONS}，實際執行 {effective} 次"
        )
    return effective


def _sample_std(values: List[float]) -> float:
    """樣本標準差（ddof=1）；樣本數不足 2 時回傳 0.0 避免 nan。"""
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1))


def _get_gpu_memory_usage() -> float:
    """獲取GPU記憶體使用量 (MB) - 多種備用方法"""
    try:
        # 方法1: 使用GPUtil
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            return gpus[0].memoryUsed
    except ImportError:
        pass
    
    try:
        # 方法2: 使用nvidia-smi
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=memory.used', 
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=2)
        
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception:
        pass
    
    # 方法3: 使用torch (備用)
    if torch.cuda.is_available():
        try:
            return torch.cuda.memory_allocated() / 1024**2
        except Exception:
            pass
    
    return 0

def _get_gpu_utilization() -> float:
    """獲取GPU使用率 (%) - 多種備用方法"""
    try:
        # 方法1: 使用pynvml
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return util.gpu
    except ImportError:
        pass
    except Exception:
        pass
    
    try:
        # 方法2: 使用GPUtil
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            return gpus[0].load * 100
    except ImportError:
        pass
    except Exception:
        pass
    
    try:
        # 方法3: 使用nvidia-smi
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=utilization.gpu', 
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=2)
        
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception:
        pass
    
    try:
        # 方法4: 使用torch (如果可用)
        return torch.cuda.utilization()
    except Exception:
        pass
    
    return 0.0  # 如果所有方法都失敗，返回0

class GPUMonitor:
    """GPU使用率監控器 - 持續監控並統計"""
    
    def __init__(self, sample_interval: float = 0.02):
        self.sample_interval = sample_interval
        self.monitoring = False
        self.utilization_readings = []
        self.memory_readings = []
        self.thread = None
        self.start_time = None
    
    def start_monitoring(self):
        """開始監控"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.utilization_readings = []
        self.memory_readings = []
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def stop_monitoring(self):
        """停止監控"""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=2.0)
    
    def _monitor_loop(self):
        """監控主迴圈"""
        while self.monitoring:
            try:
                timestamp = time.time()
                utilization = _get_gpu_utilization()
                memory_usage = _get_gpu_memory_usage()
                
                if utilization >= 0:
                    self.utilization_readings.append({
                        "timestamp": timestamp,
                        "utilization": utilization
                    })
                
                if memory_usage >= 0:
                    self.memory_readings.append({
                        "timestamp": timestamp,
                        "memory_mb": memory_usage
                    })
                    
            except Exception:
                pass
            
            time.sleep(self.sample_interval)
    
    def get_statistics(self) -> Dict:
        """獲取統計數據"""
        stats = {
            "samples": len(self.utilization_readings),
            "duration_s": (time.time() - self.start_time) if self.start_time else 0
        }
        
        if self.utilization_readings:
            utilizations = [r["utilization"] for r in self.utilization_readings]
            stats.update({
                "avg_utilization": np.mean(utilizations),
                "max_utilization": np.max(utilizations),
                "min_utilization": np.min(utilizations),
                "p95_utilization": np.percentile(utilizations, 95) if len(utilizations) > 1 else utilizations[0]
            })
        
        if self.memory_readings:
            memories = [r["memory_mb"] for r in self.memory_readings]
            stats.update({
                "avg_memory_mb": np.mean(memories),
                "max_memory_mb": np.max(memories),
                "min_memory_mb": np.min(memories)
            })
        
        return stats

class InferenceService:
    """
    模型推理服務，負責模型推理和性能測試
    """
    def __init__(self):
        """初始化推理服務"""
        # 移除單次測試資料夾，只保留自動化模型性能評估功能
        self.model_service = ModelService()
        
        # 創建執行緒池用於YOLO操作
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
        # 設定默認GPU ID
        self.gpu_id = 0
    
    def _get_task_from_model_type(self, model_type: ModelType) -> str:
        """根據模型類型確定YOLO task參數（委派至 model_tasks）"""
        return task_from_model_type(model_type)

    def predict(self, model: ModelInfo, image_path: str, confidence: float = 0.25,
               iou_threshold: float = 0.45) -> Dict[str, Any]:
        """
        使用指定模型對圖片進行預測
        """
        try:
            print(f"使用模型 {model.id} ({model.format.value}) 對圖片 {image_path} 進行預測")
            
            # 讀取圖片
            image = cv2.imread(image_path)
            if image is None:
                raise Exception(f"無法讀取圖片: {image_path}")
            
            # 根據模型格式選擇推理方法
            results = None
            if model.format.value == "pt":
                results = self._predict_with_pytorch(model, image, confidence, iou_threshold)
            elif model.format.value == "onnx":
                results = self._predict_with_onnx(model, image, confidence, iou_threshold)
            elif model.format.value == "engine":
                results = self._predict_with_tensorrt(model, image, confidence, iou_threshold)
            else:
                raise Exception(f"不支持的模型格式: {model.format.value}")
            
            return results
        
        except Exception as e:
            print(f"預測時出錯: {str(e)}")
            raise
    
    def _predict_with_pytorch(self, model: ModelInfo, image: np.ndarray,
                            confidence: float, iou_threshold: float) -> Dict[str, Any]:
        """使用PyTorch模型進行預測"""
        try:
            from ultralytics import YOLO
            
            # 加載模型，設置task
            task = self._get_task_from_model_type(model.type)
            yolo_model = YOLO(model.path, task=task)
            
            # 執行預測
            results = yolo_model(image, conf=confidence, iou=iou_threshold)[0]
            
            # 繪製結果
            plot_image = results.plot()
            
            # 將結果轉換為可序列化格式
            detections = []
            for box in results.boxes:
                detection = {
                    "bbox": box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                    "confidence": float(box.conf[0]),
                    "class_id": int(box.cls[0]),
                    "class_name": results.names[int(box.cls[0])]
                }
                detections.append(detection)
            
            # 將圖片轉換為bytes
            success, encoded_image = cv2.imencode('.jpg', plot_image)
            if not success:
                raise Exception("無法編碼結果圖片")
            
            result = {
                "detections": detections,
                "image": encoded_image.tobytes(),
                "model_id": model.id,
                "model_type": model.type.value,
                "inference_time_ms": results.speed["inference"]
            }
            
            # 釋放模型資源
            del yolo_model
            self._cleanup_gpu_resources()
            
            return result
        
        except Exception as e:
            # 確保資源被釋放
            self._cleanup_gpu_resources()
            raise Exception(f"PyTorch推理失敗: {str(e)}")
    
    def _predict_with_onnx(self, model: ModelInfo, image: np.ndarray,
                           confidence: float, iou_threshold: float) -> Dict[str, Any]:
        """使用ONNX模型進行預測"""
        try:
            import onnxruntime as ort
            from ultralytics.utils.ops import non_max_suppression
            
            # 獲取模型類型相關信息
            is_pose = model.type.value == "yolov8-pose"
            is_seg = model.type.value == "yolov8-seg"
            
            # 準備圖片
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_height, img_width = img.shape[:2]
            input_height, input_width = 640, 640  # 默認輸入尺寸
            
            # 預處理
            img_processed = cv2.resize(img, (input_width, input_height))
            img_processed = img_processed.astype(np.float32) / 255.0
            img_processed = img_processed.transpose(2, 0, 1)  # HWC -> CHW
            img_processed = np.expand_dims(img_processed, axis=0)  # 添加batch維度
            
            # 創建推理會話
            sess = ort.InferenceSession(model.path)
            input_name = sess.get_inputs()[0].name
            
            # 計時
            start_time = time.time()
            
            # 執行推理
            outputs = sess.run(None, {input_name: img_processed})
            
            # 後處理
            if is_pose:
                # 處理姿態模型輸出
                boxes_kpts = outputs[0]
                results = non_max_suppression(boxes_kpts, confidence, iou_threshold, nc=1, nkpt=17)
            elif is_seg:
                # 處理分割模型輸出
                boxes, masks = outputs[0], outputs[1]
                results = non_max_suppression(boxes, confidence, iou_threshold)
                # 需要額外處理掩碼
                proto = outputs[1]
            else:
                # 標準目標檢測
                results = non_max_suppression(outputs[0], confidence, iou_threshold)
            
            # 計算推理時間
            inference_time_ms = (time.time() - start_time) * 1000
            
            # 將結果轉換回原始圖片尺寸
            scale_x, scale_y = img_width / input_width, img_height / input_height
            
            # 構建結果
            result_img = image.copy()
            detections = []
            
            # 如果有任何檢測結果
            if len(results) > 0 and len(results[0]) > 0:
                for det in results[0]:
                    # 還原到原始圖片尺寸
                    x1, y1, x2, y2 = det[:4]
                    x1, x2 = x1 * scale_x, x2 * scale_x
                    y1, y2 = y1 * scale_y, y2 * scale_y
                    
                    # 繪製邊界框
                    cv2.rectangle(result_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # 添加置信度文字
                    conf = float(det[4])
                    class_id = int(det[5])
                    cv2.putText(result_img, f"{class_id}: {conf:.2f}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # 保存檢測結果
                    detection = {
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": conf,
                        "class_id": class_id,
                        "class_name": str(class_id)  # 這裡需要一個類別映射
                    }
                    detections.append(detection)
            
            # 將圖片轉換為bytes
            success, encoded_image = cv2.imencode('.jpg', result_img)
            if not success:
                raise Exception("無法編碼結果圖片")
            
            return {
                "detections": detections,
                "image": encoded_image.tobytes(),
                "model_id": model.id,
                "model_type": model.type.value,
                "inference_time_ms": inference_time_ms
            }
        
        except Exception as e:
            raise Exception(f"ONNX推理失敗: {str(e)}")
    
    def _predict_with_tensorrt(self, model: ModelInfo, image: np.ndarray,
                              confidence: float, iou_threshold: float) -> Dict[str, Any]:
        """使用TensorRT引擎進行推理"""
        try:
            import tensorrt as trt
            import torch
            from ultralytics.utils.ops import non_max_suppression
            
            # 確保有GPU可用
            if not torch.cuda.is_available():
                raise Exception("TensorRT推理需要GPU")
            
            # 準備圖片
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_height, img_width = img.shape[:2]
            input_height, input_width = 640, 640  # 默認輸入尺寸
            
            # 預處理
            img_processed = cv2.resize(img, (input_width, input_height))
            img_processed = img_processed.astype(np.float32) / 255.0
            img_processed = img_processed.transpose(2, 0, 1)  # HWC -> CHW
            img_processed = np.expand_dims(img_processed, axis=0)  # 添加batch維度
            
            # 獲取引擎路徑
            engine_path = model.path
            
            # 如果路徑不存在，嘗試在模型目錄中查找
            if not os.path.exists(engine_path):
                model_dir = os.path.dirname(model.path)
                print(f"引擎路徑不存在: {engine_path}，嘗試在目錄 {model_dir} 中查找")
                
                # 查找 .engine 或 .plan 文件
                engine_files = []
                if os.path.exists(model_dir):
                    for file in os.listdir(model_dir):
                        if file.endswith(".engine") or file.endswith(".plan"):
                            engine_files.append(os.path.join(model_dir, file))
                
                if engine_files:
                    engine_path = engine_files[0]
                    print(f"找到引擎文件: {engine_path}")
                else:
                    raise Exception(f"在目錄 {model_dir} 中找不到TensorRT引擎文件")
            
            if not engine_path or not os.path.exists(engine_path):
                raise Exception(f"找不到TensorRT引擎文件: {engine_path}")
            
            # 加載引擎
            try:
                print(f"正在加載TensorRT引擎: {engine_path}")
                with open(engine_path, "rb") as f:
                    engine_data = f.read()
                
                if not engine_data:
                    raise Exception("引擎文件為空")
                
                logger = trt.Logger(trt.Logger.WARNING)
                runtime = trt.Runtime(logger)
                engine = runtime.deserialize_cuda_engine(engine_data)
                
                if engine is None:
                    raise Exception("無法反序列化引擎，可能是引擎文件損壞或版本不兼容")
                
                context = engine.create_execution_context()
                if context is None:
                    raise Exception("無法創建執行上下文")
                    
            except Exception as e:
                raise Exception(f"加載TensorRT引擎失敗: {str(e)}")
            
            # 使用torch來處理GPU內存和同步
            device = torch.device('cuda:0')
            
            # 準備輸入張量
            input_idx = engine.get_binding_index("images")
            output_idx = engine.get_binding_index("output0")
            
            # 獲取輸入輸出形狀
            input_shape = engine.get_binding_shape(input_idx)
            output_shape = engine.get_binding_shape(output_idx)
            
            # 使用torch分配GPU內存
            input_tensor = torch.from_numpy(img_processed).to(device)
            output_tensor = torch.empty(output_shape, dtype=torch.float32, device=device)
            
            # 創建緩衝區指針列表
            bindings = [input_tensor.data_ptr(), output_tensor.data_ptr()]
            
            # 測量推理時間
            torch.cuda.synchronize()
            start_time = time.time()
            
            # 執行推理
            context.execute_v2(bindings=bindings)
            
            # 同步並計算時間
            torch.cuda.synchronize()
            inference_time_ms = (time.time() - start_time) * 1000
            
            # 將輸出從GPU複製到CPU
            output_data = output_tensor.cpu().numpy()
            
            # 後處理 - 使用非極大值抑制
            results = non_max_suppression(torch.from_numpy(output_data), confidence, iou_threshold)
            
            # 將結果轉換回原始圖片尺寸
            scale_x, scale_y = img_width / input_width, img_height / input_height
            
            # 構建結果
            result_img = image.copy()
            detections = []
            
            # 如果有任何檢測結果
            if len(results) > 0 and len(results[0]) > 0:
                for det in results[0]:
                    # 還原到原始圖片尺寸
                    x1, y1, x2, y2 = det[:4]
                    x1, x2 = x1 * scale_x, x2 * scale_x
                    y1, y2 = y1 * scale_y, y2 * scale_y
                    
                    # 繪製邊界框
                    cv2.rectangle(result_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # 添加置信度文字
                    conf = float(det[4])
                    class_id = int(det[5]) if len(det) > 5 else 0
                    cv2.putText(result_img, f"{class_id}: {conf:.2f}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # 保存檢測結果
                    detection = {
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": conf,
                        "class_id": class_id,
                        "class_name": str(class_id)  # 這裡需要一個類別映射
                    }
                    detections.append(detection)
            
            # 將圖片轉換為bytes
            success, encoded_image = cv2.imencode('.jpg', result_img)
            if not success:
                raise Exception("無法編碼結果圖片")
            
            result = {
                "detections": detections,
                "image": encoded_image.tobytes(),
                "model_id": model.id,
                "model_type": model.type.value,
                "inference_time_ms": inference_time_ms
            }
            
            # 釋放資源
            del context
            del engine
            self._cleanup_gpu_resources()
            
            return result
        
        except Exception as e:
            raise Exception(f"TensorRT推理失敗: {str(e)}")

    def _get_device_info(self) -> Dict[str, Any]:
        """獲取設備信息"""
        device_info = {
            "platform": "unknown",
            "device": "unknown",
            "cuda_version": "unknown"
        }
        
        try:
            import platform
            device_info["platform"] = platform.platform()
            
            try:
                import torch
                if torch.cuda.is_available():
                    device_info["device"] = torch.cuda.get_device_name(0)
                    device_info["cuda_version"] = torch.version.cuda
            except ImportError:
                pass
            
            try:
                import tensorrt as trt
                device_info["tensorrt_version"] = trt.__version__
            except ImportError:
                pass
        
        except Exception as e:
            print(f"獲取設備信息時出錯: {str(e)}")
        
        return device_info

    def generate_comparison(self, results: List[ModelPerformance]) -> Dict[str, Any]:
        """生成模型性能比較結果（委派至 comparison）"""
        return compute_comparison(results)

    async def validate_model(self, model_id: str, dataset_id: str, batch_size: int = 1, custom_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        在指定數據集上驗證模型性能
        
        Args:
            model_id: 模型ID
            dataset_id: 數據集ID
            batch_size: 批次大小
            custom_params: 自定義參數
        """
        from app.services.model_service import ModelService
        model_service = ModelService()
        
        try:
            # 獲取模型信息
            model = model_service.get_model_by_id(model_id)
            if not model:
                raise Exception(f"找不到模型: {model_id}")
            
            # 對於Engine模型，從metadata中提取實際的batch_size
            actual_batch_size = batch_size
            if model.format.value == "engine" and model.metadata:
                # 從模型名稱中提取batch size
                model_name = model.metadata.get("display_name", model.name)
                import re
                batch_match = re.search(r'batch(\d+)', model_name)
                if batch_match:
                    actual_batch_size = int(batch_match.group(1))
                    print(f"從模型metadata中提取batch size: {actual_batch_size}")
            
            # 獲取數據集信息
            datasets_dir = os.path.join(os.getcwd(), "data", "datasets")
            datasets_metadata_path = os.path.join(datasets_dir, "datasets.json")
            
            dataset_name = None
            dataset_path = None
            if os.path.exists(datasets_metadata_path):
                with open(datasets_metadata_path, 'r', encoding='utf-8') as f:
                    datasets = json.load(f)
                    dataset_info = next((d for d in datasets if d["id"] == dataset_id), None)
                    if dataset_info:
                        dataset_name = dataset_info["name"]
                        dataset_path = dataset_info["path"]
            
            if not dataset_name:
                raise Exception(f"找不到數據集: {dataset_id}")
            
            print(f"開始在數據集 {dataset_name} 上驗證模型 {model.name}，使用batch size: {actual_batch_size}")
            
            # 根據模型格式進行不同的驗證
            if model.format.value == "pt":
                # 對於PT模型，直接驗證
                return await self._validate_pytorch_model(model, dataset_id, dataset_name, dataset_path, actual_batch_size, custom_params)
            elif model.format.value in ["onnx", "engine"]:
                # 對於轉換後的模型，需要找到源PyTorch模型
                source_model = None
                
                # 策略1: 通過metadata中的source_model_name查找
                if model.metadata and model.metadata.get("source_model_name"):
                    source_name = model.metadata["source_model_name"]
                    source_model = model_service.get_model_by_name(source_name)
                    if source_model and source_model.format.value == "pt":
                        print(f"通過source_model_name找到源模型: {source_model.name}")
                
                # 策略2: 通過模型名稱推斷源模型名稱
                if not source_model:
                    # 從轉換後的模型名稱中提取源模型名稱
                    # 例如: test_engine_fp32_batch1_size640 -> test
                    model_name_parts = model.name.split('_')
                    if len(model_name_parts) >= 1:
                        # 取第一部分作為源模型名稱
                        potential_source_name = model_name_parts[0]
                        source_model = model_service.get_model_by_name(potential_source_name)
                        if source_model and source_model.format.value == "pt":
                            print(f"通過名稱推斷找到源模型: {source_model.name}")
                
                # 策略3: 查找同類型的PT模型
                if not source_model:
                    all_models = model_service.get_models(skip=0, limit=1000)
                    for candidate in all_models:
                        if (candidate.format.value == "pt" and 
                            candidate.type == model.type and
                            candidate.name != model.name):
                            source_model = candidate
                            print(f"通過類型匹配找到源模型: {source_model.name}")
                            break
                
                if not source_model:
                    raise Exception(f"找不到源PyTorch模型進行驗證。模型: {model.name}")
                
                # 對於engine格式，使用YOLO直接載入和驗證
                if model.format.value == "engine":
                    return await self._validate_engine_model(model, source_model, dataset_id, dataset_name, dataset_path, actual_batch_size, custom_params)
                else:
                    return await self._validate_converted_model(model, source_model, dataset_id, dataset_name, dataset_path, actual_batch_size, custom_params)
            else:
                raise Exception(f"不支持的模型格式: {model.format}")
                
        except Exception as e:
            print(f"模型驗證失敗: {str(e)}")
            # 返回錯誤結果而不是模擬結果
            return {
                "model_id": model_id,
                "model_name": model.name if model else "unknown",
                "dataset_id": dataset_id,
                "dataset_name": dataset_name or "unknown",
                "batch_size": actual_batch_size if 'actual_batch_size' in locals() else batch_size,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
                "status": "failed"
            }
    
    async def _validate_pytorch_model(self, model: ModelInfo, dataset_id: str, dataset_name: str, dataset_path: str, batch_size: int, custom_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """驗證PyTorch模型"""
        try:
            # 查找數據集的YAML配置文件
            yaml_file = self._find_dataset_yaml(dataset_name, dataset_path)
            if not yaml_file:
                raise Exception(f"找不到數據集 {dataset_name} 的YAML配置文件")
            
            # 使用執行緒池運行YOLO驗證操作（非阻塞），包含VRAM測量
            validation_params = {
                "data": yaml_file,
                "device": 0 if torch.cuda.is_available() else "cpu",  # 使用GPU進行驗證
                "batch": batch_size,
                "verbose": custom_params.get("verbose", False) if custom_params else False
            }
            validation_func = partial(self._run_pytorch_yolo_validation, model.path, model.type, validation_params)
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor, validation_func
            )
            
            # 從結果中獲取metrics和GPU使用量信息
            metrics = result['metrics']
            memory_usage_mb = result['memory_usage_mb']
            avg_gpu_load = result['avg_gpu_load']
            max_gpu_load = result.get('max_gpu_load', 0.0)
            monitoring_samples = result.get('monitoring_samples', 0)
            model_vram_mb = result.get('model_vram_mb', 0.0)
            monitoring_duration_s = result.get('monitoring_duration_s', 0.0)
            
            # 獲取驗證結果 - 使用results_dict和屬性兩種方式
            validation_results = {}
            
            # 方式1: 直接從metrics物件取得屬性值
            if hasattr(metrics, "pose"):
                validation_results["metrics/mAP50(P)"] = float(metrics.pose.map50)
                validation_results["metrics/mAP50-95(P)"] = float(metrics.pose.map)
                validation_results["metrics/precision(P)"] = float(getattr(metrics.pose, 'p', 0.0))
                validation_results["metrics/recall(P)"] = float(getattr(metrics.pose, 'r', 0.0))
            elif hasattr(metrics, "box"):
                validation_results["metrics/mAP50(B)"] = float(metrics.box.map50)
                validation_results["metrics/mAP50-95(B)"] = float(metrics.box.map)
                validation_results["metrics/precision(B)"] = float(getattr(metrics.box, 'p', 0.0))
                validation_results["metrics/recall(B)"] = float(getattr(metrics.box, 'r', 0.0))
            
            # 方式2: 從results_dict取得值（如果available）
            if hasattr(metrics, 'results_dict') and metrics.results_dict:
                validation_results.update(metrics.results_dict)
            
            # 添加GPU使用量信息到驗證結果
            validation_results["memory_usage_mb"] = memory_usage_mb
            validation_results["avg_gpu_load"] = avg_gpu_load
            validation_results["max_gpu_load"] = max_gpu_load
            validation_results["monitoring_samples"] = monitoring_samples
            validation_results["model_vram_mb"] = model_vram_mb
            validation_results["monitoring_duration_s"] = monitoring_duration_s
            
            # 格式化結果
            return self._format_validation_results(model.id, model.name, dataset_id, dataset_name, batch_size, validation_results)
            
        except Exception as e:
            print(f"PyTorch模型驗證失敗: {str(e)}")
            raise
    
    async def _validate_converted_model(self, model: ModelInfo, source_model: ModelInfo, dataset_id: str, dataset_name: str, dataset_path: str, batch_size: int, custom_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """驗證轉換後的模型（ONNX/TensorRT）"""
        try:
            # 查找數據集的YAML配置文件
            yaml_file = self._find_dataset_yaml(dataset_name, dataset_path)
            if not yaml_file:
                raise Exception(f"找不到數據集 {dataset_name} 的YAML配置文件")
                
            # 準備驗證參數 - 使用數字設備名稱
            device = "cpu"  # 默認使用CPU
            if torch.cuda.is_available():
                device = 0  # 如果有GPU，使用第一個GPU
                
            validation_params = {
                "batch": batch_size,
                "imgsz": custom_params.get("imgsz", 640) if custom_params else 640,
                "conf": custom_params.get("conf", 0.25) if custom_params else 0.25,
                "iou": custom_params.get("iou", 0.7) if custom_params else 0.7,
                "verbose": custom_params.get("verbose", False) if custom_params else False,
                "device": device,  # 使用數字設備名稱
            }
            
            # 使用源PyTorch模型加載，但使用轉換後的推理引擎
            from ultralytics import YOLO
            
            # 首先用源模型創建驗證器，設置task
            task = self._get_task_from_model_type(source_model.type)
            yolo_model = YOLO(source_model.path, task=task)
                
            # 然後設置使用轉換後的模型進行推理
            if model.format.value == "onnx":
                validation_params["format"] = "onnx"
                validation_params["model"] = model.path
            elif model.format.value == "engine":
                validation_params["format"] = "engine"
                validation_params["model"] = model.path
                
            # 使用執行緒池運行YOLO驗證操作（非阻塞）
            validation_func = partial(self._run_yolo_validation_with_source, source_model.path, source_model.type, model.path, model.format.value, yaml_file, validation_params)
            validation_results = await asyncio.get_event_loop().run_in_executor(
                self.executor, validation_func
            )
            
            # 格式化結果
            return self._format_validation_results(model.id, model.name, dataset_id, dataset_name, batch_size, validation_results)
            
        except Exception as e:
            print(f"轉換模型驗證失敗: {str(e)}")
            raise
    
    def _find_dataset_yaml(self, dataset_name: str, dataset_path: str) -> str:
        """查找數據集的YAML配置文件"""
        # 可能的YAML文件名
        possible_yaml_names = [
            f"{dataset_name}.yaml",
            f"{dataset_name}_pose.yaml",
            f"{dataset_name}_detect.yaml",
            "data.yaml",
            "dataset.yaml"
        ]
        
        # 可能的搜索路徑
        search_paths = []
        if dataset_path:
            search_paths.append(dataset_path)
            search_paths.append(os.path.dirname(dataset_path))
        
        datasets_dir = os.path.join(os.getcwd(), "data", "datasets")
        search_paths.append(datasets_dir)
        
        for search_path in search_paths:
            if not search_path or not os.path.exists(search_path):
                continue
                
            for yaml_name in possible_yaml_names:
                yaml_path = os.path.join(search_path, yaml_name)
                if os.path.exists(yaml_path):
                    print(f"找到數據集YAML文件: {yaml_path}")
                    return yaml_path
        
        # 如果找不到，返回None
        return None
    
    def _format_validation_results(self, model_id: str, model_name: str, dataset_id: str, dataset_name: str, batch_size: int, validation_results: dict) -> Dict[str, Any]:
        """格式化驗證結果"""
        formatted_results = {
            "model_id": model_id,
            "model_name": model_name,
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "batch_size": batch_size,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": {}
        }
        
        # 提取關鍵指標
        if validation_results:
            # 檢測指標 (Box detection)
            if "metrics/precision(B)" in validation_results:
                formatted_results["metrics"]["precision"] = float(validation_results["metrics/precision(B)"])
            
            if "metrics/recall(B)" in validation_results:
                formatted_results["metrics"]["recall"] = float(validation_results["metrics/recall(B)"])
            
            if "metrics/mAP50(B)" in validation_results:
                formatted_results["metrics"]["mAP50"] = float(validation_results["metrics/mAP50(B)"])
            
            if "metrics/mAP50-95(B)" in validation_results:
                formatted_results["metrics"]["mAP50_95"] = float(validation_results["metrics/mAP50-95(B)"])
            
            # 姿態指標 (Pose detection)
            if "metrics/precision(P)" in validation_results:
                formatted_results["metrics"]["precision"] = float(validation_results["metrics/precision(P)"])
            
            if "metrics/recall(P)" in validation_results:
                formatted_results["metrics"]["recall"] = float(validation_results["metrics/recall(P)"])
            
            if "metrics/mAP50(P)" in validation_results:
                formatted_results["metrics"]["mAP50"] = float(validation_results["metrics/mAP50(P)"])
            
            if "metrics/mAP50-95(P)" in validation_results:
                formatted_results["metrics"]["mAP50_95"] = float(validation_results["metrics/mAP50-95(P)"])
            
            # 速度指標
            if "speed/inference(ms)" in validation_results:
                formatted_results["metrics"]["inference_time_ms"] = float(validation_results["speed/inference(ms)"])
            
            if "speed/preprocess(ms)" in validation_results:
                formatted_results["metrics"]["preprocess_time_ms"] = float(validation_results["speed/preprocess(ms)"])
            
            if "speed/postprocess(ms)" in validation_results:
                formatted_results["metrics"]["postprocess_time_ms"] = float(validation_results["speed/postprocess(ms)"])
            
            # GPU使用量信息
            if "memory_usage_mb" in validation_results:
                formatted_results["memory_usage_mb"] = float(validation_results["memory_usage_mb"])
            
            if "avg_gpu_load" in validation_results:
                formatted_results["avg_gpu_load"] = float(validation_results["avg_gpu_load"])
            
            if "max_gpu_load" in validation_results:
                formatted_results["max_gpu_load"] = float(validation_results["max_gpu_load"])
            
            if "monitoring_samples" in validation_results:
                formatted_results["monitoring_samples"] = int(validation_results["monitoring_samples"])
            
            if "model_vram_mb" in validation_results:
                formatted_results["model_vram_mb"] = float(validation_results["model_vram_mb"])
            
            if "monitoring_duration_s" in validation_results:
                formatted_results["monitoring_duration_s"] = float(validation_results["monitoring_duration_s"])
        
        return formatted_results

    async def _validate_engine_model(self, model: ModelInfo, source_model: ModelInfo, dataset_id: str, dataset_name: str, dataset_path: str, batch_size: int, custom_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """驗證TensorRT Engine模型"""
        try:
            # 查找數據集的YAML配置文件
            yaml_file = self._find_dataset_yaml(dataset_name, dataset_path)
            if not yaml_file:
                raise Exception(f"找不到數據集 {dataset_name} 的YAML配置文件")
            
            # 使用執行緒池運行YOLO驗證操作（非阻塞），包含VRAM測量
            validation_params = {
                "data": yaml_file,
                "device": 0 if torch.cuda.is_available() else "cpu",  # 使用GPU進行驗證
                "batch": batch_size,
                "verbose": custom_params.get("verbose", False) if custom_params else False
            }
            validation_func = partial(self._run_engine_yolo_validation, model.path, model.type, validation_params)
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor, validation_func
            )
            
            # 從結果中獲取metrics和GPU使用量信息
            metrics = result['metrics']
            memory_usage_mb = result['memory_usage_mb']
            avg_gpu_load = result['avg_gpu_load']
            max_gpu_load = result.get('max_gpu_load', 0.0)
            monitoring_samples = result.get('monitoring_samples', 0)
            model_vram_mb = result.get('model_vram_mb', 0.0)
            monitoring_duration_s = result.get('monitoring_duration_s', 0.0)
            
            # 獲取驗證結果 - 使用results_dict和屬性兩種方式
            validation_results = {}
            
            # 方式1: 直接從metrics物件取得屬性值
            if hasattr(metrics, "pose"):
                validation_results["metrics/mAP50(P)"] = float(metrics.pose.map50)
                validation_results["metrics/mAP50-95(P)"] = float(metrics.pose.map)
                validation_results["metrics/precision(P)"] = float(getattr(metrics.pose, 'p', 0.0))
                validation_results["metrics/recall(P)"] = float(getattr(metrics.pose, 'r', 0.0))
            elif hasattr(metrics, "box"):
                validation_results["metrics/mAP50(B)"] = float(metrics.box.map50)
                validation_results["metrics/mAP50-95(B)"] = float(metrics.box.map)
                validation_results["metrics/precision(B)"] = float(getattr(metrics.box, 'p', 0.0))
                validation_results["metrics/recall(B)"] = float(getattr(metrics.box, 'r', 0.0))
            
            # 方式2: 從results_dict取得值（如果available）
            if hasattr(metrics, 'results_dict') and metrics.results_dict:
                validation_results.update(metrics.results_dict)
            
            # 添加GPU使用量信息到驗證結果
            validation_results["memory_usage_mb"] = memory_usage_mb
            validation_results["avg_gpu_load"] = avg_gpu_load
            validation_results["max_gpu_load"] = max_gpu_load
            validation_results["monitoring_samples"] = monitoring_samples
            validation_results["model_vram_mb"] = model_vram_mb
            validation_results["monitoring_duration_s"] = monitoring_duration_s
            
            # 格式化結果
            return self._format_validation_results(model.id, model.name, dataset_id, dataset_name, batch_size, validation_results)
                
        except Exception as e:
            print(f"TensorRT Engine模型驗證失敗: {str(e)}")
            raise

    def _run_pytorch_yolo_validation(self, model_path: str, model_type: ModelType, validation_params: dict) -> dict:
        """在執行緒中運行PyTorch模型的YOLO驗證（同步操作），包含VRAM測量"""
        try:
            # 測量從載入到驗證的完整過程（只測量一次）
            metrics, vram_info = self._measure_model_load_and_val_vram(model_path, model_type, validation_params)
            
            # 準備返回結果，包含GPU使用量信息
            result = {
                'metrics': metrics,
                'memory_usage_mb': vram_info['memory_usage_mb'],
                'avg_gpu_load': vram_info['avg_gpu_load'],
                'max_gpu_load': vram_info.get('max_gpu_load', 0.0),
                'monitoring_samples': vram_info.get('monitoring_samples', 0),
                'model_vram_mb': vram_info.get('model_vram_mb', 0.0),
                'monitoring_duration_s': vram_info.get('monitoring_duration_s', 0.0)
            }
            
            return result
            
        except Exception as e:
            print(f"PyTorch模型驗證失敗: {e}")
            self._cleanup_gpu_resources()
            
            return {
                'metrics': None,
                'memory_usage_mb': 0.0,
                'avg_gpu_load': 0.0,
                'max_gpu_load': 0.0,
                'monitoring_samples': 0,
                'model_vram_mb': 0.0,
                'monitoring_duration_s': 0.0
            }
        finally:
            # 驗證完成後清理runs目錄
            self._cleanup_runs_directory()

    def _run_yolo_validation(self, model_path: str, model_type: ModelType, yaml_file: str, validation_params: dict) -> dict:
        """在執行緒中運行YOLO驗證（同步操作）"""
        from ultralytics import YOLO
        
        try:
            task = self._get_task_from_model_type(model_type)
            yolo_model = YOLO(model_path, task=task)
            metrics = yolo_model.val(data=yaml_file, **validation_params)
            return metrics.results_dict
        finally:
            # 驗證完成後清理runs目錄
            self._cleanup_runs_directory()
    
    def _run_yolo_validation_with_source(self, source_path: str, source_type: ModelType, model_path: str, model_format: str, yaml_file: str, validation_params: dict) -> dict:
        """在執行緒中運行轉換模型的YOLO驗證（同步操作）"""
        from ultralytics import YOLO
        
        try:
            task = self._get_task_from_model_type(source_type)
            yolo_model = YOLO(source_path, task=task)
            
            # 根據格式設置參數
            if model_format == "onnx":
                validation_params["format"] = "onnx"
                validation_params["model"] = model_path
            elif model_format == "engine":
                validation_params["format"] = "engine"
                validation_params["model"] = model_path
            
            metrics = yolo_model.val(data=yaml_file, **validation_params)
            return metrics.results_dict
        finally:
            # 驗證完成後清理runs目錄
            self._cleanup_runs_directory()
    
    def _run_engine_yolo_validation(self, model_path: str, model_type: ModelType, validation_params: dict) -> dict:
        """在執行緒中運行Engine模型的YOLO驗證（同步操作），包含VRAM測量"""
        try:
            # 測量從載入到驗證的完整過程（只測量一次）
            metrics, vram_info = self._measure_model_load_and_val_vram(model_path, model_type, validation_params)
            
            # 準備返回結果，包含GPU使用量信息
            result = {
                'metrics': metrics,
                'memory_usage_mb': vram_info['memory_usage_mb'],
                'avg_gpu_load': vram_info['avg_gpu_load'],
                'max_gpu_load': vram_info.get('max_gpu_load', 0.0),
                'monitoring_samples': vram_info.get('monitoring_samples', 0),
                'model_vram_mb': vram_info.get('model_vram_mb', 0.0),
                'monitoring_duration_s': vram_info.get('monitoring_duration_s', 0.0)
            }
            
            return result
            
        except Exception as e:
            print(f"Engine模型驗證失敗: {e}")
            self._cleanup_gpu_resources()
            
            return {
                'metrics': None,
                'memory_usage_mb': 0.0,
                'avg_gpu_load': 0.0,
                'max_gpu_load': 0.0,
                'monitoring_samples': 0,
                'model_vram_mb': 0.0,
                'monitoring_duration_s': 0.0
            }
        finally:
            # 驗證完成後清理runs目錄
            self._cleanup_runs_directory()

    def _get_torch_vram_usage(self):
        """
        獲取當前VRAM使用量 - 使用多種備用方法
        
        Returns:
            float: 當前VRAM使用量（MB）
        """
        if not torch.cuda.is_available():
            return 0.0
        
        try:
            # 優先使用全局記憶體獲取函數
            vram_used_mb = _get_gpu_memory_usage()
            print(f"當前VRAM使用量: {vram_used_mb:.1f} MB")
            return vram_used_mb
        except Exception as e:
            print(f"獲取VRAM使用量失敗: {e}")
            return 0.0

    def _measure_model_load_and_val_vram(self, model_path: str, model_type: ModelType, validation_params: dict):
        """
        測量模型載入和驗證過程的VRAM使用量和GPU負載（只測量一次）
        
        Args:
            model_path: 模型路徑
            model_type: 模型類型
            validation_params: 驗證參數
            
        Returns:
            tuple: (metrics, vram_info) 驗證結果和VRAM信息
        """
        if not torch.cuda.is_available():
            # CPU模式下執行
            from ultralytics import YOLO
            task = self._get_task_from_model_type(model_type)
            model = YOLO(model_path, task=task)            
            metrics = model.val(**validation_params)
            del model
            
            return metrics, {
                'memory_usage_mb': 0.0,
                'avg_gpu_load': 0.0,
                'max_gpu_load': 0.0,
                'monitoring_samples': 0,
                'model_vram_mb': 0.0,
                'monitoring_duration_s': 0.0
            }
        
        # 初始化GPU監控器
        monitor = GPUMonitor(sample_interval=0.02)
        
        try:
            # 測量前清理GPU資源
            print("清理GPU資源...")
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(0.2)
            
            # 獲取基準記憶體使用量
            vram_baseline = _get_gpu_memory_usage()
            print(f"基準VRAM: {vram_baseline:.1f} MB")
            
            # 載入模型
            from ultralytics import YOLO
            task = self._get_task_from_model_type(model_type)
            print(f"載入模型: {model_path}, task: {task}")
            model = YOLO(model_path, task=task)
            print(f"模型載入完成, 類型: {type(model)}")
            

            # 對於PT模型，移動到GPU
            if hasattr(model, 'model') and hasattr(model.model, 'to'):
                if not model_path.endswith('.engine') and not model_path.endswith('.plan'):
                    model.to(f'cuda:{self.gpu_id}')
                    torch.cuda.synchronize()
                    print("PT模型已移動至GPU")
                else:
                    print("Engine模型載入完成")
            
            # 測量載入後記憶體
            vram_after_load = _get_gpu_memory_usage()
            model_loading_vram = max(0, vram_after_load - vram_baseline)
            print(f"模型載入VRAM: {model_loading_vram:.1f} MB")
            
            # 對於Engine模型，需要執行一次推理來觸發VRAM分配
            if model_path.endswith('.engine') :
                print("執行一次推理以觸發Engine模型VRAM分配...")
                import numpy as np
                
                # 從validation_params獲取batch size
                batch_size = validation_params.get('batch', 1)
                
                # 準備正確數量的dummy圖片
                if batch_size > 1:
                    dummy_images = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8) for _ in range(batch_size)]
                else:
                    dummy_images = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                
                try:
                    _ = model.predict(dummy_images, device=self.gpu_id, verbose=False)
                    torch.cuda.synchronize()
                    
                    # 重新測量記憶體
                    vram_after_warmup = _get_gpu_memory_usage()
                    print(f"Engine預熱後VRAM: {vram_after_warmup:.1f} MB")
                except Exception as warmup_error:
                    print(f"Engine模型預熱失敗: {warmup_error}")
                    # 如果預熱失敗，繼續執行但記錄錯誤
                    pass
            
            # 開始GPU監控
            monitor.start_monitoring()
            
            # 執行驗證
            print("開始執行驗證...")
            print(f"驗證參數: {validation_params}")
            metrics = model.val(**validation_params)
            
            # 等待驗證完成
            torch.cuda.synchronize()
            time.sleep(0.1)  # 讓監控器收集一些數據
            
            # 停止監控
            monitor.stop_monitoring()
            
            # 獲取最終記憶體使用量
            final_vram = _get_gpu_memory_usage()
            total_vram_usage = max(0, final_vram - vram_baseline)
            
            # 獲取GPU使用率統計
            gpu_stats = monitor.get_statistics()
            
            print(f"最終VRAM使用量: {final_vram:.1f} MB")
            print(f"總VRAM使用量: {total_vram_usage:.1f} MB")
            
            # 準備VRAM信息
            vram_info = {
                'memory_usage_mb': final_vram,
                'model_vram_mb': total_vram_usage,
                'avg_gpu_load': gpu_stats.get('avg_utilization', 0.0),
                'max_gpu_load': gpu_stats.get('max_utilization', 0.0),
                'monitoring_samples': gpu_stats.get('samples', 0),
                'monitoring_duration_s': gpu_stats.get('duration_s', 0.0)
            }
            
            if gpu_stats.get('samples', 0) > 0:
                print(f"GPU使用率 - 平均: {vram_info['avg_gpu_load']:.1f}%, 峰值: {vram_info['max_gpu_load']:.1f}%")
                print(f"監控數據: {vram_info['monitoring_samples']} 個樣本, {vram_info['monitoring_duration_s']:.1f}秒")
            else:
                print("未能收集到GPU使用率數據")
            
            return metrics, vram_info
            
        except Exception as e:
            print(f"測量模型載入和驗證VRAM失敗: {e}")
            # 停止監控
            monitor.stop_monitoring()
            
            # 確保資源被釋放
            if 'model' in locals():
                del model
            self._cleanup_gpu_resources()
            
            # 返回錯誤的預設值
            return None, {
                'memory_usage_mb': 0.0,
                'avg_gpu_load': 0.0,
                'max_gpu_load': 0.0,
                'monitoring_samples': 0,
                'model_vram_mb': 0.0,
                'monitoring_duration_s': 0.0
            }
        
        finally:
            # 確保監控停止
            monitor.stop_monitoring()
            
            # 測量後清理資源
            print("清理模型資源...")
            if 'model' in locals():
                del model
            torch.cuda.empty_cache()
            gc.collect()
            print("資源清理完成")
            
            # 清理runs目錄
            self._cleanup_runs_directory()

    def _cleanup_gpu_resources(self):
        """清理GPU資源"""
        if torch.cuda.is_available():
            try:
                # 強制回收Python物件
                gc.collect()
                # 清空CUDA快取
                torch.cuda.empty_cache()
                # 再次強制回收（有些物件需要兩次回收）
                gc.collect()
                
                # 檢查清理後的VRAM使用量
                vram_after_cleanup = _get_gpu_memory_usage()
                print(f"GPU資源已清理，剩餘VRAM: {vram_after_cleanup:.1f} MB")
            except Exception as e:
                print(f"清理GPU資源時出錯: {e}")
        else:
            print("GPU不可用，跳過資源清理")

    def _get_dataset_path(self, dataset_id: str = None) -> str:
        """根據資料集ID獲取資料集路徑"""
        if not dataset_id:
            # 使用默認資料集；與下方正常分支一致，目錄存在還不夠，必須真的含有圖像檔，
            # 否則回傳 None 讓呼叫端退回模擬資料，而不是稍後在 images[0] 拋 IndexError
            default_path = os.path.join(os.getcwd(), "data", "datasets", "coco.zip_a8bedb81", "images", "val2017")
            if os.path.isdir(default_path):
                images = [f for f in os.listdir(default_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                if images:
                    return default_path
                print(f"警告: 預設資料集目錄存在但沒有圖像檔，改用模擬資料: {default_path}")
            return None
        
        # 從 datasets.json 獲取資料集資訊
        datasets_dir = os.path.join(os.getcwd(), "data", "datasets")
        datasets_metadata_path = os.path.join(datasets_dir, "datasets.json")
        
        if os.path.exists(datasets_metadata_path):
            with open(datasets_metadata_path, 'r', encoding='utf-8') as f:
                datasets = json.load(f)
                dataset_info = next((d for d in datasets if d["id"] == dataset_id), None)
                if dataset_info:
                    dataset_path = os.path.join(os.getcwd(), dataset_info["path"])
                    # 尋找images子目錄
                    possible_paths = [
                        os.path.join(dataset_path, "images", "val2017"),
                        os.path.join(dataset_path, "images", "val"),
                        os.path.join(dataset_path, "images"),
                        os.path.join(dataset_path, "val2017"),
                        os.path.join(dataset_path, "val"),
                        dataset_path
                    ]
                    for path in possible_paths:
                        if os.path.exists(path) and os.listdir(path):
                            # 檢查是否包含圖像文件
                            files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                            if files:
                                return path
        
        # 如果找不到，返回None
        return None

    async def benchmark_model(self, model_id: str, batch_size: int = 1, iterations: int = 100, 
                            num_iterations: int = None, image_size: int = 640, test_dataset: str = None, 
                            custom_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        對模型進行性能基準測試
        
        Args:
            model_id: 模型ID
            batch_size: 批次大小
            iterations: 推理迭代次數 (首選參數名)
            num_iterations: 推理迭代次數 (向後兼容)
            image_size: 圖像尺寸
            test_dataset: 測試數據集ID
            custom_params: 自定義參數
        """
        from app.services.model_service import ModelService
        model_service = ModelService()
        
        # 處理參數向後兼容性
        if num_iterations is not None:
            iterations = num_iterations
        
        # 獲取模型信息
        model = model_service.get_model_by_id(model_id)
        if not model:
            raise Exception(f"找不到模型: {model_id}")
        
        # 對於Engine模型，從metadata中提取實際的batch_size
        actual_batch_size = batch_size
        if model.format.value == "engine" and model.metadata:
            # 從模型名稱中提取batch size
            model_name = model.metadata.get("display_name", model.name)
            import re
            batch_match = re.search(r'batch(\d+)', model_name)
            if batch_match:
                actual_batch_size = int(batch_match.group(1))
                print(f"從模型metadata中提取batch size: {actual_batch_size}")
        
        print(f"開始對模型 {model.name} 進行性能測試，批次大小: {actual_batch_size}, 迭代次數: {iterations}, 資料集: {test_dataset}")
        
        try:
            # 初始化GPU監測（如果可用）
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                print("成功初始化GPU監測")
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            
            # 對於TensorRT Engine模型，使用特殊處理
            if model.format.value == "engine":
                return await self._benchmark_engine_model(model, actual_batch_size, iterations, test_dataset, custom_params)
            else:
                # 對於其他格式，使用原有邏輯
                return await self._benchmark_general_model(model, actual_batch_size, iterations, test_dataset, custom_params)
                
        except Exception as e:
            print(f"性能測試失敗: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return {
                "model_id": model_id,
                "model_name": model.name,
                "batch_size": actual_batch_size if 'actual_batch_size' in locals() else batch_size,
                "iterations": iterations,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
                "status": "failed"
            }
    
    async def _benchmark_engine_model(self, model: ModelInfo, batch_size: int, iterations: int, test_dataset: str = None, custom_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """對TensorRT Engine模型進行基準測試"""
        try:
            import numpy as np
            
            # 準備設備參數
            device = "cpu"  # 默認使用CPU
            if torch.cuda.is_available():
                device = 0  # 如果有GPU，使用第一個GPU
            
            # 根據資料集ID獲取測試圖像路徑
            test_data_path = self._get_dataset_path(test_dataset)
            if not test_data_path:
                print(f"警告: 找不到資料集 {test_dataset}，使用模擬數據")
                # 創建模擬測試數據
                test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            else:
                print(f"使用資料集路徑: {test_data_path}")
                # 獲取第一張圖像作為備用
                images = sorted(f for f in os.listdir(test_data_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')))
                if images:
                    test_image = os.path.join(test_data_path, images[0])
                else:
                    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # 使用執行緒池運行推論測試（非阻塞）
            effective_iterations = _resolve_iterations(iterations)
            benchmark_func = partial(
                self._run_engine_benchmark,
                model.path,
                model.type,
                test_data_path if test_data_path else test_image,
                device,
                batch_size,
                effective_iterations
            )
            benchmark_results = await asyncio.get_event_loop().run_in_executor(
                self.executor, benchmark_func
            )
            
            inference_times = benchmark_results["inference_times"]
            throughputs = benchmark_results["throughputs"]
            
            # 計算統計數據 - 轉換為單次請求的平均時間
            avg_inference_time = float(np.mean(inference_times)) / batch_size  # 除以batch_size得到單次請求時間
            std_inference_time = _sample_std(inference_times) / batch_size  # 樣本標準差（ddof=1）
            min_inference_time = float(np.min(inference_times)) / batch_size
            max_inference_time = float(np.max(inference_times)) / batch_size
            avg_throughput = float(np.mean(throughputs))
            
            return {
                "model_id": model.id,
                "model_name": model.name,
                "batch_size": batch_size,
                "iterations": len(inference_times),  # 實際執行次數
                "requested_iterations": iterations,  # 使用者請求次數（可能被上限截斷）
                "max_iterations": MAX_BENCHMARK_ITERATIONS,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "avg_inference_time_ms": avg_inference_time,
                "std_inference_time_ms": std_inference_time,
                "min_inference_time_ms": min_inference_time,
                "max_inference_time_ms": max_inference_time,
                "all_inference_times": [t / batch_size for t in inference_times],  # 轉換為單次請求時間
                "avg_throughput_fps": avg_throughput,
                "status": "success"
            }
            
        except Exception as e:
            print(f"TensorRT Engine基準測試失敗: {str(e)}")
            raise
    
    async def _benchmark_general_model(self, model: ModelInfo, batch_size: int, iterations: int, test_dataset: str = None, custom_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """對一般模型進行基準測試（PT、ONNX格式）"""
        try:
            import numpy as np
            
            # 準備設備參數
            device = "cpu"  # 默認使用CPU
            if torch.cuda.is_available():
                device = 0  # 如果有GPU，使用第一個GPU
            
            # 根據資料集ID獲取測試圖像路徑
            test_data_path = self._get_dataset_path(test_dataset)
            if not test_data_path:
                print(f"警告: 找不到資料集 {test_dataset}，使用模擬數據")
                # 創建模擬測試數據
                test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            else:
                print(f"使用資料集路徑: {test_data_path}")
                # 獲取第一張圖像作為備用
                images = sorted(f for f in os.listdir(test_data_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')))
                if images:
                    test_image = os.path.join(test_data_path, images[0])
                else:
                    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # 使用執行緒池運行推論測試（非阻塞）
            effective_iterations = _resolve_iterations(iterations)
            benchmark_func = partial(
                self._run_general_benchmark,
                model.path,
                model.type,
                test_data_path if test_data_path else test_image,
                device,
                batch_size,
                effective_iterations
            )
            benchmark_results = await asyncio.get_event_loop().run_in_executor(
                self.executor, benchmark_func
            )
            
            inference_times = benchmark_results["inference_times"]
            throughputs = benchmark_results["throughputs"]
            
            # 計算統計數據 - 轉換為單次請求的平均時間
            avg_inference_time = float(np.mean(inference_times)) / batch_size  # 除以batch_size得到單次請求時間
            std_inference_time = _sample_std(inference_times) / batch_size  # 樣本標準差（ddof=1）
            min_inference_time = float(np.min(inference_times)) / batch_size
            max_inference_time = float(np.max(inference_times)) / batch_size
            avg_throughput = float(np.mean(throughputs))
            
            return {
                "model_id": model.id,
                "model_name": model.name,
                "batch_size": batch_size,
                "iterations": len(inference_times),  # 實際執行次數
                "requested_iterations": iterations,  # 使用者請求次數（可能被上限截斷）
                "max_iterations": MAX_BENCHMARK_ITERATIONS,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "avg_inference_time_ms": avg_inference_time,
                "std_inference_time_ms": std_inference_time,
                "min_inference_time_ms": min_inference_time,
                "max_inference_time_ms": max_inference_time,
                "all_inference_times": [t / batch_size for t in inference_times],  # 轉換為單次請求時間
                "avg_throughput_fps": avg_throughput,
                "status": "success"
            }
            
        except Exception as e:
            print(f"一般模型基準測試失敗: {str(e)}")
            raise

    def _run_engine_benchmark(self, model_path: str, model_type: ModelType, test_data_or_image, device, batch_size: int, iterations: int) -> dict:
        """在執行緒中運行Engine模型的基準測試（同步操作）"""
        from ultralytics import YOLO
        import time
        import os
        
        try:
            inference_times = []
            throughputs = []
            
            # 使用傳入的test_data_or_image
            test_data_path = test_data_or_image
            
            # 只載入一次模型用於所有迭代
            task = self._get_task_from_model_type(model_type)
            yolo_model = YOLO(model_path, task=task)
            
            # 對於Engine模型，先檢查是否支援請求的batch size
            # 透過嘗試一次預測來檢測batch size相容性
            try:
                # 準備測試數據
                if isinstance(test_data_path, str) and os.path.isdir(test_data_path):
                    # 如果是目錄，獲取多張圖片進行batch測試
                    images = sorted(f for f in os.listdir(test_data_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')))
                    if len(images) >= batch_size:
                        test_images = [os.path.join(test_data_path, images[i]) for i in range(batch_size)]
                    else:
                        # 如果圖片不夠，重複使用第一張
                        test_images = [os.path.join(test_data_path, images[0])] * batch_size
                else:
                    # 如果是單一圖像，根據batch_size重複
                    if batch_size > 1:
                        test_images = [test_data_path] * batch_size
                    else:
                        test_images = [test_data_path]
                
                # 嘗試預熱（使用指定的batch size）
                _ = yolo_model.predict(test_images, device=device, verbose=False)
                print(f"Engine模型支援batch size {batch_size}")
                
            except Exception as batch_error:
                error_msg = str(batch_error)
                if "not equal to max model size" in error_msg or "input size" in error_msg:
                    # 提取模型支援的batch size
                    import re
                    pattern = r'\((\d+),.*?\)'
                    match = re.search(pattern, error_msg)
                    if match:
                        supported_batch = int(match.group(1))
                        raise Exception(f"TensorRT Engine模型僅支援batch size {supported_batch}，但測試要求batch size {batch_size}")
                    else:
                        raise Exception(f"TensorRT Engine模型不支援batch size {batch_size}: {error_msg}")
                else:
                    raise Exception(f"Engine模型預熱失敗: {error_msg}")
            
            for run in range(iterations):
                print(f"運行推理測試 {run+1}/{iterations}...")
                
                # 測量推理時間
                start_time = time.time()
                
                # 執行推理 - 使用準備好的測試數據
                _ = yolo_model.predict(
                    test_images,
                    device=device, 
                    verbose=False, 
                    task=task,
                    stream=False
                )
                
                inference_time = (time.time() - start_time) * 1000  # 轉換為毫秒
                
                inference_times.append(inference_time)
                
                # 計算吞吐量 (images per second)
                throughput = batch_size / (inference_time / 1000.0)  # FPS
                throughputs.append(throughput)
                
                print(f"第{run+1}次：推論時間={inference_time:.2f}ms (batch {batch_size})")
            
            return {
                "inference_times": inference_times,
                "throughputs": throughputs
            }
            
        finally:
            # 清理資源
            if 'yolo_model' in locals():
                del yolo_model
            self._cleanup_gpu_resources()
            # 清理runs目錄
            self._cleanup_runs_directory()
    
    def _run_general_benchmark(self, model_path: str, model_type: ModelType, test_data_or_image, device, batch_size: int, iterations: int) -> dict:
        """在執行緒中運行一般模型的基準測試（同步操作）"""
        from ultralytics import YOLO
        import time
        import os
        
        try:
            inference_times = []
            throughputs = []
            
            # 使用傳入的test_data_or_image
            test_data_path = test_data_or_image
            
            # 只載入一次模型用於所有迭代
            task = self._get_task_from_model_type(model_type)
            yolo_model = YOLO(model_path, task=task)
            
            # 準備測試數據
            if isinstance(test_data_path, str) and os.path.isdir(test_data_path):
                # 如果是目錄，獲取多張圖片進行batch測試
                images = sorted(f for f in os.listdir(test_data_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')))
                if len(images) >= batch_size:
                    test_images = [os.path.join(test_data_path, images[i]) for i in range(batch_size)]
                else:
                    # 如果圖片不夠，重複使用第一張
                    test_images = [os.path.join(test_data_path, images[0])] * batch_size
            else:
                # 如果是單一圖像，根據batch_size重複
                if batch_size > 1:
                    test_images = [test_data_path] * batch_size
                else:
                    test_images = [test_data_path]
            
            # 預熱（使用測試數據）
            _ = yolo_model.predict(test_images, device=device, verbose=False)
            
            for run in range(iterations):
                print(f"運行推理測試 {run+1}/{iterations}...")
                
                # 測量推理時間
                start_time = time.time()
                _ = yolo_model.predict(
                    test_images,
                    device=device, 
                    verbose=False, 
                    task=task,
                    stream=False
                )
                inference_time = (time.time() - start_time) * 1000  # 轉換為毫秒
                
                inference_times.append(inference_time)
                
                # 計算吞吐量 (images per second)
                throughput = batch_size / (inference_time / 1000.0)  # FPS
                throughputs.append(throughput)
                
                print(f"第{run+1}次：推論時間={inference_time:.2f}ms (batch {batch_size})")
            
            return {
                "inference_times": inference_times,
                "throughputs": throughputs
            }
            
        finally:
            # 清理資源
            if 'yolo_model' in locals():
                del yolo_model
            self._cleanup_gpu_resources()
            # 清理runs目錄
            self._cleanup_runs_directory()

    def _cleanup_runs_directory(self):
        """清理YOLO生成的runs目錄"""
        try:
            runs_dir = os.path.join(os.getcwd(), "runs")
            if os.path.exists(runs_dir):
                print(f"清理runs目錄: {runs_dir}")
                shutil.rmtree(runs_dir)
                print("runs目錄已清理完成")
        except Exception as e:
            print(f"清理runs目錄時出錯: {e}")