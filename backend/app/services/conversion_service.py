import os
import json
import time
import uuid
import subprocess
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone, timedelta

from app.models import ConversionJob, ConversionStatus, ModelFormat, PrecisionType, ModelInfo, ModelType
from app.services.model_service import ModelService

class ConversionService:
    """
    模型轉換服務，負責模型格式轉換和轉換任務管理
    """
    def __init__(self):
        """初始化轉換服務"""
        self.jobs: Dict[str, ConversionJob] = {}
        self.data_file = "data/conversion_jobs.json"
        self.model_service = ModelService()
        
        # 確保數據目錄存在
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        
        # 載入已有任務數據
        self._load_jobs()
    
    def _get_task_from_model_type(self, model_type: ModelType) -> str:
        """根據模型類型確定YOLO task參數"""
        if model_type == ModelType.YOLOV8_POSE:
            return "pose"
        elif model_type == ModelType.YOLOV8_SEG:
            return "segment"
        elif model_type == ModelType.YOLOV8:
            return "detect"
        else:
            return "detect"  # 默認為檢測任務
    
    def get_jobs(self, status: Optional[ConversionStatus] = None, 
                 skip: int = 0, limit: int = 10) -> List[ConversionJob]:
        """獲取轉換任務列表，支持過濾和分頁"""
        jobs = list(self.jobs.values())
        
        # 按狀態過濾
        if status:
            jobs = [j for j in jobs if j.status == status]
        
        # 確保所有 datetime 對象都是 aware 的
        for job in jobs:
            if job.created_at and job.created_at.tzinfo is None:
                # 如果是 naive datetime，轉換為 aware datetime (UTC+8)
                job.created_at = job.created_at.replace(tzinfo=timezone(timedelta(hours=8)))
        
        # 按創建時間排序（最新的先顯示）
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        
        # 分頁
        return jobs[skip:skip + limit]
    
    def get_job_by_id(self, job_id: str) -> Optional[ConversionJob]:
        """根據ID獲取轉換任務"""
        return self.jobs.get(job_id)
    
    def save_job(self, job: ConversionJob) -> ConversionJob:
        """保存轉換任務信息"""
        self.jobs[job.id] = job
        self._save_jobs()
        return job
    
    def delete_job(self, job_id: str) -> bool:
        """刪除轉換任務"""
        if job_id in self.jobs:
            # 刪除前記錄日誌
            print(f"刪除轉換任務: {job_id}, 當前狀態: {self.jobs[job_id].status.value}")
            
            # 從字典中刪除任務
            del self.jobs[job_id]
            
            # 立即保存任務數據，確保更改被持久化
            self._save_jobs()
            
            # 刪除後確認任務已不存在
            if job_id not in self.jobs:
                print(f"轉換任務 {job_id} 已成功刪除")
                return True
            else:
                print(f"轉換任務 {job_id} 刪除失敗，仍然存在於任務字典中")
                return False
        
        print(f"找不到要刪除的轉換任務: {job_id}")
        return False
    
    def process_conversion_job(self, job_id: str):
        """處理轉換任務，在後台執行"""
        job = self.get_job_by_id(job_id)
        if not job:
            print(f"找不到轉換任務: {job_id}")
            return
        
        # 更新任務狀態為處理中
        job.status = ConversionStatus.PROCESSING
        self._save_jobs()
        
        # 使用多進程執行轉換任務
        try:
            import multiprocessing
            # 創建一個進程來執行轉換任務
            conversion_process = multiprocessing.Process(
                target=self._run_conversion_task,
                args=(job_id,)
            )
            conversion_process.start()
            print(f"已在獨立進程中啟動轉換任務 {job_id}，進程ID: {conversion_process.pid}")
        except Exception as e:
            # 如果無法創建獨立進程，則在當前線程中執行
            print(f"無法在獨立進程中啟動轉換任務: {str(e)}，將在當前線程中執行")
            self._run_conversion_task(job_id)
    
    def _run_conversion_task(self, job_id: str):
        """在獨立進程中執行實際的轉換任務"""
        try:
            # 重新初始化服務以確保獨立進程有自己的數據副本
            self._load_jobs()
            
            job = self.get_job_by_id(job_id)
            if not job:
                print(f"找不到轉換任務: {job_id}")
                return
            
            # 檢查任務狀態
            if job.status != ConversionStatus.PROCESSING:
                job.status = ConversionStatus.PROCESSING
                self._save_jobs()
            
            # 模型服務在獨立進程中需要重新初始化
            self.model_service = ModelService()
            
            print(f"轉換任務開始，源模型ID: {job.source_model_id}")
            
            # 獲取源模型信息
            source_model = self.model_service.get_model_by_id(job.source_model_id)
            
            # 如果直接通過ID找不到，嘗試通過參數中的模型名稱查找
            if not source_model and 'model_name' in job.parameters:
                model_name = job.parameters['model_name']
                print(f"通過ID找不到源模型，嘗試通過名稱查找: {model_name}")
                source_model = self.model_service.get_model_by_name(model_name)
                if source_model:
                    print(f"通過模型名稱找到源模型: {source_model.name} (ID: {source_model.id})")
                    # 更新任務的源模型ID
                    job.source_model_id = source_model.id
                    self._save_jobs()
            
            if not source_model:
                raise Exception(f"無法找到源模型，ID: {job.source_model_id}")
            
            print(f"開始處理轉換任務: {job_id}, 源模型: {source_model.name} (ID: {source_model.id})")
            
            # 執行轉換
            target_model_path = self._convert_model(
                source_model=source_model,
                target_format=job.target_format,
                precision=job.precision,
                parameters=job.parameters or {}
            )
            
            # 創建目標模型信息
            target_model = self._create_target_model(
                source_model=source_model,
                target_path=target_model_path,
                target_format=job.target_format,
                precision=job.precision
            )
            
            # 更新任務狀態為已完成
            job.status = ConversionStatus.COMPLETED
            job.completed_at = datetime.now(timezone(timedelta(hours=8)))
            job.target_model_id = target_model.id
            
            print(f"轉換任務 {job_id} 成功完成，目標模型ID: {target_model.id}")
            
        except Exception as e:
            print(f"轉換任務 {job_id} 失敗: {str(e)}")
            import traceback
            print(traceback.format_exc())
            
            try:
                # 重新加載任務以防止數據不一致
                self._load_jobs()
                job = self.get_job_by_id(job_id)
                if job:
                    # 更新任務狀態為失敗
                    job.status = ConversionStatus.FAILED
                    job.error_message = str(e)
            except Exception as update_error:
                print(f"更新任務狀態時出錯: {str(update_error)}")
        
        finally:
            # 保存任務更新
            try:
                self._save_jobs()
            except Exception as save_error:
                print(f"保存任務更新時出錯: {str(save_error)}")
    
    def _convert_model(self, source_model: ModelInfo, target_format: ModelFormat,
                       precision: PrecisionType, parameters: Dict[str, Any]) -> str:
        """執行模型轉換"""
        if source_model.format == target_format:
            raise Exception(f"源模型已經是 {target_format.value} 格式，無需轉換")
        
        # 獲取源模型名稱和參數以創建目標目錄名稱
        source_name = source_model.name
        safe_name = source_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        
        # 添加轉換參數到模型名
        param_suffix = f"_{target_format.value}_{precision.value}"
        if parameters.get("batch_size"):
            param_suffix += f"_batch{parameters.get('batch_size')}"
        if parameters.get("imgsz"):
            param_suffix += f"_size{parameters.get('imgsz')}"
        
        # 創建目標目錄，確保包含源模型ID以避免命名衝突
        # 使用源模型ID的前8位作為唯一標識符
        source_id_short = source_model.id[:8]
        target_model_name = f"{safe_name}_{source_id_short}{param_suffix}"
            
        target_dir = os.path.join("model_repository", target_model_name)
        
        # 確保目標目錄乾淨，如果已存在則先清理
        if os.path.exists(target_dir):
            print(f"目標目錄已存在，清理: {target_dir}")
            import shutil
            shutil.rmtree(target_dir)
        
        # 創建Triton目錄結構
        version_dir = os.path.join(target_dir, "1")  # 使用版本1
        os.makedirs(version_dir, exist_ok=True)
        
        # 記錄轉換參數到任務
        parameters["model_name"] = source_name
        parameters["source_model_id"] = source_model.id
        
        # 根據源格式和目標格式決定轉換方法
        target_model_path = None
        if source_model.format == ModelFormat.PT:
            if target_format == ModelFormat.ONNX:
                target_model_path = self._convert_pt_to_onnx(source_model, version_dir, precision, parameters)
            elif target_format == ModelFormat.ENGINE:
                # 直接從PyTorch轉換到TensorRT引擎
                target_model_path = self._convert_pt_to_tensorrt(source_model, version_dir, precision, parameters)
        
        elif source_model.format == ModelFormat.ONNX:
            if target_format == ModelFormat.ENGINE:
                target_model_path = self._convert_onnx_to_tensorrt(source_model.path, version_dir, precision, parameters)
        
        if not target_model_path:
            raise Exception(f"不支持從 {source_model.format.value} 轉換到 {target_format.value}")
        
        # 創建config.pbtxt配置文件
        config_path = os.path.join(target_dir, "config.pbtxt")
        
        # 根據模型類型和格式生成配置文件
        platform = ""
        model_filename = os.path.basename(target_model_path)
        
        if target_format == ModelFormat.PT:
            platform = "pytorch_libtorch"
        elif target_format == ModelFormat.ONNX:
            platform = "onnxruntime_onnx"
        elif target_format == ModelFormat.ENGINE:
            platform = "tensorrt_plan"
            # 對於 TensorRT Engine，使用 .plan 檔案
            model_filename = "model.plan"
        
        # 獲取輸入圖像尺寸
        img_size = parameters.get("imgsz", 640)
        batch_size = parameters.get("batch_size", 1)
        
        # 根據模型類型設置不同的輸出維度
        output_dims = "[ 84, 8400 ]"  # 默認為YOLOv8檢測模型
        if source_model.type.value == "yolov8_seg":
            # 分割模型有不同的輸出 - 實際輸出是 [batch, 56, anchors]
            # 當max_batch_size > 0時，Triton會自動添加batch維度
            # 所以配置中只需要指定 [56, anchors] 兩個維度
            output_dims = "[ 56, -1 ]"  # 第一維固定56，第二維動態
        elif source_model.type.value == "yolov8_pose":
            # 姿態估計模型有不同的輸出 - 實際輸出是 [batch, pose_dim, anchors]
            # 配置中只需要指定非batch維度
            output_dims = "[ 56, -1 ]"  # 兩個動態維度
        
        # 根據精度設置data_type
        data_type = "TYPE_FP16" if precision == PrecisionType.FP16 else "TYPE_FP32"
        
        # 生成基本配置 - max_batch_size 設定為原始 batch 大小
        config_content = f"""name: "{target_model_name}"
platform: "{platform}"
max_batch_size: {batch_size}
input [
  {{
    name: "images"
    data_type: {data_type}
    dims: [ 3, {img_size}, {img_size} ]
  }}
]
output [
  {{
    name: "output0"
    data_type: {data_type}
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
        with open(config_path, "w", encoding="utf-8") as config_file:
            config_file.write(config_content)
        
        print(f"創建Triton配置文件: {config_path}")
        print(f"平台: {platform}, 模型檔案: {model_filename}, max_batch_size: {batch_size}")
        print(f"配置內容: \n{config_content}")
        
        # 創建元數據文件，包含原始模型ID，以便後續能找到關聯
        metadata_file = os.path.join(target_dir, "metadata.json")
        try:
            # 生成新的模型ID
            new_model_id = str(uuid.uuid4())
            
            with open(metadata_file, 'w', encoding="utf-8") as f:
                json.dump({
                    "model_id": new_model_id,
                    "source_model_id": source_model.id,
                    "display_name": target_model_name,
                    "type": source_model.type.value,
                    "format": target_format.value,
                    "precision": precision.value,
                    "conversion_parameters": parameters
                }, f, indent=2)
            print(f"創建轉換模型元數據文件: {metadata_file}")
        except Exception as e:
            print(f"創建轉換模型元數據文件時出錯: {str(e)}")
        
        return target_model_path
    
    def _convert_pt_to_onnx(self, source_model: ModelInfo, target_dir: str,
                          precision: PrecisionType, parameters: Dict[str, Any]) -> str:
        """將PyTorch模型轉換為ONNX格式"""
        print(f"開始將 {source_model.path} 轉換為ONNX格式...")
        
        # 使用Ultralytics API轉換
        try:
            import sys
            import shutil
            import json
            import torch
            from ultralytics import YOLO
            
            # 載入模型，設置task
            print(f"載入模型文件: {source_model.path}")
            task = self._get_task_from_model_type(source_model.type)
            model = YOLO(source_model.path, task=task)
            
            # 設置導出參數
            imgsz = parameters.get("imgsz", 640)
            half = precision == PrecisionType.FP16
            opset = parameters.get("opset", 12)
            dynamic = parameters.get("dynamic", True)
            
            export_params = {
                "format": "onnx",
                "imgsz": imgsz,
                "half": half,
                "simplify": True,
                "opset": opset,
                "dynamic": dynamic,
                "device": 0 if torch.cuda.is_available() else "cpu",
            }
            
            print(f"ONNX導出參數: {export_params}")
            
            # 執行導出
            model.export(**export_params)
            
            # 獲取源模型目錄作為ONNX文件的查找位置
            source_model_dir = os.path.dirname(source_model.path)
            onnx_path = os.path.join(source_model_dir, "model.onnx")
            
            # 確保目標目錄存在
            os.makedirs(target_dir, exist_ok=True)
            
            # 移動文件到目標目錄
            target_path = os.path.join(target_dir, "model.onnx")
            
            print(f"將ONNX文件從 {onnx_path} 移動到 {target_path}")
            shutil.move(onnx_path, target_path)
            
            # 檢查文件是否成功處理
            if os.path.exists(target_path):
                file_size = os.path.getsize(target_path)
                print(f"ONNX文件大小: {file_size / (1024*1024):.2f} MB")
            else:
                raise Exception(f"ONNX文件不存在: {target_path}")
            
            print(f"PyTorch模型成功轉換為ONNX，保存到: {target_path}")
            return target_path
            
        except Exception as e:
            error_msg = f"PyTorch轉ONNX失敗: {str(e)}"
            print(error_msg)
            import traceback
            print(traceback.format_exc())
            raise Exception(error_msg)
    
    def _convert_onnx_to_tensorrt(self, onnx_path: str, target_dir: str,
                               precision: PrecisionType, parameters: Dict[str, Any]) -> str:
        """將ONNX模型轉換為TensorRT格式 - 使用trtexec生成.engine和.plan檔案"""
        print(f"開始將 {onnx_path} 轉換為TensorRT格式...")
        print(f"目標目錄: {target_dir}")
        
        try:
            import subprocess
            
            # 設置TensorRT轉換參數
            target_engine_path = os.path.join(target_dir, "model.engine")
            target_plan_path = os.path.join(target_dir, "model.plan")
            
            # 確保目標目錄存在
            os.makedirs(target_dir, exist_ok=True)
            
            # 檢查ONNX文件是否存在
            if not os.path.exists(onnx_path):
                raise Exception(f"ONNX文件不存在: {onnx_path}")
            
            # 根據精度類型設置trtexec參數
            precision_flag = ""
            if precision == PrecisionType.FP16:
                precision_flag = "--fp16"
            
            # 設置batch size和workspace
            batch_size = parameters.get("batch_size", 1)
            workspace = parameters.get("workspace", 4)  # GB
            min_size = parameters.get("min_size", 640)
            opt_size = parameters.get("opt_size", 640) 
            max_size = parameters.get("max_size", 1280)
            
            # 第一步：使用trtexec生成 .engine 檔案
            print("第一步：使用trtexec生成 .engine 檔案...")
            
            cmd_engine = [
                "trtexec", 
                f"--onnx={onnx_path}",
                f"--saveEngine={target_engine_path}",
                precision_flag,
                f"--workspace={workspace*1024}",
                f"--minShapes=images:1x3x{min_size}x{min_size}",
                f"--optShapes=images:{batch_size}x3x{opt_size}x{opt_size}",
                f"--maxShapes=images:{batch_size}x3x{max_size}x{max_size}"
            ]
            
            # 過濾掉空參數
            cmd_engine = [c for c in cmd_engine if c]
            
            print(f"執行TensorRT轉換命令（生成.engine）: {' '.join(cmd_engine)}")
            
            # 執行命令
            process = subprocess.Popen(
                cmd_engine, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()
            
            stdout_str = stdout.decode('utf-8')
            stderr_str = stderr.decode('utf-8')
            
            print(f"TensorRT轉換輸出（生成.engine）: {stdout_str}")
            
            if stderr_str:
                print(f"TensorRT轉換錯誤（生成.engine）: {stderr_str}")
            
            if process.returncode != 0:
                raise Exception(f"TensorRT轉換失敗（生成.engine），錯誤代碼: {process.returncode}, 錯誤信息: {stderr_str}")
            
            # 檢查 .engine 文件是否成功生成
            if os.path.exists(target_engine_path):
                file_size = os.path.getsize(target_engine_path)
                print(f".engine 檔案大小: {file_size / (1024*1024):.2f} MB")
            else:
                raise Exception(f".engine 檔案不存在: {target_engine_path}")
            
            # 第二步：使用trtexec生成 .plan 檔案
            print("第二步：使用trtexec生成 .plan 檔案...")
            
            cmd_plan = [
                "trtexec", 
                f"--onnx={onnx_path}",
                f"--saveEngine={target_plan_path}",
                precision_flag,
                f"--workspace={workspace*1024}",
                f"--minShapes=images:1x3x{min_size}x{min_size}",
                f"--optShapes=images:{batch_size}x3x{opt_size}x{opt_size}",
                f"--maxShapes=images:{batch_size}x3x{max_size}x{max_size}"
            ]
            
            # 過濾掉空參數
            cmd_plan = [c for c in cmd_plan if c]
            
            print(f"執行TensorRT轉換命令（生成.plan）: {' '.join(cmd_plan)}")
            
            # 執行命令
            process = subprocess.Popen(
                cmd_plan, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()
            
            stdout_str = stdout.decode('utf-8')
            stderr_str = stderr.decode('utf-8')
            
            print(f"TensorRT轉換輸出（生成.plan）: {stdout_str}")
            
            if stderr_str:
                print(f"TensorRT轉換錯誤（生成.plan）: {stderr_str}")
            
            if process.returncode != 0:
                raise Exception(f"TensorRT轉換失敗（生成.plan），錯誤代碼: {process.returncode}, 錯誤信息: {stderr_str}")
            
            # 檢查 .plan 文件是否成功生成
            if os.path.exists(target_plan_path):
                file_size = os.path.getsize(target_plan_path)
                print(f".plan 檔案大小: {file_size / (1024*1024):.2f} MB")
            else:
                raise Exception(f".plan 檔案不存在: {target_plan_path}")
            
            print(f"ONNX轉TensorRT完成：")
            print(f"  - .engine 檔案（用trtexec生成，用於驗證和測試）: {target_engine_path}")
            print(f"  - .plan 檔案（用trtexec生成，用於 Triton 部署）: {target_plan_path}")
            
            # 返回 .engine 文件路徑（保持向後相容性）
            return target_engine_path
            
        except Exception as e:
            error_msg = f"ONNX轉TensorRT失敗: {str(e)}"
            print(error_msg)
            import traceback
            print(traceback.format_exc())
            raise Exception(error_msg)
    
    def _convert_pt_to_tensorrt(self, source_model: ModelInfo, target_dir: str,
                              precision: PrecisionType, parameters: Dict[str, Any]) -> str:
        """將PyTorch模型轉換為TensorRT格式 - 使用yolo.export生成.engine + trtexec生成.plan"""
        print(f"開始將 {source_model.path} 轉換為TensorRT格式（yolo.export + trtexec流程）...")
        print(f"目標目錄: {target_dir}")
        
        try:
            import sys
            import os
            import shutil
            import json
            import torch
            import subprocess
            from ultralytics import YOLO
            
            # 載入模型，設置task
            print(f"載入模型文件: {source_model.path}")
            task = self._get_task_from_model_type(source_model.type)
            model = YOLO(source_model.path, task=task)
            
            # 設置導出參數
            imgsz = parameters.get("imgsz", 640)
            half = precision == PrecisionType.FP16
            workspace = parameters.get("workspace", 4)  # GB
            batch = parameters.get("batch_size", 1)
            
            # 確保目標目錄存在
            os.makedirs(target_dir, exist_ok=True)
            
            # 第一步：使用 YOLO export 直接生成 .engine 檔案
            print("第一步：執行 YOLO export (.pt -> .engine)...")
            
            # 設置導出參數
            export_params = {
                "format": "engine",  # 直接導出為TensorRT引擎
                "imgsz": imgsz,
                "half": half,
                "batch": batch,      # 明確設置batch參數
                "workspace": workspace,
                "simplify": True,
                "device": 0 if torch.cuda.is_available() else "cpu",
            }
            
            print(f"TensorRT導出參數: {export_params}")
            
            # 執行導出
            model.export(**export_params)
  
            # 獲取源模型目錄作為引擎文件的查找位置
            source_model_dir = os.path.dirname(source_model.path)
            engine_path = os.path.join(source_model_dir, "model.engine")  
            
            # 移動引擎文件到目標目錄
            target_engine_path = os.path.join(target_dir, "model.engine")
            print(f"將引擎文件從 {engine_path} 移動到 {target_engine_path}")      
            shutil.move(engine_path, target_engine_path)
            
            # 檢查 .engine 文件是否成功生成
            if os.path.exists(target_engine_path):
                file_size = os.path.getsize(target_engine_path)
                print(f"引擎文件大小: {file_size / (1024*1024):.2f} MB")
            else:
                raise Exception(f"引擎文件不存在: {target_engine_path}")
            
            # 第二步：生成 ONNX 然後用 trtexec 生成 .plan 檔案
            print("第二步：執行 YOLO export (.pt -> .onnx) + trtexec (.onnx -> .plan)...")
            
            # 重新載入模型進行ONNX轉換
            model = YOLO(source_model.path, task=task)
            
            # 設置ONNX導出參數
            onnx_export_params = {
                "format": "onnx",
                "imgsz": imgsz,
                "half": half,
                "simplify": True,
                "opset": 12,
                "dynamic": True,
                "device": 0 if torch.cuda.is_available() else "cpu",
            }
            
            print(f"ONNX導出參數: {onnx_export_params}")
            
            # 執行ONNX導出
            model.export(**onnx_export_params)
            
            # 獲取ONNX文件路徑
            onnx_path = os.path.join(source_model_dir, "model.onnx")
            temp_onnx_path = os.path.join(target_dir, "temp_model.onnx")
            
            # 複製ONNX文件到目標目錄（暫時使用）
            print(f"複製ONNX文件從 {onnx_path} 到 {temp_onnx_path}")
            shutil.copy2(onnx_path, temp_onnx_path)
            
            # 檢查ONNX文件是否存在
            if not os.path.exists(temp_onnx_path):
                raise Exception(f"ONNX文件不存在: {temp_onnx_path}")
            
            # 使用 trtexec 將 ONNX 轉換為 .plan 檔案
            print("使用 trtexec 將 ONNX 轉換為 .plan 檔案...")
            
            plan_path = os.path.join(target_dir, "model.plan")
            
            # 根據精度類型設置trtexec參數
            precision_flag = ""
            if precision == PrecisionType.FP16:
                precision_flag = "--fp16"
            
            # 設置batch size和workspace
            min_size = parameters.get("min_size", 640)
            opt_size = parameters.get("opt_size", 640) 
            max_size = parameters.get("max_size", 1280)
            
            cmd = [
                "trtexec", 
                f"--onnx={temp_onnx_path}",
                f"--saveEngine={plan_path}",
                precision_flag,
                f"--workspace={workspace*1024}",
                f"--minShapes=images:1x3x{min_size}x{min_size}",
                f"--optShapes=images:{batch}x3x{opt_size}x{opt_size}",
                f"--maxShapes=images:{batch}x3x{max_size}x{max_size}"
            ]
            
            # 過濾掉空參數
            cmd = [c for c in cmd if c]
            
            print(f"執行TensorRT轉換命令: {' '.join(cmd)}")
            
            # 執行命令
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()
            
            stdout_str = stdout.decode('utf-8')
            stderr_str = stderr.decode('utf-8')
            
            print(f"TensorRT轉換輸出: {stdout_str}")
            
            if stderr_str:
                print(f"TensorRT轉換錯誤: {stderr_str}")
            
            if process.returncode != 0:
                raise Exception(f"TensorRT轉換失敗，錯誤代碼: {process.returncode}, 錯誤信息: {stderr_str}")
            
            # 檢查 .plan 文件是否成功生成
            if os.path.exists(plan_path):
                file_size = os.path.getsize(plan_path)
                print(f".plan 文件大小: {file_size / (1024*1024):.2f} MB")
            else:
                raise Exception(f".plan 文件不存在: {plan_path}")
            
            # 清理臨時檔案
            try:
                os.remove(temp_onnx_path)
                os.remove(onnx_path)  # 清理源目錄中的ONNX檔案
                print(f"清理臨時ONNX文件")
            except Exception as e:
                print(f"清理ONNX文件時出錯: {str(e)}")
            
            print(f"轉換流程完成：")
            print(f"  - .engine 檔案（用YOLO export生成，用於驗證和測試）: {target_engine_path}")
            print(f"  - .plan 檔案（用trtexec生成，用於 Triton 部署）: {plan_path}")
            
            # 返回 .engine 文件路徑（保持向後相容性）
            return target_engine_path
        
        except Exception as e:
            error_msg = f"PyTorch轉換TensorRT失敗: {str(e)}"
            print(error_msg)
            import traceback
            print(traceback.format_exc())
            raise Exception(error_msg)
    
    def _create_target_model(self, source_model: ModelInfo, target_path: str,
                            target_format: ModelFormat, precision: PrecisionType) -> ModelInfo:
        """創建目標模型信息"""
        try:
            # 計算文件大小
            file_size_mb = os.path.getsize(target_path) / (1024 * 1024)
            
            # 取得模型目錄（跳過版本目錄）
            model_dir = os.path.dirname(os.path.dirname(target_path))
            model_name = os.path.basename(model_dir)  # 使用目錄名作為模型名稱
            
            # 生成新的唯一模型ID
            model_id = str(uuid.uuid4())
            
            # 創建模型信息
            target_model = ModelInfo(
                id=model_id,
                name=model_name,
                type=source_model.type,
                format=target_format,
                path=target_path,
                size_mb=file_size_mb,
                created_at=datetime.now(timezone.utc),
                description=f"轉換後的模型: {model_name}",
                metadata={
                    "model_id": model_id,
                    "display_name": model_name,
                    "type": source_model.type.value,
                    "format": target_format.value,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "is_trt_model": True,
                    "triton_model_name": model_name,
                    "triton_model_dir": model_dir,
                    "version": "1",
                    "platform": "tensorrt_plan" if target_format == ModelFormat.ENGINE else "onnxruntime_onnx",
                    # 保存源模型信息（使用名稱而非ID）
                    "source_model_name": source_model.name,
                    "source_model_type": source_model.type.value,
                    "source_model_format": source_model.format.value,
                    # 轉換信息
                    "conversion_precision": precision.value,
                    "conversion_target_format": target_format.value
                }
            )
            
            # 保存目標模型到model_service
            self.model_service.save_model(target_model)
            
            return target_model
            
        except Exception as e:
            print(f"創建目標模型信息失敗: {str(e)}")
            raise Exception(f"創建目標模型信息失敗: {str(e)}")
    
    def _load_jobs(self):
        """從文件載入任務數據"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    jobs_data = json.load(f)
                    # 清空現有任務，確保重新載入
                    self.jobs = {}
                    for job_data in jobs_data:
                        job = ConversionJob.parse_obj(job_data)
                        self.jobs[job.id] = job
            except Exception as e:
                print(f"載入轉換任務數據時出錯: {str(e)}")
    
    def _save_jobs(self):
        """保存任務數據到文件"""
        try:
            with open(self.data_file, 'w', encoding="utf-8") as f:
                jobs_data = [job.dict() for job in self.jobs.values()]
                # 處理datetime序列化
                for job_data in jobs_data:
                    job_data['created_at'] = job_data['created_at'].isoformat()
                    if job_data['completed_at']:
                        job_data['completed_at'] = job_data['completed_at'].isoformat()
                json.dump(jobs_data, f, indent=2)
        except Exception as e:
            print(f"保存轉換任務數據時出錯: {str(e)}")

    async def create_job(self, source_model_id: str, target_format: 'ModelFormat', precision: 'PrecisionType', parameters: Optional[Dict[str, Any]] = None) -> 'ConversionJob':
        """
        創建一個新的轉換任務
        """
        from app.models import ConversionJob, ConversionStatus
        import uuid
        
        # 檢查源模型是否存在
        source_model = self.model_service.get_model_by_id(source_model_id)
        if not source_model:
            raise Exception(f"找不到原始模型: {source_model_id}")
        
        # 確保parameters不為None
        if parameters is None:
            parameters = {}
        
        # 添加預設值
        if "batch_size" not in parameters:
            parameters["batch_size"] = 1
        if "imgsz" not in parameters:
            parameters["imgsz"] = 640
        if "workspace" not in parameters:
            parameters["workspace"] = 4
        
        # 創建轉換任務
        job_id = str(uuid.uuid4())
        conversion_job = ConversionJob(
            id=job_id,
            source_model_id=source_model.id,
            target_format=target_format,
            precision=precision,
            status=ConversionStatus.PENDING,
            created_at=datetime.now(timezone(timedelta(hours=8))),
            parameters=parameters
        )
        
        # 保存任務信息
        saved_job = self.save_job(conversion_job)
        
        # 在背景中執行轉換任務
        self.process_conversion_job(job_id=job_id)
        
        return saved_job

    async def wait_for_job_completion(self, job_id: str, timeout: int = 600) -> Optional[str]:
        """
        等待轉換任務完成，並返回轉換後的模型ID
        """
        from app.models import ConversionStatus
        import asyncio
        import time
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # 每次檢查前重新載入任務，確保獲取最新狀態
            self._load_jobs()
            
            # 檢查任務狀態
            job = self.get_job_by_id(job_id)
            
            if not job:
                print(f"找不到轉換任務: {job_id}")
                return None
            
            if job.status == ConversionStatus.COMPLETED:
                print(f"轉換任務完成，目標模型ID: {job.target_model_id}")
                return job.target_model_id
            
            if job.status == ConversionStatus.FAILED:
                print(f"轉換任務失敗: {job.error_message}")
                return None
            
            # 檢查檔案系統，看目標模型是否已實際創建
            # 如果已創建但任務狀態尚未更新，則更新狀態
            if job.target_model_id:
                print(f"任務 {job_id} 的目標模型ID已設置，但狀態為 {job.status}，嘗試主動檢查目標模型")
                model = self.model_service.get_model_by_id(job.target_model_id)
                if model:
                    print(f"找到目標模型 {job.target_model_id}，強制將任務標記為完成")
                    job.status = ConversionStatus.COMPLETED
                    job.completed_at = datetime.now(timezone(timedelta(hours=8)))
                    self._save_jobs()
                    return job.target_model_id
            
            # 掃描模型目錄，檢查是否存在新的轉換模型
            try:
                self.model_service._scan_model_repository()
                print(f"已掃描模型目錄，繼續等待任務 {job_id} 完成")
            except Exception as e:
                print(f"掃描模型目錄時出錯: {str(e)}")
            
            # 等待10秒再檢查
            await asyncio.sleep(10)
        
        print(f"轉換任務等待超時: {job_id}")
        return None 

    def list_active_conversion_tasks(self):
        """
        獲取所有正在進行中的轉換任務
        
        Returns:
            正在進行中的轉換任務列表（狀態為pending或processing）
        """
        try:
            # 確保任務數據是最新的
            self._load_jobs()
            
            active_tasks = []
            for job_id, job in self.jobs.items():
                if job.status in [ConversionStatus.PENDING, ConversionStatus.PROCESSING]:
                    active_tasks.append(job)
            
            print(f"找到 {len(active_tasks)} 個活躍的轉換任務")
            return active_tasks
        except Exception as e:
            print(f"列出活躍轉換任務時出錯: {str(e)}")
            return [] 