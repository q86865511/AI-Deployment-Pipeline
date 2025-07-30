import os
import json
import uuid
import time
import asyncio
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

from app.models import ModelFormat, PrecisionType
from app.services.model_service import ModelService
from app.services.conversion_service import ConversionService
from app.services.inference_service import InferenceService

class TestManager:
    """
    測試管理服務，負責處理自動化管線流程
    包括模型轉換、驗證和推論測試
    """
    def __init__(self):
        """初始化測試管理服務"""
        self.model_service = ModelService()
        self.conversion_service = ConversionService()
        self.inference_service = InferenceService()
        
        # 測試任務數據存儲路徑
        self.data_dir = "data/test_tasks"
        self.tasks_file = os.path.join(self.data_dir, "test_tasks.json")
        self.inference_results_dir = os.path.join(self.data_dir, "inference_results")
        
        # 確保數據目錄存在
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.inference_results_dir, exist_ok=True)
        
        # 測試任務字典
        self.tasks = {}
        
        # 載入現有任務
        self._load_tasks()
    
    def _load_tasks(self):
        """從文件載入任務數據"""
        if os.path.exists(self.tasks_file):
            try:
                with open(self.tasks_file, 'r', encoding='utf-8') as f:
                    saved_tasks = json.load(f)
                    
                # 重建完整的任務結構
                self.tasks = {}
                for task_id, saved_task in saved_tasks.items():
                    # 重建 combinations 數組（如果任務還在進行中）
                    combinations = []
                    if saved_task["status"] in ["pending", "processing"]:
                        # 根據保存的參數重建組合
                        # 這裡我們需要從保存的數據推斷批次大小和精度
                        # 由於我們沒有保存這些詳細信息，所以只能創建空的組合數組
                        for i in range(saved_task["total_combinations"]):
                            combinations.append({
                                "batch_size": None,
                                "precision": None,
                                "image_size": None,
                                "status": "pending",
                                "conversion_job_id": None,
                                "target_model_id": None,
                                "inference_results": None,
                                "error": None
                            })
                    
                    # 重建完整的任務結構
                    task = {
                        "id": saved_task["id"],
                        "model_id": saved_task["model_id"],
                        "model_name": saved_task["model_name"],
                        "model_type": saved_task.get("model_type", "object"),
                        "created_at": saved_task["created_at"],
                        "status": saved_task["status"],
                        "current_step": saved_task.get("current_step", "conversion"),
                        "current_combination_index": saved_task.get("current_combination_index", 0),
                        "total_combinations": saved_task["total_combinations"],
                        "completed_combinations": saved_task.get("completed_combinations", 0),
                        "combinations": combinations,
                        "dataset_id": saved_task["dataset_id"],
                        "iterations": saved_task["iterations"],
                        "custom_params": saved_task.get("custom_params", {}),
                        "completed_at": saved_task.get("completed_at"),
                        "error": saved_task.get("error"),
                        "partial_success": saved_task.get("partial_success"),
                        "success_count": saved_task.get("success_count"),
                        "fail_count": saved_task.get("fail_count")
                    }
                    
                    self.tasks[task_id] = task
                    
            except Exception as e:
                print(f"載入測試任務數據時出錯: {str(e)}")
                self.tasks = {}
    
    def _save_tasks(self):
        """保存任務狀態到文件（只保存必要的設定參數）"""
        # 創建簡化版的任務數據，只保存設定參數
        simplified_tasks = {}
        for task_id, task in self.tasks.items():
            simplified_tasks[task_id] = {
                "id": task["id"],
                "model_id": task["model_id"],
                "model_name": task["model_name"],
                "model_type": task.get("model_type", "object"),
                "dataset_id": task["dataset_id"],
                "iterations": task["iterations"],
                "custom_params": task["custom_params"],
                "created_at": task["created_at"],
                "completed_at": task.get("completed_at"),
                "status": task["status"],
                "current_step": task.get("current_step", "conversion"),
                "current_combination_index": task.get("current_combination_index", 0),
                "total_combinations": task["total_combinations"],
                "completed_combinations": task.get("completed_combinations", 0),
                "error": task.get("error"),
                "partial_success": task.get("partial_success"),
                "success_count": task.get("success_count"),
                "fail_count": task.get("fail_count")
            }
        
        with open(self.tasks_file, 'w', encoding='utf-8') as f:
            json.dump(simplified_tasks, f, indent=2, ensure_ascii=False)
    
    def create_test_task(self, model_id: str, batch_sizes: List[int], precisions: List[str], 
                         image_size: int, iterations: int, dataset_id: Optional[str] = None,
                         model_type: str = "object", 
                         custom_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        創建新的測試任務
        
        Args:
            model_id: 源模型ID
            batch_sizes: 批次大小列表
            precisions: 精度選項列表
            image_size: 圖像尺寸
            iterations: 迭代次數
            dataset_id: 數據集ID
            model_type: 模型類型，'object'或'pose'
            custom_params: 自定義參數
        """
        # 生成任務ID
        task_id = str(uuid.uuid4())
        
        # 獲取模型信息
        model = self.model_service.get_model_by_id(model_id)
        if not model:
            raise Exception(f"找不到模型: {model_id}")
        
        # 檢查模型類型
        if model_type not in ["object", "pose"]:
            raise Exception(f"不支持的模型類型: {model_type}，必須為 'object' 或 'pose'")
        
        # 生成所有測試組合，包含原始模型
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
        
        # 創建任務數據
        task = {
            "id": task_id,
            "model_id": model_id,
            "model_name": model.name,
            "model_type": model_type,  # 添加模型類型
            "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
            "status": "pending",
            "current_step": "conversion",  # conversion, validation, inference
            "current_combination_index": 0,
            "total_combinations": len(combinations),
            "completed_combinations": 0,
            "combinations": combinations,
            "dataset_id": dataset_id,
            "iterations": iterations,
            "custom_params": custom_params or {},
            "completed_at": None,
            "error": None
        }
        
        # 保存任務
        self.tasks[task_id] = task
        self._save_tasks()
        
        # 返回任務信息
        return task
    
    async def start_test_task(self, task_id: str):
        """
        啟動測試任務
        """
        task = self.tasks.get(task_id)
        if not task:
            raise Exception(f"找不到測試任務: {task_id}")
        
        # 更新任務狀態
        task["status"] = "processing"
        self._save_tasks()
        
        try:
            # 開始處理轉換任務
            await self._process_conversion_step(task)
            
            # 如果轉換成功，進行驗證步驟
            if task["status"] != "failed" and task["status"] != "aborted":
                task["current_step"] = "validation"
                task["current_combination_index"] = 0
                self._save_tasks()
                
                await self._process_validation_step(task)
            
            # 如果驗證成功，進行推理測試步驟
            if task["status"] != "failed" and task["status"] != "aborted":
                task["current_step"] = "inference"
                task["current_combination_index"] = 0
                self._save_tasks()
                
                await self._process_inference_step(task)
            
            # 完成所有步驟，更新任務狀態
            if task["status"] != "failed" and task["status"] != "aborted":
                # 統計成功和失敗的組合數
                successful_combinations = 0
                failed_combinations = 0
                for combination in task["combinations"]:
                    # 檢查轉換是否成功
                    if combination.get("status") == "completed":
                        # 檢查驗證和推理狀態
                        validation_success = combination.get("validation_status") == "completed"
                        inference_success = combination.get("inference_status") == "completed"
                        
                        # 需要驗證和推理都成功才算成功
                        if validation_success and inference_success:
                            successful_combinations += 1
                        else:
                            failed_combinations += 1
                    else:
                        failed_combinations += 1
                
                # 如果至少有一個組合成功，則任務視為完成（部分成功）
                if successful_combinations > 0:
                    task["status"] = "completed"
                    task["completed_at"] = datetime.now(timezone(timedelta(hours=8))).isoformat()
                    if failed_combinations > 0:
                        task["partial_success"] = True
                        task["success_count"] = successful_combinations
                        task["fail_count"] = failed_combinations
                        print(f"任務 {task_id} 部分完成：成功 {successful_combinations} 個，失敗 {failed_combinations} 個")
                    else:
                        print(f"任務 {task_id} 全部完成")
                    self._save_tasks()
                else:
                    # 如果所有組合都失敗，則任務失敗
                    task["status"] = "failed"
                    task["error"] = "所有測試組合都失敗"
                    task["completed_at"] = datetime.now(timezone(timedelta(hours=8))).isoformat()
                    self._save_tasks()
                    print(f"任務 {task_id} 失敗：所有組合都未能成功完成")
            
        except Exception as e:
            print(f"處理測試任務出錯: {str(e)}")
            # 更新任務狀態為失敗
            task["status"] = "failed"
            task["error"] = str(e)
            self._save_tasks()
    
    async def _process_conversion_step(self, task: Dict[str, Any]):
        """處理模型轉換步驟"""
        model_id = task["model_id"]
        combinations = task["combinations"]
        
        # 遍歷所有組合
        for i, combination in enumerate(combinations):
            # 檢查任務是否被中止
            if task["status"] == "aborted":
                print(f"任務 {task['id']} 已被中止")
                return
            
            # 更新當前處理的組合索引
            task["current_combination_index"] = i
            self._save_tasks()
            
            # 允許其他異步操作執行
            await asyncio.sleep(0.1)
            
            try:
                # 獲取組合參數
                batch_size = combination["batch_size"]
                precision = combination["precision"]
                image_size = combination["image_size"]
                is_original = combination.get("is_original", False)
                
                print(f"處理組合 {i+1}/{len(combinations)}: 批次={batch_size}, 精度={precision}")
                
                # 如果是原始模型，跳過轉換步驟
                if is_original:
                    print(f"跳過原始模型轉換: 組合 {i+1}")
                    combination["status"] = "completed"
                    task["completed_combinations"] += 1
                    self._save_tasks()
                    continue
                
                # 準備轉換參數
                precision_enum = PrecisionType.FP16 if precision.lower() == 'fp16' else PrecisionType.FP32
                parameters = {
                    "batch_size": batch_size,
                    "imgsz": image_size,
                    "workspace": 4  # 預設工作空間大小
                }
                
                # 檢查是否已存在相同參數的模型
                existing_model = self._find_existing_model(model_id, precision_enum, parameters)
                if existing_model:
                    print(f"找到現有模型: {existing_model.name} (ID: {existing_model.id})")
                    combination["target_model_id"] = existing_model.id
                    combination["status"] = "completed"
                    task["completed_combinations"] += 1
                    self._save_tasks()
                    continue
                
                # 檢查是否有活躍的相同參數轉換任務
                existing_job = self._find_active_conversion_job(model_id, precision_enum, parameters)
                if existing_job:
                    print(f"找到進行中的轉換任務: {existing_job.id}")
                    combination["conversion_job_id"] = existing_job.id
                    combination["status"] = "processing"
                    self._save_tasks()
                    
                    # 等待轉換任務完成
                    target_model_id = await self.conversion_service.wait_for_job_completion(existing_job.id)
                    
                    if target_model_id:
                        combination["target_model_id"] = target_model_id
                        combination["status"] = "completed"
                        task["completed_combinations"] += 1
                        self._save_tasks()
                    else:
                        raise Exception(f"轉換任務 {existing_job.id} 失敗")
                    
                    continue
                
                # 創建新的轉換任務
                combination["status"] = "processing"
                self._save_tasks()
                
                conversion_job = await self.conversion_service.create_job(
                    source_model_id=model_id,
                    target_format=ModelFormat.ENGINE,
                    precision=precision_enum,
                    parameters=parameters
                )
                
                combination["conversion_job_id"] = conversion_job.id
                self._save_tasks()
                
                # 等待轉換任務完成
                target_model_id = await self.conversion_service.wait_for_job_completion(conversion_job.id)
                
                if target_model_id:
                    combination["target_model_id"] = target_model_id
                    combination["status"] = "completed"
                    task["completed_combinations"] += 1
                    self._save_tasks()
                else:
                    raise Exception(f"轉換任務 {conversion_job.id} 失敗")
                
            except Exception as e:
                print(f"處理組合 {i+1} 時出錯: {str(e)}")
                combination["status"] = "failed"
                combination["error"] = str(e)
                self._save_tasks()
                
                # 繼續下一個組合，不中斷整個任務
    
    async def _process_validation_step(self, task: Dict[str, Any]):
        """處理模型驗證步驟"""
        combinations = task["combinations"]
        dataset_id = task["dataset_id"]
        model_type = task.get("model_type", "object")  # 獲取模型類型，預設為object
        
        # 打印調試信息
        print(f"\n=== 開始驗證步驟 ===")
        print(f"任務ID: {task['id']}")
        print(f"模型類型: {model_type}")
        print(f"數據集ID: {dataset_id}")
        
        # 打印所有組合的狀態
        for i, comb in enumerate(combinations):
            print(f"組合 {i+1}: 狀態={comb.get('status')}, 目標模型ID={comb.get('target_model_id')}")
        
        # 驗證數據集現在是必須的
        if not dataset_id:
            raise Exception(f"任務 {task['id']} 未提供數據集，無法進行驗證")
        
        # 獲取數據集信息
        datasets_dir = os.path.join("data", "datasets")
        datasets_json = os.path.join(datasets_dir, "datasets.json")
        
        dataset_info = None
        if os.path.exists(datasets_json):
            with open(datasets_json, "r", encoding="utf-8") as f:
                datasets = json.load(f)
                for dataset in datasets:
                    if dataset["id"] == dataset_id:
                        dataset_info = dataset
                        break
        
        if not dataset_info:
            raise Exception(f"找不到數據集: {dataset_id}")
        
        # 根據模型類型和數據集類型選擇適當的YAML文件
        yaml_file = None
        dataset_path = dataset_info["path"]
        
        # 檢查數據集是否有與模型類型匹配的YAML文件
        if model_type == "object":
            object_yaml = os.path.join(dataset_path, "coco_object.yaml")
            if os.path.exists(object_yaml):
                yaml_file = object_yaml
        elif model_type == "pose":
            pose_yaml = os.path.join(dataset_path, "coco_pose.yaml")
            if os.path.exists(pose_yaml):
                yaml_file = pose_yaml
        
        # 如果沒有找到匹配的YAML文件，使用默認的
        if not yaml_file:
            # 嘗試使用數據集中的任何YAML文件
            for root, _, files in os.walk(dataset_path):
                for file_name in files:
                    if file_name.endswith('.yaml'):
                        yaml_file = os.path.join(root, file_name)
                        print(f"使用默認YAML文件: {yaml_file}")
                        break
                if yaml_file:
                    break
        
        if not yaml_file:
            raise Exception(f"數據集 {dataset_id} 中找不到YAML配置文件")
        
        print(f"開始執行模型驗證步驟，使用YAML文件: {yaml_file}")
        
        # 遍歷所有組合
        for i, combination in enumerate(combinations):
            # 檢查任務是否被中止
            if task["status"] == "aborted":
                print(f"任務 {task['id']} 已被中止")
                return
            
            # 更新當前處理的組合索引
            task["current_combination_index"] = i
            self._save_tasks()
            
            # 允許其他異步操作執行
            await asyncio.sleep(0.1)
            
            # 只處理轉換成功的組合
            if combination["status"] != "completed" or not combination["target_model_id"]:
                continue
            
            try:
                # 獲取組合參數
                target_model_id = combination["target_model_id"]
                batch_size = combination["batch_size"]
                
                print(f"驗證組合 {i+1}/{len(combinations)}: 模型ID={target_model_id}, 批次={batch_size}")
                
                # 驗證模型是否存在
                self.model_service._scan_model_repository()  # 重新掃描確保最新
                target_model = self.model_service.get_model_by_id(target_model_id)
                if not target_model:
                    raise Exception(f"找不到模型: {target_model_id}")
                
                # 設置驗證狀態
                combination["validation_status"] = "processing"
                self._save_tasks()
                
                # 將YAML文件路徑添加到自定義參數中
                custom_params = task["custom_params"].copy()
                custom_params["yaml_file"] = yaml_file
                custom_params["model_type"] = model_type
                
                # 確保InferenceService已經初始化
                inference_service = InferenceService()
                inference_service.model_service = self.model_service  # 使用相同的模型服務實例
                
                # 執行模型驗證（添加異步睡眠以允許其他操作）
                try:
                    # 測試連接和引用
                    print(f"準備執行驗證: 模型ID={target_model_id}, 模型名稱={target_model.name}, 數據集ID={dataset_id}")
                    
                    # 短暫睡眠以允許其他異步操作執行
                    await asyncio.sleep(0.1)
                    
                    validation_results = await inference_service.validate_model(
                        model_id=target_model_id,
                        dataset_id=dataset_id,
                        batch_size=batch_size,
                        custom_params=custom_params
                    )
                    
                    # 保存驗證結果
                    combination["validation_results"] = validation_results
                    combination["validation_status"] = "completed"
                    print(f"驗證完成: 模型ID={target_model_id}, 結果={validation_results}")
                except Exception as validation_error:
                    print(f"驗證執行錯誤: {str(validation_error)}")
                    # 如果驗證執行失敗，記錄錯誤但繼續進行
                    combination["validation_error"] = str(validation_error)
                    combination["validation_status"] = "failed"
                    # 創建一個模擬的驗證結果，以便流程可以繼續
                    combination["validation_results"] = {
                        "model_id": target_model_id,
                        "dataset_id": dataset_id,
                        "error": str(validation_error),
                        "metrics": {
                            "precision": 0.0,
                            "recall": 0.0,
                            "mAP50": 0.0,
                            "mAP50_95": 0.0
                        }
                    }
                
                self._save_tasks()
                
            except Exception as e:
                print(f"驗證組合 {i+1} 時出錯: {str(e)}")
                combination["validation_status"] = "failed"
                combination["validation_error"] = str(e)
                self._save_tasks()
                
                # 繼續下一個組合，不中斷整個任務
            
        print(f"模型驗證步驟完成，完成組合數: {task['completed_combinations']}/{len(combinations)}")
    
    async def _process_inference_step(self, task: Dict[str, Any]):
        """處理推論測試步驟"""
        combinations = task["combinations"]
        iterations = task["iterations"]
        
        print(f"開始執行推論測試步驟，迭代次數: {iterations}")
        
        # 重置推論階段的完成計數，使用獨立計數器
        inference_completed = 0
        
        # 遍歷所有組合
        for i, combination in enumerate(combinations):
            # 檢查任務是否被中止
            if task["status"] == "aborted":
                print(f"任務 {task['id']} 已被中止")
                return
            
            # 更新當前處理的組合索引
            task["current_combination_index"] = i
            self._save_tasks()
            
            # 允許其他異步操作執行
            await asyncio.sleep(0.1)
            
            # 只處理轉換成功的組合
            if combination["status"] != "completed" or not combination["target_model_id"]:
                continue
            
            try:
                # 獲取組合參數
                target_model_id = combination["target_model_id"]
                batch_size = combination["batch_size"]
                image_size = combination["image_size"]
                
                print(f"推論測試組合 {i+1}/{len(combinations)}: 模型ID={target_model_id}, 批次={batch_size}")
                
                # 驗證模型是否存在
                self.model_service._scan_model_repository()  # 重新掃描確保最新
                target_model = self.model_service.get_model_by_id(target_model_id)
                if not target_model:
                    raise Exception(f"找不到模型: {target_model_id}")
                
                # 設置推理測試狀態
                combination["inference_status"] = "processing"
                self._save_tasks()
                
                # 確保InferenceService已經初始化
                inference_service = InferenceService()
                inference_service.model_service = self.model_service  # 使用相同的模型服務實例
                
                # 執行推論測試（添加異步睡眠以允許其他操作）
                try:
                    # 測試連接和引用
                    print(f"準備執行推論測試: 模型ID={target_model_id}, 模型名稱={target_model.name}, 批次={batch_size}")
                    
                    # 短暫睡眠以允許其他異步操作執行
                    await asyncio.sleep(0.1)
                    
                    inference_results = await inference_service.benchmark_model(
                        model_id=target_model_id,
                        batch_size=batch_size,
                        num_iterations=iterations,
                        image_size=image_size,
                        test_dataset=task["dataset_id"]
                    )
                    
                    # 保存推理結果
                    combination["inference_results"] = inference_results
                    
                    # 檢查推理狀態
                    if inference_results.get("status") == "failed":
                        combination["inference_status"] = "failed"
                        combination["inference_error"] = inference_results.get("error", "推論測試失敗")
                        print(f"推論測試失敗: 模型ID={target_model_id}, 錯誤={inference_results.get('error')}")
                    else:
                        combination["inference_status"] = "completed"
                        print(f"推論測試完成: 模型ID={target_model_id}, 結果={inference_results}")
                        
                except Exception as inference_error:
                    print(f"推論測試執行錯誤: {str(inference_error)}")
                    # 如果推論測試執行失敗，記錄錯誤但繼續進行
                    combination["inference_error"] = str(inference_error)
                    combination["inference_status"] = "failed"
                    # 創建一個模擬的推論測試結果
                    combination["inference_results"] = {
                        "model_id": target_model_id,
                        "error": str(inference_error),
                        "avg_inference_time_ms": 0.0,
                        "throughput_fps": 0.0,
                        "status": "failed"
                    }
                
                # 更新推論完成計數（無論成功或失敗）
                inference_completed += 1
                
                # 即時保存任務狀態，確保前端能看到最新結果
                self._save_tasks()
                
                # 為前端顯示產生當前組合的臨時結果
                self._update_current_combination_result(task, i, combination)
                
                # 保存結果到獨立文件（使用新的目錄結構）
                task_results_dir = os.path.join(
                    self.inference_results_dir, 
                    f"task_{task['id']}"
                )
                os.makedirs(task_results_dir, exist_ok=True)
                
                result_file = os.path.join(
                    task_results_dir,
                    f"combination_{i+1}_batch{batch_size}_{combination['precision']}.json"
                )
                
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "task_id": task["id"],
                        "combination_index": i,
                        "model_id": target_model_id,
                        "model_name": target_model.name if target_model else "unknown",
                        "batch_size": batch_size,
                        "precision": combination["precision"],
                        "image_size": image_size,
                        "iterations": iterations,
                        "validation_results": combination.get("validation_results"),
                        "inference_results": combination["inference_results"],
                        "timestamp": datetime.now(timezone(timedelta(hours=8))).isoformat()
                    }, f, indent=2, ensure_ascii=False)
                
            except Exception as e:
                print(f"性能測試組合 {i+1} 時出錯: {str(e)}")
                import traceback
                print(traceback.format_exc())
                combination["inference_status"] = "failed"
                combination["inference_error"] = str(e)
                inference_completed += 1  # 即使失敗也計入完成數
                self._save_tasks()
                
                # 繼續下一個組合，不中斷整個任務
        
        print(f"推論測試步驟完成，完成組合數: {inference_completed}/{len(combinations)}")
        
        # 生成最終的測試結果JSON文件
        self._generate_final_results(task)
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """獲取測試任務信息"""
        return self.tasks.get(task_id)
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """獲取所有測試任務信息"""
        return list(self.tasks.values())
    
    def abort_task(self, task_id: str) -> bool:
        """中止測試任務"""
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        # 檢查任務是否已完成
        if task["status"] in ["completed", "failed", "aborted"]:
            return True
        
        # 更新任務狀態為中止
        task["status"] = "aborted"
        task["completed_at"] = datetime.now(timezone(timedelta(hours=8))).isoformat()
        
        # 終止相關的轉換任務
        for combination in task["combinations"]:
            if combination["status"] == "processing" and combination["conversion_job_id"]:
                try:
                    job = self.conversion_service.get_job_by_id(combination["conversion_job_id"])
                    if job:
                        print(f"終止轉換任務: {job.id}")
                        self.conversion_service.delete_job(job.id)
                except Exception as e:
                    print(f"終止轉換任務時出錯: {str(e)}")
        
        # 保存任務狀態
        self._save_tasks()
        return True
    
    def delete_task(self, task_id: str) -> bool:
        """刪除測試任務"""
        if task_id not in self.tasks:
            return False
        
        # 檢查任務是否正在進行中
        task = self.tasks[task_id]
        if task["status"] == "processing":
            # 先中止任務
            self.abort_task(task_id)
        
        # 刪除任務相關的結果文件
        try:
            # 刪除任務專屬的結果目錄
            task_results_dir = os.path.join(self.inference_results_dir, f"task_{task_id}")
            if os.path.exists(task_results_dir):
                import shutil
                shutil.rmtree(task_results_dir)
                print(f"已刪除任務結果目錄: {task_results_dir}")
            
            # 刪除舊版本的結果文件（如果存在）
            # 這是為了兼容性，可以在之後移除
            old_files = [
                f"final_results_{task_id}.json",
                f"performance_analysis_{task_id}.json"
            ]
            for filename in old_files:
                old_file = os.path.join(self.inference_results_dir, filename)
                if os.path.exists(old_file):
                    os.remove(old_file)
                    print(f"已刪除舊版本文件: {old_file}")
            
            # 刪除可能存在的單個結果文件
            for file in os.listdir(self.inference_results_dir):
                if file.startswith(f"{task_id}_") and file.endswith(".json"):
                    file_path = os.path.join(self.inference_results_dir, file)
                    os.remove(file_path)
                    print(f"已刪除結果文件: {file_path}")
        
        except Exception as e:
            print(f"刪除任務文件時出錯: {str(e)}")
        
        # 從字典中刪除任務
        del self.tasks[task_id]
        
        # 保存任務狀態
        self._save_tasks()
        return True
    
    def _find_existing_model(self, source_model_id: str, precision: PrecisionType, parameters: Dict[str, Any]) -> Optional[Any]:
        """查找是否已存在相同參數的模型"""
        # 確保模型服務已刷新
        self.model_service._scan_model_repository()
        
        batch_size = parameters.get("batch_size", 1)
        image_size = parameters.get("imgsz", 640)
        
        # 獲取源模型
        source_model = self.model_service.get_model_by_id(source_model_id)
        if not source_model:
            print(f"找不到源模型: {source_model_id}")
            return None
        
        # 構建可能的目標模型名稱模式
        model_prefix = source_model.name.split('_')[0] if source_model.name else ""
        precision_str = precision.value
        
        print(f"查找現有模型: 源模型={source_model.name}, 精度={precision_str}, 批次={batch_size}, 尺寸={image_size}")
        
        # 查找符合條件的模型
        for model_id, model in self.model_service.models.items():
            # 檢查模型格式
            if model.format != ModelFormat.ENGINE:
                continue
            
            # 檢查模型元數據
            metadata = model.metadata or {}
            
            # 檢查批次大小和圖像尺寸
            if (metadata.get("source_model_id") == source_model_id and
                metadata.get("precision") == precision.value and
                metadata.get("conversion_parameters", {}).get("batch_size") == batch_size and
                metadata.get("conversion_parameters", {}).get("imgsz") == image_size):
                print(f"通過元數據找到匹配模型: {model.name} (ID: {model.id})")
                return model
            
            # 檢查模型名稱模式 - 更靈活的匹配
            # 例如: test_engine_fp32_batch1_size640
            # 或者: test_engine_fp16_batch1_size640
            model_name_lower = (model.name or "").lower()
            
            # 檢查是否包含必要的參數信息
            has_precision = precision_str.lower() in model_name_lower
            has_batch = f"batch{batch_size}" in model_name_lower or f"batch_{batch_size}" in model_name_lower
            has_size = f"size{image_size}" in model_name_lower or f"size_{image_size}" in model_name_lower
            
            # 檢查是否是同一個源模型的轉換結果
            # 可能的名稱格式：
            # 1. {source_name}_engine_{precision}_batch{batch}_size{size}
            # 2. {source_name}_engine_{precision}_batch_{batch}_size_{size}
            # 3. 其他變體
            is_from_same_source = False
            
            # 檢查名稱前綴
            if model_prefix and model_name_lower.startswith(model_prefix.lower()):
                is_from_same_source = True
            
            # 或者檢查元數據中的源模型信息（即使元數據不完整）
            if metadata.get("source_model_id") == source_model_id:
                is_from_same_source = True
            
            # 如果所有條件都滿足，則認為找到了匹配的模型
            if has_precision and has_batch and has_size and is_from_same_source:
                print(f"通過名稱模式找到匹配模型: {model.name} (ID: {model.id})")
                return model
        
        print(f"未找到匹配的現有模型")
        return None
    
    def _find_active_conversion_job(self, source_model_id: str, precision: PrecisionType, parameters: Dict[str, Any]) -> Optional[Any]:
        """查找是否有活躍的相同參數轉換任務"""
        # 確保轉換服務已刷新
        self.conversion_service._load_jobs()
        
        batch_size = parameters.get("batch_size", 1)
        image_size = parameters.get("imgsz", 640)
        
        # 查找符合條件的活躍任務
        for job_id, job in self.conversion_service.jobs.items():
            from app.models import ConversionStatus
            
            # 只檢查活躍的任務
            if job.status not in [ConversionStatus.PENDING, ConversionStatus.PROCESSING]:
                continue
            
            # 檢查參數是否匹配
            if (job.source_model_id == source_model_id and
                job.precision == precision and
                job.parameters.get("batch_size") == batch_size and
                job.parameters.get("imgsz") == image_size):
                return job
        
        return None
    
    def _generate_final_results(self, task: Dict[str, Any]):
        """生成最終的測試結果JSON文件"""
        try:
            # 創建任務專屬的結果目錄
            task_results_dir = os.path.join(self.inference_results_dir, f"task_{task['id']}")
            os.makedirs(task_results_dir, exist_ok=True)
            
            # 收集所有測試結果
            results = []
            for i, combination in enumerate(task["combinations"]):
                # 判斷組合的狀態
                combination_status = "pending"
                errors = {}
                
                # 檢查轉換狀態
                if combination.get("status") == "failed":
                    combination_status = "failed"
                    errors["conversion"] = combination.get("error")
                elif combination.get("status") == "completed":
                    # 檢查驗證狀態
                    if combination.get("validation_status") == "failed":
                        combination_status = "failed"
                        errors["validation"] = combination.get("validation_error")
                    
                    # 檢查推理狀態
                    elif combination.get("inference_status") == "failed":
                        combination_status = "failed"
                        errors["inference"] = combination.get("inference_error")
                    
                    # 如果都成功，則標記為成功
                    elif (combination.get("validation_status") == "completed" and 
                          combination.get("inference_status") == "completed"):
                        combination_status = "completed"
                    else:
                        # 部分完成或仍在處理中
                        combination_status = "failed"
                
                # 構建結果記錄
                result = {
                    "combination_index": i,
                    "batch_size": combination["batch_size"],
                    "precision": combination["precision"],
                    "image_size": combination["image_size"],
                    "status": combination_status,
                    "model_id": combination.get("target_model_id"),
                    "errors": errors if errors else None,
                }
                
                # 如果有驗證結果，添加關鍵指標和GPU監控數據
                validation_results = combination.get("validation_results")
                if validation_results and combination.get("validation_status") == "completed":
                    metrics = validation_results.get("metrics", {})
                    result["validation_metrics"] = {
                        "mAP50": metrics.get("mAP50", 0.0),
                        "mAP50_95": metrics.get("mAP50_95", 0.0),
                        "precision": metrics.get("precision", 0.0),
                        "recall": metrics.get("recall", 0.0)
                    }
                
                # 如果有推理結果，添加性能指標
                inference_results = combination.get("inference_results")
                if inference_results and combination.get("inference_status") == "completed":
                    result["performance_metrics"] = {
                        "avg_inference_time_ms": inference_results.get("avg_inference_time_ms", 0.0),
                        "throughput_fps": inference_results.get("avg_throughput_fps", 0.0),
                        # 優先使用驗證結果中的GPU監控數據，如果沒有則使用推理結果中的
                        "memory_usage_mb": validation_results.get("memory_usage_mb", inference_results.get("memory_usage_mb", 0.0)) if validation_results else inference_results.get("memory_usage_mb", 0.0),
                        "avg_gpu_load": validation_results.get("avg_gpu_load", inference_results.get("avg_gpu_load", 0.0)) if validation_results else inference_results.get("avg_gpu_load", 0.0),
                        # 添加更多GPU監控字段
                        "max_gpu_load": validation_results.get("max_gpu_load", 0.0) if validation_results else 0.0,
                        "model_vram_mb": validation_results.get("model_vram_mb", 0.0) if validation_results else 0.0,
                        "monitoring_samples": validation_results.get("monitoring_samples", 0) if validation_results else 0,
                        "monitoring_duration_s": validation_results.get("monitoring_duration_s", 0.0) if validation_results else 0.0
                    }
                
                results.append(result)
            
            # 構建最終的結果文件
            final_results = {
                "task_id": task["id"],
                "model_name": task["model_name"],
                "model_type": task.get("model_type", "object"),
                "dataset_id": task["dataset_id"],
                "created_at": task["created_at"],
                "completed_at": task.get("completed_at", datetime.now(timezone(timedelta(hours=8))).isoformat()),
                "status": "completed",  # 修正：始終設為completed，因為此方法只在任務完成時調用
                "total_combinations": task["total_combinations"],
                "completed_combinations": task["completed_combinations"],
                "partial_success": task.get("partial_success", False),
                "success_count": task.get("success_count", 0),
                "fail_count": task.get("fail_count", 0),
                "test_configurations": {
                    "iterations": task["iterations"],
                    "custom_params": task["custom_params"]
                },
                "results": results
            }
            
            # 同時更新內存中的任務狀態
            task["status"] = "completed"
            if not task.get("completed_at"):
                task["completed_at"] = datetime.now(timezone(timedelta(hours=8))).isoformat()
            self._save_tasks()
            
            # 保存最終結果
            final_results_file = os.path.join(task_results_dir, "final_results.json")
            with open(final_results_file, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False)
            
            print(f"已生成最終測試結果文件: {final_results_file}")
            
            # 生成結果分析格式的文件（供可視化使用）
            self._generate_performance_analysis_format(task, final_results, task_results_dir)
            
        except Exception as e:
            print(f"生成最終結果文件時出錯: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    def _generate_performance_analysis_format(self, task: Dict[str, Any], results: Dict[str, Any], task_results_dir: str):
        """生成結果分析頁面所需的格式"""
        try:
            performance_data = {}
            
            # 只包含成功的組合
            for result in results["results"]:
                if result["status"] == "completed" and result.get("model_id"):
                    model_id = result["model_id"]
                    
                    if model_id not in performance_data:
                        # 獲取模型信息
                        model_info = self.model_service.get_model_by_id(model_id)
                        model_name = model_info.name if model_info else model_id
                        
                        performance_data[model_id] = {
                            "model_name": model_name,
                            "benchmarks": []
                        }
                    
                    # 構建基準測試記錄
                    benchmark = {
                        "batch_size": result["batch_size"],
                        "precision": result["precision"],
                        "image_size": result["image_size"],
                        "timestamp": datetime.now(timezone(timedelta(hours=8))).isoformat()
                    }
                    
                    # 添加驗證指標
                    if result.get("validation_metrics"):
                        benchmark.update({
                            "map50": result["validation_metrics"]["mAP50"],
                            "map50_95": result["validation_metrics"]["mAP50_95"],
                            "precision": result["validation_metrics"]["precision"],
                            "recall": result["validation_metrics"]["recall"]
                        })
                    
                    # 添加性能指標
                    if result.get("performance_metrics"):
                        benchmark.update({
                            "avg_inference_time_ms": result["performance_metrics"]["avg_inference_time_ms"],
                            "throughput_fps": result["performance_metrics"]["throughput_fps"],
                            "memory_usage_mb": result["performance_metrics"]["memory_usage_mb"],
                            "avg_gpu_load": result["performance_metrics"]["avg_gpu_load"],
                            "max_gpu_load": result["performance_metrics"].get("max_gpu_load", 0.0),
                            "model_vram_mb": result["performance_metrics"].get("model_vram_mb", 0.0),
                            "monitoring_samples": result["performance_metrics"].get("monitoring_samples", 0),
                            "monitoring_duration_s": result["performance_metrics"].get("monitoring_duration_s", 0.0)
                        })
                    
                    performance_data[model_id]["benchmarks"].append(benchmark)
            
            # 保存結果分析格式的文件
            performance_file = os.path.join(task_results_dir, "performance_analysis.json")
            with open(performance_file, 'w', encoding='utf-8') as f:
                json.dump(performance_data, f, indent=2, ensure_ascii=False)
            
            print(f"已生成結果分析文件: {performance_file}")
            
        except Exception as e:
            print(f"生成結果分析文件時出錯: {str(e)}")
    
    def _update_current_combination_result(self, task: Dict[str, Any], combination_index: int, combination: Dict[str, Any]):
        """更新當前組合的結果供前端實時顯示"""
        try:
            # 為前端提供當前組合的即時結果
            task["current_combination_result"] = {
                "index": combination_index,
                "batch_size": combination["batch_size"],
                "precision": combination["precision"],
                "status": combination.get("inference_status", "unknown"),
                "validation_results": combination.get("validation_results"),
                "inference_results": combination.get("inference_results")
            }
        except Exception as e:
            print(f"更新當前組合結果時出錯: {str(e)}") 