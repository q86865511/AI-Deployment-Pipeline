from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query, Body, Form, UploadFile, File
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone, timedelta
import json
import os
import shutil
import uuid
import zipfile

from app.services.test_manager import TestManager

router = APIRouter()
test_manager = TestManager()

# 添加數據集相關端點
@router.get("/datasets")
async def get_datasets():
    """獲取所有可用的數據集"""
    try:
        # 從datasets.json讀取數據集列表
        datasets_file = os.path.join("data", "datasets", "datasets.json")
        if not os.path.exists(datasets_file):
            return {"datasets": []}
        
        with open(datasets_file, "r", encoding="utf-8") as f:
            datasets = json.load(f)
        
        return {"datasets": datasets}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"獲取數據集列表失敗: {str(e)}")

@router.post("/datasets/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    dataset_type: str = Form("object")  # 新增參數，預設為物體檢測類型
):
    """
    上傳數據集文件（ZIP格式）
    系統會自動解壓並根據數據集類型(object或pose)生成對應的YAML配置文件
    
    Args:
        file: ZIP格式的數據集文件
        dataset_type: 數據集類型，可選值為 "object" 或 "pose"
    """
    # 設置臨時目錄變數
    temp_dir = None
    
    try:
        # 檢查數據集類型
        if dataset_type not in ["object", "pose"]:
            raise HTTPException(status_code=400, detail="數據集類型必須為 object 或 pose")
            
        # 檢查文件類型
        if not file.filename.endswith('.zip'):
            raise HTTPException(status_code=400, detail="只接受ZIP格式的數據集文件")
        
        # 保存原始文件名
        original_filename = file.filename
        
        # 生成唯一ID和目錄
        dataset_id = str(uuid.uuid4())
        dataset_short_id = dataset_id.split('-')[0]
        
        # 創建臨時目錄和目標目錄
        datasets_dir = os.path.join("data", "datasets")
        os.makedirs(datasets_dir, exist_ok=True)
        
        temp_dir = os.path.join("temp", f"dataset_{dataset_short_id}")
        os.makedirs(temp_dir, exist_ok=True)
        
        # 目標目錄名稱
        dataset_dir_name = f"{original_filename}_{dataset_short_id}"
        dataset_dir = os.path.join(datasets_dir, dataset_dir_name)
        
        # 保存上傳的ZIP文件
        zip_path = os.path.join(temp_dir, original_filename)
        with open(zip_path, "wb") as f:
            contents = await file.read()
            f.write(contents)
        
        # 解壓文件
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
        
        # 檢查是否為COCO格式數據集並創建YAML文件
        is_coco = False
        yaml_file = None
        
        # 檢查是否已包含YAML文件
        for root, _, files in os.walk(dataset_dir):
            for file_name in files:
                if file_name.endswith('.yaml'):
                    yaml_file = os.path.join(root, file_name)
                    is_coco = True
                    break
            if yaml_file:
                break
        
        # 如果沒有YAML文件，檢查是否可以識別為COCO格式
        if not yaml_file:
            # 先檢查annotations目錄
            annotations_dir = None
            for root, dirs, _ in os.walk(dataset_dir):
                if 'annotations' in dirs:
                    annotations_dir = os.path.join(root, 'annotations')
                    break
            
            # 檢查train2017.txt或val2017.txt文件
            train_txt = os.path.join(dataset_dir, 'train2017.txt')
            val_txt = os.path.join(dataset_dir, 'val2017.txt')
            has_txt_files = os.path.exists(train_txt) or os.path.exists(val_txt)
            
            # 檢查images/labels目錄
            has_images_dir = False
            has_labels_dir = False
            for root, dirs, _ in os.walk(dataset_dir):
                if 'images' in dirs:
                    has_images_dir = True
                if 'labels' in dirs:
                    has_labels_dir = True
            
            # 判斷是否為COCO格式
            if (annotations_dir and os.path.exists(annotations_dir)) or (has_txt_files and (has_images_dir or has_labels_dir)):
                is_coco = True
                
                # 檢查是否有instances_val.json
                coco_json = None
                if annotations_dir:
                    coco_json = os.path.join(annotations_dir, 'instances_val.json')
                    if not os.path.exists(coco_json):
                        # 嘗試其他可能的文件名
                        other_json_files = ['instances_train.json', 'annotations.json']
                        for json_file in other_json_files:
                            test_path = os.path.join(annotations_dir, json_file)
                            if os.path.exists(test_path):
                                coco_json = test_path
                                break
                
                # 根據數據集類型創建不同的YAML配置
                if dataset_type == "object":
                    # 物體檢測模型 - 創建完整的COCO類別列表
                    coco_classes = {
                        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
                        5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
                        10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 
                        14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
                        20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
                        25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
                        30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
                        35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 
                        39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
                        45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
                        50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
                        55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
                        60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
                        65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
                        70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
                        75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
                    }
                    
                    # 構建YAML類別部分
                    classes_yaml = "names:\n"
                    for class_id, class_name in coco_classes.items():
                        classes_yaml += f"  {class_id}: {class_name}\n"
                    
                    # 創建YAML文件
                    yaml_content = f"""
# COCO dataset http://cocodataset.org
path: {dataset_dir}  # dataset root dir
train: train2017.txt  # train images
val: val2017.txt  # val images
test: test-dev2017.txt  # test images

# Classes
{classes_yaml}
"""
                    yaml_file = os.path.join(dataset_dir, "coco_object.yaml")
                    
                else:  # dataset_type == "pose"
                    # 姿態估計模型 - 創建COCO關鍵點配置
                    keypoints = [
                        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                        "left_wrist", "right_wrist", "left_hip", "right_hip",
                        "left_knee", "right_knee", "left_ankle", "right_ankle"
                    ]
                    
                    # 構建關鍵點YAML配置
                    keypoints_yaml = "keypoints:\n"
                    for i, kp in enumerate(keypoints):
                        keypoints_yaml += f"  {i}: {kp}\n"
                    
                    # 創建姿態估計YAML文件
                    yaml_content = f"""
# COCO keypoints dataset for pose estimation
path: {dataset_dir}  # dataset root dir
train: train2017.txt  # train images
val: val2017.txt  # val images
test: test-dev2017.txt  # test images

# Keypoints
kpt_shape: [17, 3]  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

# Classes (只有一個類別：人)
names:
  0: person

# Keypoints
{keypoints_yaml}

# Skeleton connections (關鍵點連接)
skeleton:
  - [16, 14]  # right ankle -> right knee
  - [14, 12]  # right knee -> right hip
  - [17, 15]  # left ankle -> left knee
  - [15, 13]  # left knee -> left hip
  - [12, 13]  # right hip -> left hip
  - [6, 12]   # right shoulder -> right hip
  - [7, 13]   # left shoulder -> left hip
  - [6, 7]    # right shoulder -> left shoulder
  - [6, 8]    # right shoulder -> right elbow
  - [7, 9]    # left shoulder -> left elbow
  - [8, 10]   # right elbow -> right wrist
  - [9, 11]   # left elbow -> left wrist
  - [2, 3]    # right eye -> right ear
  - [1, 3]    # left eye -> right ear
  - [1, 2]    # left eye -> right eye
  - [0, 1]    # nose -> left eye
  - [0, 2]    # nose -> right eye
"""
                    yaml_file = os.path.join(dataset_dir, "coco_pose.yaml")
                
                with open(yaml_file, "w", encoding="utf-8") as f:
                    f.write(yaml_content)
        
        # 獲取數據集大小
        dataset_size = 0
        for root, dirs, files in os.walk(dataset_dir):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                dataset_size += os.path.getsize(file_path)
        
        # 更新數據集記錄
        dataset_info = {
            "id": dataset_id,
            "name": original_filename.replace('.zip', ''),
            "path": dataset_dir,
            "yaml_file": yaml_file,
            "size": dataset_size,
            "is_coco": is_coco,
            "type": dataset_type,  # 新增：保存數據集類型
            "uploaded_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
            "extracted": True
        }
        
        # 讀取現有數據集列表
        datasets = []
        datasets_json = os.path.join(datasets_dir, "datasets.json")
        if os.path.exists(datasets_json):
            with open(datasets_json, "r", encoding="utf-8") as f:
                datasets = json.load(f)
        
        # 添加新數據集
        datasets.append(dataset_info)
        
        # 保存更新後的數據集列表
        with open(datasets_json, "w", encoding="utf-8") as f:
            json.dump(datasets, f, indent=2, ensure_ascii=False)
        
        return {
            "message": "數據集上傳成功",
            "dataset_id": dataset_id,
            "dataset_name": dataset_info["name"],
            "is_coco": is_coco,
            "yaml_file": yaml_file,
            "dataset_type": dataset_type
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"上傳數據集失敗: {str(e)}")
    finally:
        # 確保在任何情況下都清理臨時目錄
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                print(f"已清理臨時目錄: {temp_dir}")
            except Exception as e:
                print(f"清理臨時目錄失敗: {temp_dir}, 錯誤: {str(e)}")

@router.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """刪除數據集"""
    try:
        # 讀取數據集列表
        datasets_dir = os.path.join("data", "datasets")
        datasets_json = os.path.join(datasets_dir, "datasets.json")
        
        if not os.path.exists(datasets_json):
            raise HTTPException(status_code=404, detail="找不到數據集列表")
        
        with open(datasets_json, "r", encoding="utf-8") as f:
            datasets = json.load(f)
        
        # 尋找要刪除的數據集
        dataset_to_delete = None
        updated_datasets = []
        
        for dataset in datasets:
            if dataset["id"] == dataset_id:
                dataset_to_delete = dataset
            else:
                updated_datasets.append(dataset)
        
        if not dataset_to_delete:
            raise HTTPException(status_code=404, detail=f"找不到數據集: {dataset_id}")
        
        # 刪除數據集目錄
        dataset_path = dataset_to_delete["path"]
        if os.path.exists(dataset_path):
            shutil.rmtree(dataset_path, ignore_errors=True)
        
        # 更新數據集列表
        with open(datasets_json, "w", encoding="utf-8") as f:
            json.dump(updated_datasets, f, indent=2, ensure_ascii=False)
        
        return {"message": "數據集已刪除", "dataset_id": dataset_id}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"刪除數據集失敗: {str(e)}")

@router.post("/create")
async def create_test_task(
    model_id: str = Form(...),
    batch_sizes: str = Form(...),
    precisions: List[str] = Form(...),
    image_size: int = Form(...),
    iterations: int = Form(...),
    dataset_id: str = Form(...),  # 必需的
    model_type: str = Form("object"),  # 新增：模型類型，預設為物體檢測
    custom_params: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    創建自動化管線任務
    
    Args:
        model_id: 源模型ID
        batch_sizes: 批次大小列表，以逗號分隔
        precisions: 精度選項列表 ['fp32', 'fp16']
        image_size: 圖像尺寸
        iterations: 迭代次數
        dataset_id: 數據集ID (必需)
        model_type: 模型類型 ['object', 'pose']
        custom_params: 自定義參數JSON字符串 (可選)
    """
    try:
        # 檢查模型類型
        if model_type not in ["object", "pose"]:
            raise HTTPException(status_code=400, detail="模型類型必須為 object 或 pose")
            
        # 解析批次大小
        try:
            batch_sizes_list = [int(bs.strip()) for bs in batch_sizes.split(',') if bs.strip()]
            if not batch_sizes_list:
                raise ValueError("必須提供至少一個有效的批次大小")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"批次大小格式無效: {str(e)}")
        
        # 解析自定義參數
        custom_params_dict = {}
        if custom_params:
            try:
                custom_params_dict = json.loads(custom_params)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="自定義參數必須是有效的JSON格式")
        
        # 創建測試任務
        task = test_manager.create_test_task(
            model_id=model_id,
            batch_sizes=batch_sizes_list,
            precisions=precisions,
            image_size=image_size,
            iterations=iterations,
            dataset_id=dataset_id,
            model_type=model_type,  # 傳遞模型類型
            custom_params=custom_params_dict
        )
        
        # 在背景啟動測試任務
        background_tasks.add_task(test_manager.start_test_task, task["id"])
        
        return {
            "task_id": task["id"],
            "status": "pending",
            "message": "測試任務已創建，即將開始處理",
            "total_combinations": task["total_combinations"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"創建測試任務失敗: {str(e)}")

@router.get("/tasks")
async def get_all_tasks():
    """獲取所有測試任務列表"""
    try:
        tasks = test_manager.get_all_tasks()
        
        # 格式化任務數據以便前端顯示
        formatted_tasks = []
        for task in tasks:
            formatted_tasks.append({
                "task_id": task["id"],
                "model_name": task["model_name"],
                "created_at": task["created_at"],
                "completed_at": task.get("completed_at"),  # 添加完成時間
                "status": task["status"],
                "current_step": task.get("current_step", "conversion"),  # 添加默認值
                "total_combinations": task["total_combinations"],
                "completed_combinations": task.get("completed_combinations", 0),  # 添加默認值
                "error": task.get("error"),
                "partial_success": task.get("partial_success", False),
                "success_count": task.get("success_count"),
                "fail_count": task.get("fail_count"),
                "current_combination_index": task.get("current_combination_index", -1)
            })
            
        # 按創建時間降序排序
        formatted_tasks.sort(key=lambda x: x["created_at"], reverse=True)
        
        return {"tasks": formatted_tasks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"獲取測試任務列表失敗: {str(e)}")

@router.get("/task/{task_id}")
async def get_task_detail(task_id: str):
    """獲取測試任務詳細信息"""
    task = test_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"找不到測試任務: {task_id}")
    
    return task

@router.post("/abort/{task_id}")
async def abort_task(task_id: str):
    """中止測試任務"""
    try:
        if not test_manager.abort_task(task_id):
            raise HTTPException(status_code=404, detail=f"找不到測試任務: {task_id}")
        
        return {"message": "測試任務已中止", "task_id": task_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"中止測試任務失敗: {str(e)}")

@router.delete("/task/{task_id}")
async def delete_task(task_id: str):
    """刪除測試任務"""
    try:
        if not test_manager.delete_task(task_id):
            raise HTTPException(status_code=404, detail=f"找不到測試任務: {task_id}")
        
        return {"message": "測試任務已刪除", "task_id": task_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"刪除測試任務失敗: {str(e)}")

@router.get("/results/{task_id}")
async def get_test_results(task_id: str):
    """獲取測試任務結果"""
    task = test_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"找不到測試任務: {task_id}")
    
    # 優先嘗試從文件讀取結果（確保數據持久化）
    try:
        final_results_file = os.path.join(
            "backend/data/test_tasks/inference_results", 
            f"task_{task_id}", 
            "final_results.json"
        )
        
        if os.path.exists(final_results_file):
            with open(final_results_file, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
                # 確保狀態是completed
                file_data["status"] = "completed"
                return file_data
    except Exception as e:
        print(f"讀取final_results.json失敗: {str(e)}")
    
    # 如果任務未完成，返回當前狀態
    if task["status"] != "completed":
        return {
            "task_id": task["id"],
            "status": task["status"],
            "message": f"測試任務尚未完成，當前狀態: {task['status']}",
            "current_step": task["current_step"],
            "completed_combinations": task["completed_combinations"],
            "total_combinations": task["total_combinations"]
        }
    
    # 如果文件不存在，從內存中提取測試結果
    results = []
    for i, combination in enumerate(task["combinations"]):
        result = {
            "combination_index": i,
            "batch_size": combination["batch_size"],
            "precision": combination["precision"],
            "image_size": combination.get("image_size", 640),
            "status": combination["status"],
            "model_id": combination.get("target_model_id"),
            "errors": None,
        }
        
        # 處理驗證結果 - 確保數據結構一致，包含GPU監控數據
        if combination.get("validation_results"):
            validation_results = combination["validation_results"]
            result["validation_results"] = {
                "model_id": validation_results.get("model_id"),
                "model_name": validation_results.get("model_name"),
                "dataset_id": validation_results.get("dataset_id"),
                "dataset_name": validation_results.get("dataset_name"),
                "batch_size": validation_results.get("batch_size"),
                "timestamp": validation_results.get("timestamp"),
                "metrics": validation_results.get("metrics", {}),
                # 添加GPU監控數據
                "memory_usage_mb": validation_results.get("memory_usage_mb"),
                "avg_gpu_load": validation_results.get("avg_gpu_load"),
                "max_gpu_load": validation_results.get("max_gpu_load"),
                "monitoring_samples": validation_results.get("monitoring_samples"),
                "model_vram_mb": validation_results.get("model_vram_mb"),
                "monitoring_duration_s": validation_results.get("monitoring_duration_s")
            }
        
        # 處理推論結果 - 確保數據結構一致
        if combination.get("inference_results"):
            inference_results = combination["inference_results"]
            result["inference_results"] = {
                "model_id": inference_results.get("model_id"),
                "model_name": inference_results.get("model_name"),
                "batch_size": inference_results.get("batch_size"),
                "iterations": inference_results.get("iterations"),
                "timestamp": inference_results.get("timestamp"),
                "avg_inference_time_ms": inference_results.get("avg_inference_time_ms"),
                "std_inference_time_ms": inference_results.get("std_inference_time_ms"),
                "min_inference_time_ms": inference_results.get("min_inference_time_ms"),
                "max_inference_time_ms": inference_results.get("max_inference_time_ms"),
                "all_inference_times": inference_results.get("all_inference_times", []),
                "avg_throughput_fps": inference_results.get("avg_throughput_fps"),
                "avg_vram_usage_mb": inference_results.get("avg_vram_usage_mb"),
                "all_vram_usages": inference_results.get("all_vram_usages", []),
                "status": inference_results.get("status")
            }
        
        # 收集錯誤信息
        errors = {}
        if combination.get("error"):
            errors["conversion"] = combination["error"]
        if combination.get("validation_error"):
            errors["validation"] = combination["validation_error"]
        if combination.get("inference_error"):
            errors["inference"] = combination["inference_error"]
        
        if errors:
            result["errors"] = errors
        
        results.append(result)
    
    return {
        "task_id": task["id"],
        "model_name": task["model_name"],
        "model_type": task.get("model_type", "object"),
        "dataset_id": task.get("dataset_id"),
        "status": "completed",  # 確保返回completed狀態
        "created_at": task["created_at"],
        "completed_at": task.get("completed_at"),
        "total_combinations": task["total_combinations"],
        "completed_combinations": task.get("completed_combinations", 0),
        "test_configurations": {
            "iterations": task.get("iterations", 100),
            "custom_params": task.get("custom_params", {})
        },
        "results": results
    }

@router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """獲取測試任務當前狀態"""
    task = test_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"找不到測試任務: {task_id}")
    
    # 獲取當前處理的組合信息
    current_combination = None
    current_combination_index = task.get("current_combination_index", 0)
    if 0 <= current_combination_index < len(task.get("combinations", [])):
        current_combination = task["combinations"][current_combination_index]
    
    return {
        "task_id": task["id"],
        "status": task["status"],
        "current_step": task.get("current_step", "conversion"),  # 添加默認值
        "current_combination_index": current_combination_index,
        "completed_combinations": task.get("completed_combinations", 0),  # 添加默認值
        "total_combinations": task["total_combinations"],
        "current_combination": current_combination,
        "error": task.get("error")
    }

# 增加系統狀態API
@router.get("/system-state")
async def get_system_state():
    """獲取測試系統當前狀態"""
    # 獲取所有測試任務
    tasks = test_manager.get_all_tasks()
    
    # 檢查是否有正在處理中的任務
    active_task = None
    for task in tasks:
        if task["status"] == "processing":
            active_task = task
            break
    
    # 檢查是否有正在進行的轉換任務
    conversion_jobs = []
    try:
        from app.services.conversion_service import ConversionService
        from app.models import ConversionStatus
        conversion_service = ConversionService()
        conversion_jobs = conversion_service.get_jobs()
        
        active_conversion = None
        for job in conversion_jobs:
            if job.status == ConversionStatus.PROCESSING:
                active_conversion = job
                break
                
        # 如果有活動的轉換任務但沒有活動的測試任務
        if active_conversion and not active_task:
            return {
                "is_testing": False,
                "is_converting": True,
                "active_task_id": None,
                "active_conversion_id": active_conversion.id,
                "conversion_model_name": active_conversion.source_model_name if hasattr(active_conversion, "source_model_name") else None,
                "conversion_status": active_conversion.status.value,
                "current_step": None,
                "current_model": None,
                "current_batch_size": None,
                "current_precision": None,
                "completed_combinations": 0,
                "total_combinations": 0
            }
    except Exception as e:
        print(f"獲取轉換任務時出錯: {str(e)}")
    
    # 如果沒有活動中的任務，返回空閒狀態
    if not active_task:
        return {
            "is_testing": False,
            "is_converting": False,
            "active_task_id": None,
            "active_conversion_id": None,
            "current_step": None,
            "current_model": None,
            "current_batch_size": None,
            "current_precision": None,
            "completed_combinations": 0,
            "total_combinations": 0
        }
    
    # 獲取當前處理的組合信息
    current_combination = None
    current_combination_index = active_task.get("current_combination_index", 0)
    if 0 <= current_combination_index < len(active_task.get("combinations", [])):
        current_combination = active_task["combinations"][current_combination_index]
    
    # 構建系統狀態返回數據
    return {
        "is_testing": True,
        "is_converting": False,
        "active_task_id": active_task["id"],
        "active_conversion_id": None,
        "current_step": active_task.get("current_step", "conversion"),
        "current_step_name": get_step_name(active_task.get("current_step", "conversion")),
        "current_model": active_task["model_name"],
        "current_batch_size": current_combination["batch_size"] if current_combination else None,
        "current_precision": current_combination["precision"] if current_combination else None,
        "completed_combinations": active_task.get("completed_combinations", 0),
        "total_combinations": active_task["total_combinations"]
    }

def get_step_name(step_code):
    """獲取步驟的中文名稱"""
    step_names = {
        "conversion": "模型轉換",
        "validation": "模型驗證",
        "inference": "性能測試"
    }
    return step_names.get(step_code, step_code)

@router.get("/combination/{task_id}/{combination_index}")
async def get_combination_detail(task_id: str, combination_index: int):
    """獲取特定測試組合的詳細信息"""
    task = test_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"找不到測試任務: {task_id}")
    
    if combination_index < 0 or combination_index >= len(task["combinations"]):
        raise HTTPException(status_code=400, detail=f"組合索引無效: {combination_index}")
    
    combination = task["combinations"][combination_index]
    
    # 構建詳細結果
    result = {
        "task_id": task["id"],
        "model_name": task["model_name"],
        "model_type": task.get("model_type", "object"),
        "combination_index": combination_index,
        "batch_size": combination["batch_size"],
        "precision": combination["precision"],
        "image_size": combination["image_size"],
        "conversion": {
            "status": combination["status"],
            "target_model_id": combination.get("target_model_id"),
            "error": combination.get("error")
        },
        "validation": {
            "status": combination.get("validation_status", "pending"),
            "results": combination.get("validation_results"),
            "error": combination.get("validation_error")
        },
        "inference": {
            "status": combination.get("inference_status", "pending"),
            "results": combination.get("inference_results"),
            "error": combination.get("inference_error")
        }
    }
    
    return result

@router.get("/tasks/{task_id}/performance-analysis")
async def get_task_performance_analysis(task_id: str):
    """獲取測試任務的性能分析數據（JSON格式）"""
    # 檢查任務是否存在
    task = test_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"找不到測試任務: {task_id}")
    
    # 檢查任務是否已完成
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"測試任務尚未完成，當前狀態: {task['status']}")
    
    # 構建性能分析文件路徑
    result_file = os.path.join(
        test_manager.inference_results_dir,
        f"task_{task_id}",
        "performance_analysis.json"
    )
    
    if not os.path.exists(result_file):
        raise HTTPException(status_code=404, detail="找不到性能分析文件")
    
    # 讀取並返回JSON數據
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            performance_data = json.load(f)
        return performance_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"讀取性能分析文件失敗: {str(e)}")

@router.get("/download-results/{task_id}")
async def download_test_results(task_id: str):
    """下載測試任務的最終結果JSON文件"""
    from fastapi.responses import FileResponse
    
    # 檢查任務是否存在
    task = test_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"找不到測試任務: {task_id}")
    
    # 檢查任務是否已完成
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"測試任務尚未完成，當前狀態: {task['status']}")
    
    # 構建結果文件路徑（使用新的目錄結構）
    result_file = os.path.join(
        test_manager.inference_results_dir,
        f"task_{task_id}",
        "final_results.json"
    )
    
    # 檢查文件是否存在
    if not os.path.exists(result_file):
        raise HTTPException(status_code=404, detail="找不到測試結果文件")
    
    # 返回文件
    return FileResponse(
        path=result_file,
        filename=f"test_results_{task['model_name']}_{task_id}.json",
        media_type="application/json"
    )

@router.get("/download-performance-analysis/{task_id}")
async def download_performance_analysis(task_id: str):
    """下載性能分析格式的JSON文件"""
    from fastapi.responses import FileResponse
    
    # 檢查任務是否存在
    task = test_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"找不到測試任務: {task_id}")
    
    # 構建性能分析文件路徑（使用新的目錄結構）
    performance_file = os.path.join(
        test_manager.inference_results_dir,
        f"task_{task_id}",
        "performance_analysis.json"
    )
    
    # 檢查文件是否存在
    if not os.path.exists(performance_file):
        raise HTTPException(status_code=404, detail="找不到性能分析文件")
    
    # 返回文件
    return FileResponse(
        path=performance_file,
        filename=f"performance_analysis_{task['model_name']}_{task_id}.json",
        media_type="application/json"
    ) 