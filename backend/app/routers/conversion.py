from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query, Body
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone, timedelta
import os

from app.models import ConversionJob, ConversionJobList, ModelFormat, PrecisionType, ConversionStatus
from app.services.model_service import ModelService
from app.services.conversion_service import ConversionService

router = APIRouter()
model_service = ModelService()
conversion_service = ConversionService()

@router.get("/", response_model=ConversionJobList)
async def list_conversion_jobs(
    status: Optional[ConversionStatus] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100)
):
    """
    獲取轉換任務列表，可按狀態過濾
    """
    # 重新載入任務數據，確保狀態更新
    conversion_service._load_jobs()
    
    # 檢查處理中的任務是否已經超時（預設30分鐘，可透過環境變數CONVERSION_TIMEOUT_MINUTES調整）
    import os
    timeout_minutes = int(os.getenv('CONVERSION_TIMEOUT_MINUTES', 30))
    jobs_to_update = {}
    for job_id, job in conversion_service.jobs.items():
        if job.status == ConversionStatus.PROCESSING:
            # 檢查任務是否超過設定時間沒有完成
            if job.created_at:
                if datetime.now(timezone(timedelta(hours=8))) - job.created_at > timedelta(minutes=timeout_minutes):
                    print(f"任務 {job_id} 已超時，標記為失敗")
                    job.status = ConversionStatus.FAILED
                    job.error_message = "任務處理超時，可能已失敗"
                    jobs_to_update[job_id] = job
    
    # 檢查是否有任務狀態需要更新
    if jobs_to_update:
        for job_id, job in jobs_to_update.items():
            conversion_service.jobs[job_id] = job
        conversion_service._save_jobs()
    
    # 檢查模型庫中是否存在新的TensorRT模型，這些模型可能是由自動測試流程創建的
    # 但對應的轉換任務可能仍顯示為"進行中"
    model_service._scan_model_repository()  # 刷新模型庫
    
    # 檢查轉換任務的目標模型是否已存在於模型庫中
    for job_id, job in list(conversion_service.jobs.items()):  # 使用list創建副本以避免修改迭代對象
        if job.status == ConversionStatus.PROCESSING:
            # 構建可能的目標模型路徑
            source_model = model_service.get_model_by_id(job.source_model_id)
            if source_model and job.parameters:
                # 構建可能的目標模型名稱
                source_name = source_model.name
                safe_name = source_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
                batch_size = job.parameters.get("batch_size", 1)
                img_size = job.parameters.get("imgsz", 640)
                
                # 使用新的命名方式，包含源模型ID
                source_id_short = job.source_model_id[:8]
                param_suffix = f"_{job.target_format.value}_{job.precision.value}"
                param_suffix += f"_batch{batch_size}"
                param_suffix += f"_size{img_size}"
                target_model_name = f"{safe_name}_{source_id_short}{param_suffix}"
                
                # 檢查目標模型是否已存在
                target_model_path = f"model_repository/{target_model_name}"
                existing_models = [m for m in model_service.models.values() 
                                if m.metadata and m.metadata.get("triton_model_dir") == target_model_path]
                
                if existing_models:
                    print(f"發現進行中的轉換任務 {job_id} 的目標模型已存在，自動更新任務狀態為完成")
                    # 更新任務狀態為已完成
                    job.status = ConversionStatus.COMPLETED
                    job.completed_at = datetime.now(timezone(timedelta(hours=8)))
                    job.target_model_id = existing_models[0].id
                    conversion_service.jobs[job_id] = job
                    conversion_service._save_jobs()
    
    jobs = conversion_service.get_jobs(status, skip, limit)
    return {"jobs": jobs, "total": len(jobs)}

@router.get("/{job_id}", response_model=ConversionJob)
async def get_conversion_job(job_id: str):
    """
    獲取特定轉換任務的詳細信息
    """
    job = conversion_service.get_job_by_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="找不到指定轉換任務")
    return job

@router.post("/", response_model=ConversionJob)
async def create_conversion_job(
    background_tasks: BackgroundTasks,
    source_model_id: str = Body(...),
    target_format: ModelFormat = Body(...),
    precision: PrecisionType = Body(PrecisionType.FP32),
    parameters: Optional[Dict[str, Any]] = Body(None)
):
    """
    創建新的模型轉換任務
    """
    # 強制重新掃描模型目錄，確保最新上傳的模型被識別
    print(f"轉換前強制重新掃描模型目錄...")
    model_service._scan_model_repository()
    
    # 檢查源模型是否存在
    source_model = model_service.get_model_by_id(source_model_id)
    
    # 如果找不到模型，嘗試通過元數據中的original_id查找
    if not source_model:
        print(f"直接通過ID找不到源模型: {source_model_id}，嘗試通過原始ID查找")
        # 嘗試查找帶有對應original_id的模型
        for m_id, model in model_service.models.items():
            if (model.metadata and 
                (model.metadata.get("original_id") == source_model_id or 
                model.id == source_model_id)):
                source_model = model
                print(f"通過original_id找到了模型: {model.id}")
                break
    
    if not source_model:
        print(f"找不到源模型，ID: {source_model_id}")
        print(f"當前所有模型ID: {list(model_service.models.keys())}")
        raise HTTPException(status_code=404, detail=f"找不到原始模型: {source_model_id}")
    
    # 檢查轉換配置是否有效
    if source_model.format == target_format:
        raise HTTPException(status_code=400, detail="原始模型已經是目標格式，無需轉換")
    
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
    
    # 檢查是否已存在相同參數的模型
    # 構建目標模型名稱，與conversion_service中的邏輯保持一致
    source_name = source_model.name
    safe_name = source_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    batch_size = parameters.get("batch_size", 1)
    img_size = parameters.get("imgsz", 640)
    
    # 使用新的命名方式，包含源模型ID
    source_id_short = source_model_id[:8]
    param_suffix = f"_{target_format.value}_{precision.value}"
    if batch_size:
        param_suffix += f"_batch{batch_size}"
    if img_size:
        param_suffix += f"_size{img_size}"
    target_model_name = f"{safe_name}_{source_id_short}{param_suffix}"
    
    # 檢查目標模型是否已存在
    target_model_path = f"model_repository/{target_model_name}"
    existing_models = [m for m in model_service.models.values() 
                      if m.metadata and m.metadata.get("triton_model_dir") == target_model_path]
    
    if existing_models:
        existing_model = existing_models[0]
        raise HTTPException(
            status_code=400, 
            detail=f"已存在相同參數的轉換模型 (ID: {existing_model.id}, 名稱: {existing_model.name})。"
                  f"請使用不同的參數或直接使用現有模型。"
        )
    
    # 創建轉換任務
    job_id = str(uuid.uuid4())
    conversion_job = ConversionJob(
        id=job_id,
        source_model_id=source_model.id,  # 使用實際找到的模型ID
        target_format=target_format,
        precision=precision,
        status=ConversionStatus.PENDING,
        created_at=datetime.now(timezone(timedelta(hours=8))),
        parameters=parameters
    )
    
    # 保存任務信息
    saved_job = conversion_service.save_job(conversion_job)
    
    # 在背景中執行轉換任務
    background_tasks.add_task(
        conversion_service.process_conversion_job,
        job_id=job_id
    )
    
    return saved_job

@router.delete("/{job_id}")
async def delete_conversion_job(job_id: str):
    """
    刪除指定轉換任務
    """
    job = conversion_service.get_job_by_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="找不到指定轉換任務")
    
    # 檢查是否正在處理中
    if job.status == ConversionStatus.PROCESSING:
        raise HTTPException(status_code=400, detail="轉換任務正在處理中，無法刪除")
    
    try:
        # 如果任務已完成，先嘗試刪除相關的轉換模型
        if job.status == ConversionStatus.COMPLETED and job.target_model_id:
            try:
                # 先獲取模型信息
                target_model = model_service.get_model_by_id(job.target_model_id)
                
                if target_model:
                    print(f"嘗試刪除轉換任務 {job_id} 的目標模型: {job.target_model_id}")
                    
                    # 刪除模型記錄和文件
                    model_service.delete_model(job.target_model_id)
                    
                    # 如果是Triton模型，確保目錄被完全刪除
                    if target_model.metadata and target_model.metadata.get("triton_model_dir"):
                        triton_dir = target_model.metadata.get("triton_model_dir")
                        if os.path.exists(triton_dir):
                            import shutil
                            try:
                                shutil.rmtree(triton_dir)
                                print(f"刪除Triton模型目錄: {triton_dir}")
                            except Exception as e:
                                print(f"刪除Triton模型目錄時出錯: {str(e)}")
            except Exception as e:
                print(f"刪除目標模型時出錯: {str(e)}")
                # 繼續刪除任務，即使刪除模型失敗
    
        # 刪除任務信息
        conversion_service.delete_job(job_id)
            
        # 刷新模型列表，確保UI顯示最新狀態
        model_service._scan_model_repository()

        return {"message": "轉換任務已成功刪除"}
    except Exception as e:
        print(f"刪除轉換任務時出錯: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"刪除轉換任務時出錯: {str(e)}")

@router.get("/list", response_model=List[ConversionJob])
@router.get("/list/", response_model=List[ConversionJob])
async def list_conversion_jobs_api():
    """
    獲取所有轉換任務
    """
    try:
        jobs = list(conversion_service.jobs.values())
        print(f"返回 {len(jobs)} 個轉換任務")
        return jobs
    except Exception as e:
        print(f"獲取轉換任務列表出錯: {str(e)}")
        return [] 