from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Query
from typing import List, Optional
import os
import uuid
import shutil
from datetime import datetime, timezone, timedelta
from pydantic import ValidationError
from fastapi.responses import FileResponse
import tempfile
import json

from app.models import ModelInfo, ModelList, ModelType, ModelFormat
from app.services.model_service import ModelService
from app.utils.path_safety import sanitize_filename

router = APIRouter()
model_service = ModelService()

@router.get("/", response_model=ModelList)
async def list_models(
    model_type: Optional[ModelType] = None,
    format: Optional[ModelFormat] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000)
):
    """
    獲取模型列表，可按類型和格式過濾
    """
    # 先獲取所有模型來計算總數
    all_models = model_service.get_models(model_type, format, 0, 1000)  # 獲取所有模型計算總數
    total_count = len(all_models)
    
    # 再獲取分頁數據
    models = model_service.get_models(model_type, format, skip, limit)
    return {"models": models, "total": total_count}

@router.get("/refresh")
async def refresh_models():
    """
    重新掃描模型目錄，刷新模型列表
    """
    print("收到模型刷新請求...")
    try:
        # 掃描模型儲存庫
        new_model_ids = model_service._scan_model_repository()
        # 保存更新的模型信息
        # 模型服務不需要保存，因為它是通過掃描動態載入的
        
        # 確保new_model_ids不為None
        if new_model_ids is None:
            new_model_ids = []
        
        print(f"模型刷新成功，找到 {len(new_model_ids)} 個模型")
        return {"success": True, "message": "模型存儲庫已刷新", "new_model_ids": new_model_ids}
    except Exception as e:
        import traceback
        print(f"刷新模型列表時出錯: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"刷新模型列表失敗: {str(e)}")

@router.get("/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str):
    """
    獲取特定模型的詳細信息
    """
    model = model_service.get_model_by_id(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="找不到指定模型")
    return model

@router.post("/", response_model=ModelInfo)
async def upload_model(
    model_file: UploadFile = File(...),
    model_name: str = Form(...),
    model_type: str = Form("yolov8"),  # 改為字符串類型，稍後手動轉換為枚舉
    description: Optional[str] = Form(None)  # 添加description參數
):
    """
    上傳模型文件並創建模型記錄
    """
    try:
        # 驗證模型類型
        valid_types = ["yolov8", "yolov8_pose", "yolov8_seg", "custom"]
        if model_type not in valid_types:
            raise HTTPException(status_code=400, detail=f"無效的模型類型: {model_type}")
        
        # 檢查文件格式
        if not model_file.filename or not model_file.filename.endswith(('.pt', '.pth')):
            raise HTTPException(status_code=400, detail="只支持 .pt 或 .pth 格式的PyTorch模型文件")

        # 模型名稱會直接用來組模型庫目錄，必須先清洗；與清洗結果不符即視為非法名稱
        # （空字串與清洗後為空的名稱也要擋，否則會以空名組出模型庫根目錄本身）
        if not model_name or not sanitize_filename(model_name, default="") or model_name != sanitize_filename(model_name, default=""):
            raise HTTPException(
                status_code=400,
                detail="模型名稱只允許英數字、底線、連字號與點，且不可包含路徑分隔符"
            )

        # 檢查模型名稱是否已存在
        existing_model = model_service.get_model_by_name(model_name)
        if existing_model:
            raise HTTPException(status_code=400, detail=f"模型名稱 '{model_name}' 已存在")
        
        # 創建臨時文件
        import tempfile
        import os
        temp_file = None
        
        try:
            # 保存上傳文件到臨時位置
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
                content = await model_file.read()
                tmp.write(content)
                temp_file = tmp.name
            
            # 使用ModelService創建模型
            model = model_service.create_model_from_upload(
                name=model_name,
                model_type=model_type,
                file_path=temp_file,
                description=description
            )
            
            return model
            
        finally:
            # 清理臨時文件（如果沒有被移動）
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass
                
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"上傳模型失敗: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"上傳模型失敗: {str(e)}")

@router.delete("/{model_id}")
async def delete_model(model_id: str):
    """
    刪除模型
    """
    try:
        if model_service.delete_model(model_id):
            return {"message": "模型已刪除", "model_id": model_id}
        else:
            raise HTTPException(status_code=404, detail=f"找不到模型: {model_id}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"刪除模型失敗: {str(e)}")

@router.get("/{model_id}/download")
async def download_model(model_id: str):
    """
    下載模型文件
    """
    try:
        from fastapi.responses import FileResponse
        
        model = model_service.get_model_by_id(model_id)
        if not model:
            raise HTTPException(status_code=404, detail=f"找不到模型: {model_id}")
        
        if not os.path.exists(model.path):
            raise HTTPException(status_code=404, detail="模型文件不存在")
        
        # 獲取文件名，優先使用原始名稱
        filename = f"{model.name}.{model.format.value}"
        
        return FileResponse(
            path=model.path,
            filename=filename,
            media_type="application/octet-stream"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"下載模型失敗: {str(e)}") 