from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from app.routers import models, conversion, inference, benchmark, triton

# 創建FastAPI應用
app = FastAPI(
    title="YOLO模型自動化轉換與測試系統",
    description="提供YOLO模型轉換為TensorRT格式並進行測試的系統",
    version="1.0.0"
)

# 配置CORS：允許的來源由環境變數 CORS_ORIGINS（逗號分隔）注入，
# 預設僅允許本機前端，避免 allow_origins=["*"] 搭配 allow_credentials 的風險。
_cors_origins_env = os.environ.get("CORS_ORIGINS", "http://localhost:3000")
allowed_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 掛載靜態文件目錄
uploads_dir = os.path.join(os.getcwd(), "uploads")
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)
app.mount("/uploads", StaticFiles(directory=uploads_dir), name="uploads")

# 掛載路由
app.include_router(models.router, prefix="/api/models", tags=["模型管理"])
app.include_router(conversion.router, prefix="/api/conversion", tags=["模型轉換"])
app.include_router(inference.router, prefix="/api/inference", tags=["模型推理"])
app.include_router(benchmark.router, prefix="/api/benchmark", tags=["性能基準測試"])
app.include_router(triton.router, prefix="/api/triton", tags=["Triton服務器管理"])

# 根路由
@app.get("/", tags=["健康檢查"])
async def root():
    """
    API根路徑，可用於健康檢查
    """
    return {"status": "online", "message": "YOLO模型自動化轉換與測試系統API已運行"}

# 應用啟動與關閉事件
@app.on_event("startup")
async def startup_event():
    """應用啟動時執行"""
    print("應用已啟動，初始化服務...")

@app.on_event("shutdown")
async def shutdown_event():
    """應用關閉時執行"""
    print("應用即將關閉，清理資源...") 