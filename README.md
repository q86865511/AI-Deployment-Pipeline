# 具自動化測試模型優化和即時推論資源監控之AI部署平台

## 系統概述

本系統是一個企業級AI模型部署與性能監控平台，專為YOLO系列模型設計。系統整合了模型轉換、自動化測試、性能基準測試和推理服務部署等功能，提供完整的模型生命週期管理解決方案。

### 核心功能

1. **模型管理與格式轉換**
   - 支援PT、ONNX、TensorRT格式模型上傳與管理
   - 自動化模型格式轉換（PT → ONNX → TensorRT）
   - 支援FP32、FP16精度轉換
   - 模型版本控制與元數據管理

2. **自動化管線與模型性能評估**
   - 批量模型轉換與驗證管線
   - 多批次大小、多精度組合測試
   - 模型準確度驗證（mAP50、mAP50-95等指標）
   - 推理性能自動化模型性能評估（延遲、吞吐量、GPU使用率）

3. **推理服務部署**
   - 基於NVIDIA Triton Inference Server的生產級推理服務
   - 動態模型掛載/卸載
   - 即時推理性能監控
   - RESTful API推理服務

4. **智能分析與可視化**
   - 多維度性能分析與比較
   - 互動式圖表與數據導出
   - 測試結果自動化報告生成
   - 性能優化建議

5. **即時監控與資源管理**
   - 基於 Prometheus + Grafana 的監控棧
   - CPU 使用率、記憶體使用率即時監控
   - GPU 負載、GPU VRAM 使用率追蹤
   - 系統資源視覺化儀表板

## 系統架構

### 服務組件

- **前端界面**: React + Ant Design 響應式Web界面
- **後端API**: FastAPI 高性能異步API服務
- **推理引擎**: NVIDIA Triton Inference Server
- **模型轉換**: YOLOv8 + ONNX + TensorRT工具鏈
- **數據分析**: 內建性能分析與可視化引擎
- **監控系統**: Prometheus 指標收集 + Grafana 視覺化

### 部署架構

```
┌─────────────────────────────────────────────────────────┐
│                    Web Browser                         │
│      主界面(3000) | 監控界面(3001)                        │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────┼───────────────────────────────────┐
│              Frontend Container                        │
│              React + Ant Design                        │
└─────────────────────┬───────────────────────────────────┘
                      │ HTTP API
┌─────────────────────┼───────────────────────────────────┐
│              Backend Container                         │
│              FastAPI + Python                          │
│          (localhost:8000)                              │
└─────────────────────┬───────────────────────────────────┘
                      │ HTTP/gRPC
┌─────────────────────┼───────────────────────────────────┐
│           Triton Inference Server                      │
│              (localhost:8001/8002)                     │
└─────────────────────┬───────────────────────────────────┘
                      │
           ┌──────────┴──────────┐
           │   Model Repository  │
           │   Shared Volume     │
           └─────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                 監控系統架構                              │
├─────────────────────┬───────────────────────────────────┤
│    Grafana (3001)   │    Prometheus (9090)             │
│    視覺化儀表板        │    指標收集與儲存                    │
└─────────────────────┼───────────────────────────────────┘
                      │ 指標查詢
        ┌─────────────┼─────────────┐
        │             │             │
┌───────▼──┐  ┌──────▼──┐  ┌──────▼──────┐
│Node      │  │GPU      │  │Triton       │
│Exporter  │  │Exporter │  │Metrics      │
│(9100)    │  │(9445)   │  │(8002)       │
└──────────┘  └─────────┘  └─────────────┘
```

## 系統要求

### 硬體需求
- **GPU**: NVIDIA GPU（支援CUDA 11.4+）
- **記憶體**: 至少16GB RAM
- **儲存**: 至少50GB可用空間
- **處理器**: Intel/AMD x64處理器

### 軟體需求
- **作業系統**: Windows 10/11、Ubuntu 18.04+、CentOS 7+
- **Docker**: Docker Desktop 或 Docker CE
- **Docker Compose**: v2.0+
- **NVIDIA Container Toolkit**: 支援GPU容器化

## 快速部署

### Windows用戶

1. 安裝Docker Desktop並啟用GPU支援
2. 執行啟動腳本：
```cmd
startup.bat
```

### Linux用戶

1. 安裝Docker和NVIDIA Container Toolkit：
```bash
# 安裝Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 安裝NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

2. 啟動系統：
```bash
# 克隆專案（如果需要）
git clone <repository-url>
cd <project-directory>

# 啟動所有服務
docker-compose up -d

# 檢查服務狀態
docker-compose ps

# 開啟Web界面
xdg-open http://localhost:3000
```

### 存取界面

- **主界面**: http://localhost:3000
- **API文檔**: http://localhost:8000/docs
- **Triton監控**: http://localhost:8002/metrics
- **系統監控**: http://localhost:3001 (Grafana 儀表板)
- **Prometheus**: http://localhost:9090

## 使用指南

### 模型上傳與管理

1. 前往「模型管理」頁面
2. 點擊「上傳模型」，選擇YOLO模型文件
3. 填寫模型資訊（名稱、類型、描述）
4. 系統自動掃描並建立模型索引

### 自動化管線與模型性能評估

1. 前往「自動化轉換與測試」頁面
2. 選擇原始模型（PT格式）
3. 配置測試參數：
   - 精度選項（FP32/FP16）
   - 批次大小組合
   - 測試迭代次數
   - 驗證數據集
4. 啟動管線流程，系統將自動執行：
   - 模型格式轉換
   - 準確度驗證
   - 自動化模型性能評估

### 推理服務部署

1. 前往「部署平台監控」頁面
2. 選擇已轉換的TensorRT模型
3. 點擊「掛載到Triton」
4. 模型即可通過API提供推理服務

### 結果分析

1. 前往「自動化結果分析」頁面
2. 選擇測試任務或上傳測試數據
3. 查看多維度性能分析：
   - 推理時間比較
   - GPU使用率分析
   - 準確度對比
   - 效能權衡圖
4. 導出分析報告（PDF/Excel）

### 系統資源監控

1. 前往「部署平台監控」頁面
2. 點擊「系統資源監控」按鈕
3. 自動開啟 Grafana 監控儀表板
4. 查看即時系統狀態：
   - CPU 使用率趨勢
   - 記憶體使用率變化
   - GPU 負載狀況
   - GPU VRAM 使用情況
5. 預設登入資訊：用戶名 `admin`，密碼 `admin`

## 監控與運維

### 監控功能

#### 應用層監控
- **即時服務狀態**: Triton服務器健康狀態監控
- **模型運行統計**: 推理次數、平均延遲、成功率
- **性能指標追蹤**: 推理延遲分佈、吞吐量統計

#### 系統層監控 (Prometheus + Grafana)

**已實現功能**:
- **系統資源監控**: CPU 使用率、記憶體使用率即時追蹤
- **GPU 狀態監控**: GPU 負載、VRAM 使用率詳細統計
- **Triton 服務監控**: 推理服務器性能指標收集
- **視覺化儀表板**: 四象限監控面板，5秒刷新頻率

**Prometheus監控指標收集**:
- Node Exporter (9100): 系統 CPU、記憶體、磁碟、網路指標
- NVIDIA GPU Exporter (9445): GPU 使用率、VRAM、溫度等指標
- Triton 服務器指標（http://localhost:8002/metrics）
- 自動指標發現與收集

#### 告警系統

**告警規則配置**:
- 推理服務異常告警
- 模型性能下降告警
- 資源使用超標告警
- 任務失敗率異常告警

**通知渠道**:
- Email通知
- Slack/Teams整合
- Webhook自定義通知

## 目錄結構

```
├── backend/                    # 後端服務
│   ├── app/                   # 應用程式碼
│   │   ├── models/           # 數據模型定義
│   │   ├── routers/          # API路由
│   │   │   ├── benchmark.py  # 測試任務管理
│   │   │   ├── conversion.py # 模型轉換
│   │   │   ├── inference.py  # 推理服務
│   │   │   ├── models.py     # 模型管理
│   │   │   └── triton.py     # Triton服務整合
│   │   ├── services/         # 業務邏輯服務
│   │   │   ├── conversion_service.py  # 轉換服務
│   │   │   ├── inference_service.py   # 推理服務
│   │   │   ├── model_service.py       # 模型服務
│   │   │   ├── test_manager.py        # 測試管理器
│   │   │   └── triton_service.py      # Triton服務
│   │   └── utils/            # 工具函數
│   ├── Dockerfile           # 後端容器配置
│   └── requirements.txt     # Python依賴
├── frontend/                 # 前端應用
│   ├── public/              # 靜態資源
│   ├── src/                 # 前端原始碼
│   │   ├── components/      # React組件
│   │   ├── pages/           # 頁面組件
│   │   │   ├── BenchmarkPage.js        # 自動化測試
│   │   │   ├── ConversionPage.js       # 模型轉換
│   │   │   ├── DeploymentMonitorPage.js # 部署監控
│   │   │   ├── ModelsPage.js           # 模型管理
│   │   │   ├── PerformanceAnalyzerPage.js # 性能分析
│   │   │   └── TestResultsPage.js      # 測試結果
│   │   └── services/        # API服務
│   └── Dockerfile          # 前端容器配置
├── model_repository/        # 模型存儲倉庫
├── docker-compose.yml      # 服務編排配置
├── startup.bat             # Windows啟動腳本
├── startup.sh              # Linux啟動腳本
└── README.md              # 專案說明文檔
```

## API文檔

### 核心API端點

- **模型管理**: `/api/models/`
- **模型轉換**: `/api/conversion/`
- **自動化測試**: `/api/benchmark/`
- **推理服務**: `/api/inference/`
- **Triton整合**: `/api/triton/`

### 完整API文檔

啟動系統後可訪問：http://localhost:8000/docs

## 故障排除

### 常見問題

**Q: Docker容器啟動失敗**
```bash
# 檢查NVIDIA Container Toolkit安裝
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# 檢查Docker Compose配置
docker-compose config
```

**Q: Triton服務無法連接**
```bash
# 檢查Triton容器狀態
docker-compose logs triton

# 測試Triton API
curl http://localhost:8001/v2/health/ready
```

**Q: 模型轉換失敗**
- 檢查模型格式是否支援
- 確認GPU記憶體充足
- 查看後端日誌：`docker-compose logs backend`

**Q: GPU 監控服務無法啟動**
- 檢查是否安裝 NVIDIA Container Toolkit
- 查看 GPU exporter 日誌：`docker-compose logs nvidia-gpu-exporter`
- 如果沒有 GPU，可以在 docker-compose.yml 中註解掉 nvidia-gpu-exporter 服務

**Q: 測試任務執行異常**
- 檢查數據集格式是否正確
- 確認GPU驅動版本相容性
- 查看詳細錯誤信息

### 效能調優

**GPU記憶體優化**:
- 降低批次大小
- 使用FP16精度
- 調整TensorRT工作空間大小

**推理效能優化**:
- 啟用動態批次處理
- 配置模型實例數量
- 調整輸入數據預處理

## 開發指南

### 開發環境設置

1. 克隆專案並安裝依賴：
```bash
git clone <repository-url>
cd <project-directory>

# 後端開發
cd backend
pip install -r requirements.txt

# 前端開發
cd frontend
npm install
```

2. 配置開發環境變數：
```bash
export MODEL_REPOSITORY_PATH=/path/to/model_repository
export TRITON_URL=http://localhost:8001
export REACT_APP_API_URL=http://localhost:8000
```

### 擴展開發

**新增模型格式支援**:
1. 擴展`ModelFormat`枚舉
2. 實作對應的轉換服務
3. 更新推理服務邏輯

**新增監控指標**:
1. 在相應服務中收集指標
2. 更新API端點返回數據
3. 在前端添加可視化組件

## 技術棧

### 後端技術
- **FastAPI**: 高性能異步Web框架
- **PyTorch**: 深度學習框架
- **ONNX**: 開放神經網路交換格式
- **TensorRT**: NVIDIA推理優化引擎
- **Ultralytics**: YOLOv8官方實作

### 前端技術
- **React 18**: 現代化前端框架
- **Ant Design 5**: 企業級UI組件庫
- **ECharts**: 數據可視化圖表庫
- **Axios**: HTTP客戶端

### 基礎設施
- **Docker**: 容器化部署
- **NVIDIA Triton**: 推理服務器
- **Docker Compose**: 服務編排

## 授權協議

本專案採用MIT開源協議，詳見[LICENSE](LICENSE)文件。

## 貢獻指南

歡迎提交Issue和Pull Request來改進本專案。請確保：

1. 遵循現有代碼風格
2. 添加適當的測試用例
3. 更新相關文檔
4. 提供詳細的變更說明

## 聯絡資訊

如有任何問題或建議，請通過以下方式聯絡：

- 專案Issues: [GitHub Issues](https://github.com/your-repo/issues)
- 技術支援: support@your-domain.com 