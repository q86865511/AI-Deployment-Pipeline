# 具自動化模型優化評估和即時推論資源監控之AI部署平台

## 系統概述

本系統是一個企業級AI模型部署與性能監控平台，專為YOLO系列模型設計。系統整合了模型轉換、自動化管線、性能評估和推理服務部署等功能，提供完整的模型生命週期管理解決方案。

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
│                    Web Browser                          │
│         主界面(3000) | 監控界面(3001)                    |
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────┼───────────────────────────────────┐
│              Frontend Container                         │
│              React + Ant Design                         │
└─────────────────────┬───────────────────────────────────┘
                      │ HTTP API
┌─────────────────────┼───────────────────────────────────┐
│              Backend Container                          │
│              FastAPI + Python                           │
│          (localhost:8000)                               │
└─────────────────────┬───────────────────────────────────┘
                      │ HTTP/gRPC
┌─────────────────────┼───────────────────────────────────┐
│           Triton Inference Server                       │
│              (localhost:8001/8002)                      │
└─────────────────────┬───────────────────────────────────┘
                      │
           ┌──────────┴──────────┐
           │   Model Repository  │
           │   Shared Volume     │
           └─────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                 監控系統架構                             │
├─────────────────────┬───────────────────────────────────┤
│    Grafana (3001)   │    Prometheus (9090)              │
│    視覺化儀表板      │    指標收集與儲存                   │
└─────────────────────┼───────────────────────────────────┘
                      │ 指標查詢
        ┌─────────────┼─────────────┐
        │             │             │
   ┌────▼─────┐  ┌────▼────┐  ┌─────▼──────┐
   │Node      │  │GPU      │  │Triton      │
   │Exporter  │  │Exporter │  │Metrics     │
   │(9100)    │  │(9445)   │  │(8002)      │
   └──────────┘  └─────────┘  └────────────┘
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

## 安裝與部署

### 前置需求

1. **Docker Desktop** (Windows/Mac) 或 **Docker Engine** (Linux)
2. **NVIDIA Container Toolkit** (用於 GPU 支援)
3. **至少 16GB RAM** 和 **50GB 可用硬碟空間**
4. **NVIDIA GPU** (支援 CUDA 11.4+)

### Windows 用戶安裝步驟

1. **安裝 Docker Desktop**
   - 下載並安裝 [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop)
   - 在設定中啟用 WSL 2 後端
   - 啟用 GPU 支援（設定 → Resources → WSL Integration）

2. **克隆專案**
   ```cmd
   git clone <https://github.com/q86865511/AI-Deployment-Pipeline>
   cd Sys
   ```

3. **啟動系統**
   ```cmd
   startup.bat
   ```

### Linux 用戶安裝步驟

1. **安裝 Docker 和 NVIDIA Container Toolkit**
   ```bash
   # 安裝 Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker $USER

   # 安裝 NVIDIA Container Toolkit
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
   sudo apt-get update && sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

2. **克隆專案**
   ```bash
   git clone <https://github.com/q86865511/AI-Deployment-Pipeline>
   cd Sys
   ```

3. **啟動系統**
   ```bash
   chmod +x startup.sh
   sudo ./startup.sh
   ```

### 啟動腳本說明

**Windows (startup.bat)**
- 自動檢查 Docker 是否安裝和運行
- 檢測並建構基礎映像（如有更新）
- 啟動所有服務容器
- 自動開啟網頁介面

**Linux (startup.sh)**
- 支援快速啟動模式：`./startup.sh --quick` (跳過基礎映像檢查)
- 自動處理權限和依賴檢查
- 提供即時日誌查看指令

### 服務端點

啟動成功後，可通過以下 URL 訪問各項服務：

| 服務 | URL | 說明 |
|------|-----|------|
| 主界面 | http://localhost:3000 | React 前端應用程式 |
| API 服務 | http://localhost:8000 | FastAPI 後端服務 |
| API 文檔 | http://localhost:8000/docs | Swagger UI 互動式文檔 |
| Grafana 監控 | http://localhost:3001 | 系統資源監控儀表板 |
| Prometheus | http://localhost:9090 | 指標數據查詢介面 |
| Triton Metrics | http://localhost:8002/metrics | 推理服務器指標 |

### Grafana 監控系統登入資訊

**預設登入帳號**：
- 帳號：`admin`
- 密碼：`admin`
- 首次登入後會要求更改密碼，可以跳過或設定新密碼

**儀表板說明**：
- 系統會自動載入預設的系統監控儀表板
- 監控項目包含：CPU 使用率、記憶體使用率、GPU 負載、VRAM 使用率
- 資料刷新頻率：5 秒更新一次

## 使用指南

### 案例使用說明

本系統提供完整的 AI 模型效能評估案例，採用統一的測試標準確保結果一致性和可比較性。

**案例設計標準**：
- **基礎架構**：統一使用 YOLOv8n-Pose 模型作為測試基準
- **訓練資料集**：在 COCO 2017-pose (val 2336 張圖片) 資料集上完成訓練
- **測試參數組合**：
  - 精度選項：FP32（單精度）、FP16（半精度）
  - 批次大小：1, 2, 4, 8, 16, 32
- **測試重複性**：每個配置進行 100 次重複推論測量
- **穩定性評估**：記錄平均值與標準差以評估性能穩定性

**資料集配置要求**：
- 驗證資料集圖片數量必須能被所有批次大小整除
- 建議使用 COCO val2017 子集，確保總圖片數為 32 的倍數
- 例如：2304 張圖片（可被 1, 2, 4, 8, 16, 32 整除）

**測試輸出指標**：
- **準確度指標**：mAP50、mAP50-95、各類別 AP 值
- **性能指標**：平均推理時間、FPS、GPU 使用率、VRAM 使用量
- **穩定性分析**：標準差、變異係數、置信區間

### 自動化管線完整流程

#### 階段一：準備與設定（模型管理頁面）
1. **上傳基礎模型**：
   - 進入「模型管理」頁面
   - 上傳 YOLOv8n-Pose.pt 訓練完成的模型文件
   - 填寫模型資訊：名稱、類型（detection-pose）、描述

2. **準備驗證資料集**：
   - 準備 COCO 2017-pose 格式的 JSON 標註文件
   - 確保圖片數量為 2304 張（32 的倍數，可被所有批次大小整除）
   - 驗證標註格式包含關鍵點座標資訊

#### 階段二：啟動自動化管線（自動化管線頁面）
1. **進入管線配置**：
   - 點擊左側選單「自動化管線」
   - 選擇已上傳的 YOLOv8n-Pose 基礎模型

2. **配置測試參數**：
   ```
   基礎模型：YOLOv8n-Pose.pt
   精度選項：☑ FP32  ☑ FP16
   批次大小：[1, 2, 4, 8, 16, 32]
   驗證數據集：上傳 COCO-pose val.json
   測試迭代次數：100
   工作空間大小：4 GB
   ```

3. **啟動自動化流程**：
   - 點擊「開始自動化管線」
   - 系統自動生成 12 個測試組合（2 精度 × 6 批次）

#### 階段三：自動化轉換過程
**系統自動執行以下步驟**：

1. **PT → ONNX 轉換**：
   - FP32 精度：生成 `model-fp32-batch{1,2,4,8,16,32}.onnx`
   - FP16 精度：生成 `model-fp16-batch{1,2,4,8,16,32}.onnx`

2. **ONNX → TensorRT 轉換**：
   - 自動優化生成 `model-fp32-batch{1,2,4,8,16,32}.engine`
   - 自動優化生成 `model-fp16-batch{1,2,4,8,16,32}.engine`

3. **Triton 模型部署**：
   - 自動配置 Triton 模型倉庫
   - 動態掛載轉換完成的模型
   - 生成對應的 config.pbtxt 配置文件

#### 階段四：性能評估執行
**對每個模型組合執行**：

1. **準確度驗證**：
   - 在 2304 張驗證圖片上執行推論
   - 計算 mAP50、mAP50-95 等關鍵點檢測指標
   - 生成各類別準確度分析

2. **性能測試**：
   - 執行 100 次重複推論測量
   - 記錄推理時間、GPU 使用率、VRAM 消耗
   - 計算平均值、標準差、95% 置信區間

3. **實時監控**：
   - 管線執行期間可在「部署平台監控」頁面查看進度
   - Grafana 儀表板顯示即時系統資源使用狀況

#### 階段五：結果分析與輸出
1. **測試結果查看頁面**：
   - 自動跳轉到「測試結果查看」頁面
   - 顯示完整的測試摘要和詳細結果表格
   - 提供 JSON 格式原始數據下載

2. **自動化結果分析頁面**：
   - 多維度性能分析圖表：
     - 推理時間對比圖（不同批次大小 vs 推理延遲）
     - 準確度對比圖（FP32 vs FP16 精度損失分析）
     - GPU 資源使用圖（VRAM vs 批次大小關係）
     - 效能權衡分析（速度 vs 準確度帕累托前沿）
   - 智能篩選和排序功能
   - 導出 PDF 報告和 Excel 數據表

#### 完整案例執行時間預估
- **模型轉換階段**：約 15-30 分鐘（12 個模型組合）
- **準確度驗證階段**：約 45-60 分鐘（2304 張圖片 × 12 組合）
- **性能測試階段**：約 30-45 分鐘（100 次 × 12 組合）
- **總執行時間**：約 1.5-2.5 小時（視 GPU 性能而定）

### 1. 模型管理頁面

**功能說明**：集中管理所有 AI 模型，支援上傳、下載、刪除和詳情查看。

**使用步驟**：
1. 點擊左側選單「模型管理」進入頁面
2. **上傳模型**：
   - 點擊「上傳模型」按鈕
   - 選擇模型文件（支援 .pt、.onnx、.engine、.plan 格式）
   - 填寫模型資訊（名稱、類型、描述）
   - 點擊「確定」完成上傳
3. **模型操作**：
   - 詳情：查看模型詳細資訊
   - 下載：下載模型文件到本地
   - 掛載/卸載：將模型部署到 Triton 推理服務器
   - 刪除：永久刪除模型文件

### 2. 模型優化頁面

**功能說明**：將 PyTorch 模型轉換為優化的 ONNX 或 TensorRT 格式。

**使用步驟**：
1. 點擊左側選單「模型優化」進入頁面
2. **創建轉換任務**：
   - 選擇來源模型（PT 格式）
   - 選擇目標格式（ONNX 或 ENGINE）
   - 配置轉換參數：
     - 精度：FP32（預設）或 FP16（半精度）
     - 批次大小：推理時的批次大小
     - 工作空間：TensorRT 優化使用的記憶體大小（GB）
   - 點擊「提交轉換」
3. **查看轉換進度**：
   - 轉換任務列表顯示即時狀態
   - 點擊「詳情」查看詳細日誌
   - 完成後可在模型管理頁面查看新模型

### 3. 自動化管線頁面

**功能說明**：批量執行模型轉換、性能評估和準確度驗證的完整管線。

**使用步驟**：
1. 點擊左側選單「自動化管線」進入頁面
2. **配置管線參數**：
   - 選擇基礎模型（PT 格式）
   - 設定精度選項：FP32、FP16 或兩者都測試
   - 批次大小組合：例如 [1, 4, 8, 16]
   - 上傳驗證數據集（COCO 格式 JSON）
   - 測試迭代次數：建議 100-1000 次
3. **啟動管線**：
   - 點擊「開始自動化管線」
   - 系統自動執行：
     - 模型格式轉換（PT → ONNX → TensorRT）
     - 準確度驗證（計算 mAP）
     - 性能評估（推理延遲、吞吐量、GPU 使用率）
4. **監控進度**：
   - 即時查看當前執行階段
   - 查看各組合的測試進度
   - 完成後自動跳轉到結果頁面

### 4. 測試結果查看頁面

**功能說明**：查看自動化管線的詳細測試結果。

**使用步驟**：
1. 點擊左側選單「測試結果查看」進入頁面
2. 選擇或搜尋測試任務
3. 查看結果包含：
   - 基本資訊：模型名稱、測試時間、參數配置
   - 準確度指標：mAP50、mAP50-95、各類別 AP
   - 性能指標：平均推理時間、FPS、GPU 使用率
   - 詳細結果表格：所有測試組合的完整數據

### 5. 自動化結果分析頁面

**功能說明**：多維度可視化分析測試結果，提供性能優化建議。

**使用步驟**：
1. 點擊左側選單「自動化結果分析」進入頁面
2. 選擇測試任務或上傳 JSON 結果文件
3. **查看分析圖表**：
   - 推理時間對比圖：不同配置的延遲比較
   - 準確度對比圖：各配置的 mAP 比較
   - GPU 資源使用圖：顯卡負載和記憶體使用
   - 效能權衡分析圖：速度與準確度的帕累托前沿
4. **篩選和排序**：
   - 設定推理時間上限
   - 設定準確度下限
   - 設定 GPU 資源限制
       - 選擇權衡模式（速度優先/準確度優先/權衡）
5. **導出報告**：
   - PDF 格式：包含所有圖表的完整報告
   - Excel 格式：原始數據表格

### 6. 部署平台監控頁面

**功能說明**：監控 Triton 推理服務器上的模型運行狀態。

**使用步驟**：
1. 點擊左側選單「部署平台監控」進入頁面
2. **查看服務狀態**：
   - Triton 服務器健康狀態
   - 已掛載模型數量和列表
   - 各模型的運行統計
3. **模型管理**：
   - 即時查看推理請求次數
   - 監控平均延遲和最後推理時間
   - 卸載不需要的模型
4. **系統資源監控**：
   - 點擊「系統資源監控」按鈕
   - 自動開啟 Grafana 儀表板（http://localhost:3001）
   - 登入帳號：admin / admin
   - 查看 CPU、記憶體、GPU、VRAM 即時狀態

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


## 專案結構

### 目錄結構說明

```
Sys/
├── backend/                     # 後端服務目錄
│   ├── app/                    # FastAPI 應用程式
│   │   ├── __init__.py
│   │   ├── main.py            # 應用程式入口
│   │   ├── models/            # 數據模型定義
│   │   │   ├── __init__.py
│   │   │   └── model.py       # 資料庫模型
│   │   ├── routers/           # API 路由端點
│   │   │   ├── __init__.py
│   │   │   ├── benchmark.py   # 自動化管線 API
│   │   │   ├── conversion.py  # 模型轉換 API
│   │   │   ├── inference.py   # 推理服務 API
│   │   │   ├── models.py      # 模型管理 API
│   │   │   └── triton.py      # Triton 整合 API
│   │   ├── services/          # 業務邏輯層
│   │   │   ├── __init__.py
│   │   │   ├── conversion_service.py   # 模型轉換服務
│   │   │   ├── inference_service.py    # 推理執行服務
│   │   │   ├── model_service.py        # 模型管理服務
│   │   │   ├── test_manager.py         # 自動化管線管理
│   │   │   └── triton_service.py       # Triton 服務介面
│   │   └── utils/             # 工具函數
│   │       ├── __init__.py
│   │       ├── tensorrt_utils.py  # TensorRT 工具
│   │       └── timezone.py        # 時區處理
│   ├── data/                  # 臨時數據目錄
│   ├── uploads/               # 上傳文件暫存
│   ├── model_repository/      # 模型文件存儲
│   ├── Dockerfile             # 後端容器映像
│   ├── Dockerfile.base        # 後端基礎映像
│   └── requirements.txt       # Python 套件依賴
│
├── frontend/                   # 前端服務目錄
│   ├── public/                # 公開靜態資源
│   │   ├── index.html        # HTML 入口
│   │   └── manifest.json     # PWA 配置
│   ├── src/                   # React 原始碼
│   │   ├── App.js            # 主應用程式組件
│   │   ├── App.css           # 全域樣式
│   │   ├── index.js          # 應用程式入口
│   │   ├── components/       # 共用組件
│   │   ├── hooks/            # 自定義 Hooks
│   │   ├── pages/            # 頁面組件
│   │   │   ├── HomePage.js              # 首頁
│   │   │   ├── ModelsPage.js            # 模型管理
│   │   │   ├── ModelDetailPage.js       # 模型詳情
│   │   │   ├── ConversionPage.js        # 模型優化
│   │   │   ├── ConversionDetailPage.js  # 轉換詳情
│   │   │   ├── BenchmarkPage.js         # 自動化管線
│   │   │   ├── TestResultsPage.js       # 測試結果查看
│   │   │   ├── PerformanceAnalyzerPage.js # 自動化結果分析
│   │   │   ├── DeploymentMonitorPage.js # 部署平台監控
│   │   │   └── SettingsPage.js          # 系統設定
│   │   ├── services/         # API 服務層
│   │   └── utils/            # 工具函數
│   ├── Dockerfile            # 前端容器映像
│   ├── Dockerfile.base       # 前端基礎映像
│   ├── package.json          # NPM 套件配置
│   └── package-lock.json     # NPM 套件鎖定
│
├── monitoring/                # 監控系統配置
│   ├── prometheus/           # Prometheus 配置
│   │   └── prometheus.yml    # 指標收集配置
│   └── grafana/              # Grafana 配置
│       └── provisioning/     # 自動配置
│           ├── dashboards/   # 儀表板定義
│           │   ├── dashboard.yml
│           │   └── system-monitoring.json
│           └── datasources/  # 數據源配置
│               └── prometheus.yml
│
├── model_repository/         # Triton 模型倉庫（共享卷）
│
├── docker-compose.yml        # Docker 服務編排
├── startup.bat              # Windows 啟動腳本
├── startup.sh               # Linux 啟動腳本
└── README.md               # 專案說明文檔
```

### 重要文件說明

| 文件 | 說明 |
|------|------|
| docker-compose.yml | 定義所有服務容器的配置和網路 |
| backend/requirements.txt | 後端 Python 依賴套件列表 |
| frontend/package.json | 前端 Node.js 依賴套件列表 |
| monitoring/prometheus/prometheus.yml | Prometheus 監控目標配置 |
| monitoring/grafana/provisioning/ | Grafana 自動化配置目錄 |

## API文檔

### 核心API端點

- **模型管理**: `/api/models/`
- **模型轉換**: `/api/conversion/`
- **自動化管線**: `/api/benchmark/`
- **推理服務**: `/api/inference/`
- **Triton整合**: `/api/triton/`

### 完整API文檔

啟動系統後可訪問：http://localhost:8000/docs

## 故障排除

### 常見問題

**Q: 啟動腳本執行失敗（Windows）**
- 確認 Docker Desktop 已安裝並運行
- 以管理員權限執行命令提示字元
- 如果腳本有編碼問題，直接執行：
  ```cmd
  docker-compose build backend-base frontend-base
  docker-compose up -d
  ```

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


## 貢獻指南

歡迎提交Issue和Pull Request來改進本專案。請確保：

1. 遵循現有代碼風格
2. 添加適當的測試用例
3. 更新相關文檔
4. 提供詳細的變更說明

## 聯絡資訊

如有任何問題或建議，請通過以下方式聯絡：

- 專案Issues: [GitHub Issues](https://github.com/q86865511/AI-Deployment-Pipeline/issues)
- 技術支援: q86865511@gmail.com 