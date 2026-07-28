#!/bin/bash

echo "具自動化模型優化評估和即時推論資源監控之 AI 部署平台啟動腳本"
echo "========================================================"

# 移除 sudo 檢查，允許用戶直接使用 sudo 運行

# 檢查是否使用快速啟動模式
QUICK_START=0
if [ "$1" = "--quick" ] || [ "$1" = "-q" ]; then
    QUICK_START=1
    echo "[信息] 使用快速啟動模式，跳過基礎映像檢查"
fi

# 檢查Docker是否安裝
if ! command -v docker &> /dev/null; then
    echo "[錯誤] 未檢測到Docker，請先安裝Docker"
    echo "安裝指令: curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh"
    exit 1
fi

# 檢查Docker Compose是否安裝
if ! command -v docker-compose &> /dev/null; then
    echo "[錯誤] 未檢測到Docker Compose，請先安裝Docker Compose"
    echo "安裝指令: sudo apt-get update && sudo apt-get install docker-compose-plugin"
    exit 1
fi

# 檢查Docker是否正在運行
if ! docker info &> /dev/null; then
    echo "[錯誤] Docker沒有運行，請啟動Docker服務"
    echo "啟動指令: sudo systemctl start docker"
    exit 1
fi

# 移除 docker 組檢查，用戶可以直接使用 sudo

# 如果不是快速啟動模式，檢查配置文件是否有更新
if [ "$QUICK_START" = "0" ]; then
    REBUILD_BASE=0

    echo "[信息] 檢查後端基礎映像..."
    for file in "backend/requirements.txt" "backend/Dockerfile.base"; do
        cache_file=".docker_cache/$(basename $file)"
        if [ ! -f "$cache_file" ]; then
            echo "[信息] 檢測到 $(basename $file) 文件更新"
            REBUILD_BASE=1
        elif ! diff -q "$cache_file" "$file" &> /dev/null; then
            echo "[信息] 檢測到 $(basename $file) 文件更新"
            REBUILD_BASE=1
        fi
    done

    echo "[信息] 檢查前端基礎映像..."
    for file in "frontend/package.json" "frontend/Dockerfile.base"; do
        cache_file=".docker_cache/$(basename $file)"
        if [ ! -f "$cache_file" ]; then
            echo "[信息] 檢測到 $(basename $file) 文件更新"
            REBUILD_BASE=1
        elif ! diff -q "$cache_file" "$file" &> /dev/null; then
            echo "[信息] 檢測到 $(basename $file) 文件更新"
            REBUILD_BASE=1
        fi
    done

    if [ "$REBUILD_BASE" = "1" ]; then
        echo "[信息] 構建基礎映像..."
        docker-compose build backend-base frontend-base
        
        # 創建快取目錄並保存當前文件副本
        mkdir -p .docker_cache
        cp backend/requirements.txt .docker_cache/requirements.txt
        cp backend/Dockerfile.base .docker_cache/Dockerfile.base
        cp frontend/package.json .docker_cache/package.json
        cp frontend/Dockerfile.base .docker_cache/Dockerfile.base
        
        # 修復權限問題（如果使用 sudo 運行）
        if [ "$EUID" -eq 0 ]; then
            # 如果是 root 用戶運行，將緩存目錄權限改為原始用戶
            if [ ! -z "$SUDO_USER" ]; then
                chown -R $SUDO_USER:$SUDO_USER .docker_cache
            fi
        fi
    else
        echo "[信息] 基礎映像無需重建"
    fi
fi

echo "[信息] 構建並啟動容器..."
if [ "$QUICK_START" = "1" ]; then
    # 快速啟動模式：只重啟容器，不重建
    docker-compose up -d
else
    # 正常模式：構建並啟動
    docker-compose build backend frontend
    docker-compose up -d
fi

echo "[信息] 系統服務已啟動，請稍候..."
sleep 10

echo "[信息] 正在打開網頁界面..."
# 檢查是否有GUI環境
if command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:3000 &> /dev/null &
elif command -v gnome-open &> /dev/null; then
    gnome-open http://localhost:3000 &> /dev/null &
else
    echo "[信息] 無法自動打開瀏覽器，請手動訪問 http://localhost:3000"
fi

echo "[信息] 系統已啟動："
echo "- 前端界面：http://localhost:3000"
echo "- 後端API：http://localhost:8000"
echo "- API文檔：http://localhost:8000/docs"
echo "- 系統監控：http://localhost:3001 (Grafana儀表板，用戶名 admin，密碼為 .env 中設定的 GF_SECURITY_ADMIN_PASSWORD)"
echo ""
echo "[信息] 查看服務日誌："
echo "- 查看所有日誌: docker-compose logs -f"
echo "- 查看後端日誌: docker-compose logs -f backend"
echo "- 查看前端日誌: docker-compose logs -f frontend"
echo ""
echo "[信息] 使用提示："
echo "- 快速啟動（只修改程式碼時）: ./startup.sh --quick 或 ./startup.sh -q"
echo "- 完整啟動（修改依賴時）: ./startup.sh"
echo ""
echo "[信息] 按 Ctrl+C 停止服務，或在新終端執行 docker-compose down"

# 等待用戶中斷
trap 'echo; echo "[信息] 接收到停止信號，正在停止系統服務..."; docker-compose down; echo "[信息] 操作完成"; exit 0' INT

# 持續運行直到收到中斷信號
while true; do
    sleep 1
done 