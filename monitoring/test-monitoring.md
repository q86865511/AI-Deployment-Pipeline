# 監控系統測試指南

## 服務端口檢查

啟動系統後，請檢查以下端點是否正常運行：

### Prometheus (http://localhost:9090)
- 檢查 Status > Targets 頁面
- 確認所有 targets 都是 UP 狀態：
  - node-exporter (9100/tcp)
  - nvidia-gpu-exporter (9445/tcp) 
  - triton (8002/tcp)
  - prometheus (9090/tcp)

### Grafana (http://localhost:3001)
- 登入資訊：用戶名 `admin`，密碼 `admin`
- 檢查 Data Sources 是否正確連接到 Prometheus
- 打開 "系統資源監控" 儀表板
- 確認四個面板都有數據：
  - CPU 使用率
  - 記憶體使用率
  - GPU 負載
  - GPU VRAM 使用率

## 故障排除

### GPU監控問題
如果 GPU 相關指標無法顯示：
1. 確認 NVIDIA Container Toolkit 正確安裝
2. 檢查 nvidia-gpu-exporter 容器日誌：
   ```bash
   docker logs nvidia-gpu-exporter
   ```

### Node Exporter 問題
如果系統資源指標無法顯示：
1. 檢查 node-exporter 容器日誌：
   ```bash
   docker logs node-exporter
   ```

### Prometheus 連接問題
1. 檢查 prometheus 容器日誌：
   ```bash
   docker logs prometheus
   ```
2. 確認配置文件 `/monitoring/prometheus/prometheus.yml` 正確

## 測試數據生成

為了測試監控是否正常工作，可以：
1. 運行模型推理任務以增加 GPU 負載
2. 在系統上運行一些 CPU 密集型任務
3. 觀察 Grafana 儀表板中的指標變化 