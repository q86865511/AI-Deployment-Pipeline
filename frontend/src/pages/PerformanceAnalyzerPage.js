import React, { useState, useEffect, useRef } from 'react';
import { useParams } from 'react-router-dom';
import { 
  Card, Select, Button, Table, Form, Input, Switch, Space, 
  Statistic, Row, Col, Typography, Upload, Divider, Radio, 
  Slider, Checkbox, Tabs, message, Tooltip, Collapse
} from 'antd';
import { 
  UploadOutlined, DownloadOutlined, BarChartOutlined, 
  LineChartOutlined, PieChartOutlined, AreaChartOutlined,
  ReloadOutlined, CaretRightOutlined
} from '@ant-design/icons';
import axios from 'axios';
import ReactECharts from 'echarts-for-react';
import { jsPDF } from 'jspdf';
import html2canvas from 'html2canvas';
import * as XLSX from 'xlsx';

const { Title, Text } = Typography;
const { Option } = Select;
const { TabPane } = Tabs;
const { Panel } = Collapse;

// 注意：缺少以下依賴項，需要安裝：
// - echarts
// - echarts-for-react 
// - jspdf
// - html2canvas
// - xlsx
// 將在安裝完畢後再導入這些組件

const ModelPerformanceAnalyzer = () => {
  // 添加CSS樣式
  React.useEffect(() => {
    const style = document.createElement('style');
    style.textContent = `
      .gray-row {
        background-color: #f5f5f5 !important;
        color: #999999 !important;
      }
      .gray-row:hover {
        background-color: #e8e8e8 !important;
      }
      .gray-row td {
        color: #999999 !important;
      }
    `;
    document.head.appendChild(style);
    return () => document.head.removeChild(style);
  }, []);

  // 獲取路由參數
  const { taskId } = useParams();
  
  // 狀態管理
  const [testData, setTestData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [availableResults, setAvailableResults] = useState([]);
  const [selectedResult, setSelectedResult] = useState('');
  
  // 每個圖表分別的排序設定
  const [inferenceTimeSortOrder, setInferenceTimeSortOrder] = useState('asc');
  const [mapSortOrder, setMapSortOrder] = useState('desc');
  const [vramSortOrder, setVramSortOrder] = useState('desc');
  const [gpuLoadSortOrder, setGpuLoadSortOrder] = useState('asc');
  
  const [filterType, setFilterType] = useState('default');
  const [vramLimit, setVramLimit] = useState(8192);
  const [gpuLoadLimit, setGpuLoadLimit] = useState(90);
  const [metricType, setMetricType] = useState('speed');
  
  // 新增的篩選狀態
  const [inferenceTimeLimit, setInferenceTimeLimit] = useState(10);
  const [mapLimit, setMapLimit] = useState(0.4);
  const [modelVramLimit, setModelVramLimit] = useState(2048);
  
  // 在狀態管理部分添加極值倍率控制
  const [speedupMultiplier, setSpeedupMultiplier] = useState(1.2);
  const [accuracyDropMultiplier, setAccuracyDropMultiplier] = useState(1.0);
  
  // 預設值
  const defaultFilterValues = {
    inferenceTimeLimit: 10,
    mapLimit: 0.4,
    vramLimit: 8192,
    modelVramLimit: 2048,
    gpuLoadLimit: 90,
    speedupMultiplier: 1.2,
    accuracyDropMultiplier: 1.0  // 確保預設值在新範圍內
  };
  const [chartsCollapsed, setChartsCollapsed] = useState(true);
  const [sortConfig, setSortConfig] = useState({ key: 'modelType', direction: 'asc' });
  const [isDefaultSort, setIsDefaultSort] = useState(true);
  const [includeOriginalBatches, setIncludeOriginalBatches] = useState(true);
  
  // 檔案上傳參考
  const uploadRef = useRef();
  const chartContainerRef = useRef();
  const inferenceTimeChartRef = useRef();
  const vramUsageChartRef = useRef();
  const mAPChartRef = useRef();
  const gpuLoadChartRef = useRef();
  
  // 自動載入任務數據
  useEffect(() => {
    if (taskId) {
      loadTaskData(taskId);
    }
    // 載入可用的測試結果列表
    loadAvailableResults();
  }, [taskId]);
  
  // 載入可用的測試結果列表
  const loadAvailableResults = async () => {
    try {
      const response = await axios.get('http://localhost:8000/api/benchmark/tasks');
      const tasks = response.data.tasks || [];
      const results = tasks
        .filter(task => task.status === 'completed')
        .map(task => ({
          taskId: task.task_id,
          displayName: `${task.model_name} - ${new Date(task.created_at).toLocaleDateString()}`
        }));
      setAvailableResults(results);
    } catch (error) {
      console.error('載入測試結果列表失敗:', error);
    }
  };
  
  // 處理選擇結果變更
  const handleResultSelection = (taskId) => {
    setSelectedResult(taskId);
    if (taskId) {
      loadTaskData(taskId);
    }
  };
  
  // 清理顯示名稱 - 移除數字後綴
  const cleanDisplayName = (name, batchSize = null) => {
    if (!name) return 'Unknown';
    
    // 清理模型名稱，生成簡短標籤
    const lowerName = name.toLowerCase();
    
    // 檢查是否為原始模型
    if (lowerName.includes('original') || !lowerName.includes('_engine_')) {
      // 對於原始模型，包含batch size信息
      const batch = batchSize || '1';
      return `原始模型_batch${batch}`;
    }
    
    // 提取精度和批次大小
    let precision = 'fp32';  // 預設值
    let extractedBatchSize = batchSize || '1';     // 預設值
    
    if (lowerName.includes('fp16')) {
      precision = 'fp16';
    } else if (lowerName.includes('int8')) {
      precision = 'int8';
    }
    
    // 提取批次大小
    const batchMatch = lowerName.match(/batch[_]?(\d+)/);
    if (batchMatch) {
      extractedBatchSize = batchMatch[1];
    }
    
    // 移除數字後綴（如 _0.83...）
    return `${precision}_${extractedBatchSize}`.replace(/_\d+\.\d+.*$/, '');
  };

  // 獲取模型標籤顏色
  const getModelColor = (modelType) => {
    // 只有原始模型batch1為紫色
    if (modelType === '原始模型_batch1') {
      return '#722ed1'; // 紫色
    }
    // 其他模型使用預設顏色
    const colors = ['#1890ff', '#52c41a', '#fa8c16', '#eb2f96', '#13c2c2', '#f5222d'];
    const hash = modelType.split('').reduce((a, b) => {
      a = ((a << 5) - a) + b.charCodeAt(0);
      return a & a;
    }, 0);
    return colors[Math.abs(hash) % colors.length];
  };

  // 清理GPU負載標籤（移除數字部分）
  const cleanGpuLabel = (label) => {
    // 移除類似 "0.83..." 的數字部分
    return label.replace(/\d+\.\d+.*$/, '').trim();
  };

  // 處理 JSON 數據
  const processTestData = (data) => {
    if (!data) return null;

    // 檢查數據格式
    if (typeof data === 'object' && !Array.isArray(data)) {
      // 格式1：從後端性能分析API獲取的數據 (performance_analysis.json)
      if (Object.keys(data).some(key => data[key] && data[key].model_name && data[key].benchmarks)) {
        const processedData = [];
        
        Object.keys(data).forEach(modelId => {
          const modelData = data[modelId];
          
          modelData.benchmarks.forEach((benchmark, index) => {
            const batchSize = benchmark.batch_size || 1;
            const cleanedName = cleanDisplayName(modelData.model_name, batchSize);
            const key = `${cleanedName}`;
            
            processedData.push({
              key,
              modelType: cleanedName,
              batchSize: batchSize,
              avgInferenceTime: benchmark.avg_inference_time_ms || 0,
              stdInferenceTime: benchmark.std_inference_time_ms || 0,
              mAP5095: benchmark.map50_95 || 0,
              vramUsage: benchmark.memory_usage_mb || 0,
              modelVram: benchmark.model_vram_mb || 0,
              gpuLoad: benchmark.avg_gpu_load || 0,
              maxGpuLoad: benchmark.max_gpu_load || 0,
              monitoringSamples: benchmark.monitoring_samples || 0,
              monitoringDuration: benchmark.monitoring_duration_s || 0,
              speedupRatio: 0,
              accuracyDropRatio: 0,
              isOriginal: benchmark.precision === 'original' || benchmark.is_original || cleanedName.startsWith('原始模型')
            });
          });
        });
        
        return calculateRelativeMetrics(processedData);
      }
      
      // 格式2：從下載獲取的完整測試結果 (final_results.json)
      if (data.task_id && data.results && Array.isArray(data.results)) {
        const processedData = [];
        
        data.results.forEach((result, index) => {
          // 只處理成功完成的結果
          if (result.status === 'completed') {
            // 構建模型名稱
            const precisionName = result.precision === 'float32' ? 'fp32' : 
                                result.precision === 'float16' ? 'fp16' : 
                                result.precision === 'int8' ? 'int8' : 
                                result.precision;
            
            const batchSize = result.batch_size || 1;
            const cleanedName = result.precision === 'original' ? `原始模型_batch${batchSize}` : 
                              `${precisionName}_${batchSize}`;
            
            const key = `${cleanedName}`;
            
            // 提取性能指標
            const performanceMetrics = result.performance_metrics || {};
            const validationMetrics = result.validation_metrics || {};
            
            processedData.push({
              key,
              modelType: cleanedName,
              batchSize: batchSize,
              avgInferenceTime: performanceMetrics.avg_inference_time_ms || 0,
              stdInferenceTime: 0, // final_results格式中沒有標準差
              mAP5095: validationMetrics.mAP50_95 || 0,
              vramUsage: performanceMetrics.memory_usage_mb || 0,
              modelVram: performanceMetrics.model_vram_mb || 0,
              gpuLoad: performanceMetrics.avg_gpu_load || 0,
              maxGpuLoad: performanceMetrics.max_gpu_load || 0,
              monitoringSamples: performanceMetrics.monitoring_samples || 0,
              monitoringDuration: performanceMetrics.monitoring_duration_s || 0,
              speedupRatio: 0,
              accuracyDropRatio: 0,
              isOriginal: result.precision === 'original' || cleanedName.startsWith('原始模型')
            });
          }
        });
        
        return calculateRelativeMetrics(processedData);
      }
    }
    
    return null;
  };

  // 計算相對指標的輔助函數
  const calculateRelativeMetrics = (processedData) => {
    // 計算相對指標（以原始模型batch1為基準）
    const originalModel = processedData.find(item => 
      item.isOriginal && item.batchSize === 1
    );
        
        if (originalModel && originalModel.avgInferenceTime > 0 && originalModel.mAP5095 > 0) {
            const baseInferenceTime = originalModel.avgInferenceTime;
            const baseAccuracy = originalModel.mAP5095;
            
            processedData.forEach(item => {
                // 推論時間加速比率：(LatencyBase – Latency)/LatencyBase
                if (item.avgInferenceTime > 0) {
                    item.speedupRatio = (baseInferenceTime - item.avgInferenceTime) / baseInferenceTime;
                } else {
                    item.speedupRatio = 0;
                }
                
                // mAP下降比率：(Accuracy – AccuracyBase) /AccuracyBase
                if (baseAccuracy > 0) {
                    item.accuracyDropRatio = (item.mAP5095 - baseAccuracy) / baseAccuracy;
                } else {
                    item.accuracyDropRatio = 0;
                }
                
                console.log(`${item.modelType}: 推論時間=${item.avgInferenceTime}ms, 加速比率=${item.speedupRatio.toFixed(4)}, 準確率下降比率=${item.accuracyDropRatio.toFixed(4)}`);
            });
        } else {
        console.warn('找不到原始模型batch1或原始模型數據不完整，所有比率設為0');
            processedData.forEach(item => {
                item.speedupRatio = 0;
                item.accuracyDropRatio = 0;
            });
        }
        
        return processedData;
  };
  
  // 載入特定任務的數據
  const loadTaskData = async (taskId) => {
    setLoading(true);
    
    try {
      const response = await axios.get(`http://localhost:8000/api/benchmark/tasks/${taskId}/performance-analysis`);
      const data = response.data;
      setTestData(processTestData(data));
      message.success('已載入任務數據');
    } catch (error) {
      console.error('載入任務數據失敗:', error);
      message.error('載入任務數據失敗');
    } finally {
      setLoading(false);
    }
  };
  
  // 載入選中的測試結果
  const loadSelectedResult = () => {
    if (selectedResult) {
      loadTaskData(selectedResult);
    } else {
      message.warning('請先選擇要載入的測試結果');
    }
  };
  
  // 處理文件上傳
  const handleFileUpload = (file) => {
    setLoading(true);
    const reader = new FileReader();
    
    reader.onload = (e) => {
      try {
        const data = JSON.parse(e.target.result);
        const processed = processTestData(data);
        
        if (processed && processed.length > 0) {
        setTestData(processed);
          message.success(`數據載入成功！解析到 ${processed.length} 個測試結果`);
        } else {
          console.error('JSON數據格式不支援:', data);
          message.error('不支援的JSON格式。請確保檔案是從測試結果頁面下載的 final_results.json 或 performance_analysis.json');
        }
      } catch (error) {
        console.error('解析 JSON 失敗:', error);
        if (error instanceof SyntaxError) {
          message.error('JSON 格式錯誤，請檢查檔案內容');
        } else {
          message.error(`數據解析失敗: ${error.message}`);
        }
      } finally {
        setLoading(false);
      }
    };
    
    reader.onerror = () => {
      message.error('讀取文件失敗');
      setLoading(false);
    };
    
    reader.readAsText(file);
    return false; // 阻止默認上傳行為
  };
  
  // 篩選邏輯
  const getFilteredDataForDisplay = () => {
    if (!testData) return [];
    
    let filteredData = [...testData];
    
    // 是否排除原始模型的不同batch（除了batch1）
    if (!includeOriginalBatches) {
      filteredData = filteredData.filter(item => 
        !(item.isOriginal && item.batchSize !== 1)
      );
    }
    
    // 根據不同的篩選類型進行篩選
    if (filterType === 'filter') {
      // 篩選模式：多條件篩選
      filteredData = filteredData.filter(item => 
        item.avgInferenceTime <= inferenceTimeLimit &&
        item.mAP5095 >= mapLimit &&
        item.vramUsage <= vramLimit &&
        item.modelVram <= modelVramLimit &&
        item.gpuLoad <= gpuLoadLimit
      );
      
      return filteredData; // 返回所有符合條件的數據，排序交給表格處理
    } 
    else if (filterType === 'balanced') {
      // 權衡模式：篩選符合資源限制的數據
      filteredData = filteredData.filter(item => 
        item.avgInferenceTime <= inferenceTimeLimit &&
        item.mAP5095 >= mapLimit &&
        item.vramUsage <= vramLimit &&
        item.modelVram <= modelVramLimit &&
        item.gpuLoad <= gpuLoadLimit
      );
      
      if (filteredData.length === 0) return [];
      
      // 先計算原始數據的極值
      const accuracyDropRatios = filteredData.map(item => -item.accuracyDropRatio);
      const maxOriginalSpeedupRatio = Math.max(...filteredData.map(item => item.speedupRatio));
      const maxOriginalAccuracyDropRatio = Math.max(...accuracyDropRatios);
      
      // 計算極值點值（使用用戶設定的倍率）
      const extremeSpeedupRatio = maxOriginalSpeedupRatio * speedupMultiplier;
      const extremeAccuracyDropValue = maxOriginalAccuracyDropRatio * accuracyDropMultiplier;
      
      // 使用包含極值的Min-Max標準化
      const maxSpeedupRatio = Math.max(extremeSpeedupRatio, ...filteredData.map(item => item.speedupRatio));
      const minSpeedupRatio = Math.min(...filteredData.map(item => item.speedupRatio));
      const maxAccuracyDropRatio = Math.max(extremeAccuracyDropValue, ...accuracyDropRatios);
      const minAccuracyDropRatio = Math.min(...accuracyDropRatios);
      
      // 創建新的數據對象，避免修改原始數據
      const balancedData = filteredData.map(item => {
        // X軸：推論時間加速比的標準化
        const normSpeedupRatio = maxSpeedupRatio > minSpeedupRatio ? 
          (item.speedupRatio - minSpeedupRatio) / (maxSpeedupRatio - minSpeedupRatio) : 0.5;
        // Y軸：準確率下降比的標準化（使用轉換後的正值）
        const accuracyDropValue = -item.accuracyDropRatio; // 轉為正值
        const normAccuracyDropRatio = maxAccuracyDropRatio > minAccuracyDropRatio ? 
          (accuracyDropValue - minAccuracyDropRatio) / (maxAccuracyDropRatio - minAccuracyDropRatio) : 0.5;
        
        // 計算點到y=x軸的垂直距離
        const distanceToLine = Math.abs(normSpeedupRatio - normAccuracyDropRatio) / Math.sqrt(2);
        
        return {
          ...item, // 複製所有原始屬性
          normSpeedupRatio,
          normAccuracyDropRatio,
          distanceToLine
        };
      });
      
      // 極值點直接設置在圖表右上角
      const normExtremeSpeedupRatio = 1.0;
      const normExtremeAccuracyDropRatio = 1.0;
      
      const extremeBalancedPoint = {
        key: '極值參考點',
        modelType: '極值參考點',
        batchSize: 0,
        avgInferenceTime: 0,
        stdInferenceTime: 0,
        mAP5095: 0,
        vramUsage: 0,
        modelVram: 0,
        gpuLoad: 0,
        maxGpuLoad: 0,
        monitoringSamples: 0,
        monitoringDuration: 0,
        speedupRatio: extremeSpeedupRatio,
        accuracyDropRatio: -extremeAccuracyDropValue,
        isOriginal: false,
        normSpeedupRatio: normExtremeSpeedupRatio,
        normAccuracyDropRatio: normExtremeAccuracyDropRatio,
        distanceToLine: Math.abs(normExtremeSpeedupRatio - normExtremeAccuracyDropRatio) / Math.sqrt(2)
      };
      
      balancedData.push(extremeBalancedPoint);
      
      // 按距離Y=X線排序 (越小表示越權衡)
      balancedData.sort((a, b) => a.distanceToLine - b.distanceToLine);
      return balancedData; // 返回所有符合條件的配置
    } 
    
    return filteredData;
  };

  // 為不同圖表獲取排序後的數據
  const getSortedDataForChart = (sortField, sortOrder) => {
    if (!testData) return [];
    
    let data = [...testData];
    
    // 是否排除原始模型的不同batch（除了batch1）
    if (!includeOriginalBatches) {
      data = data.filter(item => 
        !(item.isOriginal && item.batchSize !== 1)
      );
    }
    
    // 在篩選模式和權衡模式下，圖表也需要篩選
    if (filterType === 'filter' || filterType === 'balanced') {
      data = data.filter(item => 
        item.avgInferenceTime <= inferenceTimeLimit &&
        item.mAP5095 >= mapLimit &&
        item.vramUsage <= vramLimit &&
        item.modelVram <= modelVramLimit &&
        item.gpuLoad <= gpuLoadLimit
      );
    }
    
    data.sort((a, b) => {
      if (sortOrder === 'asc') {
        return a[sortField] - b[sortField];
      } else {
        return b[sortField] - a[sortField];
      }
    });
    
    return data;
  };

  // 處理表格排序
  const handleTableSort = (columnKey) => {
    let newDirection = 'asc';
    if (sortConfig.key === columnKey && sortConfig.direction === 'asc') {
      newDirection = 'desc';
    }
    setSortConfig({ key: columnKey, direction: newDirection });
    setIsDefaultSort(false);
  };

  // 重置表格到預設排序
  const resetTableSort = () => {
    setSortConfig({ key: 'modelType', direction: 'asc' });
    setIsDefaultSort(true);
  };

  // 重置篩選條件到預設值
  const resetFilterToDefault = () => {
    setInferenceTimeLimit(defaultFilterValues.inferenceTimeLimit);
    setMapLimit(defaultFilterValues.mapLimit);
    setVramLimit(defaultFilterValues.vramLimit);
    setModelVramLimit(defaultFilterValues.modelVramLimit);
    setGpuLoadLimit(defaultFilterValues.gpuLoadLimit);
    setSpeedupMultiplier(defaultFilterValues.speedupMultiplier);
    setAccuracyDropMultiplier(defaultFilterValues.accuracyDropMultiplier);
  };

  // 獲取排序後的表格數據
  const getSortedTableData = () => {
    const filteredData = getFilteredDataForDisplay();
    
    if (filterType === 'balanced') {
      // 權衡模式只能按距離排序
      return filteredData.sort((a, b) => a.distanceToLine - b.distanceToLine);
    }
    
    // 篩選模式：如果選擇了模型建議方式，按該方式排序
    if (filterType === 'filter' && metricType) {
      let sortedData = [...filteredData];
      
      if (metricType === 'speed') {
        sortedData.sort((a, b) => a.avgInferenceTime - b.avgInferenceTime);
      } else if (metricType === 'map') {
        sortedData.sort((a, b) => b.mAP5095 - a.mAP5095);
      } else if (metricType === 'vram') {
        sortedData.sort((a, b) => a.vramUsage - b.vramUsage);
      } else if (metricType === 'model_vram') {
        sortedData.sort((a, b) => a.modelVram - b.modelVram);
      } else if (metricType === 'gpu_load') {
        sortedData.sort((a, b) => a.maxGpuLoad - b.maxGpuLoad);
      }
      
      return sortedData;
    }
    
    // 預設模式或其他情況：按表格欄位排序
    return filteredData.sort((a, b) => {
      const aValue = a[sortConfig.key];
      const bValue = b[sortConfig.key];
      
      if (typeof aValue === 'string') {
        return sortConfig.direction === 'asc' 
          ? aValue.localeCompare(bValue)
          : bValue.localeCompare(aValue);
      }
      
      return sortConfig.direction === 'asc' 
        ? aValue - bValue 
        : bValue - aValue;
    });
  };
  
  // 準備圖表數據
  const getInferenceTimeChartOption = () => {
    if (!testData || testData.length === 0) return {};
    
    const data = getSortedDataForChart('avgInferenceTime', inferenceTimeSortOrder);
    const xAxisData = data.map(item => item.modelType);
    
    const option = {
      title: {
        text: '推論時間比較',
        subtext: '單位: 毫秒 (ms)',
        left: 'center',
        top: 5,
        textStyle: { fontSize: 16 },
        subtextStyle: { fontSize: 12 }
      },
      tooltip: {
        trigger: 'axis',
        formatter: function(params) {
          const item = params[0];
          const dataItem = data[item.dataIndex];
          return `${dataItem.modelType}<br/>
                 推論時間: ${dataItem.avgInferenceTime.toFixed(2)} ms<br/>
                 標準差: ${dataItem.stdInferenceTime.toFixed(2)} ms<br/>
                 加速比率: ${(dataItem.speedupRatio * 100).toFixed(2)}%<br/>
                 平均GPU負載: ${dataItem.gpuLoad.toFixed(1)}%<br/>
                 峰值GPU負載: ${dataItem.maxGpuLoad.toFixed(1)}%<br/>
                 監控樣本: ${dataItem.monitoringSamples}`;
        }
      },
      toolbox: {
        feature: {
          saveAsImage: { show: true, title: '保存為圖片' }
        },
        right: 20,
        top: 15
      },
      grid: {
        left: '8%',
        right: '8%',
        bottom: '15%',
        top: '15%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        data: xAxisData,
        axisLabel: {
          rotate: 45,
          interval: 0
        }
      },
      yAxis: {
        type: 'value',
        name: '推論時間 (ms)'
      },
      series: [
        {
          name: '推論時間',
          type: 'bar',
          data: data.map(item => item.avgInferenceTime),
          itemStyle: {
            color: function(params) {
              const dataItem = data[params.dataIndex];
              return getModelColor(dataItem.modelType);
            }
          }
        }
      ]
    };

    return option;
  };
  
  const getVRamUsageChartOption = () => {
    if (!testData || testData.length === 0) return {};
    
    const data = getSortedDataForChart('vramUsage', vramSortOrder);
    const xAxisData = data.map(item => item.modelType);
    
    // 使用篩選條件的值作為限制線
    const markLineData = [];
    if (filterType === 'resource_limit' || filterType === 'balanced') {
      markLineData.push({
        yAxis: vramLimit,
        name: `VRAM限制 (${(vramLimit/1024).toFixed(1)}GB)`,
        lineStyle: { color: '#e74c3c', type: 'dashed', width: 2 }
      });
    }
    
    const option = {
      title: {
        text: 'VRAM 使用量比較',
        subtext: '單位: 兆位元組 (MB)',
        left: 'center',
        top: 10,
        textStyle: { fontSize: 16 },
        subtextStyle: { fontSize: 12 }
      },
      tooltip: {
        trigger: 'axis',
        formatter: function(params) {
          const item = params[0];
          const dataItem = data[item.dataIndex];
          return `${dataItem.modelType}<br/>
                 總VRAM: ${dataItem.vramUsage.toFixed(2)} MB<br/>
                 模型VRAM: ${dataItem.modelVram.toFixed(2)} MB<br/>
                 監控時間: ${dataItem.monitoringDuration.toFixed(1)}秒`;
        }
      },
      toolbox: {
        feature: {
          saveAsImage: { show: true, title: '保存為圖片' }
        },
        right: 20,
        top: 15
      },
      grid: {
        left: '8%',
        right: '8%',
        bottom: '15%',
        top: '20%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        data: xAxisData,
        axisLabel: {
          rotate: 45,
          interval: 0
        }
      },
      yAxis: {
        type: 'value',
        name: 'VRAM (MB)'
      },
      series: [
        {
          name: 'VRAM使用量',
          type: 'bar',
          data: data.map(item => item.vramUsage),
          itemStyle: {
            color: function(params) {
              const dataItem = data[params.dataIndex];
              return getModelColor(dataItem.modelType);
            }
          },
          markLine: markLineData.length > 0 ? { data: markLineData } : undefined
        }
      ]
    };

    return option;
  };
  
  const getMapChartOption = () => {
    if (!testData || testData.length === 0) return {};
    
    const data = getSortedDataForChart('mAP5095', mapSortOrder);
    const xAxisData = data.map(item => item.modelType);
    
    const option = {
      title: {
        text: '準確率比較',
        subtext: 'mAP@50-95',
        left: 'center',
        top: 10,
        textStyle: { fontSize: 16 },
        subtextStyle: { fontSize: 12 }
      },
      tooltip: {
        trigger: 'axis',
        formatter: function(params) {
          const dataItem = data[params[0].dataIndex];
          return `${dataItem.modelType}<br/>
                 mAP@50-95: ${dataItem.mAP5095.toFixed(4)}<br/>
                 mAP下降比率: ${(dataItem.accuracyDropRatio * 100).toFixed(2)}%<br/>`;
        }
      },
      toolbox: {
        feature: {
          saveAsImage: { show: true, title: '保存為圖片' }
        },
        right: 20,
        top: 15
      },
      grid: {
        left: '8%',
        right: '8%',
        bottom: '15%',
        top: '20%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        data: xAxisData,
        axisLabel: {
          rotate: 45,
          interval: 0
        }
      },
      yAxis: {
        type: 'value',
        name: '準確率',
        min: 0,
        max: Math.round(Math.max(...data.map(item => item.mAP5095)) * 1.05 * 100) / 100
      },
      series: [
        {
          name: 'mAP@50-95',
          type: 'bar',
          data: data.map(item => item.mAP5095),
          itemStyle: {
            color: function(params) {
              const dataItem = data[params.dataIndex];
              return getModelColor(dataItem.modelType);
            }
          }
        }
      ]
    };

    return option;
  };
  
  const getGpuLoadChartOption = () => {
    if (!testData || testData.length === 0) return {};
    
    const data = getSortedDataForChart('gpuLoad', gpuLoadSortOrder);
    const xAxisData = data.map(item => item.modelType);
    
    // 使用篩選條件的值作為限制線
    const markLineData = [];
    if (filterType === 'resource_limit' || filterType === 'balanced') {
      markLineData.push({
        yAxis: gpuLoadLimit,
        name: `GPU負載限制 (${gpuLoadLimit}%)`,
        lineStyle: { color: '#e74c3c', type: 'dashed', width: 2 }
      });
    }
    
    const option = {
      title: {
        text: 'GPU負載比較',
        subtext: '單位: 百分比 (%)',
        left: 'center',
        top: 10,
        textStyle: { fontSize: 16 },
        subtextStyle: { fontSize: 12 }
      },
      tooltip: {
        trigger: 'axis',
        formatter: function(params) {
          const dataItem = data[params[0].dataIndex];
          return `${dataItem.modelType}<br/>
                 平均GPU負載: ${dataItem.gpuLoad.toFixed(2)}%<br/>
                 峰值GPU負載: ${dataItem.maxGpuLoad.toFixed(2)}%<br/>
                 監控樣本: ${dataItem.monitoringSamples}<br/>
                 監控時間: ${dataItem.monitoringDuration.toFixed(1)}秒`;
        }
      },
      toolbox: {
        feature: {
          saveAsImage: { show: true, title: '保存為圖片' }
        },
        right: 20,
        top: 15
      },
      grid: {
        left: '8%',
        right: '8%',
        bottom: '15%',
        top: '20%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        data: xAxisData,
        axisLabel: {
          rotate: 45,
          interval: 0
        }
      },
      yAxis: {
        type: 'value',
        name: 'GPU負載 (%)',
        min: 0,
        max: Math.round(Math.max(...data.map(item => item.gpuLoad)) * 1.1)
      },
      series: [
        {
          name: 'GPU負載',
          type: 'bar',
          data: data.map(item => item.gpuLoad),
          itemStyle: {
            color: function(params) {
              const dataItem = data[params.dataIndex];
              return getModelColor(dataItem.modelType);
            }
          },
          markLine: markLineData.length > 0 ? { data: markLineData } : undefined
        }
      ]
    };

    return option;
  };
  
  // 獲取權衡模式散點圖配置
  const getBalanceScatterChartOption = () => {
    if (!testData || testData.length === 0) return {};
    
    let filteredData = [...testData];
    
    // 是否排除原始模型的不同batch（除了batch1）
    if (!includeOriginalBatches) {
      filteredData = filteredData.filter(item => 
        !(item.isOriginal && item.batchSize !== 1)
      );
    }
    
    filteredData = filteredData.filter(item => 
      item.avgInferenceTime <= inferenceTimeLimit &&
      item.mAP5095 >= mapLimit &&
      item.vramUsage <= vramLimit &&
      item.modelVram <= modelVramLimit &&
      item.gpuLoad <= gpuLoadLimit
    );
    
    if (filteredData.length === 0) return {};
    
    // 先計算原始數據的極值
    const accuracyDropRatios = filteredData.map(item => -item.accuracyDropRatio);
    const maxOriginalSpeedupRatio = Math.max(...filteredData.map(item => item.speedupRatio));
    const maxOriginalAccuracyDropRatio = Math.max(...accuracyDropRatios);
    
    // 計算極值點值（使用用戶設定的倍率）
    const extremeSpeedupRatio = maxOriginalSpeedupRatio * speedupMultiplier;
    const extremeAccuracyDropValue = maxOriginalAccuracyDropRatio * accuracyDropMultiplier;
    
    // 使用包含極值的Min-Max標準化
    const maxSpeedupRatio = Math.max(extremeSpeedupRatio, ...filteredData.map(item => item.speedupRatio));
    const minSpeedupRatio = Math.min(...filteredData.map(item => item.speedupRatio));
    const maxAccuracyDropRatio = Math.max(extremeAccuracyDropValue, ...accuracyDropRatios);
    const minAccuracyDropRatio = Math.min(...accuracyDropRatios);
    
    const scatterData = filteredData.map(item => {
      // X軸：推論時間加速比的標準化
      const normSpeedupRatio = maxSpeedupRatio > minSpeedupRatio ? 
        (item.speedupRatio - minSpeedupRatio) / (maxSpeedupRatio - minSpeedupRatio) : 0.5;
      // Y軸：準確率下降比的標準化（使用轉換後的正值）
      const accuracyDropValue = -item.accuracyDropRatio; // 轉為正值
      const normAccuracyDropRatio = maxAccuracyDropRatio > minAccuracyDropRatio ? 
        (accuracyDropValue - minAccuracyDropRatio) / (maxAccuracyDropRatio - minAccuracyDropRatio) : 0.5;
      
      // 計算點到y=x軸的垂直距離
      const distanceToLine = Math.abs(normSpeedupRatio - normAccuracyDropRatio) / Math.sqrt(2);
      
      return {
        value: [normSpeedupRatio, normAccuracyDropRatio],
        name: item.modelType,
        distance: distanceToLine,
        originalData: item,  // 保留原始數據
        itemStyle: {
          color: getModelColor(item.modelType)
        }
      };
    });
    
            // 排序找出最接近權衡線的點，取前5個
    scatterData.sort((a, b) => a.distance - b.distance);
    const closestPoints = scatterData.slice(0, 5);
    
    // 添加極值點 - 現在使用正確的標準化計算
    const normExtremeSpeedupRatio = maxSpeedupRatio > minSpeedupRatio ? 
      (extremeSpeedupRatio - minSpeedupRatio) / (maxSpeedupRatio - minSpeedupRatio) : 0.5;
    const normExtremeAccuracyDropRatio = maxAccuracyDropRatio > minAccuracyDropRatio ? 
      (extremeAccuracyDropValue - minAccuracyDropRatio) / (maxAccuracyDropRatio - minAccuracyDropRatio) : 0.5;
    
    // 極值點直接設置在圖表右上角
    const extremePoint = {
      value: [normExtremeSpeedupRatio, normExtremeAccuracyDropRatio],
      name: '極值參考點',
      distance: Math.abs(normExtremeSpeedupRatio - normExtremeAccuracyDropRatio) / Math.sqrt(2),
      originalData: {
        avgInferenceTime: 0,
        mAP5095: 0,
        speedupRatio: extremeSpeedupRatio,
        accuracyDropRatio: -extremeAccuracyDropValue // 轉換回負值
      },
      itemStyle: {
        color: '#000000', // 黑色
        borderWidth: 2,  // 從 3 改回 2
        borderColor: '#000000'
      },
      symbolSize: 12  // 從 25 改為 12，與其他點相同
    };
    
    // 將極值點添加到散點數據中
    scatterData.push(extremePoint);
    
    // 調試：檢查極值點是否正確添加
    console.log('極值點已添加:', extremePoint);
    console.log('散點數據總數:', scatterData.length);
    
    const option = {
      title: {
        text: '權衡模式分析 - 推論時間加速比 vs 準確率下降比',
        subtext: '最靠近對角線(y=x)的為最佳權衡',
        left: 'center',
        top: 10,
        textStyle: { fontSize: 16 },
        subtextStyle: { fontSize: 12 }
            },
              tooltip: {
          trigger: 'item',
          formatter: function(params) {
            const data = params.data;
            if (!data || !data.value) {
              return '無數據';
            }
            
            // 極值點特殊顯示
            if (data.name === '極值參考點') {
              return `${data.name}<br/>
                     標準化加速比: ${data.value[0].toFixed(3)}<br/>
                     標準化準確率下降比: ${data.value[1].toFixed(3)}<br/>
                     距離權衡線: ${(data.distance || 0).toFixed(4)}<br/>
                     理論最大加速比: ${data.originalData.speedupRatio.toFixed(4)}<br/>
                     理論最大下降比: ${Math.abs(data.originalData.accuracyDropRatio).toFixed(4)}<br/>
                     <strong>極值參考點 (性能上限)</strong>`;
            }
            
            // 檢查是否為最接近的前5個點
            const rank = closestPoints.findIndex(point => point.name === data.name) + 1;
            const rankText = rank > 0 ? `<br/>排名: 第${rank}個最接近點` : '';
            
            return `${data.name}<br/>
                   標準化加速比: ${data.value[0].toFixed(3)}<br/>
                   標準化準確率下降比: ${data.value[1].toFixed(3)}<br/>
                   距離權衡線: ${(data.distance || 0).toFixed(4)}<br/>
                   原始推論時間: ${data.originalData.avgInferenceTime.toFixed(2)}ms<br/>
                   原始mAP: ${data.originalData.mAP5095.toFixed(4)}<br/>
                   加速比率: ${data.originalData.speedupRatio.toFixed(4)}${rankText}`;
          }
        },
      toolbox: {
        feature: {
          saveAsImage: { show: true, title: '保存為圖片' }
        },
        right: 20,
        top: 15
      },
      grid: {
        left: 70,
        right: 70,
        bottom: 45,  // 從 70 減少到 45
        top: 80,
        containLabel: true,
        width: 500,
        height: 520  // 從 500 增加到 520，補償底部空間
      },
      xAxis: {
        type: 'value',
        name: '標準化推論時間加速比',
        nameLocation: 'middle',
        nameGap: 25,
        nameTextStyle: {
          fontSize: 12,
          color: '#666'
        },
        min: 0,
        max: 1,
        interval: 0.2,
        splitNumber: 5,
        axisLabel: {
          formatter: function(value) {
            if (value === 1) {
              return '1\n(越高越好)';
            }
            return value;
          },
          align: 'center',
          verticalAlign: 'top',
          fontSize: 10
        }
      },
      yAxis: {
        type: 'value',
        name: '標準化準確率下降比 (越高越差)',
        min: 0,
        max: 1,
        interval: 0.2,
        splitNumber: 5,
        axisLabel: {
          formatter: '{value}'
        }
      },
      series: [
        {
          name: '模型表現',
          type: 'scatter',
          data: scatterData,
          symbolSize: function(data) {
            // 極值點使用相同大小
            if (data.name === '極值參考點') {
              return 12;  // 從 25 改為 12
            }
            // 最接近權衡線的前5個點顯示較大
            const isClosest = closestPoints.some(point => point.name === data.name);
            return isClosest ? 20 : 12;
          },
          symbol: function(data) {
            // 極值點使用菱形符號
            if (data.name === '極值參考點') {
              return 'diamond';
            }
            return 'circle';
          },
          markLine: {
            data: [
              // 主要權衡線 y=x (實線)
              [
                { coord: [0, 0], symbol: 'none' },
                { coord: [1, 1], symbol: 'none' }
              ],
              // 前5個最接近點到權衡線的垂直距離線 (虛線)
              ...closestPoints.map((point, index) => {
                const x = point.value[0];
                const y = point.value[1];
                // 計算點到 y=x 線的垂直投影點
                // 對於直線 y=x，點(x,y)的垂直投影點為 ((x+y)/2, (x+y)/2)
                const projectionX = (x + y) / 2;
                const projectionY = (x + y) / 2;
                
                return [
                  { 
                    coord: [x, y], 
                    symbol: 'none', 
                    name: `第${index + 1}個最接近點: ${point.name}`,
                    value: `TOP${index + 1}`
                  },
                  { 
                    coord: [projectionX, projectionY], 
                    symbol: 'none' 
                  }
                ];
              })
            ],
            lineStyle: function(params) {
              // 第一條線（權衡線）為實線，其他為虛線
              if (params.dataIndex === 0) {
                return { color: '#666', type: 'solid', width: 2, opacity: 0.8 };
              } else {
                return { color: '#ff4d4f', type: 'dashed', width: 2, opacity: 0.8 };
              }
            },
            label: { 
              show: true, 
              position: 'middle', 
              formatter: function(params) {
                // 只在權衡線上顯示標籤
                if (params.dataIndex === 0) {
                  return '權衡線 (y=x)';
                }
                return '';
              }
            },
            emphasis: {
              lineStyle: {
                width: 3
              }
            },
            tooltip: {
              show: true,
              formatter: function(params) {
                if (params.dataIndex === 0) {
                  return '權衡線 (y=x)';
                } else {
                  const index = params.dataIndex;
                  const point = closestPoints[index - 1];
                  if (point) {
                    return `第${index}個最接近點<br/>模型: ${point.name}<br/>距離: ${point.distance.toFixed(4)}`;
                  }
                  return '';
                }
              }
            }
          }
        }
      ]
    };

    return option;
  };
  
  // 匯出圖表為 PDF
  const exportChartsToPDF = () => {
    if (!chartContainerRef.current) {
      message.error('無法獲取圖表容器');
      return;
    }
    
    message.loading('正在生成 PDF，請稍候...', 0);
    
    const input = chartContainerRef.current;
    const pdf = new jsPDF('p', 'mm', 'a4');
    const pdfWidth = pdf.internal.pageSize.getWidth();
    const pdfHeight = pdf.internal.pageSize.getHeight();
    
    html2canvas(input, { scale: 2 }).then(canvas => {
      // 計算合適的縮放比例
      const imgWidth = canvas.width;
      const imgHeight = canvas.height;
      const ratio = Math.min(pdfWidth / imgWidth, pdfHeight / imgHeight);
      
      const imgData = canvas.toDataURL('image/png');
      pdf.addImage(imgData, 'PNG', 0, 0, imgWidth * ratio, imgHeight * ratio);
      
      // 保存 PDF
      pdf.save('AI模型測試結果分析報告.pdf');
      message.destroy();
      message.success('PDF 生成成功');
    }).catch(error => {
      message.destroy();
      message.error('PDF 生成失敗: ' + error.message);
    });
  };
  
  // 匯出數據為 Excel
  const exportDataToExcel = () => {
    if (!testData || testData.length === 0) {
      message.error('無數據可匯出');
      return;
    }
    
    try {
      const data = getFilteredDataForDisplay();
      const baseMapping = {
        '模型': item => item.key,
        '模型類型': item => item.modelType,
        '批次大小': item => item.batchSize,
        '推論時間 (ms)': item => item.avgInferenceTime.toFixed(2),
        '標準差 (ms)': item => item.stdInferenceTime.toFixed(2),
        'mAP@50-95': item => item.mAP5095.toFixed(4),
        'VRAM (MB)': item => item.vramUsage.toFixed(2),
        '模型VRAM (MB)': item => item.modelVram.toFixed(2),
        '平均GPU負載 (%)': item => item.gpuLoad.toFixed(2),
        '峰值GPU負載 (%)': item => item.maxGpuLoad.toFixed(2),
        '監控樣本數': item => item.monitoringSamples,
        '監控時間 (秒)': item => item.monitoringDuration.toFixed(1),
        '加速比': item => item.speedup.toFixed(2),
        '準確率比率': item => item.accuracyRatio.toFixed(4)
      };
      
      // 如果是權衡模式，添加距離欄位
      if (filterType === 'balanced') {
        baseMapping['距離Y=X線'] = item => item.distanceToLine ? item.distanceToLine.toFixed(4) : '';
      }
      
      const worksheet = XLSX.utils.json_to_sheet(data.map(item => {
        const mappedItem = {};
        Object.keys(baseMapping).forEach(key => {
          mappedItem[key] = baseMapping[key](item);
        });
        return mappedItem;
      }));
      
      const workbook = XLSX.utils.book_new();
      XLSX.utils.book_append_sheet(workbook, worksheet, '模型性能數據');
      
      // 生成並下載 Excel 文件
      XLSX.writeFile(workbook, 'AI模型測試結果分析數據.xlsx');
      message.success('Excel 檔案已匯出');
    } catch (error) {
      console.error('匯出 Excel 失敗:', error);
      message.error('匯出 Excel 失敗: ' + error.message);
    }
  };
  
  // 結果數據表格列定義
  const getTableColumns = () => {
    // 篩選模式下選擇了模型建議方式時，禁用手動排序
    const canSort = filterType !== 'balanced' && !(filterType === 'filter' && metricType);
    
    // 獲取當前排序依據的欄位key
    const getCurrentSortKey = () => {
      if (filterType === 'filter' && metricType) {
        switch (metricType) {
          case 'speed': return 'avgInferenceTime';
          case 'map': return 'mAP5095';
          case 'vram': return 'vramUsage';
          case 'model_vram': return 'modelVram';
          case 'gpu_load': return 'maxGpuLoad';
          default: return null;
        }
      }
      return canSort && sortConfig.key;
    };
    
    const currentSortKey = getCurrentSortKey();
    
    const baseColumns = [
      {
        title: '模型',
        dataIndex: 'key',
        key: 'modelType',
        sorter: canSort,
        sortOrder: sortConfig.key === 'modelType' ? sortConfig.direction : null,
        onHeaderCell: () => ({
          onClick: canSort ? () => handleTableSort('modelType') : undefined,
          style: { 
            cursor: canSort ? 'pointer' : 'default',
            color: (canSort && sortConfig.key === 'modelType') || currentSortKey === 'modelType' ? '#ff4d4f' : 'inherit'
          }
        }),
      },
      {
        title: '批次大小',
        dataIndex: 'batchSize',
        key: 'batchSize',
        sorter: canSort,
        sortOrder: sortConfig.key === 'batchSize' ? sortConfig.direction : null,
        onHeaderCell: () => ({
          onClick: canSort ? () => handleTableSort('batchSize') : undefined,
          style: { 
            cursor: canSort ? 'pointer' : 'default',
            color: (canSort && sortConfig.key === 'batchSize') || currentSortKey === 'batchSize' ? '#ff4d4f' : 'inherit'
          }
        }),
      },
      {
        title: '推論時間 (ms)',
        dataIndex: 'avgInferenceTime',
        key: 'avgInferenceTime',
        render: (text) => text.toFixed(2),
        sorter: canSort,
        sortOrder: sortConfig.key === 'avgInferenceTime' ? sortConfig.direction : null,
        onHeaderCell: () => ({
          onClick: canSort ? () => handleTableSort('avgInferenceTime') : undefined,
          style: { 
            cursor: canSort ? 'pointer' : 'default',
            color: (canSort && sortConfig.key === 'avgInferenceTime') || currentSortKey === 'avgInferenceTime' ? '#ff4d4f' : 'inherit'
          }
        }),
      },
      {
        title: '推論時間加速比率',
        dataIndex: 'speedupRatio',
        key: 'speedupRatio',
        render: (text) => (text * 100).toFixed(2) + '%',
        sorter: canSort,
        sortOrder: sortConfig.key === 'speedupRatio' ? sortConfig.direction : null,
        onHeaderCell: () => ({
          onClick: canSort ? () => handleTableSort('speedupRatio') : undefined,
          style: { 
            cursor: canSort ? 'pointer' : 'default',
            color: (canSort && sortConfig.key === 'speedupRatio') || currentSortKey === 'speedupRatio' ? '#ff4d4f' : 'inherit'
          }
        }),
      },
      {
        title: 'mAP@50-95',
        dataIndex: 'mAP5095',
        key: 'mAP5095',
        render: (text) => text.toFixed(4),
        sorter: canSort,
        sortOrder: sortConfig.key === 'mAP5095' ? sortConfig.direction : null,
        onHeaderCell: () => ({
          onClick: canSort ? () => handleTableSort('mAP5095') : undefined,
          style: { 
            cursor: canSort ? 'pointer' : 'default',
            color: (canSort && sortConfig.key === 'mAP5095') || currentSortKey === 'mAP5095' ? '#ff4d4f' : 'inherit'
          }
        }),
      },
      {
        title: 'mAP下降比率',
        dataIndex: 'accuracyDropRatio',
        key: 'accuracyDropRatio',
        render: (text) => (text * 100).toFixed(2) + '%',
        sorter: canSort,
        sortOrder: sortConfig.key === 'accuracyDropRatio' ? sortConfig.direction : null,
        onHeaderCell: () => ({
          onClick: canSort ? () => handleTableSort('accuracyDropRatio') : undefined,
          style: { 
            cursor: canSort ? 'pointer' : 'default',
            color: (canSort && sortConfig.key === 'accuracyDropRatio') || currentSortKey === 'accuracyDropRatio' ? '#ff4d4f' : 'inherit'
          }
        }),
      },
      {
        title: 'VRAM (MB)',
        dataIndex: 'vramUsage',
        key: 'vramUsage',
        render: (text) => text.toFixed(2),
        sorter: canSort,
        sortOrder: sortConfig.key === 'vramUsage' ? sortConfig.direction : null,
        onHeaderCell: () => ({
          onClick: canSort ? () => handleTableSort('vramUsage') : undefined,
          style: { 
            cursor: canSort ? 'pointer' : 'default',
            color: (canSort && sortConfig.key === 'vramUsage') || currentSortKey === 'vramUsage' ? '#ff4d4f' : 'inherit'
          }
        }),
      },
      {
        title: '模型VRAM (MB)',
        dataIndex: 'modelVram',
        key: 'modelVram',
        render: (text) => text.toFixed(2),
        sorter: canSort,
        sortOrder: sortConfig.key === 'modelVram' ? sortConfig.direction : null,
        onHeaderCell: () => ({
          onClick: canSort ? () => handleTableSort('modelVram') : undefined,
          style: { 
            cursor: canSort ? 'pointer' : 'default',
            color: (canSort && sortConfig.key === 'modelVram') || currentSortKey === 'modelVram' ? '#ff4d4f' : 'inherit'
          }
        }),
      },
      {
        title: '峰值GPU負載 (%)',
        dataIndex: 'maxGpuLoad',
        key: 'maxGpuLoad',
        render: (text) => text.toFixed(2),
        sorter: canSort,
        sortOrder: sortConfig.key === 'maxGpuLoad' ? sortConfig.direction : null,
        onHeaderCell: () => ({
          onClick: canSort ? () => handleTableSort('maxGpuLoad') : undefined,
          style: { 
            cursor: canSort ? 'pointer' : 'default',
            color: (canSort && sortConfig.key === 'maxGpuLoad') || currentSortKey === 'maxGpuLoad' ? '#ff4d4f' : 'inherit'
          }
        }),
      }
    ];

          // 權衡模式添加距離欄位
      if (filterType === 'balanced') {
        baseColumns.push({
        title: '與y=x軸的距離',
        dataIndex: 'distanceToLine',
        key: 'distanceToLine',
        render: (text) => typeof text === 'number' ? text.toFixed(4) : '0.0000',
        sorter: false,
        onHeaderCell: () => ({
          style: { 
            color: '#ff4d4f', // 權衡模式下只能按此欄位排序，始终显示红色
            cursor: 'default'
          }
        }),
      });
    }

    return baseColumns;
  };
  
  return (
    <div style={{ padding: '20px' }}>
      <Title level={2}>自動化模型性能評估結果分析儀表板</Title>
      <Divider />
      
      {/* 上傳與控制區域 */}
      <Row gutter={24} style={{ marginBottom: '20px' }}>
        <Col span={8}>
          <Card title="結果載入">
            <div style={{ marginBottom: '10px' }}>
              <Select
                style={{ width: '100%' }}
                placeholder="選擇測試結果..."
                value={selectedResult}
                onChange={handleResultSelection}
                allowClear
              >
                {availableResults.map(result => (
                  <Option key={result.taskId} value={result.taskId}>
                    {result.displayName}
                  </Option>
                ))}
              </Select>
            </div>
            <Upload
              ref={uploadRef}
              beforeUpload={handleFileUpload}
              showUploadList={false}
              accept=".json"
            >
              <Button icon={<UploadOutlined />} loading={loading}>
                上傳 JSON 檔案
              </Button>
            </Upload>
          </Card>
        </Col>
        
        <Col span={16}>
          <Card title="模型建議-選擇模式">
            <Radio.Group 
              value={filterType} 
              onChange={(e) => setFilterType(e.target.value)}
              buttonStyle="solid"
              style={{ marginBottom: '20px' }}
            >
              <Tooltip title="提供圖表以及可排序的表格讓使用者排序">
                <Radio.Button value="default">預設模式</Radio.Button>
              </Tooltip>
              <Tooltip title="提供篩選各指標的模式">
                <Radio.Button value="filter">篩選模式</Radio.Button>
              </Tooltip>
              <Tooltip title="提供交換最穩定或是最權衡的模型">
                <Radio.Button value="balanced">權衡模式</Radio.Button>
              </Tooltip>
            </Radio.Group>
            
            {(filterType === 'filter' || filterType === 'balanced') && (
              <div style={{ marginTop: '10px' }}>
                <div style={{ marginBottom: '15px' }}>
                  <Space>
                    <Checkbox 
                      checked={includeOriginalBatches}
                      onChange={(e) => setIncludeOriginalBatches(e.target.checked)}
                    >
                      是否加入原始模型(僅調整batch)比較
                    </Checkbox>
                    <Button 
                      size="small" 
                      onClick={resetFilterToDefault}
                      icon={<ReloadOutlined />}
                    >
                      重置篩選條件
                    </Button>
                  </Space>
                </div>
                {(filterType === 'filter' || filterType === 'balanced') && (
                  <>
                <Row gutter={16}>
                  <Col span={8}>
                    <div style={{ marginBottom: '15px' }}>
                      <Text>推論時間上限: {inferenceTimeLimit}</Text>
                      <Row gutter={8} style={{ marginTop: '8px' }}>
                        <Col span={14}>
                          <Slider 
                            min={1} 
                            max={1000} 
                            value={inferenceTimeLimit} 
                            onChange={setInferenceTimeLimit}
                            step={0.1}
                          />
                        </Col>
                        <Col span={10}>
                          <Input
                            type="number"
                            min={1}
                            max={1000}
                            value={inferenceTimeLimit}
                            onChange={(e) => {
                              const value = parseFloat(e.target.value);
                              if (!isNaN(value) && value >= 1 && value <= 1000) {
                                setInferenceTimeLimit(value);
                              }
                            }}
                            addonAfter="ms"
                            style={{ width: '100%' }}
                          />
                        </Col>
                      </Row>
                    </div>
                    <div style={{ marginBottom: '15px' }}>
                      <Text>mAP下限: {mapLimit}</Text>
                      <Row gutter={8} style={{ marginTop: '8px' }}>
                        <Col span={14}>
                          <Slider 
                            min={0} 
                            max={1} 
                            value={mapLimit} 
                            onChange={setMapLimit}
                            step={0.01}
                          />
                        </Col>
                        <Col span={10}>
                          <Input
                            type="number"
                            min={0}
                            max={1}
                            value={mapLimit}
                            onChange={(e) => {
                              const value = parseFloat(e.target.value);
                              if (!isNaN(value) && value >= 0 && value <= 1) {
                                setMapLimit(value);
                              }
                            }}
                            step={0.01}
                            style={{ width: '100%' }}
                          />
                        </Col>
                      </Row>
                    </div>
                  </Col>
                  <Col span={8}>
                    <div style={{ marginBottom: '15px' }}>
                      <Text>VRAM上限: {vramLimit}</Text>
                      <Row gutter={8} style={{ marginTop: '8px' }}>
                        <Col span={14}>
                          <Slider 
                            min={0} 
                            max={32768} 
                            value={vramLimit} 
                            onChange={setVramLimit}
                            step={1}
                          />
                        </Col>
                        <Col span={10}>
                          <Input
                            type="number"
                            min={0}
                            max={32768}
                            value={vramLimit}
                            onChange={(e) => {
                              const value = parseInt(e.target.value);
                              if (!isNaN(value) && value >= 0 && value <= 32768) {
                                setVramLimit(value);
                              }
                            }}
                            addonAfter="MB"
                            style={{ width: '100%' }}
                          />
                        </Col>
                      </Row>
                    </div>
                    <div style={{ marginBottom: '15px' }}>
                      <Text>模型VRAM上限: {modelVramLimit}</Text>
                      <Row gutter={8} style={{ marginTop: '8px' }}>
                        <Col span={14}>
                          <Slider 
                            min={0} 
                            max={16384} 
                            value={modelVramLimit} 
                            onChange={setModelVramLimit}
                            step={1}
                          />
                        </Col>
                        <Col span={10}>
                          <Input
                            type="number"
                            min={0}
                            max={16384}
                            value={modelVramLimit}
                            onChange={(e) => {
                              const value = parseInt(e.target.value);
                              if (!isNaN(value) && value >= 0 && value <= 16384) {
                                setModelVramLimit(value);
                              }
                            }}
                            addonAfter="MB"
                            style={{ width: '100%' }}
                          />
                        </Col>
                      </Row>
                    </div>
                  </Col>
                  <Col span={8}>
                    <div style={{ marginBottom: '15px' }}>
                      <Text>GPU負載上限: {gpuLoadLimit}</Text>
                      <Row gutter={8} style={{ marginTop: '8px' }}>
                        <Col span={14}>
                          <Slider 
                            min={0} 
                            max={100} 
                            value={gpuLoadLimit} 
                            onChange={setGpuLoadLimit}
                            step={1}
                          />
                        </Col>
                        <Col span={10}>
                          <Input
                            type="number"
                            min={0}
                            max={100}
                            value={gpuLoadLimit}
                            onChange={(e) => {
                              const value = parseInt(e.target.value);
                              if (!isNaN(value) && value >= 0 && value <= 100) {
                                setGpuLoadLimit(value);
                              }
                            }}
                            addonAfter="%"
                            style={{ width: '100%' }}
                          />
                        </Col>
                      </Row>
                    </div>
                  </Col>
                </Row>
                
                    {filterType === 'filter' && (
                      <div style={{ marginTop: '15px' }}>
                        <Text>模型建議方式 (選擇指標作為排序目標): </Text>
                        <Radio.Group 
                          value={metricType} 
                          onChange={(e) => {
                            setMetricType(e.target.value);
                            // 重置排序狀態，讓排序重新生效
                            setIsDefaultSort(false);
                          }}
                          buttonStyle="solid"
                          style={{ marginTop: '8px' }}
                        >
                          <Radio.Button value="speed">以推論時間為目標 (越低越好)</Radio.Button>
                          <Radio.Button value="map">以mAP為目標 (越高越好)</Radio.Button>
                          <Radio.Button value="vram">以VRAM為目標 (越低越好)</Radio.Button>
                          <Radio.Button value="model_vram">以模型VRAM為目標 (越低越好)</Radio.Button>
                          <Radio.Button value="gpu_load">以GPU負載為目標 (越低越好)</Radio.Button>
                        </Radio.Group>
                      </div>
                    )}
                  </>
                )}
                
                {filterType === 'balanced' && (
                  <div style={{ marginTop: '20px', padding: '15px', border: '1px solid #d9d9d9', borderRadius: '6px', backgroundColor: '#fafafa' }}>
                    <Text strong>極值參考點設置</Text>
                    <Row gutter={16} style={{ marginTop: '10px' }}>
                      <Col span={12}>
                        <div>
                          <Text>推論時間加速比的倍率(相較MAX值): {speedupMultiplier}</Text>
                          <Row gutter={8} style={{ marginTop: '8px' }}>
                            <Col span={14}>
                              <Slider 
                                min={1.0} 
                                max={2.0} 
                                value={speedupMultiplier} 
                                onChange={setSpeedupMultiplier}
                                step={0.1}
                              />
                            </Col>
                            <Col span={10}>
                              <Input
                                type="number"
                                min={1.0}
                                max={2.0}
                                value={speedupMultiplier}
                                onChange={(e) => {
                                  const value = parseFloat(e.target.value);
                                  if (!isNaN(value) && value >= 1.0 && value <= 2.0) {
                                    setSpeedupMultiplier(value);
                                  }
                                }}
                                step={0.1}
                                style={{ width: '100%' }}
                              />
                            </Col>
                          </Row>
                        </div>
                      </Col>
                      <Col span={12}>
                        <div>
                          <Text>準確率下降比的倍率(相較MAX值): {accuracyDropMultiplier}</Text>
                          <Row gutter={8} style={{ marginTop: '8px' }}>
                            <Col span={14}>
                              <Slider 
                                min={1.0}  // 從 0.5 改為 1.0
                                max={2.0} 
                                value={accuracyDropMultiplier} 
                                onChange={setAccuracyDropMultiplier}
                                step={0.1}
                              />
                            </Col>
                            <Col span={10}>
                              <Input
                                type="number"
                                min={1.0}  // 從 0.5 改為 1.0
                                max={2.0}
                                value={accuracyDropMultiplier}
                                onChange={(e) => {
                                  const value = parseFloat(e.target.value);
                                  if (!isNaN(value) && value >= 1.0 && value <= 2.0) {  // 從 0.5 改為 1.0
                                    setAccuracyDropMultiplier(value);
                                  }
                                }}
                                step={0.1}
                                style={{ width: '100%' }}
                              />
                            </Col>
                          </Row>
                        </div>
                      </Col>
                    </Row>
                  </div>
                )}
              </div>
            )}
          </Card>
        </Col>
      </Row>
      
      {/* 圖表顯示區 */}
      {testData ? (
        <div ref={chartContainerRef}>
          {/* 權衡模式特殊顯示 */}
          {filterType === 'balanced' && (
            <Row gutter={24} style={{ marginBottom: '20px' }}>
              <Col span={24}>
                <Card title="權衡模式分析 - 推論時間加速比 vs 準確率下降比">
                  <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', marginBottom: '10px' }}>
                    <ReactECharts
                      option={getBalanceScatterChartOption()}
                      style={{ height: '800px', width: '800px' }}
                      notMerge={true}
                    />
                  </div>
                </Card>
              </Col>
            </Row>
          )}
          
          {/* 四個比較圖表 - 在篩選模式和權衡模式下可折疊 */}
          {filterType === 'filter' || filterType === 'balanced' ? (
            <Collapse 
              defaultActiveKey={chartsCollapsed ? [] : ['charts']}
              expandIcon={({ isActive }) => <CaretRightOutlined rotate={isActive ? 90 : 0} />}
              style={{ marginBottom: '20px' }}
            >
              <Panel header="詳細比較圖表" key="charts">
                <Row gutter={24} style={{ marginBottom: '20px' }}>
                  <Col span={12}>
                    <Card 
                      title="推論時間比較"
                      extra={
                        <Radio.Group 
                          size="small"
                          value={inferenceTimeSortOrder} 
                          onChange={(e) => setInferenceTimeSortOrder(e.target.value)}
                        >
                          <Radio.Button value="asc">升序</Radio.Button>
                          <Radio.Button value="desc">降序</Radio.Button>
                        </Radio.Group>
                      }
                    >
                      <ReactECharts
                        ref={inferenceTimeChartRef}
                        option={getInferenceTimeChartOption()}
                        style={{ height: '350px' }}
                        notMerge={true}
                      />
                    </Card>
                  </Col>
                  <Col span={12}>
                    <Card 
                      title="VRAM 使用量比較"
                      extra={
                        <Radio.Group 
                          size="small"
                          value={vramSortOrder} 
                          onChange={(e) => setVramSortOrder(e.target.value)}
                        >
                          <Radio.Button value="asc">升序</Radio.Button>
                          <Radio.Button value="desc">降序</Radio.Button>
                        </Radio.Group>
                      }
                    >
                      <ReactECharts
                        ref={vramUsageChartRef}
                        option={getVRamUsageChartOption()}
                        style={{ height: '350px' }}
                        notMerge={true}
                      />
                    </Card>
                  </Col>
                </Row>
                
                <Row gutter={24} style={{ marginBottom: '20px' }}>
                  <Col span={12}>
                    <Card 
                      title="準確率比較"
                      extra={
                        <Radio.Group 
                          size="small"
                          value={mapSortOrder} 
                          onChange={(e) => setMapSortOrder(e.target.value)}
                        >
                          <Radio.Button value="asc">升序</Radio.Button>
                          <Radio.Button value="desc">降序</Radio.Button>
                        </Radio.Group>
                      }
                    >
                      <ReactECharts
                        ref={mAPChartRef}
                        option={getMapChartOption()}
                        style={{ height: '350px' }}
                        notMerge={true}
                      />
                    </Card>
                  </Col>
                  <Col span={12}>
                    <Card 
                      title="GPU負載比較"
                      extra={
                        <Radio.Group 
                          size="small"
                          value={gpuLoadSortOrder} 
                          onChange={(e) => setGpuLoadSortOrder(e.target.value)}
                        >
                          <Radio.Button value="asc">升序</Radio.Button>
                          <Radio.Button value="desc">降序</Radio.Button>
                        </Radio.Group>
                      }
                    >
                      <ReactECharts
                        ref={gpuLoadChartRef}
                        option={getGpuLoadChartOption()}
                        style={{ height: '350px' }}
                        notMerge={true}
                      />
                    </Card>
                  </Col>
                </Row>
              </Panel>
            </Collapse>
          ) : (
            // 預設模式顯示所有圖表
            <>
              <Row gutter={24} style={{ marginBottom: '20px' }}>
                <Col span={12}>
                  <Card 
                    title="推論時間比較"
                    extra={
                      <Radio.Group 
                        size="small"
                        value={inferenceTimeSortOrder} 
                        onChange={(e) => setInferenceTimeSortOrder(e.target.value)}
                      >
                        <Radio.Button value="asc">升序</Radio.Button>
                        <Radio.Button value="desc">降序</Radio.Button>
                      </Radio.Group>
                    }
                  >
                    <ReactECharts
                      ref={inferenceTimeChartRef}
                      option={getInferenceTimeChartOption()}
                      style={{ height: '350px' }}
                      notMerge={true}
                    />
                  </Card>
                </Col>
                <Col span={12}>
                  <Card 
                    title="VRAM 使用量比較"
                    extra={
                      <Radio.Group 
                        size="small"
                        value={vramSortOrder} 
                        onChange={(e) => setVramSortOrder(e.target.value)}
                      >
                        <Radio.Button value="asc">升序</Radio.Button>
                        <Radio.Button value="desc">降序</Radio.Button>
                      </Radio.Group>
                    }
                  >
                    <ReactECharts
                      ref={vramUsageChartRef}
                      option={getVRamUsageChartOption()}
                      style={{ height: '350px' }}
                      notMerge={true}
                    />
                  </Card>
                </Col>
              </Row>
              
              <Row gutter={24} style={{ marginBottom: '20px' }}>
                <Col span={12}>
                  <Card 
                    title="準確率比較"
                    extra={
                      <Radio.Group 
                        size="small"
                        value={mapSortOrder} 
                        onChange={(e) => setMapSortOrder(e.target.value)}
                      >
                        <Radio.Button value="asc">升序</Radio.Button>
                        <Radio.Button value="desc">降序</Radio.Button>
                      </Radio.Group>
                    }
                  >
                    <ReactECharts
                      ref={mAPChartRef}
                      option={getMapChartOption()}
                      style={{ height: '350px' }}
                      notMerge={true}
                    />
                  </Card>
                </Col>
                <Col span={12}>
                  <Card 
                    title="GPU負載比較"
                    extra={
                      <Radio.Group 
                        size="small"
                        value={gpuLoadSortOrder} 
                        onChange={(e) => setGpuLoadSortOrder(e.target.value)}
                      >
                        <Radio.Button value="asc">升序</Radio.Button>
                        <Radio.Button value="desc">降序</Radio.Button>
                      </Radio.Group>
                    }
                  >
                    <ReactECharts
                      ref={gpuLoadChartRef}
                      option={getGpuLoadChartOption()}
                      style={{ height: '350px' }}
                      notMerge={true}
                    />
                  </Card>
                </Col>
              </Row>
            </>
          )}
          
          {/* 結果表格 */}
          <Card 
            title={filterType === 'filter' ? "建議部署模型列表" : 
                   filterType === 'balanced' ? "建議部署模型列表" : 
                   "模型列表"}
            style={{ marginTop: '20px' }}
            extra={
              <Space>
                {(filterType !== 'balanced' && !(filterType === 'filter' && metricType)) && (
                  <Button 
                    icon={<ReloadOutlined />} 
                    onClick={resetTableSort}
                    disabled={isDefaultSort}
                  >
                    重置排序
                  </Button>
                )}
                <Button 
                  icon={<DownloadOutlined />} 
                  onClick={exportChartsToPDF}
                >
                  導出圖表 (PDF)
                </Button>
                <Button 
                  icon={<DownloadOutlined />} 
                  onClick={exportDataToExcel}
                >
                  導出數據 (Excel)
                </Button>
              </Space>
            }
          >
            <Table 
              columns={getTableColumns()} 
              dataSource={getSortedTableData()}
              size="small"
              pagination={{ pageSize: 10 }}
              rowClassName={(record) => {
                // 原始模型_batch1和極值參考點使用灰色顯示
                if (record.modelType === '原始模型_batch1' || record.modelType === '極值參考點') {
                  return 'gray-row';
                }
                return '';
              }}
            />
          </Card>
        </div>
      ) : (
        <Card style={{ textAlign: 'center', padding: '50px 0' }}>
          <Title level={4}>請選擇測試結果或上傳 JSON 檔案</Title>
          <p>選擇測試結果後將顯示自動化模型性能評估結果分析圖表</p>
        </Card>
      )}
    </div>
  );
};

export default ModelPerformanceAnalyzer; 