import React, { useState, useEffect } from 'react';
import { Card, Select, Button, Table, Form, Input, Tag, Alert, Progress, Popconfirm, Space, Statistic, Row, Col, Typography, Upload, Modal, message, Divider, Radio, Empty, Tooltip } from 'antd';
import { DeleteOutlined, SyncOutlined, AreaChartOutlined, UploadOutlined, InboxOutlined, DownloadOutlined, QuestionCircleOutlined } from '@ant-design/icons';
import api, { API_BASE_URL } from '../api/client';
import { calculateActualProgress, getStatusColor, formatDateTime, stepNameMap, statusNameMap } from '../utils/benchmarkFormat';

const { Option } = Select;
const { Text } = Typography;
const { Dragger } = Upload;

const BenchmarkPage = () => {
  // 基本狀態
  const [models, setModels] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [testTasks, setTestTasks] = useState([]);
  const [tasksLoading, setTasksLoading] = useState(false);
  
  // 表單和當前任務狀態
  const [form] = Form.useForm();
  const [currentTask, setCurrentTask] = useState(null);
  const [taskStatus, setTaskStatus] = useState({});
  const [refreshInterval, setRefreshInterval] = useState(null);
  
  // 數據集上傳相關狀態
  const [uploadVisible, setUploadVisible] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [datasetFile, setDatasetFile] = useState(null);
  const [datasetType, setDatasetType] = useState('object');
  
  // 系統狀態
  const [systemState, setSystemState] = useState(null);
  const [systemStateLoading, setSystemStateLoading] = useState(false);
  
  // 組合詳細結果狀態
  const [combinationDetails, setCombinationDetails] = useState({});
  const [loadingCombinations, setLoadingCombinations] = useState({});

  // 進度計算移至 utils/benchmarkFormat.js（calculateActualProgress，已加 jest 測試）

  // 計算當前任務的進度數據
  const progressData = calculateActualProgress(taskStatus);

  // 初始化頁面
  useEffect(() => {
    fetchModels();
    fetchDatasets();
    fetchTestTasks();
    fetchSystemState();
    checkActiveTask();

    // 系統狀態刷新間隔（10秒）
    const systemInterval = setInterval(() => {
      fetchSystemState();
    }, 10000);

    // 清理函數
    return () => {
      clearInterval(systemInterval);
    };
  }, []);

  // 單獨處理當前任務的刷新
  useEffect(() => {
    if (!currentTask) return;

    let taskInterval = null;
    
    // 根據任務狀態決定刷新間隔
    const getRefreshInterval = () => {
      if (!taskStatus.status || taskStatus.status === 'completed' || taskStatus.status === 'failed' || taskStatus.status === 'aborted') {
        return null; // 不需要刷新
      }
      
      // 如果在轉換階段，降低刷新頻率到30秒
      if (taskStatus.current_step === 'conversion') {
        return 30000;
      }
      
      // 其他階段保持10秒
      return 10000;
    };

    const refreshInterval = getRefreshInterval();
    if (refreshInterval) {
      // 立即刷新一次
      fetchTaskStatus(currentTask.task_id);
      
      // 設置定時刷新
      taskInterval = setInterval(() => {
        fetchTaskStatus(currentTask.task_id);
      }, refreshInterval);
    }

    // 如果任務完成，清除當前任務
    if (taskStatus.status === 'completed' || taskStatus.status === 'failed' || taskStatus.status === 'aborted') {
      setTimeout(() => {
        setCurrentTask(null);
        setTaskStatus({});
      }, 2000); // 2秒後清除當前任務顯示
    }

    return () => {
      if (taskInterval) {
        clearInterval(taskInterval);
      }
    };
  }, [currentTask, taskStatus.current_step, taskStatus.status]);

  // 單獨處理任務列表的刷新（降低頻率）
  useEffect(() => {
    // 每30秒刷新一次任務列表
    const tasksInterval = setInterval(() => {
      fetchTestTasks();
    }, 30000);

    return () => {
      clearInterval(tasksInterval);
    };
  }, []);

  // 獲取模型列表
  const fetchModels = async () => {
    setLoading(true);
    try {
      const response = await api.get('/api/models/');
      // 只保留PyTorch格式的模型
      const ptModels = response.data.models.filter(model => model.format === 'pt');
      setModels(ptModels);
    } catch (error) {
      console.error('獲取模型列表失敗:', error);
    } finally {
      setLoading(false);
    }
  };

  // 獲取數據集列表
  const fetchDatasets = async () => {
    setLoading(true);
    try {
      const response = await api.get('/api/benchmark/datasets');
      setDatasets(response.data.datasets || []);
    } catch (error) {
      console.error('獲取數據集列表失敗:', error);
    } finally {
      setLoading(false);
    }
  };

  // 獲取測試任務列表
  const fetchTestTasks = async () => {
    setTasksLoading(true);
    try {
      const response = await api.get('/api/benchmark/tasks');

      // 確保返回的是數組
      const tasks = response.data.tasks || [];
      
      // 按創建時間排序
      tasks.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
      
      setTestTasks(tasks);
    } catch (error) {
      console.error('獲取測試任務列表失敗:', error);
    } finally {
      setTasksLoading(false);
    }
  };
  
  // 獲取單個任務的詳細狀態
  const fetchTaskStatus = async (taskId) => {
    try {
      const response = await api.get(`/api/benchmark/status/${taskId}`);
      const statusData = response.data;
      
      // 檢查狀態是否有變化
      const prevStatus = taskStatus;
      setTaskStatus(statusData);
      
      // 如果有當前處理的組合，獲取其詳細信息（但不在轉換階段時）
      if (statusData.current_combination_index >= 0 && 
          statusData.current_combination && 
          statusData.current_step !== 'conversion') {
        await fetchCombinationDetail(taskId, statusData.current_combination_index);
      }
      
      // 只在狀態從非錯誤變為錯誤時顯示一次錯誤消息
      if (statusData.error && statusData.status === 'processing' && 
          (!prevStatus.error || prevStatus.error !== statusData.error)) {
        message.error(`測試任務發生錯誤: ${statusData.error}`, 5);
      }
      
      return statusData;
    } catch (error) {
      console.error(`獲取任務 ${taskId} 狀態失敗:`, error);
      return null;
    }
  };
      
  // 提交表單創建測試任務
  const handleSubmit = async (values) => {
    // 檢查系統狀態，防止與模型轉換衝突
    if (systemState) {
      // 如果有測試正在運行
      if (systemState.is_testing) {
        message.error(`系統當前正在執行測試 (${systemState.current_step_name || systemState.current_step})，請等待當前測試完成後再創建新的測試任務`);
      return;
    }
    
      // 如果有轉換任務正在運行
      if (systemState.is_converting) {
        message.error(`系統當前有模型轉換任務正在運行 (${systemState.conversion_model_name || '未知模型'})，請等待轉換完成後再創建測試任務`);
      return;
    }
    }
    
    setSubmitting(true);
    try {
      // 準備表單數據
      const formData = new FormData();
      formData.append('model_id', values.model_id);
      formData.append('batch_sizes', values.batch_sizes);
      
      // 處理精度選項
      values.precisions.forEach(precision => {
        formData.append('precisions', precision);
      });
      
      formData.append('image_size', values.image_size);
      formData.append('iterations', values.iterations);
      
      // 數據集ID (必選)
      formData.append('dataset_id', values.dataset_id);
      
      // 模型類型 (新增)
      formData.append('model_type', values.model_type);
      
      // 如果有自定義參數，序列化為JSON字符串
      if (values.custom_params) {
        formData.append('custom_params', values.custom_params);
      }
      
      // 發送請求
      const response = await api.post('/api/benchmark/create', formData);
      
      // 設置當前任務並立即獲取其狀態
      const newTask = {
        task_id: response.data.task_id,
        status: response.data.status,
        total_combinations: response.data.total_combinations
      };
      
      setCurrentTask(newTask);
      fetchTaskStatus(newTask.task_id);
      
      // 刷新任務列表和系統狀態
      fetchTestTasks();
      fetchSystemState();
      
      // 顯示成功消息
      message.success('測試任務已創建，即將開始處理');
    } catch (error) {
      console.error('創建測試任務失敗:', error);
      message.error(`創建測試任務失敗: ${error.response?.data?.detail || error.message}`);
    } finally {
      setSubmitting(false);
    }
  };
      
  // 中止測試任務
  const abortTask = async (taskId) => {
    try {
      await api.post(`/api/benchmark/abort/${taskId}`);
      
      // 刷新任務列表和當前任務狀態
      fetchTestTasks();
      
      if (currentTask && currentTask.task_id === taskId) {
        fetchTaskStatus(taskId);
      }
      
      message.success('測試任務已中止');
    } catch (error) {
      console.error('中止測試任務失敗:', error);
      message.error(`中止測試任務失敗: ${error.response?.data?.detail || error.message}`);
    }
  };
  
  // 刪除測試任務
  const deleteTask = async (taskId) => {
    try {
      await api.delete(`/api/benchmark/task/${taskId}`);
      
      // 刷新任務列表
      fetchTestTasks();
      
      // 如果刪除的是當前任務，清除當前任務
      if (currentTask && currentTask.task_id === taskId) {
        setCurrentTask(null);
        setTaskStatus({});
      }
      
      message.success('測試任務已刪除');
      } catch (error) {
      console.error('刪除測試任務失敗:', error);
      message.error(`刪除測試任務失敗: ${error.response?.data?.detail || error.message}`);
    }
  };
  
  // 查看測試結果
  const viewResults = (taskId) => {
    // 跳轉到測試結果查看頁面
    window.open(`/test-results/${taskId}`, '_blank');
  };
  
  // 上傳數據集
  const uploadDataset = async () => {
    if (!datasetFile) {
      message.warning('請選擇要上傳的數據集文件');
        return;
    }
    
    setUploading(true);
    setUploadProgress(0);
    
    const formData = new FormData();
    formData.append('file', datasetFile);
    formData.append('dataset_type', datasetType);
    
    try {
      const response = await api.post('/api/benchmark/datasets/upload', formData, {
          onUploadProgress: (progressEvent) => {
            const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            setUploadProgress(percentCompleted);
          }
      });
      
      // 上傳成功
      message.success('數據集上傳成功');
      setUploadVisible(false);
      setDatasetFile(null);
      fetchDatasets();
    } catch (error) {
      console.error('上傳數據集失敗:', error);
      message.error(`上傳數據集失敗: ${error.response?.data?.detail || error.message}`);
    } finally {
      setUploading(false);
    }
  };

  // 刪除數據集
  const deleteDataset = async (datasetId) => {
    try {
      await api.delete(`/api/benchmark/datasets/${datasetId}`);
      message.success('數據集已刪除');
      fetchDatasets();
    } catch (error) {
      console.error('刪除數據集失敗:', error);
      message.error(`刪除數據集失敗: ${error.response?.data?.detail || error.message}`);
    }
  };

  // 獲取狀態標籤顏色
  // getStatusColor / formatDateTime / stepNameMap / statusNameMap 移至 utils/benchmarkFormat.js（已加 jest 測試）

  // 顯示任務錯誤詳情
  const showTaskErrorDetail = async (task) => {
    try {
      const response = await api.get(`/api/benchmark/task/${task.task_id}`);
      const taskDetail = response.data;
      
      // 收集所有錯誤的組合
      const failedCombinations = [];
      taskDetail.combinations.forEach((combo, index) => {
        const errors = [];
        
        if (combo.status === 'failed' && combo.error) {
          errors.push({ stage: '轉換', error: combo.error });
        }
        if (combo.validation_status === 'failed' && combo.validation_error) {
          errors.push({ stage: '驗證', error: combo.validation_error });
        }
        if (combo.inference_status === 'failed' && combo.inference_error) {
          errors.push({ stage: '推論測試', error: combo.inference_error });
        }
        
        if (errors.length > 0) {
          failedCombinations.push({
            index: index + 1,
            batch_size: combo.batch_size,
            precision: combo.precision,
            errors: errors
          });
        }
      });
      
      // 顯示錯誤詳情模態框
      Modal.info({
        title: '任務錯誤詳情 - ' + task.model_name,
        width: 800,
        content: (
          <div>
            {taskDetail.error && (
              <Alert 
                type="error" 
                message="任務級錯誤" 
                description={taskDetail.error}
                style={{ marginBottom: 16 }}
              />
            )}
            
            {failedCombinations.length > 0 ? (
              <div>
                <h4>失敗的組合：</h4>
                {failedCombinations.map((combo, idx) => (
                  <Card 
                    key={idx}
                    size="small"
                    title={'組合 ' + combo.index + ': 批次=' + combo.batch_size + ', 精度=' + combo.precision}
                    style={{ marginBottom: 8 }}
                  >
                    {combo.errors.map((err, errIdx) => (
                      <Alert
                        key={errIdx}
                        type="error"
                        message={err.stage + '階段錯誤'}
                        description={err.error}
                        style={{ marginBottom: errIdx < combo.errors.length - 1 ? 8 : 0 }}
                      />
                    ))}
                  </Card>
                ))}
              </div>
            ) : (
              <Empty description="沒有找到具體的錯誤信息" />
            )}
            
            <div style={{ marginTop: 16 }}>
              <Text type="secondary">
                任務ID: {taskDetail.id}<br/>
                創建時間: {formatDateTime(taskDetail.created_at)}<br/>
                總組合數: {taskDetail.total_combinations}<br/>
                完成組合數: {taskDetail.completed_combinations}
              </Text>
            </div>
          </div>
        ),
        okText: '確定'
      });
    } catch (error) {
      console.error('獲取任務詳情失敗:', error);
      message.error('獲取任務詳情失敗');
    }
  };

  // 任務列表表格列定義
  const columns = [
    {
      title: '任務ID',
      dataIndex: 'task_id',
      key: 'task_id',
      ellipsis: true,
      width: 100
    },
    {
      title: '模型',
      dataIndex: 'model_name',
      key: 'model_name'
    },
    {
      title: '創建時間',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (text) => formatDateTime(text)
    },
    {
      title: '完成時間',
      dataIndex: 'completed_at',
      key: 'completed_at',
      render: (text) => text ? formatDateTime(text) : '-'
    },
    {
      title: '狀態',
      dataIndex: 'status',
      key: 'status',
      render: (status, record) => (
        <div>
          {/* 如果有錯誤，顯示失敗狀態 */}
          {record.error ? (
            <Tag color="red">
              失敗
            </Tag>
          ) : status === 'processing' ? 
            <Tag color={getStatusColor(status)} icon={<SyncOutlined spin />}>
              {statusNameMap[status] || status}
            </Tag> :
            <Tag color={getStatusColor(status)}>
              {statusNameMap[status] || status}
            </Tag>
          }
          {/* 如果有錯誤，顯示錯誤圖標 */}
          {record.error && (
            <Tooltip title="任務執行過程中有錯誤，點擊查看詳情">
              <Button 
                type="link" 
                size="small"
                icon={<QuestionCircleOutlined />}
                style={{ color: '#ff4d4f', marginLeft: 4 }}
                onClick={() => showTaskErrorDetail(record)}
              />
            </Tooltip>
          )}
        </div>
      )
    },
    {
      title: '當前階段',
      dataIndex: 'current_step',
      key: 'current_step',
      render: (step, record) => {
        // 如果是當前正在運行的任務，使用taskStatus中的階段信息
        if (currentTask && currentTask.task_id === record.task_id && taskStatus.current_step) {
          step = taskStatus.current_step;
        }
        
        if (record.status === 'completed') {
          return '已完成';
        } else if (record.status === 'failed') {
          return '失敗';
        } else if (record.status === 'aborted') {
          return '已中止';
        }
        return stepNameMap[step] || step || '未知';
      }
    },
    {
      title: '進度',
              key: 'progress',
        render: (_, record) => {
         // 使用統一的進度計算函數
         const progressData = calculateActualProgress(record);

                  // 如果有成功/失敗統計，顯示詳細信息
          if (record.partial_success || record.success_count !== undefined) {
            return (
              <Tooltip title={'成功: ' + (record.success_count || 0) + ', 失敗: ' + (record.fail_count || 0)}>
                <span>
                 {progressData.stageProgress || `${progressData.completed}/${progressData.total}`}
                 {record.partial_success && (
                   <Tag color="orange" style={{ marginLeft: 8 }}>部分成功</Tag>
                 )}
                </span>
              </Tooltip>
            );
          }
         return <span>{progressData.stageProgress || `${progressData.completed}/${progressData.total}`}</span>;
      }
    },
    {
      title: '操作',
      key: 'action',
      width: 200,
      render: (_, record) => (
        <Space wrap>
          {record.status === 'completed' || record.error ? (
            <>
              <Button 
                type="link" 
                icon={<AreaChartOutlined />}
                onClick={() => window.open('/test-results/' + record.task_id, '_blank')}
              >
                查看結果
              </Button>
              <Button 
                type="link" 
                icon={<DownloadOutlined />}
                onClick={() => window.open(API_BASE_URL + '/api/benchmark/download-results/' + record.task_id)}
              >
                下載完整結果
              </Button>
            </>
          ) : record.status === 'processing' || record.status === 'pending' ? (
            <Popconfirm
              title="確定要中止此測試任務嗎？"
              onConfirm={() => abortTask(record.task_id)}
              okText="確定"
              cancelText="取消"
            >
              <Button type="link" danger>
                中止
              </Button>
            </Popconfirm>
          ) : null}
          
          {/* 查看當前組合的詳細結果 */}
          {record.current_combination_index >= 0 && 
           record.status === 'processing' && (
            <ViewCombinationDetailButton 
              taskId={record.task_id} 
              combinationIndex={record.current_combination_index} 
            />
          )}
          
          {record.status !== 'processing' && record.status !== 'pending' && (
            <Popconfirm
              title="確定要刪除此測試任務嗎？"
              onConfirm={() => deleteTask(record.task_id)}
              okText="確定"
              cancelText="取消"
            >
              <Button type="link" danger>
                <DeleteOutlined />
              </Button>
            </Popconfirm>
          )}
          
          {/* 使當前任務高亮 */}
          {currentTask && currentTask.task_id === record.task_id && (
            <Button type="link" disabled>
              <SyncOutlined spin />
            </Button>
          )}
        </Space>
      )
    }
  ];

  // 數據集上傳相關屬性
  const uploadProps = {
    name: 'file',
    multiple: false,
    beforeUpload: (file) => {
      // 檢查文件類型
      if (file.type !== 'application/zip' && !file.name.endsWith('.zip')) {
        message.error('只能上傳ZIP檔案!');
        return false;
      }
      
      // 保存文件
      setDatasetFile(file);
      return false;
    },
    onRemove: () => {
      setDatasetFile(null);
    },
    fileList: datasetFile ? [datasetFile] : []
  };

  // 獲取系統狀態
  const fetchSystemState = async () => {
    setSystemStateLoading(true);
    try {
      const response = await api.get('/api/benchmark/system-state');
      setSystemState(response.data);
    } catch (error) {
      console.error('獲取系統狀態失敗:', error);
    } finally {
      setSystemStateLoading(false);
    }
  };

  // 新增：檢查是否有活動中的任務
  const checkActiveTask = async () => {
    try {
      // 獲取系統狀態
      const stateResponse = await api.get('/api/benchmark/system-state');
      const systemState = stateResponse.data;
      
      // 如果有活動中的測試任務
      if (systemState.is_testing && systemState.active_task_id) {
        // 設置當前任務
        setCurrentTask({
          task_id: systemState.active_task_id,
          status: 'processing',
          total_combinations: systemState.total_combinations
        });
        
        // 獲取任務詳細狀態
        fetchTaskStatus(systemState.active_task_id);
      }
    } catch (error) {
      console.error('檢查活動任務失敗:', error);
    }
  };

  // 獲取組合詳細信息
  const fetchCombinationDetail = async (taskId, combinationIndex) => {
    const key = `${taskId}_${combinationIndex}`;
    
    // 設置加載狀態
    setLoadingCombinations(prev => ({
      ...prev,
      [key]: true
    }));
    
    try {
      const response = await api.get(`/api/benchmark/combination/${taskId}/${combinationIndex}`);
      
      // 更新組合詳細信息
      setCombinationDetails(prev => ({
        ...prev,
        [key]: response.data
      }));
      
      return response.data;
    } catch (error) {
      console.error(`獲取組合詳細信息失敗: ${taskId}, ${combinationIndex}`, error);
      message.error(`獲取組合詳細信息失敗: ${error.response?.data?.detail || error.message}`);
      return null;
    } finally {
      setLoadingCombinations(prev => ({
        ...prev,
        [key]: false
      }));
    }
  };

  // 查看組合詳細結果按鈕
  const ViewCombinationDetailButton = ({ taskId, combinationIndex }) => {
    const key = `${taskId}_${combinationIndex}`;
    const isLoading = loadingCombinations[key];
    const hasDetail = combinationDetails[key];
    
    const handleClick = async () => {
      await fetchCombinationDetail(taskId, combinationIndex);
      
      // 創建一個模態框展示詳細結果
      const detail = combinationDetails[key];
      if (!detail) return;
      
      Modal.info({
        title: '組合詳細結果 (批次: ' + detail.batch_size + ', 精度: ' + detail.precision + ')',
        width: 800,
        content: (
          <div>
            {/* 驗證結果 */}
            {detail.validation.status === 'completed' && detail.validation.results && (
              <Card size="small" title="模型驗證結果" style={{ marginBottom: 10 }}>
                {detail.validation.results.metrics ? (
                  <Row gutter={16}>
                    <Col span={6}>
                      <Statistic title="精確度 (Precision)" value={Number(detail.validation.results.metrics.precision || 0).toFixed(4)} />
                    </Col>
                    <Col span={6}>
                      <Statistic title="召回率 (Recall)" value={Number(detail.validation.results.metrics.recall || 0).toFixed(4)} />
                    </Col>
                    <Col span={6}>
                      <Statistic title="mAP@0.5" value={Number(detail.validation.results.metrics.mAP50 || 0).toFixed(4)} />
                    </Col>
                    <Col span={6}>
                      <Statistic title="mAP@0.5:0.95" value={Number(detail.validation.results.metrics.mAP50_95 || 0).toFixed(4)} />
                    </Col>
                  </Row>
                ) : (
                  <Empty description="無可用的驗證指標數據" />
                )}
              </Card>
            )}
            
            {/* 驗證錯誤 */}
            {detail.validation.status === 'failed' && detail.validation.error && (
              <Alert 
                type="error" 
                message="模型驗證失敗" 
                description={detail.validation.error}
                style={{ marginBottom: 10 }}
                showIcon
              />
            )}
            
            {/* 推論測試結果 */}
            {detail.inference.status === 'completed' && detail.inference.results && (
              <Card size="small" title="推論測試結果" style={{ marginBottom: 10 }}>
                {detail.inference.results.avg_inference_time_ms ? (
                  <Row gutter={16}>
                    <Col span={6}>
                      <Statistic 
                        title="平均推理時間" 
                        value={Number(detail.inference.results.avg_inference_time_ms || 0).toFixed(2)} 
                        suffix="ms" 
                      />
                    </Col>
                    <Col span={6}>
                      <Statistic 
                        title="吞吐量" 
                        value={Number(detail.inference.results.throughput_fps || 0).toFixed(2)} 
                        suffix="FPS" 
                      />
                    </Col>
                    <Col span={6}>
                      <Statistic 
                        title="內存使用" 
                        value={Number(detail.inference.results.memory_usage_mb || 0).toFixed(2)} 
                        suffix="MB" 
                      />
                    </Col>
                    <Col span={6}>
                      <Statistic 
                        title="GPU使用率" 
                        value={Number(detail.inference.results.avg_gpu_load || 0).toFixed(2)} 
                        suffix="%" 
                      />
                    </Col>
                  </Row>
                ) : (
                  <Empty description="無可用的推論測試數據" />
                )}
              </Card>
            )}
            
            {/* 推論測試錯誤 */}
            {detail.inference.status === 'failed' && detail.inference.error && (
              <Alert 
                type="error" 
                message="推論測試失敗" 
                description={detail.inference.error} 
                style={{ marginBottom: 10 }}
                showIcon
              />
            )}
          </div>
        ),
        onOk() {},
      });
    };

    return (
      <Button 
        type="link" 
        onClick={handleClick} 
        loading={isLoading}
        icon={<AreaChartOutlined />}
      >
        {hasDetail ? '查看詳細結果' : '載入詳細結果'}
      </Button>
    );
  };

  return (
    <div className="benchmark-page">
      {/* 當前任務狀態卡片 */}
      {currentTask && taskStatus.task_id && (
        <Card 
          title={
            <div>
              <SyncOutlined spin style={{ marginRight: 8 }} />
              測試任務正在進行中 - {stepNameMap[taskStatus.current_step] || taskStatus.current_step || '準備中'}
            </div>
          } 
          style={{ marginBottom: 16 }}
          type="inner"
          extra={
            taskStatus.status === 'processing' && (
              <Popconfirm
                title="確定要中止此測試任務嗎？"
                onConfirm={() => abortTask(taskStatus.task_id)}
                okText="確定"
                cancelText="取消"
              >
                <Button type="primary" danger>
                  中止任務
            </Button>
              </Popconfirm>
            )
          }
        >
          <Row gutter={16}>
            <Col span={8}>
              <Statistic 
                title="任務ID" 
                value={taskStatus.task_id} 
                valueStyle={{ fontSize: '14px' }} 
              />
            </Col>
            <Col span={4}>
              <Statistic 
                title="當前階段" 
                value={stepNameMap[taskStatus.current_step] || taskStatus.current_step} 
                valueStyle={{ fontSize: '14px', color: taskStatus.status === 'processing' ? '#1890ff' : 'inherit' }} 
              />
            </Col>
            <Col span={4}>
              <Statistic 
                title="狀態" 
                value={statusNameMap[taskStatus.status] || taskStatus.status} 
                valueStyle={{ 
                  fontSize: '14px',
                  color: taskStatus.status === 'processing' ? '#1890ff' :
                          taskStatus.status === 'completed' ? '#52c41a' :
                          taskStatus.status === 'failed' ? '#f5222d' : 'inherit'
                }} 
              />
            </Col>
            <Col span={8}>
              <Statistic 
                title="進度" 
                value={progressData.stageProgress || `${progressData.completed}/${progressData.total}`}
                valueStyle={{ fontSize: '14px' }} 
                suffix={
                  <Progress 
                    percent={progressData.percent} 
                    size="small" 
                    status={
                      taskStatus.status === 'processing' ? 'active' :
                      taskStatus.status === 'completed' ? 'success' :
                      taskStatus.status === 'failed' ? 'exception' : 'normal'
                    }
                  />
                }
              />
            </Col>
          </Row>
          
          {taskStatus.current_combination && (
            <div>
          <Alert
                style={{ marginTop: 16 }}
            type="info"
                message="當前處理組合"
                description={
                  <div>
                    <p>批次大小: {taskStatus.current_combination.batch_size}</p>
                    <p>精度: {taskStatus.current_combination.precision}</p>
                    <p>狀態: {statusNameMap[taskStatus.current_combination.status] || taskStatus.current_combination.status}</p>
                    <p>進度: 第 {taskStatus.current_combination_index + 1} 個組合，共 {taskStatus.total_combinations} 個</p>
                  </div>
                }
            showIcon
              />
              
              {/* 動態顯示組合詳細結果 */}
              {taskStatus.current_combination_index >= 0 && (
                <div style={{ marginTop: 16 }}>
                  {/* 實時測試結果部分已移除 */}
                </div>
              )}
            </div>
          )}
          
          {taskStatus.error && (
              <Alert
              style={{ marginTop: 16 }}
              type="error"
              message="錯誤信息"
              description={taskStatus.error}
                showIcon
              />
          )}
        </Card>
      )}

      {/* 創建測試任務表單 */}
      <Card title="創建自動化模型性能評估任務" style={{ marginBottom: 16 }}>
        <Form
          form={form}
          layout="vertical"
          onFinish={handleSubmit}
          initialValues={{
            image_size: 640,
            iterations: 10,  // 與後端 MAX_BENCHMARK_ITERATIONS 預設上限一致
            precisions: ['fp32'],
            model_type: 'object'  // 預設為物體檢測
          }}
        >
          <Form.Item
            name="model_id"
            label="選擇模型"
            rules={[{ required: true, message: '請選擇要測試的模型' }]}
          >
            <Select
              placeholder="選擇要測試的模型"
              loading={loading}
              showSearch
              optionFilterProp="children"
            >
              {models.map(model => (
                <Option key={model.id} value={model.id}>
                  {model.name} ({model.format}) 
                </Option>
              ))}
            </Select>
          </Form.Item>

          <Form.Item
            name="model_type"
            label="模型類型"
            rules={[{ required: true, message: '請選擇模型類型' }]}
            tooltip="選擇模型類型將決定使用哪種YAML配置進行驗證"
          >
            <Radio.Group>
              <Radio.Button value="object">物體檢測</Radio.Button>
              <Radio.Button value="pose">姿態估計</Radio.Button>
            </Radio.Group>
          </Form.Item>

          <Form.Item
            name="precisions"
            label="精度選項"
            rules={[{ required: true, message: '請選擇至少一個精度選項' }]}
          >
            <Select 
              mode="multiple" 
              placeholder="選擇精度選項"
            >
              <Option value="fp32">FP32</Option>
              <Option value="fp16">FP16</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="batch_sizes"
            label="批次大小列表 (以逗號分隔)"
            rules={[{ required: true, message: '請輸入批次大小列表' }]}
            tooltip="輸入以逗號分隔的批次大小列表，例如: 1,2,4,8"
          >
            <Input placeholder="例如: 1,2,4,8" />
          </Form.Item>

          <Form.Item
            name="image_size"
            label="圖像尺寸"
            rules={[{ required: true, message: '請選擇圖像尺寸' }]}
          >
            <Select placeholder="選擇圖像尺寸">
              <Option value={512}>512</Option>
              <Option value={640}>640</Option>
              <Option value={768}>768</Option>
              <Option value={1024}>1024</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="iterations"
            label="迭代次數"
            rules={[{ required: true, message: '請選擇迭代次數' }]}
            tooltip="每個「精度 × 批次大小」組態重複量測的次數"
            extra="後端有單組態上限（環境變數 MAX_BENCHMARK_ITERATIONS，預設 10 次）；超過上限時只會執行到上限次數，結果檔的 iterations 欄位記錄實際執行次數。"
          >
            <Select placeholder="選擇迭代次數">
              <Option value={10}>10</Option>
              <Option value={50}>50</Option>
              <Option value={100}>100</Option>
              <Option value={500}>500</Option>
              <Option value={1000}>1000</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="dataset_id"
            label="驗證數據集"
            rules={[{ required: true, message: '請選擇驗證數據集' }]}
            tooltip="選擇數據集用於模型驗證，將測量mAP等精度指標"
          >
            <Select 
              placeholder="選擇數據集" 
              showSearch
              optionFilterProp="children"
              loading={loading}
              notFoundContent={datasets.length === 0 ? "無可用數據集，請上傳新數據集" : undefined}
              popupRender={menu => (
                <div>
                  {menu}
                  <Divider style={{ margin: '4px 0' }} />
                  <div style={{ padding: '4px 8px' }}>
                    <Button 
                      type="link" 
                      size="small" 
                      onClick={() => setUploadVisible(true)}
                    >
                      <UploadOutlined /> 上傳新數據集
                    </Button>
                  </div>
                </div>
              )}
            >
              {datasets.map(dataset => (
                <Option key={dataset.id} value={dataset.id}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span>
                    {dataset.name}
                      {dataset.is_coco && <Tag color="green" style={{ marginLeft: 8 }}>COCO</Tag>}
                      {dataset.type === "object" && <Tag color="blue" style={{ marginLeft: 8 }}>物體</Tag>}
                      {dataset.type === "pose" && <Tag color="purple" style={{ marginLeft: 8 }}>姿態</Tag>}
                    </span>
                    <Popconfirm
                      title="確定要刪除此數據集嗎？"
                      onConfirm={(e) => {
                        e.stopPropagation();
                        deleteDataset(dataset.id);
                      }}
                      onCancel={(e) => e.stopPropagation()}
                      okText="確定"
                      cancelText="取消"
                    >
                      <Button
                        type="link"
                        danger
                        icon={<DeleteOutlined />}
                        size="small"
                        onClick={(e) => e.stopPropagation()}
                      />
                    </Popconfirm>
                  </div>
                </Option>
              ))}
            </Select>
          </Form.Item>

                <Form.Item
                  name="custom_params"
            label="自定義參數 (JSON格式，可選)"
            tooltip="輸入JSON格式的自定義參數，例如: {'conf': 0.25, 'iou': 0.45}"
          >
            <Input.TextArea placeholder="{'conf': 0.25, 'iou': 0.45}" rows={3} />
                </Form.Item>

          <Form.Item>
            <Button 
              type="primary" 
              htmlType="submit" 
              loading={submitting} 
              disabled={
                // 如果有當前正在進行的任務，禁用提交按鈕
                (currentTask && 
                 taskStatus.status && 
                 (taskStatus.status === 'processing' || taskStatus.status === 'pending')) ||
                // 如果系統有測試任務正在進行
                (systemState && systemState.is_testing) ||
                // 如果系統有轉換任務正在進行
                (systemState && systemState.is_converting)
              }
            >
              開始測試
            </Button>
            
            {/* 如果有當前任務且正在進行中，顯示中止按鈕 */}
            {currentTask && 
             taskStatus.status && 
             (taskStatus.status === 'processing' || taskStatus.status === 'pending') && (
              <Popconfirm
                title="確定要中止當前測試任務嗎？"
                onConfirm={() => abortTask(currentTask.task_id)}
                okText="確定"
                cancelText="取消"
              >
                <Button danger style={{ marginLeft: 8 }}>
                  中止當前任務
              </Button>
              </Popconfirm>
            )}
            
            {/* 顯示系統狀態警告 */}
            {systemState && systemState.is_testing && systemState.active_task_id !== (currentTask && currentTask.task_id) && (
              <span style={{ marginLeft: 8, color: '#ff4d4f' }}>
                系統當前正在執行其他測試任務 ({systemState.current_step_name || systemState.current_step})
              </span>
            )}
            
            {/* 顯示轉換任務警告 */}
            {systemState && systemState.is_converting && (
              <span style={{ marginLeft: 8, color: '#ff4d4f' }}>
                系統當前有模型轉換任務正在運行
              </span>
            )}
          </Form.Item>
        </Form>
      </Card>

      {/* 測試任務列表 */}
      <Card 
        title={
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span>測試任務列表</span>
        <Button 
          type="primary" 
          icon={<SyncOutlined />} 
          onClick={fetchTestTasks}
              loading={tasksLoading}
              size="small"
        >
              刷新
        </Button>
      </div>
        }
      >
      <Table 
        dataSource={testTasks} 
          columns={columns}
        rowKey="task_id"
          loading={tasksLoading}
          pagination={{ pageSize: 10 }}
        />
      </Card>
      
      {/* 數據集上傳模態框 */}
      <Modal
        title="上傳數據集"
        open={uploadVisible}
        onCancel={() => {
          setUploadVisible(false);
          setDatasetFile(null);
        }}
        footer={[
          <Button key="back" onClick={() => {
            setUploadVisible(false);
            setDatasetFile(null);
          }}>
            取消
          </Button>,
          <Button 
            key="submit" 
            type="primary" 
            loading={uploading} 
            onClick={uploadDataset}
            disabled={!datasetFile}
          >
            上傳
          </Button>
        ]}
      >
        <p>請上傳ZIP格式的數據集文件，系統將自動處理並生成YAML配置文件</p>
        
        <Form layout="vertical" style={{ marginBottom: 16 }}>
          <Form.Item 
            label="數據集類型" 
            tooltip="選擇數據集用於何種模型類型的驗證"
          >
            <Radio.Group 
              defaultValue="object" 
              onChange={(e) => setDatasetType(e.target.value)}
            >
              <Radio value="object">物體檢測</Radio>
              <Radio value="pose">姿態估計</Radio>
            </Radio.Group>
          </Form.Item>
        </Form>
        
        <Dragger {...uploadProps}>
          <p className="ant-upload-drag-icon">
            <InboxOutlined />
          </p>
          <p className="ant-upload-text">點擊或拖拽文件到此區域上傳</p>
          <p className="ant-upload-hint">僅支持ZIP格式的數據集文件</p>
        </Dragger>
        
        {uploading && (
          <div style={{ marginTop: 16 }}>
            <Progress percent={uploadProgress} status="active" />
          </div>
        )}
      </Modal>
    </div>
  );
};

export default BenchmarkPage; 