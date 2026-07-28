import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, Form, Select, Button, Table, Progress, Tag, message, Alert, Tooltip, Input } from 'antd';
import { SyncOutlined, QuestionCircleOutlined } from '@ant-design/icons';
import api from '../api/client';

const { Option } = Select;

const ConversionPage = () => {
  const navigate = useNavigate();
  const [models, setModels] = useState([]);
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [jobsLoading, setJobsLoading] = useState(false);
  const [form] = Form.useForm();
  const [modelMap, setModelMap] = useState({});  // 用於保存ID到模型名稱的映射
  const [prevJobs, setPrevJobs] = useState([]);
  const [existingModelWarning, setExistingModelWarning] = useState(null);
  const [systemState, setSystemState] = useState(null);
  const [systemStateLoading, setSystemStateLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [duplicateModelWarning, setDuplicateModelWarning] = useState(null);
  const [selectedModelFormat, setSelectedModelFormat] = useState(null);  // 新增狀態追踪選中的模型格式

  useEffect(() => {
    fetchModels();
    fetchJobs();
    fetchSystemState();
    
    // 修改為條件性輪詢：只在有處理中任務時才高頻率輪詢
    const checkInterval = () => {
      // 檢查是否有處理中任務
      fetchJobs().then(hasProcessingJobs => {
        const interval = hasProcessingJobs ? 5000 : 15000; // 有處理中任務時5秒刷新，否則15秒
        console.log(`設置下次更新間隔: ${interval}ms`);
        setTimeout(checkInterval, interval);
      });
    };
    
    // 啟動輪詢
    const timeoutId = setTimeout(checkInterval, 5000);
    return () => clearTimeout(timeoutId);
  }, []);

  // 檢查任務變化，刷新模型列表
  useEffect(() => {
    // 檢查是否有任務從processing變為completed
    const completedNewJobs = jobs.filter(
      job => job.status === 'completed' && 
      prevJobs.some(prevJob => prevJob.id === job.id && prevJob.status === 'processing')
    );
    
    if (completedNewJobs.length > 0) {
      console.log('有任務完成，刷新模型列表');
      refreshModelRepository();
    }
    
    // 更新前一個任務列表
    setPrevJobs(jobs);
  }, [jobs]);

  const fetchModels = async () => {
    setLoading(true);
    try {
      const response = await api.get('/api/models/');
      setModels(response.data.models);
      
      // 創建模型ID到模型名稱的映射
      const idToNameMap = {};
      response.data.models.forEach(model => {
        idToNameMap[model.id] = model.name;
      });
      setModelMap(idToNameMap);
      
    } catch (error) {
      console.error('獲取模型失敗:', error);
      message.error('獲取模型列表失敗');
    } finally {
      setLoading(false);
    }
  };

  const fetchJobs = async () => {
    setJobsLoading(true);
    try {
      const response = await api.get('/api/conversion/');
      setJobs(response.data.jobs);
      
      // 檢查是否有處理中的任務
      const hasProcessingJobs = response.data.jobs.some(job => job.status === 'processing');
      return hasProcessingJobs;
    } catch (error) {
      console.error('獲取轉換任務失敗:', error);
      return false;
    } finally {
      setJobsLoading(false);
    }
  };

  // 刷新模型存儲庫
  const refreshModelRepository = async () => {
    try {
      console.log('開始刷新模型庫...');
      const response = await api.get('/api/models/refresh');
      console.log('模型庫刷新成功，返回：', response.data);
      // 刷新模型列表
      fetchModels();
    } catch (error) {
      console.error('刷新模型庫失敗:', error.response || error);
      // 即使刷新失敗，仍然嘗試獲取最新的模型列表
      fetchModels();
    }
  };

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

  const checkExistingModel = async (values) => {
    try {
      // 獲取選擇的源模型
      const sourceModel = models.find(model => model.id === values.model_id);
      if (!sourceModel) return false;

      // 建構潛在的目標模型名稱 (與後端邏輯相同)
      const safeName = sourceModel.name.replace(/[ /\\]/g, "_");
      const batchSize = parseInt(values.batch_size);
      const imgSize = parseInt(values.image_size);
      
      // 使用源模型ID的前8位
      const sourceIdShort = sourceModel.id.substring(0, 8);
      const paramSuffix = `_${values.target_format}_${values.precision}`;
      const batchSizeSuffix = `_batch${batchSize}`;
      const imgSizeSuffix = `_size${imgSize}`;
      const targetModelName = `${safeName}_${sourceIdShort}${paramSuffix}${batchSizeSuffix}${imgSizeSuffix}`;

      
      // 檢查是否存在此名稱的模型
      const existingModel = models.find(model => 
        model.name === targetModelName || 
        (model.metadata && model.metadata.triton_model_dir === `model_repository/${targetModelName}`)
      );
      
      if (existingModel) {
        setExistingModelWarning({
          modelId: existingModel.id,
          modelName: existingModel.name
        });
        return true;
      }
      
      // 沒有找到相同配置的模型
      setExistingModelWarning(null);
      return false;
    } catch (error) {
      console.error('檢查已存在模型時出錯:', error);
      return false;
    }
  };

  const handleSubmit = async (values) => {
    // 檢查系統是否在執行測試
    if (systemState && systemState.is_testing) {
      message.error('系統當前正在執行自動化模型性能評估，請等待評估完成後再創建轉換任務');
      return;
    }

    // 檢查是否有重複的模型
    const isDuplicate = await checkExistingModel(values);
    if (isDuplicate) {
      const confirm = window.confirm('已存在相同參數的模型，是否仍要繼續創建轉換任務？');
      if (!confirm) return;
    }

    setSubmitting(true);
    try {
      // 獲取選擇的模型詳細信息
      const selectedModel = models.find(model => model.id === values.model_id);
      if (!selectedModel) {
        message.error('找不到選擇的模型');
        return;
      }
      
      console.log('選擇的模型:', selectedModel);
      
      const response = await api.post('/api/conversion/', {
        source_model_id: values.model_id,
        target_format: values.target_format,
        precision: values.precision,
        parameters: {
          batch_size: parseInt(values.batch_size),
          imgsz: parseInt(values.image_size),
          workspace: parseInt(values.workspace),
          model_name: selectedModel.name  // 添加模型名稱作為額外參數
        }
      });
      
      console.log('轉換響應:', response.data);
      message.success('創建轉換任務成功');
      fetchJobs();
      form.resetFields();
      setSelectedModelFormat(null);  // 重置選中的模型格式
    } catch (error) {
      console.error('創建轉換任務失敗:', error);
      if (error.response && error.response.data && error.response.data.detail) {
        message.error(`創建轉換任務失敗: ${error.response.data.detail}`);
      } else {
        message.error('創建轉換任務失敗');
      }
    } finally {
      setSubmitting(false);
    }
  };

  const handleValuesChange = async (changedValues, allValues) => {
    // 追踪選中模型的格式
    if (changedValues.model_id) {
      const selectedModel = models.find(model => model.id === changedValues.model_id);
      if (selectedModel) {
        setSelectedModelFormat(selectedModel.format);
        // 如果選擇的是 ONNX 模型，自動設定目標格式為 engine
        if (selectedModel.format === 'onnx') {
          form.setFieldsValue({
            target_format: 'engine'
          });
        }
      }
    }
    
    // 只有當所有必要字段都有值時才進行檢查
    if (allValues.model_id && 
        allValues.target_format && 
        allValues.precision && 
        allValues.batch_size && 
        allValues.image_size) {
      await checkExistingModel(allValues);
    }
  };

  const getStatusTag = (status) => {
    switch (status) {
      case 'pending':
        return <Tag color="blue">等待中</Tag>;
      case 'processing':
        return <Tag color="orange" icon={<SyncOutlined spin />}>處理中</Tag>;
      case 'completed':
        return <Tag color="green">已完成</Tag>;
      case 'failed':
        return <Tag color="red">失敗</Tag>;
      default:
        return <Tag>{status}</Tag>;
    }
  };

  const columns = [
    {
      title: '任務ID',
      dataIndex: 'id',
      key: 'id',
      width: 220,
    },
    {
      title: '原始模型',
      dataIndex: 'source_model_id',
      key: 'source_model_id',
      render: (modelId) => modelMap[modelId] || modelId,
    },
    {
      title: '目標格式',
      dataIndex: 'target_format',
      key: 'target_format',
    },
    {
      title: '精度',
      dataIndex: 'precision',
      key: 'precision',
    },
    {
      title: '批次大小',
      dataIndex: 'parameters',
      key: 'batch_size',
      render: (params) => params?.batch_size || '1',
    },
    {
      title: '圖像尺寸',
      dataIndex: 'parameters',
      key: 'image_size',
      render: (params) => params?.imgsz || '640',
    },
    {
      title: '工作區大小',
      dataIndex: 'parameters',
      key: 'workspace',
      render: (params) => (params?.workspace ? `${params.workspace} GB` : '4 GB'),
    },
    {
      title: '狀態',
      dataIndex: 'status',
      key: 'status',
      render: (status, record) => {
        if (status === 'failed' && record.error_message) {
          return (
            <div>
              {getStatusTag(status)}
              <div style={{ color: 'red', fontSize: '12px' }}>
                {record.error_message.length > 30 
                  ? record.error_message.substring(0, 30) + '...' 
                  : record.error_message}
              </div>
            </div>
          );
        }
        return getStatusTag(status);
      },
    },
    {
      title: '創建時間',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (date) => new Date(date).toLocaleString(),
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record) => (
        <>
          <Button 
            type="link" 
            onClick={() => navigate(`/conversion/${record.id}`)}
            style={{ marginRight: 8 }}
          >
          詳情
        </Button>
          {record.status !== 'processing' && (
            <Button 
              type="link" 
              danger
              onClick={() => handleDeleteJob(record.id)}
            >
              刪除
            </Button>
          )}
        </>
      ),
    },
  ];

  // 刪除轉換任務
  const handleDeleteJob = async (jobId) => {
    const confirm = window.confirm('確定要刪除這個轉換任務嗎？如果任務已完成，相關的轉換模型也會被刪除。');
    if (!confirm) return;
    
    try {
      const response = await api.delete(`/api/conversion/${jobId}`);
      message.success('轉換任務已成功刪除');
      
      // 刷新轉換任務列表
      fetchJobs();
      
      // 刷新模型庫，確保UI同步
      refreshModelRepository();
    } catch (error) {
      console.error('刪除轉換任務失敗:', error);
      if (error.response && error.response.data && error.response.data.detail) {
        message.error(`刪除轉換任務失敗: ${error.response.data.detail}`);
      } else {
        message.error('刪除轉換任務失敗');
      }
    }
  };

  return (
    <div>
      {/* 系統狀態警告 - 只在有測試運行時顯示 */}
      {systemState && systemState.is_testing && (
        <Alert
          message="系統正在執行自動化模型性能評估"
          description={
            <div>
              <p>系統當前正在執行自動化模型性能評估，以下是當前評估信息：</p>
              <p>- 當前模型：{systemState.current_model || '準備中'}</p>
              <p>- 當前階段：{systemState.current_step_name || systemState.current_step || '準備中'}</p>
              <p>- 當前批次大小/精度：{systemState.current_batch_size ? 
                `${systemState.current_batch_size} / ${systemState.current_precision}` : '準備中'}</p>
              <p>- 進度：{systemState.completed_combinations}/{systemState.total_combinations} 組合</p>
              <p>為避免資源衝突，暫時無法創建新的轉換任務，請等待測試完成後再試。</p>
            </div>
          }
          type="warning"
          showIcon
          style={{ marginBottom: 16 }}
          icon={<SyncOutlined spin />}
        />
      )}

      <Card title="創建模型轉換任務">
        <Form
          form={form}
          layout="vertical"
          onFinish={handleSubmit}
          onValuesChange={handleValuesChange}
        >
          <Form.Item
            name="model_id"
            label="選擇模型"
            rules={[{ required: true, message: '請選擇模型' }]}
          >
            <Select
              placeholder="選擇要轉換的模型"
              loading={loading}
              disabled={systemState && systemState.is_testing}
            >
              {models
                .filter(model => model.format !== 'engine') // 過濾掉 engine 格式的模型
                .map(model => (
                <Option key={model.id} value={model.id}>
                  {model.name} ({model.format})
                </Option>
              ))}
            </Select>
          </Form.Item>
          <Form.Item
            name="target_format"
            label="目標格式"
            rules={[{ required: true, message: '請選擇目標格式' }]}
          >
            <Select 
              placeholder="選擇目標格式" 
              disabled={systemState && systemState.is_testing}
            >
              {/* 如果選中的是 ONNX 模型，只顯示 TensorRT Engine 選項 */}
              {selectedModelFormat === 'onnx' ? (
                <Option value="engine">TensorRT Engine (包含 .engine 和 .plan)</Option>
              ) : (
                <>
              <Option value="onnx">ONNX</Option>
                  <Option value="engine">TensorRT Engine (包含 .engine 和 .plan)</Option>
                </>
              )}
            </Select>
          </Form.Item>
          <Form.Item
            name="precision"
            label="精度"
            rules={[{ required: true, message: '請選擇精度' }]}
            initialValue="fp32"
          >
            <Select placeholder="選擇精度" disabled={systemState && systemState.is_testing}>
              <Option value="fp32">FP32</Option>
              <Option value="fp16">FP16</Option>
            </Select>
          </Form.Item>
          <Form.Item
            name="batch_size"
            label="批次大小"
            rules={[{ required: true, message: '請選擇批次大小' }]}
            initialValue="1"
          >
            <Select placeholder="選擇批次大小" disabled={systemState && systemState.is_testing}>
              <Option value="1">1</Option>
              <Option value="2">2</Option>
              <Option value="4">4</Option>
              <Option value="8">8</Option>
              <Option value="16">16</Option>
            </Select>
          </Form.Item>
          <Form.Item
            name="image_size"
            label="圖像尺寸"
            rules={[{ required: true, message: '請選擇圖像尺寸' }]}
            initialValue="640"
          >
            <Select placeholder="選擇圖像尺寸" disabled={systemState && systemState.is_testing}>
              <Option value="512">512</Option>
              <Option value="640">640</Option>
              <Option value="768">768</Option>
              <Option value="1024">1024</Option>
            </Select>
          </Form.Item>
          <Form.Item
            name="workspace"
            label="TensorRT工作區大小 (GB)"
            rules={[{ required: true, message: '請選擇工作區大小' }]}
            initialValue="4"
          >
            <Select placeholder="選擇工作區大小" disabled={systemState && systemState.is_testing}>
              <Option value="2">2</Option>
              <Option value="4">4</Option>
              <Option value="8">8</Option>
              <Option value="16">16</Option>
            </Select>
          </Form.Item>
          <Form.Item>
            <Button type="primary" htmlType="submit" loading={submitting} disabled={systemState && systemState.is_testing}>
              開始轉換
            </Button>
            {systemState && systemState.is_testing && (
              <span style={{ marginLeft: 8, color: '#ff4d4f' }}>
                系統正在執行自動化管線，暫時無法創建轉換任務
              </span>
            )}
          </Form.Item>
          {existingModelWarning && (
            <div style={{ color: 'red', marginTop: '8px' }}>
              警告: 已存在相同參數的轉換模型 (ID: {existingModelWarning.modelId}, 名稱: {existingModelWarning.modelName})。
              請使用不同的參數或直接使用現有模型。
            </div>
          )}
        </Form>
      </Card>

      <Card title="轉換任務列表" style={{ marginTop: 16 }}>
        <div style={{ marginBottom: 16 }}>
          <Button 
            type="primary" 
            icon={<SyncOutlined />} 
            onClick={() => fetchJobs()}
            loading={jobsLoading}
          >
            刷新任務列表
          </Button>
        </div>
        <Table
          columns={columns}
          dataSource={jobs}
          rowKey="id"
          loading={jobsLoading}
        />
      </Card>
    </div>
  );
};

export default ConversionPage; 