import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Table, Button, Upload, Form, Input, Select, Modal, Card, message, Tooltip, Badge, Space, Tag } from 'antd';
import { 
  UploadOutlined, 
  PlusOutlined, 
  InfoCircleOutlined, 
  ReloadOutlined,
  CloudUploadOutlined,
  CloudDownloadOutlined,
  LoadingOutlined
} from '@ant-design/icons';
import api, { API_BASE_URL } from '../api/client';

const { Option } = Select;

const ModelsPage = () => {
  const navigate = useNavigate();
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(false);
  const [modalVisible, setModalVisible] = useState(false);
  const [uploadForm] = Form.useForm();
  const [loadingModels, setLoadingModels] = useState(new Set()); // 跟踪正在掛載/卸載的模型
  const [uploading, setUploading] = useState(false);

  // 當組件掛載時和每次訪問頁面時刷新模型列表
  useEffect(() => {
    // 讓模型服務掃描目錄後再獲取模型列表
    const refreshAndFetch = async () => {
      try {
        // 先嘗試刷新模型庫
        await api.get('/api/models/refresh');
        console.log('模型庫刷新成功');
        // 然後獲取最新的模型列表
        await fetchModels();
      } catch (error) {
        console.error('刷新模型庫失敗，直接獲取模型列表', error);
        fetchModels();
      }
    };

    // 執行刷新和獲取
    refreshAndFetch();

    // 添加頁面可見性變化事件監聽器
    document.addEventListener('visibilitychange', handleVisibilityChange);

    // 清理函數，當組件卸載時移除事件監聽器
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, []);

  // 處理頁面可見性變化，當用戶從其他頁面返回時刷新數據
  const handleVisibilityChange = () => {
    if (document.visibilityState === 'visible') {
      // 當頁面再次變為可見時，刷新模型庫然後獲取模型列表
      api.get('/api/models/refresh')
        .then(() => fetchModels())
        .catch(() => fetchModels());
    }
  };

  const fetchModels = async () => {
    setLoading(true);
    try {
      const response = await api.get('/api/models/');
      const rawModels = response.data.models;
      
      // 過濾掉重複模型（相同triton_model_dir路徑的模型）
      const modelsByPath = {};
      const filteredModels = [];
      
      rawModels.forEach(model => {
        // 如果模型有metadata和triton_model_dir
        if (model.metadata && model.metadata.triton_model_dir) {
          const path = model.metadata.triton_model_dir;
          
          // 如果這個路徑還沒有記錄過，或這個模型更新
          if (!modelsByPath[path] || new Date(model.created_at) > new Date(modelsByPath[path].created_at)) {
            modelsByPath[path] = model;
          }
        } else {
          // 沒有triton_model_dir的模型直接保留
          filteredModels.push(model);
        }
      });
      
      // 將所有唯一路徑的最新模型添加到結果
      Object.values(modelsByPath).forEach(model => {
        filteredModels.push(model);
      });
      
      // 按創建時間排序
      filteredModels.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
      
      // 檢查Triton兼容模型的掛載狀態
      const modelsWithStatus = await Promise.all(
        filteredModels.map(async (model) => {
          if (model.metadata && model.metadata.triton_model_name && model.metadata.is_trt_model) {
            try {
              const statusResponse = await api.get(`/api/triton/models/${model.id}/status`);
              return {
                ...model,
                tritonStatus: statusResponse.data
              };
            } catch (error) {
              return {
                ...model,
                tritonStatus: { loaded: false, ready: false, error: error.message }
              };
            }
          }
          return model;
        })
      );
      
      console.log(`從 ${rawModels.length} 個模型中過濾出 ${filteredModels.length} 個不重複模型`);
      setModels(modelsWithStatus);
    } catch (error) {
      console.error('獲取模型失敗:', error);
      message.error('獲取模型列表失敗');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (modelId) => {
    try {
      await api.delete(`/api/models/${modelId}`);
      message.success('刪除模型成功');
      fetchModels();
    } catch (error) {
      console.error('刪除模型失敗:', error);
      message.error('刪除模型失敗');
    }
  };

  const handleUpload = async (values) => {
    const { name, model_type, description, fileList } = values;
    
    try {
      console.log('提交的模型類型:', model_type);
      console.log('提交的文件:', fileList);
      
      if (!fileList || !fileList[0] || !fileList[0].originFileObj) {
        message.error('請選擇有效的模型文件');
        return;
      }
      
      // 設置loading狀態
      setUploading(true);
      
      // 顯示開始上傳的消息
      const uploadMessage = message.loading('正在上傳模型文件...', 0);
      
      const formData = new FormData();
      formData.append('model_file', fileList[0].originFileObj);
      formData.append('model_name', name);
      formData.append('model_type', model_type);
      if (description) {
        formData.append('description', description);
      }
      
      // 檢查FormData內容
      for (let pair of formData.entries()) {
        console.log(pair[0] + ': ' + (pair[0] === 'model_file' ? '文件對象' : pair[1]));
      }
      
      try {
        // 更新loading消息
        uploadMessage();
        const processingMessage = message.loading('正在處理模型文件...', 0);
        
        // 如果是.pt文件，顯示TorchScript轉換消息
        const filename = fileList[0].originFileObj.name;
        if (filename.endsWith('.pt')) {
          processingMessage();
          const torchscriptMessage = message.loading('正在轉換PyTorch模型為TorchScript，請稍候...', 0);
          
          const response = await api.post('/api/models/', formData, {
            headers: {
              'Content-Type': 'multipart/form-data',
            },
            timeout: 120000, // 2分鐘超時，因為TorchScript轉換需要時間
          });
          
          torchscriptMessage();
        } else {
      const response = await api.post('/api/models/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
            timeout: 60000, // 1分鐘超時
      });
      
          processingMessage();
        }
        
        message.success('模型上傳並處理完成！');
      setModalVisible(false);
      uploadForm.resetFields();
      
      // 上傳完成後立即刷新模型列表，並設置短暫延遲確保後端處理完成
      setTimeout(() => {
        fetchModels();
      }, 1000);
        
      } catch (requestError) {
        throw requestError;
      }
      
    } catch (error) {
      console.error('上傳模型失敗:', error.response?.data || error);
      
      // 根據錯誤類型提供更詳細的錯誤信息
      let errorMessage = '上傳模型失敗';
      if (error.code === 'ECONNABORTED') {
        errorMessage = '上傳超時，請檢查文件大小或網路連接';
      } else if (error.response?.data?.detail) {
        errorMessage = `上傳失敗: ${error.response.data.detail}`;
      } else if (error.message) {
        errorMessage = `上傳失敗: ${error.message}`;
      }
      
      message.error(errorMessage);
    } finally {
      setUploading(false);
    }
  };

  // 掛載模型到Triton
  const handleLoadModel = async (modelId, modelName) => {
    setLoadingModels(prev => new Set([...prev, modelId]));
    try {
      const response = await api.post(`/api/triton/models/${modelId}/load`);
      if (response.data.success) {
        message.success(`模型 ${modelName} 掛載成功`);
        // 刷新模型狀態
        await fetchModels();
      } else {
        message.error(`掛載模型失敗: ${response.data.error}`);
      }
    } catch (error) {
      console.error('掛載模型失敗:', error);
      message.error(`掛載模型失敗: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoadingModels(prev => {
        const newSet = new Set(prev);
        newSet.delete(modelId);
        return newSet;
      });
    }
  };

  // 從Triton卸載模型
  const handleUnloadModel = async (modelId, modelName) => {
    setLoadingModels(prev => new Set([...prev, modelId]));
    try {
      const response = await api.post(`/api/triton/models/${modelId}/unload`);
      if (response.data.success) {
        message.success(`模型 ${modelName} 卸載成功`);
        // 刷新模型狀態
        await fetchModels();
      } else {
        message.error(`卸載模型失敗: ${response.data.error}`);
      }
    } catch (error) {
      console.error('卸載模型失敗:', error);
      message.error(`卸載模型失敗: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoadingModels(prev => {
        const newSet = new Set(prev);
        newSet.delete(modelId);
        return newSet;
      });
    }
  };

  // 渲染Triton狀態
  const renderTritonStatus = (record) => {
    if (!record.metadata || !record.metadata.is_trt_model) {
      return <Tag color="default">非Triton模型</Tag>;
    }

    if (!record.tritonStatus) {
      return <Tag color="processing">檢查中...</Tag>;
    }

    if (record.tritonStatus.loaded && record.tritonStatus.ready) {
      return <Badge status="success" text="已掛載" />;
    } else {
      return <Badge status="default" text="未掛載" />;
    }
  };

  // 渲染Triton操作按鈕
  const renderTritonActions = (record) => {
    if (!record.metadata || !record.metadata.is_trt_model) {
      return null;
    }

    const isLoading = loadingModels.has(record.id);
    const isLoaded = record.tritonStatus && record.tritonStatus.loaded;

    if (isLoaded) {
      return (
        <Button
          type="link"
          danger
          size="small"
          icon={isLoading ? <LoadingOutlined /> : <CloudDownloadOutlined />}
          loading={isLoading}
          onClick={() => handleUnloadModel(record.id, record.name)}
        >
          卸載
        </Button>
      );
    } else {
      return (
        <Button
          type="link"
          size="small"
          icon={isLoading ? <LoadingOutlined /> : <CloudUploadOutlined />}
          loading={isLoading}
          onClick={() => handleLoadModel(record.id, record.name)}
        >
          掛載
        </Button>
      );
    }
  };

  const columns = [
    {
      title: '名稱',
      dataIndex: 'name',
      key: 'name',
      render: (text, record) => (
        <div>
          <div>
        <Tooltip title={`ID: ${record.id}`}>
          <span>{text} <InfoCircleOutlined style={{ fontSize: '12px', color: '#1890ff' }} /></span>
        </Tooltip>
          </div>
          {record.metadata && record.metadata.is_trt_model && (
            <div style={{ fontSize: '12px', color: '#666', marginTop: '4px' }}>
              {renderTritonStatus(record)}
            </div>
          )}
        </div>
      ),
    },
    {
      title: '類型',
      dataIndex: 'type',
      key: 'type',
    },
    {
      title: '格式',
      dataIndex: 'format',
      key: 'format',
      render: (format, record) => (
        <div>
          <Tag color={format === 'engine' ? 'green' : format === 'onnx' ? 'blue' : 'default'}>
            {format.toUpperCase()}
          </Tag>
          {record.metadata && record.metadata.conversion_precision && (
            <Tag color={record.metadata.conversion_precision === 'fp16' ? 'orange' : 'default'} size="small">
              {record.metadata.conversion_precision.toUpperCase()}
            </Tag>
          )}
        </div>
      ),
    },
    {
      title: '大小',
      dataIndex: 'size_mb',
      key: 'size_mb',
      render: (size) => `${size.toFixed(1)} MB`,
    },
    {
      title: '上傳時間',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (date) => new Date(date).toLocaleString(),
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record) => (
        <Space size="small">
          <Button type="link" size="small" onClick={() => navigate(`/models/${record.id}`)}>
            詳情
          </Button>
          <Button 
            type="link" 
            size="small"
            onClick={() => window.open(`${API_BASE_URL}/api/models/${record.id}/download`)}
          >
            下載
          </Button>
          {renderTritonActions(record)}
          <Button type="link" danger size="small" onClick={() => handleDelete(record.id)}>
            刪除
          </Button>
        </Space>
      ),
    },
  ];

  return (
    <div>
      <Card
        title="模型管理"
        extra={
          <Space>
            <Button 
              onClick={fetchModels} 
              icon={<ReloadOutlined />} 
            >
              刷新
            </Button>
            <Button type="primary" icon={<PlusOutlined />} onClick={() => setModalVisible(true)}>
              上傳模型
            </Button>
          </Space>
        }
      >
        <Table
          columns={columns}
          dataSource={models}
          rowKey="id"
          loading={loading}
          pagination={{
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) => `第 ${range[0]}-${range[1]} 項，共 ${total} 項`,
          }}
        />
      </Card>

      <Modal
        title="上傳模型"
        open={modalVisible}
        onCancel={() => setModalVisible(false)}
        footer={null}
      >
        <Form
          form={uploadForm}
          layout="vertical"
          onFinish={handleUpload}
        >
          <Form.Item
            name="fileList"
            label="模型文件"
            rules={[{ required: true, message: '請上傳模型文件' }]}
            valuePropName="fileList"
            getValueFromEvent={(e) => {
              if (Array.isArray(e)) {
                return e;
              }
              return e?.fileList;
            }}
          >
            <Upload 
              accept=".pt,.onnx,.engine" 
              beforeUpload={() => false}
              maxCount={1}
              listType="text"
            >
              <Button icon={<UploadOutlined />}>選擇文件</Button>
            </Upload>
          </Form.Item>
          <Form.Item
            name="name"
            label="模型名稱"
            rules={[{ required: true, message: '請輸入模型名稱' }]}
          >
            <Input />
          </Form.Item>
          <Form.Item
            name="model_type"
            label="模型類型"
            rules={[{ required: true, message: '請選擇模型類型' }]}
          >
            <Select>
              <Option value="yolov8">YOLOv8</Option>
              <Option value="yolov8_pose">YOLOv8-Pose</Option>
              <Option value="yolov8_seg">YOLOv8-Seg</Option>
              <Option value="custom">自定義</Option>
            </Select>
          </Form.Item>
          <Form.Item
            name="description"
            label="描述"
          >
            <Input.TextArea rows={4} />
          </Form.Item>
          <Form.Item>
            <Button 
              type="primary" 
              htmlType="submit" 
              loading={uploading}
              disabled={uploading}
            >
              {uploading ? '處理中...' : '上傳'}
            </Button>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default ModelsPage; 