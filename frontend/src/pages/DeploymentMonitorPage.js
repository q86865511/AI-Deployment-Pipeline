import React, { useState, useEffect } from 'react';
import {
  Card,
  Table,
  Typography,
  Alert,
  Spin,
  Tag,
  Row,
  Col,
  Button,
  Space,
  Tooltip,
  Statistic,
  Badge,
  message
} from 'antd';
import { 
  ReloadOutlined,
  CloudDownloadOutlined,
  MemoryIcon,
  DashboardOutlined,
  BarChartOutlined,
  ExclamationCircleOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const DeploymentMonitorPage = () => {
  const [loadedModels, setLoadedModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [tritonHealth, setTritonHealth] = useState(null);
  const [refreshing, setRefreshing] = useState(false);

  // 獲取Triton健康狀態
  const checkTritonHealth = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/triton/health`);
      const data = await response.json();
      setTritonHealth(data);
    } catch (error) {
      console.error('檢查Triton健康狀態失敗:', error);
      setTritonHealth({ success: false, error: error.message });
    }
  };

  // 獲取部署監控數據
  const fetchDeploymentData = async () => {
    try {
      setError('');
      const response = await fetch(`${API_BASE_URL}/api/triton/deployment/monitoring`);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      if (data.success) {
        console.log('部署監控數據:', data);
        console.log('模型數據樣本:', data.models?.[0]);
        setLoadedModels(data.models || []);
      } else {
        setError(data.error || '獲取部署監控數據失敗');
      }
    } catch (error) {
      console.error('獲取部署監控數據失敗:', error);
      setError(`無法連接到Triton服務器: ${error.message}`);
      setLoadedModels([]);
    }
  };

  // 卸載模型
  const handleUnloadModel = async (modelName) => {
    try {
      setError('');
      const response = await fetch(`${API_BASE_URL}/api/triton/models/${modelName}/unload`, {
        method: 'POST'
      });
      
      const data = await response.json();
      
      if (data.success) {
        message.success(`模型 ${modelName} 卸載成功`);
        // 重新獲取數據以更新表格
        await fetchDeploymentData();
      } else {
        message.error(`卸載模型失敗: ${data.error}`);
      }
    } catch (error) {
      console.error('卸載模型失敗:', error);
      message.error(`卸載模型失敗: ${error.message}`);
    }
  };

  // 手動刷新數據
  const handleRefresh = async () => {
    setRefreshing(true);
    await Promise.all([
      checkTritonHealth(),
      fetchDeploymentData()
    ]);
    setRefreshing(false);
  };

  // 組件掛載時獲取數據
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      await Promise.all([
        checkTritonHealth(),
        fetchDeploymentData()
      ]);
      setLoading(false);
    };

    loadData();
  }, []);

  // 定期刷新數據（每30秒）
  useEffect(() => {
    const interval = setInterval(() => {
      if (!refreshing) {
        handleRefresh();
      }
    }, 30000);

    return () => clearInterval(interval);
  }, [refreshing]);

  // 格式化推論時間
  const formatInferenceTime = (timeMs) => {
    console.log('formatInferenceTime 輸入:', timeMs, '類型:', typeof timeMs);
    if (timeMs === 0 || timeMs === null || timeMs === undefined || timeMs === 'N/A') {
      return 'N/A';
    }
    const numValue = Number(timeMs);
    if (isNaN(numValue)) {
      return 'N/A';
    }
    return `${numValue.toFixed(2)} ms`;
  };

  // 格式化最後推論時間
  const formatLastInference = (timestamp) => {
    if (!timestamp || timestamp === 0) {
      return 'N/A';
    }
    
    try {
      // timestamp已經是毫秒時間戳
      const date = new Date(timestamp);
      const now = new Date();
      const diffMs = now - date;
      const diffMins = Math.floor(diffMs / 60000);
      
      if (diffMins < 1) {
        return '剛剛';
      } else if (diffMins < 60) {
        return `${diffMins} 分鐘前`;
      } else if (diffMins < 1440) { // 24小時
        const diffHours = Math.floor(diffMins / 60);
        return `${diffHours} 小時前`;
      } else {
        const diffDays = Math.floor(diffMins / 1440);
        return `${diffDays} 天前`;
      }
    } catch (error) {
      return 'N/A';
    }
  };

  // 表格列定義
  const columns = [
    {
      title: '模型名稱',
      dataIndex: 'display_name',
      key: 'display_name',
      render: (text, record) => (
        <div>
          <div style={{ fontWeight: 'bold' }}>{text || record.model_name}</div>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            {record.model_name}
          </Text>
        </div>
      ),
    },
    {
      title: '格式',
      dataIndex: 'format',
      key: 'format',
      render: (format) => {
        const getFormatColor = (fmt) => {
          switch(fmt) {
            case 'PT': return 'default';
            case 'ENGINE': return 'green';
            case 'ONNX': return 'blue';
            default: return 'default';
          }
        };
        
        return (
          <Tag color={getFormatColor(format)}>
            {format || 'unknown'}
          </Tag>
        );
      },
    },
    {
      title: '精度',
      dataIndex: 'precision',
      key: 'precision',
      render: (precision, record) => {
        // PT原始模型顯示「預設」，其他顯示實際精度
        let displayText = 'FP32';  // 預設顯示
        
        if (precision && precision !== 'unknown') {
          displayText = precision;
        }
        
        // 對於PT格式，顯示「預設」
        if (record.format === 'PT') {
          displayText = '預設';
        }
        
        return (
          <Tag color={displayText === 'FP16' ? 'orange' : 'blue'}>
            {displayText}
          </Tag>
        );
      },
    },
    {
      title: '批次大小',
      dataIndex: 'batch_size',
      key: 'batch_size',
      render: (batchSize) => {
        // 處理預設值和未知值
        const displayBatchSize = batchSize === 'unknown' ? '1' : (batchSize || '1');
        
        return (
          <Tag color="green">
            批次: {displayBatchSize}
          </Tag>
        );
      },
    },
    {
      title: '狀態',
      dataIndex: 'state',
      key: 'state',
      align: 'center',
      render: (state) => (
        <Badge 
          status={state === 'READY' ? 'success' : 'warning'} 
          text={state || 'unknown'} 
        />
      ),
    },
    {
      title: '總請求次數',
      dataIndex: 'execution_count',
      key: 'execution_count',
      align: 'right',
      render: (count) => count || 0,
    },
    {
      title: '單張延遲(總)',
      dataIndex: 'avg_total_time_ms',
      key: 'avg_total_time_ms',
      align: 'right',
      render: (time) => formatInferenceTime(time),
    },
    {
      title: '單張延遲(推理)',
      dataIndex: 'avg_infer_time_ms',
      key: 'avg_infer_time_ms',
      align: 'right',
      render: (time) => formatInferenceTime(time),
    },
    {
      title: '最後推論',
      dataIndex: 'last_inference',
      key: 'last_inference',
      align: 'right',
      render: (timestamp) => formatLastInference(timestamp),
    },
    {
      title: '操作',
      key: 'action',
      align: 'center',
      render: (_, record) => (
        <Tooltip title="卸載模型">
          <Button
            type="link"
            danger
            size="small"
            icon={<CloudDownloadOutlined />}
            onClick={() => handleUnloadModel(record.model_name)}
          >
            卸載
          </Button>
        </Tooltip>
      ),
    },
  ];

  if (loading) {
    return (
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        minHeight: '400px' 
      }}>
        <Spin size="large" />
      </div>
    );
  }

  return (
    <div style={{ padding: '24px' }}>
      {/* 頁面標題和操作 */}
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center', 
        marginBottom: '24px' 
      }}>
        <Title level={2}>部署平台監控</Title>
        <Button
          type="primary"
          icon={<ReloadOutlined />}
          onClick={handleRefresh}
          loading={refreshing}
        >
          {refreshing ? '刷新中...' : '刷新'}
        </Button>
      </div>

      {/* 統計卡片 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} md={6}>
          <Card>
            <Statistic
              title="Triton服務器狀態"
              value={tritonHealth?.healthy ? '運行正常' : '離線'}
              prefix={<DashboardOutlined />}
              valueStyle={{ 
                color: tritonHealth?.healthy ? '#3f8600' : '#cf1322' 
              }}
            />
          </Card>
        </Col>

        <Col xs={24} md={6}>
          <Card>
            <Statistic
              title="已掛載模型數量"
              value={loadedModels.length}
              prefix={<BarChartOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>

        <Col xs={24} md={6}>
          <Card>
            <Statistic
              title="總請求次數"
              value={loadedModels.reduce((total, model) => total + (model.execution_count || 0), 0)}
              prefix={<BarChartOutlined />}
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>

        <Col xs={24} md={6}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '14px', color: '#666', marginBottom: '8px' }}>模型格式分佈</div>
              <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap' }}>
                {['PT', 'ENGINE', 'ONNX'].map(format => {
                  const count = loadedModels.filter(model => model.format === format).length;
                  const color = format === 'PT' ? '#d9d9d9' : format === 'ENGINE' ? '#52c41a' : '#1890ff';
                  return (
                    <div key={format} style={{ textAlign: 'center', margin: '4px' }}>
                      <div style={{ fontSize: '18px', fontWeight: '600', color }}>{count}</div>
                      <div style={{ fontSize: '12px', color: '#666' }}>{format}</div>
                    </div>
                  );
                })}
              </div>
            </div>
          </Card>
        </Col>
      </Row>

      {/* 錯誤信息 */}
      {error && (
        <Alert
          message="錯誤"
          description={error}
          type="error"
          showIcon
          icon={<ExclamationCircleOutlined />}
          style={{ marginBottom: '24px' }}
        />
      )}

      {/* 已掛載模型表格 */}
      <Card
        title="已掛載模型列表"
        extra={
          <Text type="secondary">
            數據每30秒自動刷新 • 最後更新: {new Date().toLocaleTimeString('zh-TW')}
          </Text>
        }
      >
        {loadedModels.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '48px' }}>
            <Text type="secondary" style={{ fontSize: '16px' }}>
              目前沒有掛載的模型
            </Text>
            <br />
            <Text type="secondary">
              請前往模型管理頁面掛載模型後再查看
            </Text>
          </div>
        ) : (
          <Table
            columns={columns}
            dataSource={loadedModels}
            rowKey="model_name"
            pagination={{
              showSizeChanger: true,
              showQuickJumper: true,
              showTotal: (total, range) => `第 ${range[0]}-${range[1]} 項，共 ${total} 項`,
            }}
          />
        )}
      </Card>
    </div>
  );
};

export default DeploymentMonitorPage; 