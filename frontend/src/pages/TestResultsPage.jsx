import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { 
  Card, Descriptions, Table, Statistic, Row, Col, Typography, 
  Button, Space, Spin, Alert, Tabs, Tag, Select
} from 'antd';
import { 
  ArrowLeftOutlined, DownloadOutlined, AreaChartOutlined
} from '@ant-design/icons';
import axios from 'axios';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;

const TestResultsPage = () => {
  const { taskId: urlTaskId } = useParams();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [testResults, setTestResults] = useState(null);
  const [error, setError] = useState(null);
  const [availableTasks, setAvailableTasks] = useState([]);
  const [selectedTaskId, setSelectedTaskId] = useState(urlTaskId || '');

  useEffect(() => {
    // 如果URL中有taskId，直接載入該任務結果
    if (urlTaskId) {
      setSelectedTaskId(urlTaskId);
      fetchTestResults(urlTaskId);
    } else {
      // 否則載入可用任務列表
      fetchAvailableTasks();
    }
  }, [urlTaskId]);

  const fetchAvailableTasks = async () => {
    setLoading(true);
    try {
      const response = await axios.get('http://localhost:8000/api/benchmark/tasks');
      const tasks = response.data.tasks || [];
      const completedTasks = tasks
        .filter(task => task.status === 'completed')
        .map(task => ({
          taskId: task.task_id,
          displayName: `${task.model_name} - ${new Date(task.created_at).toLocaleDateString()}`
        }));
      setAvailableTasks(completedTasks);
    } catch (err) {
      console.error('載入任務列表失敗:', err);
      setError('載入任務列表失敗');
    } finally {
      setLoading(false);
    }
  };

  const fetchTestResults = async (taskId = selectedTaskId) => {
    if (!taskId) return;
    
    setLoading(true);
    try {
      const response = await axios.get(`http://localhost:8000/api/benchmark/results/${taskId}`);
      setTestResults(response.data);
      setError(null);
    } catch (err) {
      console.error('載入測試結果失敗:', err);
      setError(err.response?.data?.detail || '載入測試結果失敗');
    } finally {
      setLoading(false);
    }
  };

  const handleTaskSelection = (taskId) => {
    setSelectedTaskId(taskId);
    if (taskId) {
      fetchTestResults(taskId);
    } else {
      setTestResults(null);
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'success':
      case 'completed':
        return 'green';
      case 'failed':
        return 'red';
      case 'processing':
        return 'blue';
      default:
        return 'default';
    }
  };

  const formatMetric = (value, suffix = '') => {
    if (value === null || value === undefined) return 'N/A';
    if (typeof value === 'number') {
      return `${value.toFixed(4)}${suffix}`;
    }
    return value;
  };

  const validationColumns = [
    {
      title: '指標',
      dataIndex: 'metric',
      key: 'metric',
    },
    {
      title: '數值',
      dataIndex: 'value',
      key: 'value',
      render: (value) => formatMetric(value)
    },
  ];

  const inferenceColumns = [
    {
      title: '指標',
      dataIndex: 'metric',
      key: 'metric',
    },
    {
      title: '數值',
      dataIndex: 'value',
      key: 'value',
      render: (value, record) => formatMetric(value, record.suffix || '')
    },
  ];

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '50px' }}>
        <Spin size="large" />
        <div style={{ marginTop: 16 }}>載入測試結果中...</div>
      </div>
    );
  }

  if (error) {
    return (
      <Alert
        message="載入失敗"
        description={error}
        type="error"
        showIcon
        action={
          <Button size="small" onClick={fetchTestResults}>
            重試
          </Button>
        }
      />
    );
  }

  return (
    <div style={{ padding: '24px' }}>
      <Space style={{ marginBottom: 16 }}>
        <Button 
          icon={<ArrowLeftOutlined />} 
          onClick={() => navigate('/benchmark')}
        >
          返回測試管理
        </Button>
        {selectedTaskId && (
          <>
            <Button 
              icon={<AreaChartOutlined />}
              onClick={() => navigate(`/performance/${selectedTaskId}`)}
            >
              性能分析
            </Button>
            <Button 
              icon={<DownloadOutlined />}
              onClick={() => window.open(`http://localhost:8000/api/benchmark/download-results/${selectedTaskId}`)}
            >
              下載完整結果
            </Button>
          </>
        )}
      </Space>

      {!selectedTaskId && (
        <Card style={{ marginBottom: 16 }}>
          <Title level={4}>選擇測試任務</Title>
          <Select
            style={{ width: '100%' }}
            placeholder="請選擇要查看的測試任務..."
            value={selectedTaskId}
            onChange={handleTaskSelection}
            showSearch
            optionFilterProp="children"
          >
            {availableTasks.map(task => (
              <Option key={task.taskId} value={task.taskId}>
                {task.displayName}
              </Option>
            ))}
          </Select>
        </Card>
      )}

      {testResults && (
        <Card>
          <Title level={3}>測試結果詳情</Title>
          
          <Descriptions bordered column={2} style={{ marginBottom: 24 }}>
            <Descriptions.Item label="任務ID">{testResults?.task_id}</Descriptions.Item>
            <Descriptions.Item label="模型名稱">{testResults?.model_name}</Descriptions.Item>
            <Descriptions.Item label="狀態">
              <Tag color={getStatusColor(testResults?.status)}>
                {testResults?.status === 'completed' ? '已完成' : testResults?.status}
              </Tag>
            </Descriptions.Item>
            <Descriptions.Item label="創建時間">
              {testResults?.created_at ? new Date(testResults.created_at).toLocaleString('zh-TW') : 'N/A'}
            </Descriptions.Item>
            <Descriptions.Item label="完成時間">
              {testResults?.completed_at ? new Date(testResults.completed_at).toLocaleString('zh-TW') : 'N/A'}
            </Descriptions.Item>
          </Descriptions>

          {testResults?.results?.map((result, index) => (
            <Card 
              key={index}
              title={`測試組合 ${index + 1}: 批次大小=${result.batch_size}, 精度=${result.precision}`}
              style={{ marginBottom: 16 }}
            >
              <Tabs defaultActiveKey="validation">
                <TabPane tab="模型驗證" key="validation">
                  {result.validation_results ? (
                    <>
                      <Row gutter={16} style={{ marginBottom: 16 }}>
                        <Col span={6}>
                          <Statistic
                            title="模型ID"
                            value={result.validation_results.model_id}
                            valueStyle={{ fontSize: '14px' }}
                          />
                        </Col>
                        <Col span={6}>
                          <Statistic
                            title="數據集"
                            value={result.validation_results.dataset_name || 'N/A'}
                          />
                        </Col>
                        <Col span={6}>
                          <Statistic
                            title="批次大小"
                            value={result.validation_results.batch_size}
                          />
                        </Col>
                        <Col span={6}>
                          <Statistic
                            title="狀態"
                            value={<Tag color="green">驗證完成</Tag>}
                          />
                        </Col>
                      </Row>

                      <Table
                        columns={validationColumns}
                        dataSource={[
                          { key: '1', metric: 'mAP@0.5', value: result.validation_results.metrics?.mAP50 },
                          { key: '2', metric: 'mAP@0.5:0.95', value: result.validation_results.metrics?.mAP50_95 },
                          { key: '3', metric: 'Precision', value: result.validation_results.metrics?.precision },
                          { key: '4', metric: 'Recall', value: result.validation_results.metrics?.recall },
                        ]}
                        pagination={false}
                        size="small"
                      />
                    </>
                  ) : (
                    <Alert message="無驗證結果數據" type="warning" />
                  )}
                </TabPane>

                <TabPane tab="推論測試" key="inference">
                  {result.inference_results ? (
                    <>
                      <Row gutter={16} style={{ marginBottom: 16 }}>
                        <Col span={6}>
                          <Statistic
                            title="平均推論時間"
                            value={result.inference_results.avg_inference_time_ms?.toFixed(3)}
                            suffix="ms"
                          />
                        </Col>
                        <Col span={6}>
                          <Statistic
                            title="吞吐量"
                            value={result.inference_results.avg_throughput_fps?.toFixed(2)}
                            suffix="FPS"
                          />
                        </Col>
                        <Col span={6}>
                          <Statistic
                            title="VRAM使用"
                            value={result.performance_metrics?.memory_usage_mb?.toFixed(2) || result.validation_results?.memory_usage_mb?.toFixed(2) || 'N/A'}
                            suffix="MB"
                          />
                        </Col>
                        <Col span={6}>
                          <Statistic
                            title="迭代次數"
                            value={result.inference_results.iterations}
                          />
                        </Col>
                      </Row>

                      <Row gutter={16} style={{ marginBottom: 16 }}>
                        <Col span={6}>
                          <Statistic
                            title="平均GPU負載"
                            value={result.performance_metrics?.avg_gpu_load?.toFixed(1) || result.validation_results?.avg_gpu_load?.toFixed(1) || 'N/A'}
                            suffix="%"
                          />
                        </Col>
                        <Col span={6}>
                          <Statistic
                            title="峰值GPU負載"
                            value={result.performance_metrics?.max_gpu_load?.toFixed(1) || result.validation_results?.max_gpu_load?.toFixed(1) || 'N/A'}
                            suffix="%"
                          />
                        </Col>
                        <Col span={6}>
                          <Statistic
                            title="監控樣本數"
                            value={result.performance_metrics?.monitoring_samples || result.validation_results?.monitoring_samples || 'N/A'}
                          />
                        </Col>
                        <Col span={6}>
                          <Statistic
                            title="監控時間"
                            value={result.performance_metrics?.monitoring_duration_s?.toFixed(1) || result.validation_results?.monitoring_duration_s?.toFixed(1) || 'N/A'}
                            suffix="秒"
                          />
                        </Col>
                      </Row>

                      <Table
                        columns={inferenceColumns}
                        dataSource={[
                          { key: '1', metric: '平均推論時間', value: result.inference_results.avg_inference_time_ms, suffix: ' ms' },
                          { key: '2', metric: '標準差', value: result.inference_results.std_inference_time_ms, suffix: ' ms' },
                          { key: '3', metric: '最小時間', value: result.inference_results.min_inference_time_ms, suffix: ' ms' },
                          { key: '4', metric: '最大時間', value: result.inference_results.max_inference_time_ms, suffix: ' ms' },
                          { key: '5', metric: '平均吞吐量', value: result.inference_results.avg_throughput_fps, suffix: ' FPS' },
                          { key: '6', metric: 'VRAM使用量', value: result.performance_metrics?.memory_usage_mb || result.validation_results?.memory_usage_mb, suffix: ' MB' },
                          { key: '7', metric: '模型VRAM', value: result.performance_metrics?.model_vram_mb || result.validation_results?.model_vram_mb, suffix: ' MB' },
                          { key: '8', metric: '平均GPU負載', value: result.performance_metrics?.avg_gpu_load || result.validation_results?.avg_gpu_load, suffix: ' %' },
                          { key: '9', metric: '峰值GPU負載', value: result.performance_metrics?.max_gpu_load || result.validation_results?.max_gpu_load, suffix: ' %' },
                          { key: '10', metric: '監控樣本數', value: result.performance_metrics?.monitoring_samples || result.validation_results?.monitoring_samples },
                          { key: '11', metric: '監控時間', value: result.performance_metrics?.monitoring_duration_s || result.validation_results?.monitoring_duration_s, suffix: ' 秒' },
                        ]}
                        pagination={false}
                        size="small"
                      />

                      <div style={{ marginTop: 16 }}>
                        <Text strong>所有推論時間 (ms): </Text>
                        <Text code>
                          {result.inference_results.all_inference_times?.map(t => t.toFixed(3)).join(', ')}
                        </Text>
                      </div>
                    </>
                  ) : (
                    <Alert message="無推論測試結果數據" type="warning" />
                  )}
                </TabPane>
              </Tabs>
            </Card>
          ))}
        </Card>
      )}
    </div>
  );
};

export default TestResultsPage; 