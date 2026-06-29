import React, { useState, useEffect } from 'react';
import { Typography, Card, Row, Col, Statistic } from 'antd';
import { DatabaseOutlined, SwapOutlined, LineChartOutlined } from '@ant-design/icons';
import axios from 'axios';

const { Title, Paragraph } = Typography;

const HomePage = () => {
  const [stats, setStats] = useState({
    modelCount: 0,
    conversionCount: 0,
    testCount: 0
  });

  useEffect(() => {
    fetchStatistics();
  }, []);

  const fetchStatistics = async () => {
    try {
      // 獲取模型數量
      const modelsResponse = await axios.get('http://localhost:8000/api/models/');
      
      // 獲取轉換任務數量
      const conversionsResponse = await axios.get('http://localhost:8000/api/conversion/');
      
      // 獲取自動化管線任務數量
      const pipelineResponse = await axios.get('http://localhost:8000/api/benchmark/tasks');
      
      setStats({
        modelCount: modelsResponse.data.total || 0,
        conversionCount: conversionsResponse.data.total || 0,
        testCount: pipelineResponse.data.tasks ? pipelineResponse.data.tasks.length : 0
      });
    } catch (error) {
      console.error('獲取統計數據失敗:', error);
    }
  };

  return (
    <div>
      <Typography>
        <Title level={2}>AI部署平台</Title>
        <Paragraph>
          歡迎使用具自動化模型優化評估和即時推論資源監控之AI部署平台。本系統提供完整的AI模型部署、轉換、測試和監控功能。
        </Paragraph>
      </Typography>
      
      <Row gutter={16} style={{ marginTop: 24 }}>
        <Col span={8}>
          <Card>
            <Statistic 
              title="模型倉庫" 
              value={stats.modelCount} 
              prefix={<DatabaseOutlined />} 
              suffix="個模型"
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card>
            <Statistic 
              title="轉換任務" 
              value={stats.conversionCount} 
              prefix={<SwapOutlined />} 
              suffix="個任務"
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card>
            <Statistic 
              title="自動化管線任務" 
              value={stats.testCount} 
              prefix={<LineChartOutlined />} 
              suffix="個"
            />
          </Card>
        </Col>
      </Row>
      
      <Card style={{ marginTop: 24 }}>
        <Title level={3}>開始使用</Title>
        <Paragraph>
          1. 在<strong>模型管理</strong>頁面上傳您的AI模型
        </Paragraph>
        <Paragraph>
          2. 在<strong>上傳模型</strong>頁面將模型轉換為ONNX或TensorRT格式
        </Paragraph>
        <Paragraph>
          3. 在<strong>自動化管線</strong>頁面執行批次轉換和自動化模型性能評估
        </Paragraph>
        <Paragraph>
          4. 在<strong>自動化結果分析</strong>頁面分析測試結果與性能優化
        </Paragraph>
      </Card>
    </div>
  );
};

export default HomePage; 