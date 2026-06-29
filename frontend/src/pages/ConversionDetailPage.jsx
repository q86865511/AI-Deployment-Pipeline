import React, { useState, useEffect, useRef } from 'react';
import { useParams } from 'react-router-dom';
import { Card, Descriptions, Button, Spin, message, Tag } from 'antd';
import { SyncOutlined } from '@ant-design/icons';
import axios from 'axios';

const ConversionDetailPage = () => {
  const { id } = useParams();
  const [job, setJob] = useState(null);
  const [loading, setLoading] = useState(true);
  const prevJobStatusRef = useRef(null);

  useEffect(() => {
    fetchJobDetails();
    const interval = setInterval(fetchJobDetails, 5000); // 每5秒刷新一次任務狀態
    return () => clearInterval(interval);
  }, [id]);

  // 檢查任務狀態變化，當任務完成時刷新模型列表
  useEffect(() => {
    if (prevJobStatusRef.current === 'processing' && job?.status === 'completed') {
      console.log('任務已完成，刷新模型列表...');
      refreshModelRepository();
    }
    
    // 更新前一個狀態的引用
    if (job?.status) {
      prevJobStatusRef.current = job.status;
    }
  }, [job?.status]);

  const fetchJobDetails = async () => {
    try {
      const response = await axios.get(`http://localhost:8000/api/conversion/${id}`);
      setJob(response.data);
    } catch (error) {
      console.error('獲取轉換任務詳情失敗:', error);
      message.error('獲取轉換任務詳情失敗');
    } finally {
      setLoading(false);
    }
  };

  // 刷新模型存儲庫
  const refreshModelRepository = async () => {
    try {
      console.log('開始刷新模型庫...');
      // 發出請求告知服務器重新掃描模型目錄
      const response = await axios.get('http://localhost:8000/api/models/refresh');
      console.log('模型庫刷新成功，返回：', response.data);
      message.success('已刷新模型庫，可以在模型列表中查看新模型');
    } catch (error) {
      console.error('刷新模型庫失敗:', error.response || error);
      // 即使失敗也不顯示錯誤消息，因為這只是一個輔助功能
      // 但仍然嘗試強制刷新模型列表
      try {
        await axios.get('http://localhost:8000/api/models/');
        console.log('雖然刷新失敗，但已獲取最新模型列表');
      } catch (e) {
        console.error('獲取模型列表也失敗:', e);
      }
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

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '50px' }}>
        <Spin size="large" />
      </div>
    );
  }

  if (!job) {
    return (
      <Card>
        <div style={{ textAlign: 'center' }}>
          找不到轉換任務信息
        </div>
      </Card>
    );
  }

  return (
    <Card
      title={`轉換任務詳情: ${id}`}
      extra={
        <Button
          danger
          onClick={async () => {
            try {
              await axios.delete(`http://localhost:8000/api/conversion/${id}`);
              message.success('刪除轉換任務成功');
              window.location.href = '/conversion';
            } catch (error) {
              console.error('刪除轉換任務失敗:', error);
              message.error('刪除轉換任務失敗');
            }
          }}
        >
          刪除任務
        </Button>
      }
    >
      <Descriptions bordered column={2}>
        <Descriptions.Item label="任務ID">{job.id}</Descriptions.Item>
        <Descriptions.Item label="狀態">{getStatusTag(job.status)}</Descriptions.Item>
        <Descriptions.Item label="源模型ID">{job.source_model_id}</Descriptions.Item>
        <Descriptions.Item label="目標格式">{job.target_format}</Descriptions.Item>
        <Descriptions.Item label="精度">{job.precision}</Descriptions.Item>
        <Descriptions.Item label="創建時間">{new Date(job.created_at).toLocaleString()}</Descriptions.Item>
        <Descriptions.Item label="完成時間" span={2}>
          {job.completed_at ? new Date(job.completed_at).toLocaleString() : '尚未完成'}
        </Descriptions.Item>
        {job.error_message && (
          <Descriptions.Item label="錯誤訊息" span={2}>
            <div style={{ color: 'red' }}>{job.error_message}</div>
          </Descriptions.Item>
        )}
        {job.target_model_id && (
          <Descriptions.Item label="目標模型ID" span={2}>
            <Button type="link" href={`/models/${job.target_model_id}`}>
              {job.target_model_id}
            </Button>
          </Descriptions.Item>
        )}
      </Descriptions>

      {job.parameters && (
        <Card title="轉換參數" style={{ marginTop: 16 }}>
          <pre>{JSON.stringify(job.parameters, null, 2)}</pre>
        </Card>
      )}
    </Card>
  );
};

export default ConversionDetailPage; 