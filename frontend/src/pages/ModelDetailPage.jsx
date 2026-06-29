import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { Card, Descriptions, Button, Spin, message } from 'antd';
import { DownloadOutlined, DeleteOutlined, SwapOutlined } from '@ant-design/icons';
import axios from 'axios';

const ModelDetailPage = () => {
  const { id } = useParams();
  const [model, setModel] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchModelDetails();
  }, [id]);

  const fetchModelDetails = async () => {
    try {
      const response = await axios.get(`http://localhost:8000/api/models/${id}`);
      setModel(response.data);
    } catch (error) {
      console.error('獲取模型詳情失敗:', error);
      message.error('獲取模型詳情失敗');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '50px' }}>
        <Spin size="large" />
      </div>
    );
  }

  if (!model) {
    return (
      <Card>
        <div style={{ textAlign: 'center' }}>
          找不到模型信息
        </div>
      </Card>
    );
  }

  return (
    <Card
      title={`模型詳情: ${model.name}`}
      extra={
        <div>
          <Button
            type="primary"
            icon={<DownloadOutlined />}
            style={{ marginRight: 8 }}
            onClick={() => window.open(`http://localhost:8000/api/models/${id}/download`)}
          >
            下載
          </Button>
          <Button
            type="primary"
            icon={<SwapOutlined />}
            style={{ marginRight: 8 }}
            onClick={() => window.location.href = `/conversion?model_id=${id}`}
          >
            轉換
          </Button>
          <Button
            danger
            icon={<DeleteOutlined />}
            onClick={async () => {
              try {
                await axios.delete(`http://localhost:8000/api/models/${id}`);
                message.success('刪除模型成功');
                window.location.href = '/models';
              } catch (error) {
                console.error('刪除模型失敗:', error);
                message.error('刪除模型失敗');
              }
            }}
          >
            刪除
          </Button>
        </div>
      }
    >
      <Descriptions bordered column={2}>
        <Descriptions.Item label="模型ID">{model.id}</Descriptions.Item>
        <Descriptions.Item label="模型名稱">{model.name}</Descriptions.Item>
        <Descriptions.Item label="模型類型">{model.type}</Descriptions.Item>
        <Descriptions.Item label="模型格式">{model.format}</Descriptions.Item>
        <Descriptions.Item label="文件大小">{model.size_mb} MB</Descriptions.Item>
        <Descriptions.Item label="創建時間">{new Date(model.created_at).toLocaleString()}</Descriptions.Item>
        <Descriptions.Item label="文件路徑" span={2}>{model.path}</Descriptions.Item>
        <Descriptions.Item label="描述" span={2}>{model.description || '無描述'}</Descriptions.Item>
      </Descriptions>

      {model.metadata && (
        <Card title="模型元數據" style={{ marginTop: 16 }}>
          <pre>{JSON.stringify(model.metadata, null, 2)}</pre>
        </Card>
      )}
    </Card>
  );
};

export default ModelDetailPage; 