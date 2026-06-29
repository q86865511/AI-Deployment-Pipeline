import React, { useState } from 'react';
import { Card, Form, Input, Button, Select, Divider, Switch, message } from 'antd';

const { Option } = Select;

const SettingsPage = () => {
  const [form] = Form.useForm();
  const [saving, setSaving] = useState(false);

  const handleSave = (values) => {
    setSaving(true);
    // 這裡只是模擬儲存設定
    setTimeout(() => {
      console.log('儲存設定:', values);
      message.success('設定已儲存');
      setSaving(false);
    }, 1000);
  };

  return (
    <div>
      <Card title="系統設置">
        <Form
          form={form}
          layout="vertical"
          onFinish={handleSave}
          initialValues={{
            api_url: 'http://localhost:8000',
            default_precision: 'fp16',
            default_batch_size: '1',
            default_workspace: '4',
            enable_tensorrt: true,
            enable_onnx: true,
          }}
        >
          <Divider orientation="left">API設置</Divider>
          
          <Form.Item
            name="api_url"
            label="API網址"
            rules={[{ required: true, message: '請輸入API網址' }]}
          >
            <Input placeholder="例如：http://localhost:8000" />
          </Form.Item>

          <Divider orientation="left">預設轉換設置</Divider>
          
          <Form.Item
            name="default_precision"
            label="預設精度"
            rules={[{ required: true, message: '請選擇預設精度' }]}
          >
            <Select placeholder="選擇預設精度">
              <Option value="fp32">FP32</Option>
              <Option value="fp16">FP16</Option>
              <Option value="int8">INT8</Option>
            </Select>
          </Form.Item>
          
          <Form.Item
            name="default_batch_size"
            label="預設批次大小"
            rules={[{ required: true, message: '請選擇預設批次大小' }]}
          >
            <Select placeholder="選擇預設批次大小">
              <Option value="1">1</Option>
              <Option value="2">2</Option>
              <Option value="4">4</Option>
              <Option value="8">8</Option>
            </Select>
          </Form.Item>
          
          <Form.Item
            name="default_workspace"
            label="預設TensorRT工作區大小 (GB)"
            rules={[{ required: true, message: '請選擇預設工作區大小' }]}
          >
            <Select placeholder="選擇預設工作區大小">
              <Option value="2">2</Option>
              <Option value="4">4</Option>
              <Option value="8">8</Option>
              <Option value="16">16</Option>
            </Select>
          </Form.Item>

          <Divider orientation="left">功能開關</Divider>
          
          <Form.Item
            name="enable_tensorrt"
            label="啟用TensorRT轉換"
            valuePropName="checked"
          >
            <Switch />
          </Form.Item>
          
          <Form.Item
            name="enable_onnx"
            label="啟用ONNX轉換"
            valuePropName="checked"
          >
            <Switch />
          </Form.Item>
          
          <Form.Item>
            <Button type="primary" htmlType="submit" loading={saving}>
              保存設置
            </Button>
          </Form.Item>
        </Form>
      </Card>
    </div>
  );
};

export default SettingsPage; 