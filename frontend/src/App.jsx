import React, { useState, useEffect } from 'react';
import { Route, Routes, Link, useLocation } from 'react-router-dom';
import { Layout, Menu, Typography, ConfigProvider } from 'antd';
import zhTW from 'antd/locale/zh_TW';
import {
  HomeOutlined,
  CloudUploadOutlined,
  DashboardOutlined,
  FundViewOutlined,
  BarChartOutlined,
  FileSearchOutlined,
  MonitorOutlined
} from '@ant-design/icons';
import './App.css';

// 導入頁面組件
import HomePage from './pages/HomePage';
import ModelsPage from './pages/ModelsPage';
import ModelDetailPage from './pages/ModelDetailPage';
import ConversionPage from './pages/ConversionPage';
import ConversionDetailPage from './pages/ConversionDetailPage';
import BenchmarkPage from './pages/BenchmarkPage';
import PerformanceAnalyzerPage from './pages/PerformanceAnalyzerPage';
import TestResultsPage from './pages/TestResultsPage';
import DeploymentMonitorPage from './pages/DeploymentMonitorPage';

const { Header, Sider, Content, Footer } = Layout;
const { Title } = Typography;

function App() {
  const [collapsed, setCollapsed] = useState(false);
  const [selectedKeys, setSelectedKeys] = useState(['1']);
  const location = useLocation();

  // 根據當前路由設置選中的菜單項
  useEffect(() => {
    const pathname = location.pathname;
    if (pathname === '/') {
      setSelectedKeys(['1']);
    } else if (pathname === '/models') {
      setSelectedKeys(['2']);
    } else if (pathname === '/upload') {
      setSelectedKeys(['3']);
    } else if (pathname === '/benchmark') {
      setSelectedKeys(['4']);
    } else if (pathname.startsWith('/test-results')) {
      setSelectedKeys(['5']);
    } else if (pathname.startsWith('/performance')) {
      setSelectedKeys(['6']);
    } else if (pathname.startsWith('/deployment-monitor')) {
      setSelectedKeys(['7']);
    }
  }, [location.pathname]);

  const toggleCollapsed = () => {
    setCollapsed(!collapsed);
  };

  return (
    <ConfigProvider locale={zhTW}>
      <Layout style={{ minHeight: '100vh' }}>
        <Sider 
          collapsible 
          collapsed={collapsed} 
          onCollapse={toggleCollapsed}
          theme="dark"
        >
            <div style={{ height: 64, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Title level={4} style={{ color: 'white', margin: 0 }}>
                {!collapsed ? 'AI部署平台' : 'AI'}
              </Title>
            </div>
            <Menu
              theme="dark"
              selectedKeys={selectedKeys}
              mode="inline"
              items={[
                {
                  key: '1',
                  icon: <HomeOutlined />,
                  label: <Link to="/">首頁</Link>,
                },
                {
                  key: '2',
                  icon: <DashboardOutlined />,
                  label: <Link to="/models">模型管理</Link>,
                },
                {
                  key: '3',
                  icon: <CloudUploadOutlined />,
                  label: <Link to="/upload">模型優化</Link>,
                },
                {
                  key: '4',
                  icon: <FundViewOutlined />,
                  label: <Link to="/benchmark">自動化管線</Link>,
                },
                {
                  key: '5',
                  icon: <FileSearchOutlined />,
                  label: <Link to="/test-results">測試結果查看</Link>,
                },
                {
                  key: '6',
                  icon: <BarChartOutlined />,
                  label: <Link to="/performance-analyzer">自動化結果分析</Link>,
                },
                {
                  key: '7',
                  icon: <MonitorOutlined />,
                  label: <Link to="/deployment-monitor">部署平台監控</Link>,
                }
              ]}
            />
          </Sider>
          <Layout>
            <Header style={{ 
              background: '#fff', 
              padding: '0 24px',
              boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
              display: 'flex',
              alignItems: 'center'
            }}>
              <Title level={3} style={{ margin: 0 }}>
                具自動化模型優化評估和即時推論資源監控之AI部署平台
              </Title>
            </Header>
            <Content style={{ margin: '24px 16px' }}>
              <div style={{ padding: 24, background: '#fff', minHeight: 360 }}>
                <Routes>
                  <Route path="/" element={<HomePage />} />
                  <Route path="/upload" element={<ConversionPage />} />
                  <Route path="/models" element={<ModelsPage />} />
                  <Route path="/models/:id" element={<ModelDetailPage />} />
                  <Route path="/conversion/:id" element={<ConversionDetailPage />} />
                  <Route path="/benchmark" element={<BenchmarkPage />} />
                  <Route path="/performance-analyzer" element={<PerformanceAnalyzerPage />} />
                  <Route path="/performance/:taskId" element={<PerformanceAnalyzerPage />} />
                  <Route path="/test-results" element={<TestResultsPage />} />
                  <Route path="/test-results/:taskId" element={<TestResultsPage />} />
                  <Route path="/deployment-monitor" element={<DeploymentMonitorPage />} />
                </Routes>
              </div>
            </Content>
            <Footer style={{ textAlign: 'center' }}>
              AI部署平台 ©2024
            </Footer>
          </Layout>
        </Layout>
      </ConfigProvider>
    );
}

export default App; 