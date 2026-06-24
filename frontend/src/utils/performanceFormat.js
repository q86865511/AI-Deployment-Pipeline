// Pure display/formatting helpers extracted verbatim from PerformanceAnalyzerPage
// (R-split, zero behavior change). Unit-tested in performanceFormat.test.js.

// 清理模型名稱，生成簡短標籤
export const cleanDisplayName = (name, batchSize = null) => {
  if (!name) return 'Unknown';

  // 清理模型名稱，生成簡短標籤
  const lowerName = name.toLowerCase();

  // 檢查是否為原始模型
  if (lowerName.includes('original') || !lowerName.includes('_engine_')) {
    // 對於原始模型，包含batch size信息
    const batch = batchSize || '1';
    return `原始模型_batch${batch}`;
  }

  // 提取精度和批次大小
  let precision = 'fp32';  // 預設值
  let extractedBatchSize = batchSize || '1';     // 預設值

  if (lowerName.includes('fp16')) {
    precision = 'fp16';
  } else if (lowerName.includes('int8')) {
    precision = 'int8';
  }

  // 提取批次大小
  const batchMatch = lowerName.match(/batch[_]?(\d+)/);
  if (batchMatch) {
    extractedBatchSize = batchMatch[1];
  }

  // 移除數字後綴（如 _0.83...）
  return `${precision}_${extractedBatchSize}`.replace(/_\d+\.\d+.*$/, '');
};

// 獲取模型標籤顏色
export const getModelColor = (modelType) => {
  // 只有原始模型batch1為紫色
  if (modelType === '原始模型_batch1') {
    return '#722ed1'; // 紫色
  }
  // 其他模型使用預設顏色
  const colors = ['#1890ff', '#52c41a', '#fa8c16', '#eb2f96', '#13c2c2', '#f5222d'];
  const hash = modelType.split('').reduce((a, b) => {
    a = ((a << 5) - a) + b.charCodeAt(0);
    return a & a;
  }, 0);
  return colors[Math.abs(hash) % colors.length];
};

// 清理GPU負載標籤（移除數字部分）
export const cleanGpuLabel = (label) => {
  // 移除類似 "0.83..." 的數字部分
  return label.replace(/\d+\.\d+.*$/, '').trim();
};
