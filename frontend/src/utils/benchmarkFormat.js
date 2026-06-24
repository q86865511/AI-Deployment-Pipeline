// Pure progress/status/formatting helpers extracted verbatim from BenchmarkPage
// (R-split, zero behavior change). Unit-tested in benchmarkFormat.test.js.

// 計算實際進度的函數
export const calculateActualProgress = (status) => {
  if (!status || !status.total_combinations) return { completed: 0, total: status?.total_combinations || 0, percent: 0 };

  const total = status.total_combinations;
  const currentIndex = status.current_combination_index >= 0 ? status.current_combination_index + 1 : 0;

  // 根據當前階段計算進度，每個階段都是獨立的0/total
  let completed = 0;
  let stageProgress = "";
  let totalPercent = 0;

  switch (status.current_step) {
    case 'conversion':
      // 轉換階段：顯示當前轉換進度
      completed = currentIndex;
      stageProgress = `轉換階段: ${completed}/${total}`;
      totalPercent = Math.round((completed / total) * 33.33); // 轉換階段佔33%
      break;
    case 'validation':
      // 驗證階段：顯示當前驗證進度
      completed = currentIndex;
      stageProgress = `驗證階段: ${completed}/${total}`;
      totalPercent = 33 + Math.round((completed / total) * 33.33); // 33% + 驗證階段33%
      break;
    case 'inference':
      // 推論階段：顯示當前推論進度
      completed = currentIndex;
      stageProgress = `推論階段: ${completed}/${total}`;
      totalPercent = 66 + Math.round((completed / total) * 34); // 66% + 推論階段34%
      break;
    default:
      completed = status.completed_combinations || 0;
      stageProgress = `${completed}/${total}`;
      totalPercent = status.total_combinations > 0 ? Math.round((completed / status.total_combinations) * 100) : 0;
  }

  return {
    completed,
    total,
    percent: Math.min(totalPercent, 100),
    stageProgress
  };
};

export const getStatusColor = (status) => {
  switch (status) {
    case 'pending': return 'blue';
    case 'processing': return 'orange';
    case 'completed': return 'green';
    case 'failed': return 'red';
    case 'aborted': return 'gray';
    default: return 'default';
  }
};

// 格式化日期時間
export const formatDateTime = (dateTimeStr) => {
  if (!dateTimeStr) return '';

  try {
    const date = new Date(dateTimeStr);
    return date.toLocaleString('zh-TW', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false
    });
  } catch (error) {
    return dateTimeStr;
  }
};

// 測試步驟名稱映射
export const stepNameMap = {
  'conversion': '模型轉換',
  'validation': '模型驗證',
  'inference': '推論測試'
};

// 狀態映射
export const statusNameMap = {
  'pending': '等待中',
  'processing': '處理中',
  'completed': '已完成',
  'failed': '失敗',
  'aborted': '已中止'
};
