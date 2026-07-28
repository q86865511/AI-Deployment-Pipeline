import axios from 'axios';

// 後端 API 位址：建置期由 Vite 環境變數 VITE_API_URL 注入（`.env` 中設定），
// 未設定時退回本機開發預設值。需要組完整 URL 的場合（window.open、下載連結）用這個常數。
export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// 全站共用的 axios 實例；各頁面只傳相對路徑，例如 api.get('/api/models/')。
const api = axios.create({
  baseURL: API_BASE_URL,
});

export default api;
