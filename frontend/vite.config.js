import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// CRA → Vite。含 JSX 的原始碼已改用 .jsx 副檔名，故設定保持精簡。
export default defineConfig({
  plugins: [react()],
  server: { port: 3000, host: true },
  preview: { port: 3000, host: true },
  build: { outDir: 'build' }, // 沿用 CRA 的輸出目錄名，下游 Docker/部署不需改動。
  // Vitest：沿用 jest 風格的全域 describe/test/expect。
  test: { globals: true, environment: 'node' },
});
