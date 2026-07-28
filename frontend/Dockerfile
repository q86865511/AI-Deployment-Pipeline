FROM sys-frontend-base:latest

# 後端 API 位址：Vite 只認 VITE_ 前綴，且需在建置 / dev server 啟動時就存在
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=${VITE_API_URL}

# 複製項目文件
COPY . .

# 暴露端口
EXPOSE 3000

# 啟動應用
CMD ["npm", "start"] 