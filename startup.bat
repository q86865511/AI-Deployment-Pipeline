@echo off
echo AI Deployment Platform - Quick Start
echo ====================================

REM Check Docker
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker not found
    echo Please install Docker Desktop
    pause
    exit
)

docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not running
    echo Please start Docker Desktop
    pause
    exit
)

echo.
echo Step 1: Building base images...
echo -------------------------------
docker-compose build backend-base frontend-base

echo.
echo Step 2: Building application images...
echo -------------------------------------
docker-compose build backend frontend

echo.
echo Step 3: Starting all services...
echo -------------------------------
docker-compose up -d

echo.
echo Waiting for services to start...
timeout /t 15 /nobreak >nul

echo.
echo ====================================
echo System is ready!
echo ====================================
echo Frontend: http://localhost:3000
echo Backend: http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo Grafana: http://localhost:3001
echo ====================================
echo.

start http://localhost:3000

echo Press any key to stop services...
pause >nul

echo.
echo Stopping services...
docker-compose down

pause 