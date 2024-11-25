batch
@echo off
echo Checking Docker...
docker info >nul 2>&1
if errorlevel 1 (
  echo Error: Docker is not running
  pause
  exit /b 1
)

echo Starting RAG Application...
docker-compose up -d

echo Waiting for application to start...
timeout /t 5 /nobreak >nul

docker-compose ps | find "rag-app" | find "Up" >nul
if errorlevel 1 (
  echo Error: Application failed to start
  docker-compose logs
  pause
  exit /b 1
) else (
  echo Application is running!
  echo Access the application at: http://localhost:8501
  start http://localhost:8501
  pause
)