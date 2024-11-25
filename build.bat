batch
@echo off
echo Checking Docker installation...
docker --version >nul 2>&1
if errorlevel 1 (
  echo Error: Docker is not installed
  pause
  exit /b 1
)

mkdir data 2>nul

if not exist .env (
  echo Error: .env file not found
  echo Creating example .env file...
  echo GOOGLE_API_KEY=your_api_key_here > .env
  echo Please edit .env file with your actual Google API key
  pause
  exit /b 1
)

echo Building Docker image...
docker-compose build

if errorlevel 1 (
  echo Build failed!
  pause
  exit /b 1
) else (
  echo Build successful!
  echo To start the application, run: run.bat
  pause
)