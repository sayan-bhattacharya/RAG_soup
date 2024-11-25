#!/bin/bash

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
  echo "Error: Docker is not running"
  exit 1
fi

# Start the container
echo "Starting RAG Application..."
docker-compose up -d

# Wait for application to start
echo "Waiting for application to start..."
sleep 5

# Check if container is running
if docker-compose ps | grep "rag-app" | grep "Up" >/dev/null; then
  echo "Application is running!"
  echo "Access the application at: http://localhost:8501"
  open http://localhost:8501
else
  echo "Error: Application failed to start"
  docker-compose logs
  exit 1
fi