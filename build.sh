#!/bin/bash

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
  echo "Error: Docker is not installed"
  exit 1
fi

# Create necessary directories
mkdir -p data

# Check for .env file
if [ ! -f .env ]; then
  echo "Error: .env file not found"
  echo "Creating example .env file..."
  echo "GOOGLE_API_KEY=your_api_key_here" > .env
  echo "Please edit .env file with your actual Google API key"
  exit 1
fi

# Build the Docker image
echo "Building Docker image..."
docker-compose build

# Check if build was successful
if [ $? -eq 0 ]; then
  echo "Build successful!"
  echo "To start the application, run: docker-compose up -d"
else
  echo "Build failed!"
  exit 1
fi