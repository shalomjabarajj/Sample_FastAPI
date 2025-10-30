#!/bin/bash
# Startup script for FastAPI with Uvicorn on RedHat Linux

# Load environment variables
export $(cat .env | xargs)

# Start uvicorn with production settings
uvicorn main:app \
    --host 0.0.0.0 \
    --port ${PORT:-8000} \
    --workers 4 \
    --log-level info \
    --access-log \
    --use-colors \
    --loop uvloop \
    --http httptools
