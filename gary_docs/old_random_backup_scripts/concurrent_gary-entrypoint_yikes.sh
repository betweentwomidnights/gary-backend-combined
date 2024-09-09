#!/bin/bash

# Function to start the RQ worker
start_rq_worker() {
    rq worker --url redis://redis:6379/0 &
}

# Function to clear CUDA cache
clear_cuda_cache() {
    echo "Clearing CUDA cache..."
    python3.9 -c "import torch; torch.cuda.empty_cache(); torch.cuda.synchronize();"
}

# Function to start the main application with retries
start_app_with_retries() {
    MAX_RETRIES=3
    RETRY_COUNT=0

    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        # Clear CUDA cache before each retry
        clear_cuda_cache

        # Start the main application
        exec gunicorn --workers=4 --timeout=500 concurrent_gary:app --bind 0.0.0.0:8001
        EXIT_CODE=$?

        if [ $EXIT_CODE -ne 0 ]; then
            echo "Error encountered. Retrying..."
            RETRY_COUNT=$((RETRY_COUNT + 1))
        else
            break
        fi
    done

    if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
        echo "Exceeded maximum retries. Exiting..."
        exit 1
    fi
}

# Initialize CUDA and clear cache on startup
nvidia-smi
clear_cuda_cache

# Start the RQ worker in the background
start_rq_worker

# Start the main application with retries
start_app_with_retries