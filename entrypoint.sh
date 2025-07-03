#!/bin/sh
# entrypoint.sh

# Exit immediately if a command exits with a non-zero status
set -e

# Execute the gunicorn command
exec gunicorn -k geventwebsocket.gunicorn.workers.GeventWebSocketWorker \
    -w 5 \
    --worker-connections 1000 \
    -t 600 \
    --graceful-timeout 60 \
    --limit-request-line 0 \
    --max-requests 100 \
    --max-requests-jitter 50 \
    --limit-request-field_size 0 \
    g4lwebsockets:app --bind 0.0.0.0:8000
