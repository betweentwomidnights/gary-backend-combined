#!/bin/sh
# entrypoint.sh

# Exit immediately if a command exits with a non-zero status
set -e

# Execute the gunicorn command
exec gunicorn -k geventwebsocket.gunicorn.workers.GeventWebSocketWorker -w 4 -t 300 --graceful-timeout 30 g4lwebsockets:app --bind 0.0.0.0:8000
