#!/bin/bash

# Start the RQ worker in the background
rq worker --url redis://redis:6379/0 &

# Start the main application
exec gunicorn --workers=4 --timeout=500 concurrent_gary:app --bind 0.0.0.0:8001
