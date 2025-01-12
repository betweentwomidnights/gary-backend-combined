#!/bin/bash
set -e

# Function to handle cleanup
cleanup() {
    echo "Received shutdown signal - cleaning up..."
    # Kill the Python process if it's still running
    kill -TERM "$child" 2>/dev/null
}

# Trap SIGTERM and SIGINT
trap cleanup TERM INT

# Start the Flask application
python3 flask_melodyflow_2.py &

# Store the PID
child=$!

# Wait for the process to complete
wait "$child"