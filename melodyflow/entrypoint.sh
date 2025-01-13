#!/bin/bash
set -e

# Function to handle cleanup
cleanup() {
    echo "Received shutdown signal - cleaning up..."
    if [ -n "$child" ]; then
        echo "Sending SIGTERM to Python process ($child)"
        kill -TERM "$child" 2>/dev/null || true
        
        # Wait for the process to terminate gracefully
        wait "$child" 2>/dev/null || true
    fi
    exit 0
}

# Trap more signals and ensure cleanup runs
trap cleanup SIGTERM SIGINT SIGQUIT

echo "Starting MelodyFlow service..."

# Start the Flask application
python3 flask_melodyflow_2.py &

# Store the PID
child=$!

# Wait for the process to complete
# -1 means wait for any child process
wait -n

# If we get here, the Python process has ended
echo "MelodyFlow service stopped"