#!/bin/sh
set -x  # Print every command for debugging

echo "=== START DEBUG ==="
echo "Shell: $SHELL"
echo "PATH: $PATH"
echo "PORT: ${PORT:-8501}"
echo "Python location:"
which python3 || which python || echo "Python NOT FOUND"
echo "=== END DEBUG ==="

# Use python3 explicitly
exec python3 /app/run.py
