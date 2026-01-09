#!/bin/bash

# Query Analysis Script
# This script runs the search engine with BALANCED_2_NO_PR configuration
# and performs detailed query analysis

echo "=================================="
echo "Search Engine Query Analysis"
echo "=================================="

# Set engine version
export ENGINE_VERSION=BALANCED_2_NO_PR
echo "Engine version: $ENGINE_VERSION"

# Start the search engine server
echo "Starting search engine server..."
python search_frontend.py > server.log 2>&1 &
SERVER_PID=$!

# Wait for server to start
echo "Waiting for server to initialize..."
sleep 5

# Check if server is running
if ! curl -s http://127.0.0.1:8080/search?query=test > /dev/null; then
    echo "ERROR: Server failed to start"
    exit 1
fi

echo "Server is running (PID: $SERVER_PID)"

# Run query analysis
echo ""
echo "Running query analysis..."
python analyze_queries.py

# Cleanup
echo ""
echo "Shutting down server..."
kill $SERVER_PID

echo "Analysis complete!"
```

**Remove these entirely** - they don't belong in a script file:
```
Query: "DNA double helix discovery"
Problem: Relevant docs use "deoxyribonucleic acid" instead of "DNA"
Solution: Query expansion with synonyms


