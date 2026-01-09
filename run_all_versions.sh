#!/bin/bash

VERSIONS=("BASE_BODY_NO_PR" "BASE_BODY__PR" "BASE_TITLE_NO_PR" "BASE_TITLE_PR" "TITLE_60_NO_PR" "TITLE_60_PR" "BALANCED_2_NO_PR" "BALANCED_2_PR" "BODY_50_NO_PR" "BODY_50_PR" "PR_LOW_TITLE" "RECOMMENDED_1" "RECOMMENDED_2" )
RESULTS_FILE="results.csv"
K=10

echo "version,MAP@$K,latency_ms" > $RESULTS_FILE

wait_for_server() {
  echo "Waiting for server..."
  for i in {1..30}; do
    if curl -s http://127.0.0.1:8080/search?query=test > /dev/null 2>&1; then
      echo "Server is up!"
      return 0
    fi
    sleep 1
  done
  echo "Server did not start in time"
  return 1
}

for V in "${VERSIONS[@]}"; do
  echo ""
  echo "===================================="
  echo "Running version: $V"
  echo "===================================="

  echo "Killing previous server..."
  pkill -f search_frontend.py
  sleep 2

  echo "Starting server with ENGINE_VERSION=$V..."
  ENGINE_VERSION=$V nohup ~/venv/bin/python ~/search_frontend.py > server_$V.log 2>&1 &

  if ! wait_for_server; then
    echo "    Skipping version $V"
    continue
  fi

  echo "Measuring quality..."
  MAP=$(python evaluate_quality.py $K --quiet)

  if [ -z "$MAP" ]; then
    echo "Failed to measure MAP for $V"
    MAP="0.0000"
  else
    echo "   MAP@$K: $MAP"
  fi

  echo "    Measuring latency..."
  LAT=$(python measure_latency.py --quiet)

  if [ -z "$LAT" ]; then
    echo "Failed to measure latency for $V"
    LAT="999999"
  else
    echo "   Latency: ${LAT}ms"
  fi

  # 6. Save results
  echo "$V,$MAP,$LAT" | tee -a $RESULTS_FILE

  echo "Version $V completed"
done

# Cleanup
echo ""
echo "===================================="
echo "Cleaning up..."
pkill -f search_frontend.py

echo ""
echo "===================================="
echo "All versions finished!"
echo "===================================="
echo "Results saved in: $RESULTS_FILE"
echo ""

echo "Results Summary:"
echo "------------------------------------"
cat $RESULTS_FILE | column -t -s,

echo ""
echo "Best version by MAP@$K:"
tail -n +2 $RESULTS_FILE | sort -t, -k2 -rn | head -1 | awk -F, '{print "   " $1 " with MAP@'$K'=" $2}'

echo ""


