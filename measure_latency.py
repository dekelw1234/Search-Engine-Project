import json
import requests
import time
import sys
from statistics import mean, stdev

# Configuration
ENGINE_URL = "http://127.0.0.1:8080/search"
GROUND_TRUTH_FILE = "queries_train.json"

# Number of warmup and measurement runs
WARMUP_RUNS = 1
MEASURE_RUNS = 3

# Load ground truth queries
with open(GROUND_TRUTH_FILE) as f:
    ground_truth = json.load(f)


def measure_single_query(query, warmup=True):
    """
    Measure the response time for a single query.

    Args:
        query: Query string
        warmup: Whether this is a warmup run (errors not reported)

    Returns:
        Response time in milliseconds, or None if request failed
    """
    start_time = time.time()

    try:
        response = requests.get(ENGINE_URL, params={"query": query}, timeout=10)
        response.raise_for_status()
        _ = response.json()
    except Exception as e:
        if not warmup:
            print(f"ERROR: {e}", file=sys.stderr)
        return None

    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000

    return latency_ms


def measure_latency(verbose=False):
    """
    Measure average response time across all queries.

    Args:
        verbose: Whether to print detailed progress information

    Returns:
        Dictionary containing latency statistics, or None if all queries failed
    """
    all_latencies = []
    queries = list(ground_truth.keys())

    if verbose:
        print(f"Performing {WARMUP_RUNS} warmup run(s)...")

    # Warmup runs to stabilize server/cache
    for _ in range(WARMUP_RUNS):
        for query in queries:
            measure_single_query(query, warmup=True)

    if verbose:
        print(f"Measuring {MEASURE_RUNS} run(s) for latency calculation...\n")

    # Actual measurement runs
    for run in range(MEASURE_RUNS):
        if verbose:
            print(f"Run {run + 1}/{MEASURE_RUNS}:")

        for query in queries:
            latency = measure_single_query(query, warmup=False)

            if latency is not None:
                all_latencies.append(latency)

                if verbose:
                    print(f"  '{query[:40]}...' -> {latency:.2f}ms")

    # Calculate statistics
    if not all_latencies:
        return None

    stats = {
        'mean': mean(all_latencies),
        'std': stdev(all_latencies) if len(all_latencies) > 1 else 0.0,
        'min': min(all_latencies),
        'max': max(all_latencies),
        'median': sorted(all_latencies)[len(all_latencies) // 2],
        'total_queries': len(all_latencies)
    }

    return stats


# Main execution
if __name__ == "__main__":
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    quiet = "--quiet" in sys.argv or "-q" in sys.argv

    if verbose:
        print(f"\n{'=' * 70}")
        print("Measuring average search engine response time")
        print(f"{'=' * 70}\n")

    stats = measure_latency(verbose=verbose)

    if stats is None:
        print("ERROR: Failed to measure any query successfully", file=sys.stderr)
        sys.exit(1)

    if quiet:
        # Quiet mode: output only the average latency value
        print(f"{stats['mean']:.2f}")
    else:
        # Normal mode: output formatted results
        if verbose:
            print(f"\n{'=' * 70}")
            print("Response Time Statistics")
            print(f"{'=' * 70}")

        print(f"Average Latency: {stats['mean']:.2f}ms")

        if verbose:
            print(f"Median Latency:  {stats['median']:.2f}ms")
            print(f"Std Dev:         {stats['std']:.2f}ms")
            print(f"Min Latency:     {stats['min']:.2f}ms")
            print(f"Max Latency:     {stats['max']:.2f}ms")
            print(f"Total Queries:   {stats['total_queries']}")
            print(f"{'=' * 70}\n")