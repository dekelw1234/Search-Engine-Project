import json
import requests
import sys

ENGINE_URL = "http://127.0.0.1:8080/search"
GROUND_TRUTH_FILE = "queries_train.json"

K = 10
for arg in sys.argv[1:]:
    if arg.isdigit():
        K = int(arg)

with open(GROUND_TRUTH_FILE) as f:
    ground_truth = json.load(f)


def calculate_ap_for_query(predicted, relevant, k):
    predicted_k = predicted[:k]

    score = 0.0
    hits = 0

    for i, doc_id in enumerate(predicted_k, start=1):
        if doc_id in relevant:
            hits += 1
            precision_at_i = hits / i
            score += precision_at_i

    if len(relevant) == 0:
        return 0.0

    return score / min(len(relevant), k)


def evaluate_quality(verbose=False):
    ap_scores = []

    for query, relevant_docs in ground_truth.items():
        try:
            response = requests.get(ENGINE_URL, params={"query": query}, timeout=10)
            response.raise_for_status()
            results = response.json()

            predicted = [doc_id for doc_id, _ in results]
            relevant_set = set(relevant_docs)

            ap = calculate_ap_for_query(predicted, relevant_set, K)
            ap_scores.append(ap)

            if verbose:
                hits = len(set(predicted[:K]) & relevant_set)
                print(f"Query: '{query[:40]}...' | AP@{K}={ap:.4f} | Hits={hits}/{K}")

        except Exception as e:
            if verbose:
                print(f"ERROR in query '{query}': {e}")
            continue

    if not ap_scores:
        return 0.0

    map_score = sum(ap_scores) / len(ap_scores)
    return map_score


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    quiet = "--quiet" in sys.argv or "-q" in sys.argv

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"{'=' * 70}\n")

    map_score = evaluate_quality(verbose=verbose)

    if quiet:
        print(f"{map_score:.4f}")
    else:
        if verbose:
            print(f"\n{'=' * 70}")
        print(f"MAP@{K}: {map_score:.4f} ({map_score * 100:.2f}%)")
        if verbose:
            print(f"{'=' * 70}\n")