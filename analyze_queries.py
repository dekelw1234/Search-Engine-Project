import json
import requests
import pickle
from collections import defaultdict

# Configuration
ENGINE_URL = "http://127.0.0.1:8080/search"
GROUND_TRUTH_FILE = "queries_train.json"

# Load ground truth
with open(GROUND_TRUTH_FILE) as f:
    ground_truth = json.load(f)

# Load document titles
with open('id2title.pkl', 'rb') as f:
    id_to_title = pickle.load(f)


def calculate_ap_for_query(predicted, relevant, k=10):
    """
    Calculate Average Precision at K for a single query.

    Args:
        predicted: List of predicted document IDs (ordered by relevance)
        relevant: Set of relevant document IDs
        k: Number of results to consider

    Returns:
        Average Precision score (0.0 to 1.0)
    """
    predicted_k = predicted[:k]
    score = 0.0
    hits = 0

    for i, doc_id in enumerate(predicted_k, start=1):
        if doc_id in relevant:
            hits += 1
            score += hits / i

    return score / min(len(relevant), k) if relevant else 0.0


def analyze_all_queries():
    """
    Analyze all queries and compute performance metrics.

    Returns:
        List of dictionaries containing query results and metrics
    """
    results = []

    print("Analyzing all queries...")

    for query, relevant_docs in ground_truth.items():
        try:
            # Retrieve search results
            response = requests.get(ENGINE_URL, params={"query": query}, timeout=10)
            search_results = response.json()

            # Extract top-10 document IDs
            predicted = [doc_id for doc_id, _ in search_results[:10]]
            relevant_set = set(relevant_docs)

            # Calculate metrics
            ap = calculate_ap_for_query(predicted, relevant_set, k=10)
            hits = len(set(predicted) & relevant_set)
            precision = hits / 10
            recall = hits / len(relevant_set) if relevant_set else 0

            results.append({
                'query': query,
                'ap': ap,
                'hits': hits,
                'precision': precision,
                'recall': recall,
                'predicted': predicted,
                'relevant': relevant_docs
            })

        except Exception as e:
            print(f"Error processing query '{query}': {e}")
            continue

    # Sort by Average Precision (descending)
    results.sort(key=lambda x: x['ap'], reverse=True)

    return results


def print_query_analysis(query_data, rank, total):
    """
    Print detailed analysis for a single query.

    Args:
        query_data: Dictionary containing query results and metrics
        rank: Ranking position of this query
        total: Total number of queries
    """
    print(f"\n{'=' * 80}")
    print(f"Query #{rank}/{total}: \"{query_data['query']}\"")
    print(f"{'=' * 80}")
    print(f"Performance Metrics:")
    print(f"   AP@10:     {query_data['ap']:.4f} ({query_data['ap'] * 100:.2f}%)")
    print(f"   Precision: {query_data['precision']:.4f} ({query_data['hits']}/10 hits)")
    print(f"   Recall:    {query_data['recall']:.4f} ({query_data['hits']}/{len(query_data['relevant'])} found)")

    print(f"\nTop-10 Results:")
    print(f"{'Rank':<6} {'Doc ID':<12} {'Title':<60} {'Relevant?'}")
    print("-" * 80)

    for i, doc_id in enumerate(query_data['predicted'], 1):
        title = id_to_title.get(int(doc_id), "Unknown")[:57]
        is_relevant = "YES" if doc_id in query_data['relevant'] else "NO"
        print(f"{i:<6} {doc_id:<12} {title:<60} {is_relevant}")

    print(f"\nPerformance Assessment:")
    if query_data['ap'] > 0.7:
        print("   Excellent performance - most relevant documents in top positions")
    elif query_data['ap'] > 0.5:
        print("   Good performance - some relevant documents missing or ranked low")
    elif query_data['ap'] > 0.3:
        print("   Moderate performance - many relevant documents missing from top-10")
    else:
        print("   Poor performance - very few relevant documents found")


# Main execution
if __name__ == "__main__":
    print("\nStarting query analysis...")
    print("=" * 80)

    # Analyze all queries
    results = analyze_all_queries()

    # Display top 5 best performing queries
    print("\n" + "=" * 80)
    print("TOP 5 BEST PERFORMING QUERIES")
    print("=" * 80)

    for i, query_data in enumerate(results[:5], 1):
        print(f"{i}. {query_data['query']:<50} AP@10={query_data['ap']:.4f} ({query_data['hits']}/10 hits)")

    # Display top 5 worst performing queries
    print("\n" + "=" * 80)
    print("TOP 5 WORST PERFORMING QUERIES")
    print("=" * 80)

    for i, query_data in enumerate(results[-5:], 1):
        print(f"{i}. {query_data['query']:<50} AP@10={query_data['ap']:.4f} ({query_data['hits']}/10 hits)")

    # Detailed analysis of best query
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS - BEST PERFORMING QUERY")
    print_query_analysis(results[0], 1, len(results))

    # Detailed analysis of worst query
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS - WORST PERFORMING QUERY")
    print_query_analysis(results[-1], len(results), len(results))

    # Save results to JSON file
    with open('query_analysis.json', 'w') as f:
        json.dump([{
            'query': r['query'],
            'ap': r['ap'],
            'hits': r['hits'],
            'precision': r['precision'],
            'recall': r['recall'],
            'predicted_titles': [id_to_title.get(int(doc_id), "Unknown") for doc_id in r['predicted']],
            'predicted_ids': r['predicted'],
            'relevant_ids': r['relevant']
        } for r in results], f, indent=2)

    print("\n" + "=" * 80)
    print("Analysis complete! Detailed results saved to 'query_analysis.json'")
    print("=" * 80)