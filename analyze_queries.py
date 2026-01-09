import json
import requests
from collections import defaultdict

# ====== הגדרות ======
ENGINE_URL = "http://127.0.0.1:8080/search"
GROUND_TRUTH_FILE = "queries_train.json"
ENGINE_VERSION = "BALANCED_2_NO_PR"  # הגרסה המומלצת שלך

# ====== טעינת Ground Truth ======
with open(GROUND_TRUTH_FILE) as f:
    ground_truth = json.load(f)

# ====== טעינת id_to_title ======
import pickle

with open('id2title.pkl', 'rb') as f:
    id_to_title = pickle.load(f)


def calculate_ap_for_query(predicted, relevant, k=10):
    """מחשב AP עבור שאילתה אחת"""
    predicted_k = predicted[:k]
    score = 0.0
    hits = 0

    for i, doc_id in enumerate(predicted_k, start=1):
        if doc_id in relevant:
            hits += 1
            score += hits / i

    return score / min(len(relevant), k) if relevant else 0.0


def analyze_all_queries():
    """מנתח את כל השאילתות ומוצא את הטובות והגרועות"""
    results = []

    print("🔍 Analyzing all queries...")

    for query, relevant_docs in ground_truth.items():
        try:
            # שליפת תוצאות
            response = requests.get(ENGINE_URL, params={"query": query}, timeout=10)
            search_results = response.json()

            # חילוץ top-10
            predicted = [doc_id for doc_id, _ in search_results[:10]]
            relevant_set = set(relevant_docs)

            # חישוב מדדים
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
            print(f"❌ Error with query '{query}': {e}")
            continue

    # מיון לפי AP
    results.sort(key=lambda x: x['ap'], reverse=True)

    return results


def print_query_analysis(query_data, rank):
    """מדפיס ניתוח מפורט של שאילתה"""
    print(f"\n{'=' * 80}")
    print(f"#{rank} Query: \"{query_data['query']}\"")
    print(f"{'=' * 80}")
    print(f"📊 Metrics:")
    print(f"   AP@10:     {query_data['ap']:.4f} ({query_data['ap'] * 100:.2f}%)")
    print(f"   Precision: {query_data['precision']:.4f} ({query_data['hits']}/10 hits)")
    print(f"   Recall:    {query_data['recall']:.4f} ({query_data['hits']}/{len(query_data['relevant'])} found)")

    print(f"\n📋 Top-10 Results:")
    print(f"{'Rank':<6} {'Doc ID':<12} {'Title':<60} {'Relevant?'}")
    print("-" * 80)

    for i, doc_id in enumerate(query_data['predicted'], 1):
        title = id_to_title.get(int(doc_id), "Unknown")[:57]
        is_relevant = "✅ YES" if doc_id in query_data['relevant'] else "❌ NO"
        print(f"{i:<6} {doc_id:<12} {title:<60} {is_relevant}")

    print(f"\n💡 Analysis:")
    if query_data['ap'] > 0.7:
        print("   ✅ Excellent performance! Most relevant docs in top positions.")
    elif query_data['ap'] > 0.5:
        print("   ⚠️  Good performance, but some relevant docs are missing or ranked low.")
    elif query_data['ap'] > 0.3:
        print("   ⚠️  Moderate performance, many relevant docs missing from top-10.")
    else:
        print("   ❌ Poor performance, very few relevant docs found.")


# ====== הרצה ======
if __name__ == "__main__":
    print(f"\n🚀 Starting analysis for ENGINE_VERSION={ENGINE_VERSION}")
    print("=" * 80)

    # ניתוח כל השאילתות
    results = analyze_all_queries()

    # הצגת 5 הטובות ביותר
    print("\n" + "=" * 80)
    print("🏆 TOP 5 BEST PERFORMING QUERIES")
    print("=" * 80)

    for i, query_data in enumerate(results[:5], 1):
        print(f"{i}. {query_data['query']:<50} AP@10={query_data['ap']:.4f} ({query_data['hits']}/10 hits)")

    # הצגת 5 הגרועות ביותר
    print("\n" + "=" * 80)
    print("💩 TOP 5 WORST PERFORMING QUERIES")
    print("=" * 80)

    for i, query_data in enumerate(results[-5:], 1):
        print(f"{i}. {query_data['query']:<50} AP@10={query_data['ap']:.4f} ({query_data['hits']}/10 hits)")

    # ניתוח מפורט של הטובה ביותר
    print("\n" + "=" * 80)
    print("🎯 DETAILED ANALYSIS - BEST QUERY")
    print_query_analysis(results[0], 1)

    # ניתוח מפורט של הגרועה ביותר
    print("\n" + "=" * 80)
    print("🔍 DETAILED ANALYSIS - WORST QUERY")
    print_query_analysis(results[-1], len(results))

    # שמירת תוצאות לקובץ
    import json

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
    print("✅ Analysis complete! Detailed results saved to 'query_analysis.json'")
    print("=" * 80)