import json
import requests
import pickle

# ====== טעינת נתונים ======
ENGINE_URL = "http://127.0.0.1:8080/search"

with open("queries_train.json") as f:
    ground_truth = json.load(f)

with open('id2title.pkl', 'rb') as f:
    id_to_title = pickle.load(f)


def deep_dive_analysis(query):
    """ניתוח עומק של שאילתה אחת"""

    if query not in ground_truth:
        print(f"❌ Query '{query}' not found in ground truth!")
        return

    relevant_docs = set(ground_truth[query])

    # שליפת תוצאות מהמנוע
    response = requests.get(ENGINE_URL, params={"query": query})
    search_results = response.json()

    # חילוץ top-10 ו-top-100
    top_10 = [doc_id for doc_id, _ in search_results[:10]]
    top_100 = [doc_id for doc_id, _ in search_results[:100]]

    # חישוב מדדים
    hits_10 = len(set(top_10) & relevant_docs)
    hits_100 = len(set(top_100) & relevant_docs)

    print(f"\n{'=' * 100}")
    print(f"🔍 DEEP DIVE ANALYSIS: \"{query}\"")
    print(f"{'=' * 100}")

    print(f"\n📊 Summary:")
    print(f"   Relevant documents in ground truth: {len(relevant_docs)}")
    print(f"   Hits in top-10:  {hits_10}/10 ({hits_10 / 10 * 100:.1f}%)")
    print(f"   Hits in top-100: {hits_100}/100 ({hits_100 / len(relevant_docs) * 100:.1f}% of all relevant)")
    print(f"   Missing from top-100: {len(relevant_docs) - hits_100}")

    print(f"\n📋 Top-10 Results with Analysis:")
    print(f"{'Rank':<6} {'Doc ID':<12} {'Relevant?':<12} {'Title'}")
    print("-" * 100)

    for i, doc_id in enumerate(top_10, 1):
        title = id_to_title.get(int(doc_id), "Unknown")
        is_relevant = "✅ YES" if doc_id in relevant_docs else "❌ NO"
        print(f"{i:<6} {doc_id:<12} {is_relevant:<12} {title}")

    # ניתוח מסמכים רלוונטיים שלא נמצאו ב-top-10
    missing_from_top10 = relevant_docs - set(top_10)

    if missing_from_top10:
        print(f"\n⚠️  Relevant documents MISSING from top-10:")
        print(f"{'Doc ID':<12} {'Position':<10} {'Title'}")
        print("-" * 100)

        for doc_id in list(missing_from_top10)[:10]:  # הצג עד 10 ראשונים
            # מצא את המיקום של המסמך (אם הוא ב-top-100)
            try:
                pos = top_100.index(doc_id) + 1
                position = f"#{pos}"
            except ValueError:
                position = ">100"

            title = id_to_title.get(int(doc_id), "Unknown")
            print(f"{doc_id:<12} {position:<10} {title}")

    # ניתוח מסמכים לא רלוונטיים ב-top-10
    false_positives = set(top_10) - relevant_docs

    if false_positives:
        print(f"\n❌ Non-relevant documents in top-10 (False Positives):")
        print(f"{'Rank':<6} {'Doc ID':<12} {'Title'}")
        print("-" * 100)

        for i, doc_id in enumerate(top_10, 1):
            if doc_id in false_positives:
                title = id_to_title.get(int(doc_id), "Unknown")
                print(f"{i:<6} {doc_id:<12} {title}")

    print(f"\n{'=' * 100}\n")


# ====== דוגמאות שימוש ======
if __name__ == "__main__":
    # בחר שאילתות לניתוח

    # דוגמה 1: שאילתה שעובדת טוב (תחליף בשאילתה אמיתית מהניתוח)
    print("🎯 ANALYSIS 1: Well-Performing Query")
    deep_dive_analysis("Fossil fuels climate change")

    # דוגמה 2: שאילתה שעובדת גרוע (תחליף בשאילתה אמיתית מהניתוח)
    print("🎯 ANALYSIS 2: Poorly-Performing Query")
    deep_dive_analysis("Printing press invention Gutenberg")