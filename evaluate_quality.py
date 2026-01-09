import json
import requests

ENGINE_URL = "http://127.0.0.1:8080/search"
K = 10   # נמדוד top-10

# טוענים את ה-ground truth
with open("queries_train.json") as f:
    ground_truth = json.load(f)

def average_precision(predicted, relevant):
    score = 0.0
    hits = 0
    for i, doc_id in enumerate(predicted, start=1):
        if doc_id in relevant:
            hits += 1
            score += hits / i
    return score / max(1, len(relevant))

def evaluate():
    aps = []

    for query, relevant_docs in ground_truth.items():
        r = requests.get(ENGINE_URL, params={"query": query})
        results = r.json()

        # לוקחים רק את ה-doc_id-ים של top-k
        predicted = [doc_id for doc_id, _ in results[:K]]

        ap = average_precision(predicted, set(relevant_docs))
        aps.append(ap)

    return sum(aps) / len(aps)

if __name__ == "__main__":
    map_score = evaluate()
    print("MAP@10:", map_score)


