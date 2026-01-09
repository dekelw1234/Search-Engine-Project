from flask import Flask, request, jsonify
import pickle
from inverted_index_gcp import InvertedIndex
import nltk
from nltk.corpus import stopwords
import re
import math
from collections import Counter
import heapq
import os


ENGINE_VERSION = os.getenv("ENGINE_VERSION", "BALANCED_2_NO_PR")
WEIGHT_CONFIGS = {

    "BASE_BODY_NO_PR": {"title": 0.0, "body": 1.0, "anchor": 0.0, "use_pagerank": False},
    "BASE_BODY__PR": {"title": 0.0, "body": 1.0, "anchor": 0.0, "use_pagerank": True, "pagerank_alpha": 0.02},
    "BASE_TITLE_NO_PR": {"title": 1.0, "body": 0.0, "anchor": 0.0, "use_pagerank": False},
    "BASE_TITLE_PR": {"title": 1.0, "body": 0.0, "anchor": 0.0, "use_pagerank": True, "pagerank_alpha": 0.02},
    "TITLE_60_NO_PR": {"title": 0.6, "body": 0.3, "anchor": 0.1, "use_pagerank": False},
    "TITLE_60_PR": {"title": 0.6, "body": 0.3, "anchor": 0.1, "use_pagerank": True, "pagerank_alpha": 0.02},
    "BALANCED_2_NO_PR": {"title": 0.45, "body": 0.35, "anchor": 0.2, "use_pagerank": False},
    "BALANCED_2_PR": {"title": 0.45, "body": 0.35, "anchor": 0.2, "use_pagerank": True, "pagerank_alpha": 0.02},
    "BODY_50_NO_PR": {"title": 0.3, "body": 0.5, "anchor": 0.2, "use_pagerank": False},
    "BODY_50_PR": {"title": 0.3, "body": 0.5, "anchor": 0.2, "use_pagerank": True, "pagerank_alpha": 0.02},
    "PR_LOW_TITLE": {"title": 0.5, "body": 0.3, "anchor": 0.2, "use_pagerank": True, "pagerank_alpha": 0.02},
    "RECOMMENDED_1": {"title": 0.5, "body": 0.3, "anchor": 0.2, "use_pagerank": True, "pagerank_alpha": 0.1},
    "RECOMMENDED_2": {"title": 0.45, "body": 0.35, "anchor": 0.2, "use_pagerank": True, "pagerank_alpha": 0.08},
}
print("Running engine version:", ENGINE_VERSION)

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# --- 1. Load Data ---
print("Loading indices and metadata... please wait")

# Load Indices
with open('postings_gcp/body_index.pkl', 'rb') as f:
    body_index = pickle.load(f)
with open('postings_gcp/title_index.pkl', 'rb') as f:
    title_index = pickle.load(f)
with open('postings_gcp/anchor_index.pkl', 'rb') as f:
    anchor_index = pickle.load(f)

# Load PageRank
with open('pagerank.pkl', 'rb') as f:
    pagerank_dict = pickle.load(f)

# Load Titles
with open('id2title.pkl', 'rb') as f:
    id_to_title = pickle.load(f)

# Pre-calculate boosting
global_boost_dict = {}
all_relevant_ids = set(pagerank_dict.keys())
for doc_id in all_relevant_ids:
    pr = pagerank_dict.get(doc_id, 0)
    global_boost_dict[doc_id] = 1 + math.log10(pr + 1)

# Avg Body Length (Fallback)
AVG_BODY_LEN = 500

# --- 2. Tokenizer & Setup ---
nltk.download('stopwords')
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]
all_stopwords = english_stopwords.union(corpus_stopwords)

RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)


def tokenize(text):
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    return [t for t in tokens if t not in all_stopwords]


# --- 3. Ranking Functions (Fixed) ---

def read_posting_list(index, term):
    """
    Helper to safely read posting lists handling function name mismatches.
    Tries 'read_a_posting_list' (v2) first, then 'read_posting_list' (v1).
    """
    try:
        # Try the new signature: (base_dir, term)
        return index.read_a_posting_list('postings_gcp/', term)
    except AttributeError:
        # Fallback to old signature: (term, base_dir)
        return index.read_posting_list(term, 'postings_gcp/')
    except Exception:
        return []


def get_bm25_scores(query_tokens, index, k1=1.5, b=0.75):
    scores = {}
    # Safe N calculation
    N = len(index.DL) if hasattr(index, 'DL') else len(pagerank_dict)

    for term in query_tokens:
        if term in index.df:
            df = index.df[term]
            idf = math.log10((N - df + 0.5) / (df + 0.5) + 1)

            postings = read_posting_list(index, term)

            for doc_id, tf in postings:
                # Safe DL calculation
                doc_len = index.DL.get(doc_id, AVG_BODY_LEN) if hasattr(index, 'DL') else AVG_BODY_LEN

                denominator = tf + k1 * (1 - b + b * (doc_len / AVG_BODY_LEN))
                scores[doc_id] = scores.get(doc_id, 0) + (idf * tf * (k1 + 1) / denominator)
    return scores


def get_title_scores(query_tokens, index):
    scores = {}
    for term in set(query_tokens):
        if term in index.df:
            postings = read_posting_list(index, term)
            for doc_id, tf in postings:
                scores[doc_id] = scores.get(doc_id, 0) + 1
    return scores


def get_body_scores(query_tokens, index):
    scores = {}
    query_counts = Counter(query_tokens)

    # Calculate N safely for IDF
    N = len(index.DL) if hasattr(index, 'DL') else len(pagerank_dict)

    query_norm_sq = 0
    for term, tf_q in query_counts.items():
        if term in index.df:
            df = index.df[term]
            idf = math.log10(N / df)  # Classic IDF
            w_t_q = tf_q * idf
            query_norm_sq += w_t_q ** 2

            postings = read_posting_list(index, term)

            for doc_id, tf_d in postings:
                w_t_d = tf_d * idf
                scores[doc_id] = scores.get(doc_id, 0) + (w_t_q * w_t_d)

    if not scores: return {}

    query_norm = math.sqrt(query_norm_sq)
    results = {}
    for doc_id, dot_product in scores.items():
        norm_d = index.doc_norms.get(doc_id, 1)
        results[doc_id] = dot_product / (query_norm * norm_d)
    return results


# --- 4. Routes ---




def rank_with_weights(query_tokens):
    cfg = WEIGHT_CONFIGS.get(ENGINE_VERSION, WEIGHT_CONFIGS["BALANCED_2_NO_PR"])

    body_scores = get_bm25_scores(query_tokens, body_index)
    title_scores = get_title_scores(query_tokens, title_index)
    anchor_scores = get_title_scores(query_tokens, anchor_index)

    all_docs = set(body_scores) | set(title_scores) | set(anchor_scores)
    final_scores = {}

    for doc_id in all_docs:
        text_score = (
            title_scores.get(doc_id, 0) * cfg["title"] +
            body_scores.get(doc_id, 0)  * cfg["body"]  +
            anchor_scores.get(doc_id, 0) * cfg["anchor"]
        )

        if cfg["use_pagerank"]:
            alpha = cfg.get("pagerank_alpha", 0.05)
            boost = global_boost_dict.get(doc_id, 1)
            text_score *= (1 + alpha * boost)

        final_scores[doc_id] = text_score

    return final_scores


@app.route("/search")
def search():
    query = request.args.get('query', '')
    if not query: return jsonify([])

    query_tokens = tokenize(query)
    if not query_tokens: return jsonify([])

    # 1. Gather scores
    # body_scores = get_bm25_scores(query_tokens, body_index)
    # title_scores = get_title_scores(query_tokens, title_index)
    # anchor_scores = get_title_scores(query_tokens, anchor_index)
    scores = rank_with_weights(query_tokens)
    top_docs = heapq.nlargest(100, scores.items(), key=lambda x: x[1])
    return jsonify([
        (str(doc_id), id_to_title.get(doc_id, "Unknown"))
        for doc_id, _ in top_docs
    ])
    # # 2. Combine & Boost
    # all_docs = set(body_scores.keys()) | set(title_scores.keys()) | set(anchor_scores.keys())
    # candidates = []
    #
    # for doc_id in all_docs:
    #     text_score = (title_scores.get(doc_id, 0) * 0.5) + \
    #                  (body_scores.get(doc_id, 0) * 0.3) + \
    #                  (anchor_scores.get(doc_id, 0) * 0.2)
    #
    #     final_score = text_score * global_boost_dict.get(doc_id, 1)
    #
    #     if len(candidates) < 100:
    #         heapq.heappush(candidates, (final_score, doc_id))
    #     else:
    #         if final_score > candidates[0][0]:
    #             heapq.heapreplace(candidates, (final_score, doc_id))
    #
    # sorted_res = sorted(candidates, key=lambda x: x[0], reverse=True)
    # return jsonify([(str(d), id_to_title.get(d, "Unknown")) for s, d in sorted_res])


@app.route("/search_body")
def search_body():
    query = request.args.get('query', '')
    if not query: return jsonify([])
    query_tokens = tokenize(query)
    scores = get_body_scores(query_tokens, body_index)
    sorted_res = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:100]
    return jsonify([(str(d), id_to_title.get(d, "Unknown")) for d, s in sorted_res])


@app.route("/search_title")
def search_title():
    query = request.args.get('query', '')
    if not query: return jsonify([])
    query_tokens = tokenize(query)
    scores = get_title_scores(query_tokens, title_index)
    sorted_res = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return jsonify([(str(d), id_to_title.get(d, "Unknown")) for d, s in sorted_res])


@app.route("/search_anchor")
def search_anchor():
    query = request.args.get('query', '')
    if not query: return jsonify([])
    query_tokens = tokenize(query)
    scores = get_title_scores(query_tokens, anchor_index)
    sorted_res = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return jsonify([(str(d), id_to_title.get(d, "Unknown")) for d, s in sorted_res])


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    wiki_ids = request.get_json() or []
    return jsonify([pagerank_dict.get(doc_id, 0) for doc_id in wiki_ids])


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    wiki_ids = request.get_json() or []
    return jsonify([0] * len(wiki_ids))  # Dummy return to avoid crash


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)