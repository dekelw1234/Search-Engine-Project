from flask import Flask, request, jsonify
import sys
from collections import Counter, defaultdict
import itertools
from itertools import islice, count, groupby
import pandas as pd
import os
import re
from operator import itemgetter
from time import time
from pathlib import Path
import pickle
from inverted_index_local import InvertedIndex, MultiFileReader
import math


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# --- CONFIGURATION ---
DATA_DIR = Path(__file__).parent / 'postings_gcp'

# Global Variables (Loaded at startup)
body_index = None
title_index = None
anchor_index = None
page_rank = None
page_views = None
doc_lengths = None
id2title = None  # Dictionary mapping doc_id -> title
bm25_engine = None

# --- STOPWORDS & TOKENIZER ---
english_stopwords = frozenset([
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be",
    "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does",
    "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "her",
    "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself",
    "me", "more", "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such", "than",
    "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself", "yourselves"
])

corpus_stopwords = frozenset([
    "category", "references", "also", "external", "links", "may", "first", "see", "history", "people", "one", "two",
    "part", "thumb", "including", "second", "following", "many", "however", "would", "became"
])

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)


def tokenize(text):
    return [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords]


# --- BM25 IMPLEMENTATION ---
class BM25:
    def __init__(self, index, dl_dict, k1=1.5, b=0.75):
        self.index = index
        self.dl = dl_dict
        self.N = len(dl_dict) if dl_dict else 0
        self.AVGDL = sum(dl_dict.values()) / self.N if self.N > 0 else 0
        self.k1 = k1
        self.b = b

    def search(self, query_tokens):
        scores = defaultdict(float)
        for token in query_tokens:
            if token not in self.index.df:
                continue

            # Read posting list
            posting_list = self.index.read_a_posting_list("", token, str(DATA_DIR))

            df = self.index.df[token]
            idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))

            for doc_id, tf in posting_list:
                doc_len = self.dl.get(doc_id, self.AVGDL)
                numerator = idf * tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.AVGDL))
                scores[doc_id] += numerator / denominator
        return scores


# --- STARTUP ---
@app.before_first_request
def startup():
    global body_index, title_index, anchor_index, page_rank, page_views, doc_lengths, id2title, bm25_engine
    print("Loading data...")
    try:
        # Load Indices
        body_index = InvertedIndex.read_index(str(DATA_DIR), 'index')
        title_index = InvertedIndex.read_index(str(DATA_DIR), 'title_index')
        anchor_index = InvertedIndex.read_index(str(DATA_DIR), 'anchor_index')

        # Load Aux Data
        with open(DATA_DIR / 'pr.pkl', 'rb') as f:
            page_rank = pickle.load(f)
        with open(DATA_DIR / 'pageviews.pkl', 'rb') as f:
            page_views = pickle.load(f)
        with open(DATA_DIR / 'doc_lengths.pkl', 'rb') as f:
            doc_lengths = pickle.load(f)
        with open(DATA_DIR / 'id2title.pkl', 'rb') as f:
            id2title = pickle.load(f)

        # Initialize BM25
        bm25_engine = BM25(body_index, doc_lengths)
        print("Data loaded successfully!")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please ensure you ran 'build_local_indices.py' with the latest logic to generate all .pkl files.")


# --- HELPER FUNCTIONS ---
def get_binary_score(query_tokens, index):
    scores = defaultdict(int)
    if not index: return scores
    for token in query_tokens:
        if token in index.df:
            pl = index.read_a_posting_list("", token, str(DATA_DIR))
            for doc_id, _ in pl:
                scores[doc_id] += 1
    return scores


# --- ROUTES ---

@app.route("/search")
def search():
    """
    Main Search Engine: Combines Body (BM25), Title, Anchor, PageRank, and PageViews.
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

    tokens = tokenize(query)

    # 1. Calculate Scores
    # Weights configuration
    w_body = 1.0
    w_title = 5.0
    w_anchor = 3.0
    w_pr = 0.5
    w_pv = 0.1

    body_scores = bm25_engine.search(tokens)
    title_scores = get_binary_score(tokens, title_index)
    anchor_scores = get_binary_score(tokens, anchor_index)

    # 2. Merge Candidates
    all_candidates = set(body_scores.keys()) | set(title_scores.keys()) | set(anchor_scores.keys())

    final_scores = []
    for doc_id in all_candidates:
        score = (w_body * body_scores.get(doc_id, 0)) + \
                (w_title * title_scores.get(doc_id, 0)) + \
                (w_anchor * anchor_scores.get(doc_id, 0))

        # Add PageRank
        if page_rank:
            score += w_pr * page_rank.get(doc_id, 0)

        # Add PageViews (Log scaled)
        if page_views:
            pv = page_views.get(doc_id, 0)
            score += w_pv * math.log(pv + 1, 10)

        final_scores.append((doc_id, score))

    # 3. Sort & Format
    final_scores.sort(key=lambda x: x[1], reverse=True)

    # Return (id, title) using the id2title mapping
    res = [(str(doc_id), id2title.get(doc_id, str(doc_id))) for doc_id, score in final_scores[:100]]
    return jsonify(res)


@app.route("/search_body")
def search_body():
    """ Returns BM25 scores for body """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0: return jsonify(res)

    tokens = tokenize(query)
    scores = bm25_engine.search(tokens)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    res = [(str(doc_id), id2title.get(doc_id, str(doc_id))) for doc_id, score in sorted_scores[:100]]
    return jsonify(res)


@app.route("/search_title")
def search_title():
    """ Returns binary scores for title """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0: return jsonify(res)

    tokens = tokenize(query)
    scores = get_binary_score(tokens, title_index)

    # Sort by score desc, then doc_id asc
    sorted_scores = sorted(scores.items(), key=lambda item: (-item[1], item[0]))

    res = [(str(doc_id), id2title.get(doc_id, str(doc_id))) for doc_id, score in sorted_scores]
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    """ Returns binary scores for anchor """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0: return jsonify(res)

    tokens = tokenize(query)
    scores = get_binary_score(tokens, anchor_index)

    sorted_scores = sorted(scores.items(), key=lambda item: (-item[1], item[0]))

    res = [(str(doc_id), id2title.get(doc_id, str(doc_id))) for doc_id, score in sorted_scores]
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0: return jsonify(res)

    if page_rank:
        for doc_id in wiki_ids:
            try:
                res.append(page_rank.get(int(doc_id), 0))
            except:
                res.append(0)
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0: return jsonify(res)

    if page_views:
        for doc_id in wiki_ids:
            try:
                res.append(page_views.get(int(doc_id), 0))
            except:
                res.append(0)
    return jsonify(res)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)