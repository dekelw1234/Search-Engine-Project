from flask import Flask, request, jsonify
import pickle
from inverted_index_gcp import InvertedIndex
import nltk
from nltk.corpus import stopwords
import re
import math
from collections import Counter
import heapq # Required for efficient Top-K

# Loading all necessary data once when the server starts
print("Loading indices and metadata... please wait")

# Loading Inverted Indices from the postings folder
with open('postings_gcp/body_index.pkl', 'rb') as f:
    body_index = pickle.load(f)
with open('postings_gcp/title_index.pkl', 'rb') as f:
    title_index = pickle.load(f)
with open('postings_gcp/anchor_index.pkl', 'rb') as f:
    anchor_index = pickle.load(f)

# Loading Ranking Dictionaries
with open('pagerank.pkl', 'rb') as f:
    pagerank_dict = pickle.load(f)


global_boost_dict = {}
all_relevant_ids = set(pagerank_dict.keys())

for doc_id in all_relevant_ids:
    pr = pagerank_dict.get(doc_id, 0)
    # The formula is pre-calculated here
    global_boost_dict[doc_id] = 1 + math.log10(pr + 1)
# Mapping Wiki IDs to Titles
with open('id_to_title.pkl', 'rb') as f:
    id_to_title = pickle.load(f)

# Average document length for BM25 calculation
AVG_BODY_LEN = getattr(body_index, 'avg_body_len', 500)

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# --- Global Data & Setup ---

# Tokenizer setup
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w)*""", re.UNICODE)
nltk.download('stopwords')
english_stopwords = frozenset(stopwords.words('english'))
def get_html_pattern():
  # Pattern breakdown:
    # <       - opening angle bracket
    # [^>]+   - one or more characters that are NOT >
    # >       - closing angle bracket
  return r'<[^>]+>'


def get_date_pattern():
    # Days 1-31
    day_31 = r'(0?[1-9]|[12][0-9]|3[01])'

    # Days 1-30
    day_30 = r'(0?[1-9]|[12][0-9]|30)'

    # Days 1-29 (for February)
    day_29 = r'(0?[1-9]|[12][0-9])'

    # Months with 31 days
    months_31 = r'(Jan|January|Mar|March|May|Jul|July|Aug|August|Oct|October|Dec|December)'

    # Months with 30 days
    months_30 = r'(Apr|April|Jun|June|Sep|September|Nov|November)'

    # February
    feb = r'(Feb|February)'

    # MM DD YYYY
    text_31 = rf'{months_31}\s+{day_31}(?:st|nd|rd|th)?,?\s*\d{{2,4}}'
    text_30 = rf'{months_30}\s+{day_30}(?:st|nd|rd|th)?,?\s*\d{{2,4}}'
    text_feb = rf'{feb}\s+{day_29}(?:st|nd|rd|th)?,?\s*\d{{2,4}}'

    # DD MM YYYY
    text2_31 = rf'{day_31}(?:st|nd|rd|th)?\s+{months_31},?\s+\d{{2,4}}'
    text2_30 = rf'{day_30}(?:st|nd|rd|th)?\s+{months_30},?\s+\d{{2,4}}'
    text2_feb = rf'{day_29}(?:st|nd|rd|th)?\s+{feb},?\s+\d{{2,4}}'

    # Numeric dates
    numeric_date = rf'{day_29}[/.\-]{day_29}[/.\-]\d{{2,4}}'

    # combine all patterns
    return rf'{text_31}|{text_30}|{text_feb}|{text2_31}|{text2_30}|{text2_feb}|{numeric_date}'

def get_time_pattern():
  # valid hours
    hours_24 = r'([01]?[0-9]|2[0-3])'  # 0-23
    hours_12 = r'(0?[1-9]|1[0-2])'      # 1-12

  # valid minutes and seconds
    min_sec = r'[0-5][0-9]' # 00-59

  # 24-hour without AM/PM
    time_24 = rf'(?<!\d:){hours_24}:{min_sec}(:{min_sec})?(?!:\d{{2}}|\s?[aApP]\.?[mM]\.?)'

  # 12-hour DOT + AM/PM
    time_12_dot = rf'{hours_12}\.{min_sec}[AP]M'

  # 12-hour NO SEP + a.m./p.m.
    time_12_nosep = rf'{hours_12}{min_sec}[ap]\.m\.'

   # combine all patterns
    return rf'{time_24}|{time_12_dot}|{time_12_nosep}'

def get_percent_pattern():
  # Pattern breakdown:
    # \d+  - one or more digits (without)
    # \.?  - optional decimal point
    # \d*  - zero or more digits after decimal
    # %    - literal percent sign
  return r'\d+\.?\d*%'


def get_number_pattern():
  # Pattern breakdown:
    # (?<![A-Za-z0-9_+\-,.])   - number cannot be preceded by letters, digits, signs, comma, or dot
    # (?:[+-])?                - optional sign
    # ([0-9]{1,3}(,[0-9]{3})*|[0-9]+) - integer part: valid commas or plain digits
    # (?:\.[0-9]+)?            - optional decimal part
    # (?!%|\.\d|\.[A-Za-z]|,\d|\d)-no percent, digit after dot/comma, or letters""
    sign = r"(?:[+-])?"
    int_part = r"([0-9]{1,3}(,[0-9]{3})*|[0-9]+)(?:\.[0-9]+)?"
    prefix = r"(?<![A-Za-z0-9_\+\-\,\.])"
    suffix = r"(?!%)(?!\.\d)(?!\.[A-Za-z])(?!,\d)(?!\d)"

    # combine all patterns
    return rf"{prefix}{sign}{int_part}{suffix}"


def get_word_pattern():
  # Pattern breakdown:
    # (?<!-)           - not preceded by hyphen
    # [a-zA-Z]+        - one or more letters (start)
    # (['-][a-zA-Z]+)* - zero or more groups of (apostrophe/hyphen + letters)
    # '?               - optional trailing apostrophe (for parents')
  return r"(?<!-)[a-zA-Z]+(['-][a-zA-Z]+)*'?"
RE_TOKENIZE = re.compile(rf"""
(
    # parsing html tags
     (?P<HTMLTAG>{get_html_pattern()})
    # dates
    |(?P<DATE>{get_date_pattern()})
    # time
    |(?P<TIME>{get_time_pattern()})
    # Percents
    |(?P<PERCENT>{get_percent_pattern()})
    # Numbers
    |(?P<NUMBER>{get_number_pattern()})
    # Words
    |(?P<WORD>{get_word_pattern()})
    # space
    |(?P<SPACE>[\s\t\n]+)
    # everything else
    |(?P<OTHER>.))""",  re.MULTILINE | re.IGNORECASE | re.VERBOSE | re.UNICODE)


def tokenize(text):
    tokens = [v for match in RE_TOKENIZE.finditer(text)
              for k, v in match.groupdict().items()
              if v is not None and k != 'SPACE']

    return [t.lower() for t in tokens if t.lower() not in english_stopwords]


def get_bm25_scores(query_tokens, index, k1=1.5, b=0.75):
    """
    Implements BM25 ranking logic.
    BM25 formula: $score(D, Q) = \sum_{q \in Q} IDF(q) \cdot \frac{f(q, D) \cdot (k_1 + 1)}{f(q, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{avgdl})}$
    """
    scores = {}
    N = len(index.DL)
    for term in query_tokens:
        if term in index.df:
            df = index.df[term]
            idf = math.log10((N - df + 0.5) / (df + 0.5) + 1)
            postings = index.read_posting_list(term, 'postings_gcp/')
            for doc_id, tf in postings:
                doc_len = index.DL.get(doc_id, AVG_BODY_LEN)
                denominator = tf + k1 * (1 - b + b * (doc_len / AVG_BODY_LEN))
                scores[doc_id] = scores.get(doc_id, 0) + (idf * tf * (k1 + 1) / denominator)
    return scores

def get_tfidf_scores(query_tokens, index):
    """ Classic TF-IDF Cosine Similarity for the /search_body route. """
    scores = {}
    query_counts = Counter(query_tokens)
    query_norm_sq = 0
    for term, tf_q in query_counts.items():
        if term in index.df:
            idf = math.log10(len(index.DL) / index.df[term])
            w_t_q = tf_q * idf
            query_norm_sq += w_t_q**2
            postings = index.read_posting_list(term, 'postings_gcp/')
            for doc_id, tf_d in postings:
                scores[doc_id] = scores.get(doc_id, 0) + (w_t_q * tf_d * idf)
    if not scores: return {}
    q_norm = math.sqrt(query_norm_sq)
    return {doc_id: dot / (q_norm * index.doc_norms.get(doc_id, 1)) for doc_id, dot in scores.items()}

# --- Helper Functions for Scoring ---

def get_title_scores(query_tokens, index):
    """ Helper to rank documents by count of distinct query words in title. """
    scores = {}
    for term in set(query_tokens):
        if term in index.df:
            postings = index.read_posting_list(term, 'postings_gcp/')
            for doc_id, tf in postings:
                scores[doc_id] = scores.get(doc_id, 0) + 1
    return scores


def get_body_scores(query_tokens, index):
    """ Helper to calculate TF-IDF based cosine similarity for body text. """
    scores = {}
    query_counts = Counter(query_tokens)
    query_norm_sq = 0

    # Calculate query weights and dot product
    for term, tf_q in query_counts.items():
        if term in index.df:
            idf = math.log10(len(index.DL) / index.df[term])
            w_t_q = tf_q * idf
            query_norm_sq += w_t_q ** 2

            postings = index.read_posting_list(term, 'postings_gcp/')
            for doc_id, tf_d in postings:
                w_t_d = tf_d * idf
                scores[doc_id] = scores.get(doc_id, 0) + (w_t_q * w_t_d)

    if not scores: return {}

    # Normalize by query and document norms
    query_norm = math.sqrt(query_norm_sq)
    results = {}
    for doc_id, dot_product in scores.items():
        norm_d = index.doc_norms.get(doc_id, 1)
        results[doc_id] = dot_product / (query_norm * norm_d)
    return results


# --- Routes ---

@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    query = request.args.get('query', '')
    if not query: return jsonify([])

    query_tokens = tokenize(query)
    if not query_tokens: return jsonify([])

    # 1. Gather scores from indices
    body_scores = get_bm25_scores(query_tokens, body_index)
    title_scores = get_title_scores(query_tokens, title_index)
    anchor_scores = get_title_scores(query_tokens, anchor_index)

    # 2. OPTIMIZATION 2: Efficient Combination & Boosting
    # We use a single loop and avoid unnecessary dictionary lookups
    all_docs = set(body_scores.keys()) | set(title_scores.keys()) | set(anchor_scores.keys())

    # We will use a list of tuples (score, doc_id) for the heap
    candidates = []

    for doc_id in all_docs:
        # Calculate base text score (Weights: Title 0.5, Body 0.3, Anchor 0.2)
        text_score = (title_scores.get(doc_id, 0) * 0.5) + \
                     (body_scores.get(doc_id, 0) * 0.3) + \
                     (anchor_scores.get(doc_id, 0) * 0.2)

        # Apply the pre-calculated global boost
        final_score = text_score * global_boost_dict.get(doc_id, 1)

        # OPTIMIZATION 3: Top-K using a Heap
        # This is faster than sorting the entire result set at the end
        if len(candidates) < 100:
            heapq.heappush(candidates, (final_score, doc_id))
        else:
            # If current score is better than the smallest score in the top 100
            if final_score > candidates[0][0]:
                heapq.heapreplace(candidates, (final_score, doc_id))

    # 3. Final results extraction (Heap is a min-priority queue, so we sort it once at the end)
    sorted_res = sorted(candidates, key=lambda x: x[0], reverse=True)

    return jsonify([(str(d), id_to_title.get(d, "Unknown")) for s, d in sorted_res])

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

    # BEGIN SOLUTION
    query_tokens = tokenize(query)
    scores = get_body_scores(query_tokens, body_index)
    sorted_res = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:100]
    res = [(str(doc_id), id_to_title.get(doc_id, "Unknown")) for doc_id, score in sorted_res]
    # END SOLUTION
    return jsonify(res)


@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords. For example, a document
        with a title that matches two distinct query words will be ranked before a
        document with a title that matches only one distinct query word,
        regardless of the number of times the term appeared in the title (or
        query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

    # BEGIN SOLUTION
    query_tokens = tokenize(query)
    scores = get_title_scores(query_tokens, title_index)
    sorted_res = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    res = [(str(doc_id), id_to_title.get(doc_id, "Unknown")) for doc_id, count in sorted_res]
    # END SOLUTION
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment
        3 (GCP part) to do the tokenization and remove stopwords. For example,
        a document with a anchor text that matches two distinct query words will
        be ranked before a document with anchor text that matches only one
        distinct query word, regardless of the number of times the term appeared
        in the anchor text (or query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

    # BEGIN SOLUTION
    query_tokens = tokenize(query)
    # Reusing the title score logic as anchor requirements are identical
    scores = get_title_scores(query_tokens, anchor_index)
    sorted_res = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    res = [(str(doc_id), id_to_title.get(doc_id, "Unknown")) for doc_id, count in sorted_res]
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)

    # BEGIN SOLUTION
    res = [pagerank_dict.get(doc_id, 0) for doc_id in wiki_ids]
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the
          provided list article IDs.
    '''
    wiki_ids = request.get_json()
    with open('pageviews.pkl', 'rb') as f:
        local_pv_dict = pickle.load(f)

    res = [local_pv_dict.get(doc_id, 0) for doc_id in wiki_ids]
    return jsonify(res)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)