# Wikipedia Search Engine

A search engine over the full English Wikipedia, built as a university project in Information Retrieval with a partner.

The indexes are built with Spark on Google Cloud, and queries are served by a Flask API that combines several ranking signals into one score.

Built by Dekel Winkler and [partner's name].

## How it works

### Indexing
- The Wikipedia dump is processed with Spark on GCP (`build_index_gcp.ipynb`).
- Three separate inverted indexes are built: **body text**, **titles** and **anchor text** (the text of links pointing to a page).
- Text is tokenized with a regular expression and filtered with English stopwords plus a list of Wikipedia-specific stopwords (e.g. "references", "category", "external links").
- Posting lists are stored on disk and read term by term at query time, so the full index never needs to be loaded into memory.
- PageRank scores and a document-ID-to-title mapping are precomputed.

### Ranking
The main `/search` endpoint scores each candidate document by combining:
- **BM25** on the body text
- **Title match**: how many query terms appear in the title
- **Anchor match**: how many query terms appear in anchor text pointing to the page
- An optional **PageRank** boost

Each signal has a weight. The weights are defined as named configurations in `search_frontend.py` and chosen with the `ENGINE_VERSION` environment variable, which made it easy to compare many variants on the same queries.

### Evaluation
- `evaluate_quality.py` sends the training queries (`queries_train.json`) to the running engine and computes **MAP@K** against the expected results.
- `measure_latency.py` measures query response times.
- `analyze_queries.py` helps inspect individual queries that return poor results.
- `run_all_versions.sh` runs the evaluation across the different ranking configurations.

## API

| Endpoint | Method | Description |
|---|---|---|
| `/search?query=...` | GET | Main search: weighted combination of body, title, anchor and optional PageRank |
| `/search_body?query=...` | GET | TF-IDF with cosine similarity on body text only |
| `/search_title?query=...` | GET | Ranks by number of query terms in the title |
| `/search_anchor?query=...` | GET | Ranks by number of query terms in anchor text |
| `/get_pagerank` | POST | Returns PageRank values for a JSON list of article IDs |

Each search endpoint returns up to 100 results as `(wiki_id, title)` pairs.

## Running

The engine expects the index files (`postings_gcp/`), `pagerank.pkl` and `id2title.pkl` in the working directory. These files are large and are not included in the repository.

```bash
pip install -r requirements
python search_frontend.py          # serves on port 8080
```

Choose a ranking configuration:
```bash
ENGINE_VERSION=BALANCED_2_PR python search_frontend.py
```

Evaluate it (in another terminal):
```bash
python evaluate_quality.py 10      # MAP@10
python measure_latency.py
```

On GCP, `startup_script_gcp.sh` and `run_frontend_in_gcp.sh` set up a Compute Engine instance and start the server.

## Tech stack

Python, Flask, Spark, Google Cloud (Compute Engine, Cloud Storage), NLTK.
