import pickle
import os
from pathlib import Path
from inverted_index_local import InvertedIndex

# הגדרת נתיבים
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / 'postings_gcp'
DATA_DIR.mkdir(exist_ok=True)


def create_dummy_index(name, text_dict, with_dl=False, save_titles=False):
    """
    יוצר אינדקס ומאפשר שמירה של אורך מסמך (DL) ומילון כותרות (Titles)
    """
    print(f"Creating index: {name}...")
    index = InvertedIndex()
    doc_lengths = {}

    # הוספת מסמכים לאינדקס
    for doc_id, text in text_dict.items():
        tokens = text.lower().split()
        index.add_doc(doc_id, tokens)

        if with_dl:
            doc_lengths[doc_id] = len(tokens)

    # שמירת האינדקס (bin + pkl)
    index.write_index(str(DATA_DIR), name)

    # שמירת אורכי מסמכים (עבור BM25)
    if with_dl:
        print(f"Saving doc_lengths for {name}...")
        with open(DATA_DIR / 'doc_lengths.pkl', 'wb') as f:
            pickle.dump(doc_lengths, f)

    # שמירת מילון כותרות (עבור הצגת תוצאות)
    if save_titles:
        print(f"Saving id2title for {name}...")
        # בדמי שלנו, הטקסט עצמו ישמש ככותרת
        titles = {doc_id: text for doc_id, text in text_dict.items()}
        with open(DATA_DIR / 'id2title.pkl', 'wb') as f:
            pickle.dump(titles, f)


def create_auxiliary_data():
    """
    יוצר קבצי עזר: PageRank ו-PageViews
    """
    print("Creating PageRank and PageViews...")
    # נתוני דמה
    page_rank = {1: 0.5, 2: 0.8, 3: 0.1, 4: 0.9}
    page_views = {1: 1000, 2: 5000, 3: 100, 4: 2000}

    with open(DATA_DIR / 'pr.pkl', 'wb') as f:
        pickle.dump(page_rank, f)

    with open(DATA_DIR / 'pageviews.pkl', 'wb') as f:
        pickle.dump(page_views, f)


def main():
    # 1. יצירת אינדקס גוף (Body) + שמירת אורכי מסמכים
    body_docs = {
        1: "python is a programming language excellent for data science",
        2: "machine learning using python is great",
        3: "search engine optimization is crucial for websites",
        4: "data science involves statistics and python"
    }
    # כאן אנחנו שולחים with_dl=True כדי שייווצר doc_lengths.pkl
    create_dummy_index("index", body_docs, with_dl=True)

    # 2. יצירת אינדקס כותרות (Title) + שמירת מילון כותרות
    title_docs = {
        1: "Python (programming language)",
        2: "Machine Learning",
        3: "SEO Optimization",
        4: "Data Science"
    }
    # כאן אנחנו שולחים save_titles=True כדי שייווצר id2title.pkl
    create_dummy_index("title_index", title_docs, save_titles=True)

    # 3. יצירת אינדקס עוגנים (Anchor)
    anchor_docs = {
        1: "link to python",
        2: "ml course",
        3: "google search results",
        4: "data analysis"
    }
    create_dummy_index("anchor_index", anchor_docs)

    # 4. נתונים נוספים
    create_auxiliary_data()

    print(f"Done! All data created successfully in {DATA_DIR}")


if __name__ == '__main__':
    main()