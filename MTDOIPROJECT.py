#Mark Totonjie-DOI Miner: Scientific Text Extraction & Analysis

import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import os
import re
import httpx
import pandas as pd
from springernature_api_client.meta import MetaAPI
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt

# ─── CONFIG ────────────────────────────────────────────────────────────────
ELSEVIER_API_KEY = "e808bab69cbb3adde15db0cf2c2f745b"
ELSEVIER_EMAIL   = "mtotonji@stevens.edu"
SPRINGER_API_KEY = "dd0405ffbaeca5c9e61f5c70ae9c8c59"  # metaapikey

DOI_LIST_PATH = "dois.txt"    # one DOI per line
OUTPUT_CSV    = "articles.csv"

# ─── INITIAL SETUP ─────────────────────────────────────────────────────────
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

nltk.download("punkt")
nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))

# Springer Nature Meta client
sn_client = MetaAPI(api_key=SPRINGER_API_KEY)

# HTTPX client for Elsevier fallback
elsevier_client = httpx.Client(
    headers={
        "X-ELS-APIKey": ELSEVIER_API_KEY,
        "Accept":       "application/json",
        "User-Agent":   ELSEVIER_EMAIL
    },
    timeout=10.0
)

# ─── FUNCTIONS ────────────────────────────────────────────────────────────
def fetch_springer(doi: str) -> dict:
    """Fetch title, authors, year, abstract via the Meta API."""
    q = f'doi:"{doi}"'
    resp = sn_client.search(q=q, p=1, s=1, fetch_all=False)
    records = resp.get("records", [])
    if not records:
        return {}
    rec = records[0]
    return {
        "title":    rec.get("title", ""),
        "authors":  rec.get("creator", ""),
        "year":     rec.get("publicationDate", "")[:4],
        "abstract": rec.get("abstract", "") or ""
    }

def fetch_elsevier(doi: str) -> str:
    """Elsevier fallback to get full abstract if Meta API gave none."""
    url = f"https://api.elsevier.com/content/article/doi/{doi}?view=FULL"
    r = elsevier_client.get(url)
    if r.status_code != 200:
        return ""
    data = r.json()
    core = data.get("full-text-retrieval-response", {}).get("coredata", {})
    abstract = core.get("dc:description", "") or ""
    return re.sub(r"[\r\n\t]+", " ", abstract).strip()

def preprocess(text: str) -> list[str]:
    """Tokenize, lowercase, remove stopwords & non-alpha tokens."""
    tokens = word_tokenize(text.lower())
    return [tok for tok in tokens if tok.isalpha() and tok not in STOPWORDS]

# ─── MAIN PIPELINE ─────────────────────────────────────────────────────────
def main():
    # 1) Read DOIs and fetch metadata and abstracts
    dois = [line.strip() for line in open(DOI_LIST_PATH) if line.strip()]
    records = []
    for doi in dois:
        meta = fetch_springer(doi)
        abstract = meta.get("abstract") or fetch_elsevier(doi)
        records.append({
            "doi":      doi,
            "title":    meta.get("title", ""),
            "authors":  meta.get("authors", ""),
            "year":     meta.get("year", ""),
            "abstract": abstract
        })

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"▶ Saved metadata + abstracts to {OUTPUT_CSV}")

    # 2) Preprocess abstracts
    df["tokens"] = df["abstract"].apply(preprocess)
    docs = df["tokens"].tolist()
    texts = [" ".join(doc) for doc in docs]

    # 3) TF–IDF keyword extraction
    tfidf = TfidfVectorizer(max_features=1000)
    X_tfidf = tfidf.fit_transform(texts)
    terms = tfidf.get_feature_names_out()
    sums = X_tfidf.sum(axis=0).A1
    top20 = sorted(zip(terms, sums), key=lambda x: -x[1])[:20]
    print("\nTop 20 keywords by TF–IDF:")
    for term, score in top20:
        print(f"  {term}: {score:.2f}")

    # 4) Topic modeling via scikit‑learn's LDA
    cv = CountVectorizer(max_df=0.95, min_df=2, stop_words=STOPWORDS)
    X_counts = cv.fit_transform(texts)
    lda = LatentDirichletAllocation(
        n_components=5,
        random_state=42,
        max_iter=10,
        learning_method="batch"
    )
    lda.fit(X_counts)
    feature_names = cv.get_feature_names_out()
    print("\nLDA Topics:")
    for idx, comp in enumerate(lda.components_):
        top_idxs = comp.argsort()[::-1][:5]
        top_words = [feature_names[i] for i in top_idxs]
        print(f" Topic {idx}: {', '.join(top_words)}")

    # 5) Publication-year trend plot
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    trend = df.dropna(subset=["year"]).groupby("year").size()
    trend.plot(marker="o")
    plt.title("Articles per Year")
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
