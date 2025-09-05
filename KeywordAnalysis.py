import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt

def run_keyword_term_frequency_analysis(df):
    texts = df["abstract"].tolist()

    # ---- Most common unigrams (term frequency) ----
    cv_uni = CountVectorizer(stop_words="english", max_features=1000)
    X_uni = cv_uni.fit_transform(texts)
    uni_terms  = cv_uni.get_feature_names_out()
    uni_counts = X_uni.sum(axis=0).A1
    df_uni = (
        pd.DataFrame({"term": uni_terms, "count": uni_counts})
          .sort_values("count", ascending=False)
    )
    print("Top 20 unigrams by raw frequency:")
    print(df_uni.head(20).to_string(index=False))

    # ---- Top TF–IDF terms ----
    tfidf = TfidfVectorizer(stop_words="english", max_features=1000)
    X_tfidf  = tfidf.fit_transform(texts)
    tf_terms  = tfidf.get_feature_names_out()
    tf_scores = X_tfidf.sum(axis=0).A1
    df_tfidf = (
        pd.DataFrame({"term": tf_terms, "score": tf_scores})
          .sort_values("score", ascending=False)
    )
    print("\nTop 20 terms by TF–IDF score:")
    print(df_tfidf.head(20).to_string(index=False))

    # ---- Bigrams and Trigrams ----
    cv_ng = CountVectorizer(
        stop_words="english",
        ngram_range=(2, 3),
        max_features=500
    )
    X_ng      = cv_ng.fit_transform(texts)
    ng_terms  = cv_ng.get_feature_names_out()
    ng_counts = X_ng.sum(axis=0).A1
    df_ng = (
        pd.DataFrame({"ngram": ng_terms, "count": ng_counts})
          .sort_values("count", ascending=False)
    )
    print("\nTop 20 bigrams/trigrams by frequency:")
    print(df_ng.head(20).to_string(index=False))

    # ---- the top 10 TF–IDF terms as a bar chart ----
    plt.figure(figsize=(8, 5))
    plt.barh(
        df_tfidf.head(10)["term"][::-1],
        df_tfidf.head(10)["score"][::-1]
    )
    plt.xlabel("TF–IDF Score")
    plt.title("Top 10 TF–IDF Terms")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("db_elsevier_spin_final.csv")
    df["abstract"] = df["abstract"].fillna("")
    run_keyword_term_frequency_analysis(df)