"""줄거리 TF-IDF 유사도 vs 장르 벡터 유사도 비교.

기준 영화 하나를 골라
  1) TF-IDF(overview) 코사인 유사도 상위 N편 — 장르 + 상위 키워드 5개 함께 출력
  2) 장르 원-핫 벡터 코사인 유사도 상위 N편 — 장르 + 상위 키워드 5개 함께 출력
을 나란히 보여준다. 키워드를 항상 보여주기 위해 두 리스트 모두 **overview 가 있는
영화(3,536편)로 후보를 한정**한다.

입력 : data/processed/movies_with_ratings.csv, data/processed/movies_genre_onehot.csv

사용법 (프로젝트 루트에서):
    python analysis/tfidf_similarity.py                 # 기본값: "Matrix, The"
    python analysis/tfidf_similarity.py "제목 일부" [--top 10]

필요 패키지: pandas, numpy, scikit-learn
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

MOVIES_CSV = "data/processed/movies_with_ratings.csv"
GENRE_ONEHOT_CSV = "data/processed/movies_genre_onehot.csv"
META_COLS = {"title", "release_year", "genres"}


def find_movie(df: pd.DataFrame, query: str) -> pd.Series:
    hit = df[df["title"].str.lower().str.contains(query.lower(), na=False)]
    if hit.empty:
        raise SystemExit(f"'{query}' 와 일치하는(overview 있는) 영화가 없습니다.")
    if len(hit) > 1:
        hit = hit.assign(_len=hit["title"].str.len()).sort_values("_len")
    return hit.iloc[0]


def top_keywords(tfidf_row, vocab, n=5) -> list:
    row = tfidf_row.toarray().ravel()
    idx = row.argsort()[::-1][:n]
    return [vocab[i] for i in idx if row[i] > 0]


def print_list(title: str, df: pd.DataFrame, sim_col: str, tfidf, vocab) -> None:
    print(f"\n=== {title} ===")
    for _, row in df.iterrows():
        kws = top_keywords(tfidf[row.name], vocab, 5)
        print(f"{row['title']} ({row['release_year']})  |  유사도 {row[sim_col]:.4f}  |  "
              f"장르: {row['genres']}  |  키워드: {', '.join(kws)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", default="Matrix, The")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    movies = pd.read_csv(MOVIES_CSV)
    have = movies[movies["overview"].notna()].copy()
    corpus_df = have.drop_duplicates(subset="overview", keep="first").reset_index(drop=True)

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(corpus_df["overview"])  # 행마다 L2 정규화됨
    vocab = vectorizer.get_feature_names_out()

    target = find_movie(corpus_df, args.query)
    doc_idx = int(corpus_df.index[corpus_df["movieId"] == target["movieId"]][0])
    print(f"기준 영화: {target['title']} ({target['release_year']})  |  장르: {target['genres']}")
    print(f"후보 범위: overview 보유 {len(have)}편 -> 중복 제거 {len(corpus_df)}편")

    # 1) TF-IDF 코사인 유사도: 정규화된 벡터라 내적 = 코사인 유사도
    tfidf_sims = (tfidf @ tfidf[doc_idx].T).toarray().ravel()
    corpus_df = corpus_df.assign(tfidf_sim=tfidf_sims)
    top_tfidf = (corpus_df[corpus_df.index != doc_idx]
                 .nlargest(args.top, "tfidf_sim"))
    print_list(f"TF-IDF 줄거리 유사도 상위 {args.top}", top_tfidf, "tfidf_sim", tfidf, vocab)

    # 2) 장르 원-핫 벡터 코사인 유사도 (후보를 overview 보유 영화로 한정)
    genre_matrix = pd.read_csv(GENRE_ONEHOT_CSV).set_index("movieId")
    genre_cols = [c for c in genre_matrix.columns if c not in META_COLS]
    genre_matrix = genre_matrix[genre_cols]

    sub_ids = corpus_df["movieId"].tolist()
    mat = genre_matrix.loc[sub_ids].to_numpy(dtype=float)
    vec = genre_matrix.loc[target["movieId"]].to_numpy(dtype=float)
    dot = mat @ vec
    norms = np.linalg.norm(mat, axis=1) * np.linalg.norm(vec)
    genre_sims = np.divide(dot, norms, out=np.zeros_like(dot, dtype=float), where=norms > 0)

    corpus_df = corpus_df.assign(genre_sim=genre_sims)
    top_genre = corpus_df[(corpus_df.index != doc_idx) & (corpus_df["genre_sim"] > 0)]
    top_genre = top_genre.sort_values(
        ["genre_sim", "bayesian_rating"], ascending=[False, False]
    ).head(args.top)
    print_list(f"장르 벡터 코사인 유사도 상위 {args.top} (동점은 보정 평점 순)",
               top_genre, "genre_sim", tfidf, vocab)


if __name__ == "__main__":
    main()
