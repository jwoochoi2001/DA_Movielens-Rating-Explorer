"""영화 줄거리(overview) TF-IDF — 특정 영화의 줄거리에서 TF-IDF 점수가 높은 단어 순서.

입력 : data/processed/movies_with_ratings.csv (overview 열, merge_plot_overview.py 산출)

사양
- overview 가 있는 영화만 대상으로 한다 (전체 9,737편 중 3,536편).
- 영어 불용어(stopwords) 제외: TfidfVectorizer(stop_words="english").
- 동일한 overview 텍스트가 여러 movieId 에 걸쳐 중복되면, IDF(문서 빈도) 계산이
  왜곡되지 않도록 **중복 제거 후 한 번만** 코퍼스에 포함한다.
- 지정한 영화의 overview 에 대해 TF-IDF 점수 상위 단어를 내림차순으로 출력한다.

사용법 (프로젝트 루트에서):
    python analysis/tfidf_overview.py                 # 기본값: "Matrix, The"
    python analysis/tfidf_overview.py "제목 일부" [--top 20]

필요 패키지: pandas, scikit-learn
"""
import argparse

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

MOVIES_CSV = "data/processed/movies_with_ratings.csv"


def find_movie(df: pd.DataFrame, query: str) -> pd.Series:
    hit = df[df["title"].str.lower().str.contains(query.lower(), na=False)]
    if hit.empty:
        raise SystemExit(f"'{query}' 와 일치하는(overview 있는) 영화가 없습니다.")
    if len(hit) > 1:
        hit = hit.assign(_len=hit["title"].str.len()).sort_values("_len")  # 가장 짧게 일치하는 제목 우선
    return hit.iloc[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", default="Matrix, The")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    movies = pd.read_csv(MOVIES_CSV)
    have = movies[movies["overview"].notna()].copy()

    target = find_movie(have, args.query)
    print(f"기준 영화: {target['title']} ({target['release_year']}) — overview 앞부분: "
          f"{target['overview'][:80]}...")

    # 동일 overview 중복 제거 (IDF 왜곡 방지). 대표 movieId는 첫 번째 것으로 유지.
    corpus_df = have.drop_duplicates(subset="overview", keep="first").reset_index(drop=True)
    n_dedup = len(have) - len(corpus_df)
    print(f"overview 보유 {len(have)}편 -> 중복 제거 후 코퍼스 {len(corpus_df)}편"
          f" (중복 {n_dedup}건 제거)")

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(corpus_df["overview"])
    vocab = vectorizer.get_feature_names_out()

    # 코퍼스 안에서 대상 영화의 문서 인덱스 찾기 (같은 overview 텍스트로 매칭)
    matches = corpus_df.index[corpus_df["overview"] == target["overview"]]
    if len(matches) == 0:
        raise SystemExit("대상 영화의 overview 가 코퍼스에 없습니다(예상치 못한 상태).")
    doc_idx = matches[0]

    row = tfidf[doc_idx].toarray().ravel()
    scored = pd.Series(row, index=vocab)
    scored = scored[scored > 0].sort_values(ascending=False)

    print(f"\n'{target['title']}' overview 단어별 TF-IDF 점수 (불용어 제외, 상위 {args.top}개)")
    print(scored.head(args.top).round(4).to_string())


if __name__ == "__main__":
    main()
