"""장르 벡터 코사인 유사도로 비슷한 영화 찾기.

입력 : data/processed/movies_genre_onehot.csv (장르 원-핫 기준 테이블)
       data/processed/movies_with_ratings.csv (mean_rating, rating_count, bayesian_rating)

사용법 (프로젝트 루트에서):
    python analysis/genre_similarity.py                       # 기본값: "Matrix, The"
    python analysis/genre_similarity.py "제목 일부" [--top 10]

동점(장르 벡터가 완전히 같은 영화) 처리: 코사인 유사도 내림차순, 같으면 rating_count 내림차순.
필요 패키지: pandas, numpy
"""
import argparse
import numpy as np
import pandas as pd

ONEHOT = "data/processed/movies_genre_onehot.csv"
RATINGS_TBL = "data/processed/movies_with_ratings.csv"
META_COLS = {"movieId", "title", "release_year", "genres"}


def cosine_sim(matrix: np.ndarray, vec: np.ndarray) -> np.ndarray:
    vec_norm = np.linalg.norm(vec)
    mat_norm = np.linalg.norm(matrix, axis=1)
    dot = matrix @ vec
    denom = mat_norm * vec_norm
    return np.divide(dot, denom, out=np.zeros_like(dot, dtype=float), where=denom > 0)


def find_movie(df: pd.DataFrame, query: str) -> pd.Series:
    hit = df[df["title"].str.lower().str.contains(query.lower(), na=False)]
    if hit.empty:
        raise SystemExit(f"'{query}' 와 일치하는 영화가 없습니다.")
    if len(hit) > 1:
        # 가장 짧게(정확하게) 일치하는 제목을 우선
        hit = hit.assign(_len=hit["title"].str.len()).sort_values("_len")
    return hit.iloc[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", default="Matrix, The")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    onehot = pd.read_csv(ONEHOT)
    genre_cols = [c for c in onehot.columns if c not in META_COLS]

    target = find_movie(onehot, args.query)
    print(f"기준 영화: {target['title']} ({target['release_year']}) · 장르: {target['genres']}")

    vec = target[genre_cols].to_numpy(dtype=float)
    mat = onehot[genre_cols].to_numpy(dtype=float)
    sim = cosine_sim(mat, vec)

    result = onehot[["movieId", "title", "release_year", "genres"]].copy()
    result["cosine_sim"] = sim
    result = result[result["movieId"] != target["movieId"]]

    scores = pd.read_csv(RATINGS_TBL)[
        ["movieId", "mean_rating", "rating_count", "bayesian_rating"]
    ]
    result = result.merge(scores, on="movieId", how="left")

    result = result.sort_values(
        ["cosine_sim", "rating_count"], ascending=[False, False]
    ).head(args.top)

    print(f"\n장르 코사인 유사도 상위 {args.top}편")
    print(
        result[
            ["title", "release_year", "genres", "cosine_sim",
             "rating_count", "mean_rating", "bayesian_rating"]
        ].round(4).to_string(index=False)
    )


if __name__ == "__main__":
    main()
