"""영화 평점 패턴 유사도(item-item 코사인 유사도) — 특정 영화와 평점 패턴이 비슷한 영화 찾기.

genre_similarity.py 가 장르 원-핫 벡터로 유사도를 재는 것과 달리, 이 스크립트는 영화가 실제로
받은 **평점 자체**(사용자 × 영화 희소 행렬의 열, 안 본 사람은 0)로 코사인 유사도를 계산한다 —
"사람들이 이 영화에 준 평점 패턴이 저 영화와 얼마나 닮았는가"를 재는 것으로, item_based_cf.py의
아이템 유사도와 같은 정의를 쓰되, 특정 사용자로 개인화하지 않고 전체 데이터에서 대상 영화 하나와
비슷한 영화를 찾는 범용 조회다.

- 평가 수 필터: --min-ratings(기본 5명) 미만으로 평가된 영화는 제외한다 — 평가자가 1~2명뿐이면
  우연히 코사인 유사도가 1.0(완전 동점)처럼 나올 수 있어, 최소한의 표본을 요구한다.
- 공통 평가자 수: 대상 영화와 후보 영화를 **모두** 평가한 사용자 수(이진 행렬 내적)도 함께 보여줘,
  유사도는 높아도 공통 평가자가 적어 신뢰도가 낮은 경우를 구분할 수 있게 한다.

입력 : data/processed/ratings.csv, data/processed/movies_with_ratings.csv
사용법 (프로젝트 루트에서):
    python analysis/rating_pattern_similarity.py --title "Matrix, The" --min-ratings 5 --top 10
    python analysis/rating_pattern_similarity.py --movie-id 2571 --min-ratings 5 --top 10

필요 패키지: pandas, numpy, scipy, scikit-learn
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from user_cf_neighbors import build_rating_matrix

RATINGS_CSV = "data/processed/ratings.csv"
MOVIES_CSV = "data/processed/movies_with_ratings.csv"
OUT_DIR = Path("outputs/tables")


def find_similar_movies(ratings: pd.DataFrame, movies: pd.DataFrame, target_movie_id: int,
                         min_ratings: int = 5, top: int = 10) -> pd.DataFrame:
    """target_movie_id 와 평점 패턴(코사인 유사도)이 비슷한 영화를 찾는다.
    반환: DataFrame[movieId, title, cosine_sim, n_common, rating_count], 유사도 내림차순 상위 top행."""
    mat, user_idx, user_ids = build_rating_matrix(ratings)
    movie_ids = np.sort(ratings["movieId"].unique())
    movie_idx = {m: i for i, m in enumerate(movie_ids)}
    if target_movie_id not in movie_idx:
        raise SystemExit(f"movieId {target_movie_id} 가 평점 데이터에 없습니다.")

    mat_csc = mat.tocsc()
    j = movie_idx[target_movie_id]
    target_col = mat_csc[:, j]

    sims = cosine_similarity(target_col.T, mat_csc.T).ravel()

    binary = mat_csc.copy()
    binary.data = np.ones_like(binary.data)
    common = np.asarray((binary[:, j].T @ binary).todense()).ravel()

    df = pd.DataFrame({"movieId": movie_ids, "cosine_sim": sims, "n_common": common})
    df = df[df["movieId"] != target_movie_id]
    df = df.merge(movies[["movieId", "title", "rating_count"]], on="movieId", how="left")
    df = df[df["rating_count"] >= min_ratings]
    df = df.sort_values(["cosine_sim", "n_common"], ascending=[False, False]).head(top)
    return df.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", type=str, default=None, help="영화 제목 부분일치(예: 'Matrix, The')")
    ap.add_argument("--movie-id", type=int, default=None, help="movieId 로 직접 지정")
    ap.add_argument("--min-ratings", type=int, default=5, help="평가 수 최소 기준(기본 5명)")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    ratings = pd.read_csv(RATINGS_CSV)
    movies = pd.read_csv(MOVIES_CSV)

    if args.movie_id is not None:
        target_id = args.movie_id
    elif args.title:
        matches = movies[movies["title"].str.contains(args.title, case=False, na=False)]
        if matches.empty:
            raise SystemExit(f"제목 '{args.title}' 에 해당하는 영화를 찾지 못했습니다.")
        if len(matches) > 1:
            print(f"'{args.title}' 부분일치 {len(matches)}건 중 평가 수 최다 1건을 사용합니다:")
            print(matches[["movieId", "title", "rating_count"]].sort_values("rating_count", ascending=False)
                  .to_string(index=False))
        target_id = int(matches.sort_values("rating_count", ascending=False).iloc[0]["movieId"])
    else:
        raise SystemExit("--title 또는 --movie-id 중 하나를 지정하세요.")

    target_row = movies.loc[movies["movieId"] == target_id].iloc[0]
    print(f"\n기준 영화: {target_row['title']} (movieId {target_id}, 평가 수 {int(target_row['rating_count'])}건)")
    print(f"평가 수 {args.min_ratings}건 이상인 영화만 대상 · 코사인 유사도 상위 {args.top}편\n")

    result = find_similar_movies(ratings, movies, target_id, args.min_ratings, args.top)
    print(result[["title", "cosine_sim", "n_common", "rating_count"]]
          .rename(columns={"cosine_sim": "유사도", "n_common": "공통 평가자 수", "rating_count": "전체 평가 수"})
          .to_string(index=False))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"movie{target_id}_rating_pattern_similar.csv"
    result.to_csv(out_path, index=False)
    print(f"\n저장: {out_path}")

    assert (result["rating_count"] >= args.min_ratings).all()
    assert target_id not in result["movieId"].to_numpy()
    print("검증 통과: 평가 수 최소 기준 충족, 기준 영화 자신은 제외")


if __name__ == "__main__":
    main()
