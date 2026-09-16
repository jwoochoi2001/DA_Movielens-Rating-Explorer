"""사용자 기반 협업 필터링(user-based CF) — 영화별(per-movie) 이웃 선택 버전.

user_cf_predict.py 는 대상 사용자와 평점 벡터가 가장 비슷한 이웃 k명을 **한 번 고정**해서
모든 후보 영화 예측에 그대로 재사용한다(이웃 고정형 — "나랑 비슷한 사람 k명 -> 그 사람들이
내가 안 본 영화에 준 평점의 가중평균").

이 스크립트는 반대로, **영화마다** 그 영화에 실제로 평점을 준 사람들 중에서 대상 사용자와
코사인 유사도가 가장 비슷한 상위 k명을 그때그때 다시 뽑아 그 영화만 가중 평균한다
(영화별 이웃 선택형 — "영화마다, 그 영화를 평가한 사람들 중 -> 나랑 비슷한 평점을 준 k명").
코사인 유사도 자체는 두 방식 모두 전체 평점 벡터 기준으로 동일하게 계산하고(min_common 조건도
동일), "이웃을 몇 명 뽑아서 어떤 영화들에 재사용하는가"만 다르다.

두 방식의 실질적 차이
- 이웃 고정형: 이웃 k명 중 그 영화를 평가한 사람만 쓰므로, 인기 없는 영화일수록 실제로 예측에
  쓰이는 사람 수(n_neighbor_votes)가 k보다 훨씬 적어질 수 있다(예: k=50인데 2~9명만 남음).
- 영화별 이웃 선택형: 그 영화를 평가한 사람이 충분히 많다면(min_common 조건을 만족하는 사람 중)
  대부분 k명을 꽉 채워 예측하므로 표본이 더 안정적이다. 대신 영화마다 이웃 구성이 달라지고,
  movieId 개수만큼 반복 계산해야 해서 더 오래 걸린다.

입력 : data/processed/ratings.csv, data/processed/movies_with_ratings.csv
출력 : outputs/tables/user{U}_cf_itemneighbor_candidates[_{장르}].csv
       outputs/tables/user{U}_cf_itemneighbor_recommendations[_{장르}].csv

사용법 (프로젝트 루트에서):
    python analysis/user_cf_predict_itemneighbors.py --user 414 --min-common 30 --k 50 --genre Action --top 10

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


def predict_item_conditioned(ratings: pd.DataFrame, target_user: int, min_common: int = 5, k: int = 10) -> pd.DataFrame:
    """영화마다: 그 영화를 평가한 사람들(공통 평가 영화 min_common편 이상인 사람만) 중
    코사인 유사도 상위 k명을 골라 가중 평균으로 평점을 예측한다.
    반환: DataFrame[movieId, predicted_rating, n_neighbor_votes] (대상 사용자가 이미 본 영화는 제외)."""
    mat, user_idx, user_ids = build_rating_matrix(ratings)
    if target_user not in user_idx:
        raise SystemExit(f"사용자 {target_user} 가 데이터에 없습니다.")
    u_i = user_idx[target_user]
    target_vec = mat[u_i]

    # 전체 사용자 대상 코사인 유사도 (이웃 고정형과 동일한 정의, 여기선 상위 k로 자르지 않는다)
    sims = cosine_similarity(target_vec, mat).ravel()

    binary = mat.copy()
    binary.data = np.ones_like(binary.data)
    common_counts = np.asarray((binary[u_i] @ binary.T).todense()).ravel()

    eligible = common_counts >= min_common
    eligible[u_i] = False  # 본인 제외

    seen_cols = set(mat[u_i].nonzero()[1])

    mat_csc = mat.tocsc()
    movie_ids = np.sort(ratings["movieId"].unique())

    rows_out = []
    for j, mid in enumerate(movie_ids):
        if j in seen_cols:
            continue
        col = mat_csc.getcol(j)
        raters = col.indices          # 이 영화를 평가한 사용자들의 행 인덱스
        vals = col.data                # 그 사용자들이 준 평점 (raters 와 같은 순서)
        mask = eligible[raters]
        raters = raters[mask]
        vals = vals[mask]
        if raters.size == 0:
            continue
        r_sims = sims[raters]
        if r_sims.size > k:
            top_idx = np.argpartition(-r_sims, k - 1)[:k]
            r_sims = r_sims[top_idx]
            vals = vals[top_idx]
        pred = float(np.dot(r_sims, vals) / np.abs(r_sims).sum())
        rows_out.append({
            "movieId": mid,
            "predicted_rating": round(pred, 3),
            "n_neighbor_votes": int(r_sims.size),
        })

    return pd.DataFrame(rows_out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", type=int, default=414)
    ap.add_argument("--min-common", type=int, default=30)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--min-votes", type=int, default=2,
                     help="추천 목록에 넣기 위한 최소 이웃 투표 수(기본 2)")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--genre", type=str, default=None, help="장르 부분일치 필터(예: Action)")
    args = ap.parse_args()

    ratings = pd.read_csv(RATINGS_CSV)
    movies = pd.read_csv(MOVIES_CSV)

    print(f"기준 사용자: {args.user}  ·  영화별 이웃 선택형(각 영화마다 그 영화 평가자 중 "
          f"유사도 상위 {args.k}명, 공통 평가 영화 {args.min_common}편 이상인 사람만 대상)")

    pred = predict_item_conditioned(ratings, args.user, args.min_common, args.k)
    pred = pred.merge(
        movies[["movieId", "title", "genres", "release_year",
                "mean_rating", "rating_count", "bayesian_rating"]],
        on="movieId", how="left",
    )

    suffix = ""
    if args.genre:
        before = len(pred)
        pred = pred[pred["genres"].str.contains(args.genre, case=False, na=False)]
        suffix = f"_{args.genre.lower()}"
        print(f"장르 필터 '{args.genre}': {before}편 -> {len(pred)}편")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_path = OUT_DIR / f"user{args.user}_cf_itemneighbor_candidates{suffix}.csv"
    pred.sort_values(["predicted_rating", "n_neighbor_votes"], ascending=[False, False]) \
        .to_csv(all_path, index=False)
    print(f"미평가 후보(이웃 1명 이상 투표) {len(pred)}편 -> 저장: {all_path}")

    rec = pred[pred["n_neighbor_votes"] >= args.min_votes]
    rec = rec.sort_values(["predicted_rating", "n_neighbor_votes"], ascending=[False, False])
    rec_top = rec.head(args.top)

    rec_path = OUT_DIR / f"user{args.user}_cf_itemneighbor_recommendations{suffix}.csv"
    rec_top.to_csv(rec_path, index=False)

    print(f"이웃 {args.min_votes}명 이상 투표한 후보 {len(rec)}편 중 예측 평점 상위 "
          f"{len(rec_top)}편 -> 저장: {rec_path}\n")
    print(f"[사용자 {args.user} 추천 상위 {len(rec_top)}편 (영화별 이웃 선택형)]")
    print(rec_top[["movieId", "title", "release_year", "predicted_rating",
                    "n_neighbor_votes", "bayesian_rating", "rating_count"]]
          .to_string(index=False))

    seen = set(ratings.loc[ratings.userId == args.user, "movieId"])
    assert not set(rec_top["movieId"]) & seen, "이미 본 영화가 추천에 섞임"
    assert (rec_top["predicted_rating"].between(0.5, 5.0)).all()
    assert (rec_top["n_neighbor_votes"] >= args.min_votes).all()
    assert (rec_top["n_neighbor_votes"] <= args.k).all()
    print("\n검증 통과: 추천 목록에 이미 본 영화 없음, 예측 평점 0.5~5.0 범위, "
          "이웃 수는 최소 투표 수 이상·k 이하")


if __name__ == "__main__":
    main()
