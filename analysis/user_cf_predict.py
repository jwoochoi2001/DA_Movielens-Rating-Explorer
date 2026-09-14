"""사용자 기반 협업 필터링(user-based CF) — 평점 예측 · 추천.

user_cf_neighbors.py 의 find_neighbors() 로 찾은 이웃(코사인 유사도 상위 k명,
공통 평가 영화 min_common편 이상)을 그대로 재사용해, 대상 사용자가 아직 평가하지
않은 영화의 평점을 예측하고 상위 N편을 추천한다.

예측 공식 (유사도 가중 평균, 이웃 중 그 영화를 평가한 사람만 사용)
    pred(u, m) = Σ_v sim(u, v) · rating(v, m)  /  Σ_v |sim(u, v)|
    (v 는 movieId m 을 평가한 이웃만)

- 대상 사용자가 이미 평가한 영화는 후보에서 제외한다(이미 본 영화 추천 안 함).
- 이웃 중 그 영화를 평가한 사람 수(n_neighbor_votes)를 함께 표시한다 — 1명짜리 예측은
  신뢰도가 낮으므로, 기본 추천 목록은 --min-votes(기본 2명) 이상만 추린다. 전체 후보는
  파일에 그대로 저장해 직접 확인할 수 있다.
- --genre 로 장르 부분일치 필터를 걸 수 있다(예: --genre Action). 필터는 예측을 다시
  계산하는 게 아니라, 이웃 가중 평균으로 구한 예측 평점표에서 해당 장르만 추려 순위를
  다시 매기는 것이다 — CF 예측값 자체는 장르와 무관하게 동일하게 계산된다.

입력 : data/processed/ratings.csv, data/processed/movies_with_ratings.csv
출력 : outputs/tables/user{U}_cf_candidates[_{장르}].csv      (이웃 1명 이상 투표한 전체 후보)
       outputs/tables/user{U}_cf_recommendations[_{장르}].csv (min-votes 이상, 예측 평점 상위 --top)

사용법 (프로젝트 루트에서):
    python analysis/user_cf_predict.py                              # 기본값: userId 414
    python analysis/user_cf_predict.py --user 414 --min-common 5 --k 10 --min-votes 2 --top 15
    python analysis/user_cf_predict.py --user 414 --genre Action     # 장르 필터

필요 패키지: pandas, numpy, scipy, scikit-learn
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from user_cf_neighbors import find_neighbors

RATINGS_CSV = "data/processed/ratings.csv"
MOVIES_CSV = "data/processed/movies_with_ratings.csv"
OUT_DIR = Path("outputs/tables")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", type=int, default=414)
    ap.add_argument("--min-common", type=int, default=5)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--min-votes", type=int, default=2,
                     help="추천 목록에 넣기 위한 최소 이웃 투표 수(기본 2)")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--genre", type=str, default=None, help="장르 부분일치 필터(예: Action)")
    args = ap.parse_args()

    ratings = pd.read_csv(RATINGS_CSV)
    movies = pd.read_csv(MOVIES_CSV)

    neighbors = find_neighbors(ratings, args.user, args.min_common, args.k)
    print(f"기준 사용자: {args.user}  ·  이웃 {len(neighbors)}명 "
          f"(공통 평가 영화 {args.min_common}편 이상, 코사인 유사도 상위 {args.k}명)")
    print(neighbors.to_string(index=False))

    target_seen = set(ratings.loc[ratings.userId == args.user, "movieId"])

    # 이웃들의 평점을 movieId 기준으로 모은다: movieId -> [(sim, rating), ...]
    nb_ratings = ratings[ratings.userId.isin(neighbors["neighbor_userId"])]
    nb_ratings = nb_ratings.merge(
        neighbors, left_on="userId", right_on="neighbor_userId", how="left"
    )
    nb_ratings = nb_ratings[~nb_ratings.movieId.isin(target_seen)]  # 이미 본 영화는 후보 제외

    def weighted_pred(g: pd.DataFrame) -> pd.Series:
        w = g["cosine_sim"].to_numpy()
        r = g["rating"].to_numpy()
        return pd.Series({
            "predicted_rating": float(np.dot(w, r) / np.abs(w).sum()),
            "n_neighbor_votes": len(g),
        })

    pred = (nb_ratings.groupby("movieId", group_keys=True)
            .apply(weighted_pred, include_groups=False)
            .reset_index())

    pred = pred.merge(
        movies[["movieId", "title", "genres", "release_year",
                "mean_rating", "rating_count", "bayesian_rating"]],
        on="movieId", how="left",
    )
    pred["predicted_rating"] = pred["predicted_rating"].round(3)

    suffix = ""
    if args.genre:
        before = len(pred)
        pred = pred[pred["genres"].str.contains(args.genre, case=False, na=False)]
        suffix = f"_{args.genre.lower()}"
        print(f"\n장르 필터 '{args.genre}': {before}편 -> {len(pred)}편")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_path = OUT_DIR / f"user{args.user}_cf_candidates{suffix}.csv"
    pred.sort_values(["predicted_rating", "n_neighbor_votes"], ascending=[False, False]) \
        .to_csv(all_path, index=False)
    print(f"미평가 후보(이웃 1명 이상 투표) {len(pred)}편 -> 저장: {all_path}")

    rec = pred[pred["n_neighbor_votes"] >= args.min_votes]
    rec = rec.sort_values(["predicted_rating", "n_neighbor_votes"], ascending=[False, False])
    rec_top = rec.head(args.top)

    rec_path = OUT_DIR / f"user{args.user}_cf_recommendations{suffix}.csv"
    rec_top.to_csv(rec_path, index=False)

    print(f"이웃 {args.min_votes}명 이상 투표한 후보 {len(rec)}편 중 예측 평점 상위 "
          f"{len(rec_top)}편 -> 저장: {rec_path}\n")
    print(f"[사용자 {args.user} 추천 상위 {len(rec_top)}편]")
    print(rec_top[["movieId", "title", "release_year", "predicted_rating",
                    "n_neighbor_votes", "bayesian_rating", "rating_count"]]
          .to_string(index=False))

    # 검증
    assert not set(rec_top["movieId"]) & target_seen, "이미 본 영화가 추천에 섞임"
    assert (rec_top["predicted_rating"].between(0.5, 5.0)).all()
    assert (rec_top["n_neighbor_votes"] >= args.min_votes).all()
    print("\n검증 통과: 추천 목록에 이미 본 영화 없음, 예측 평점 0.5~5.0 범위, 최소 투표 수 충족")


if __name__ == "__main__":
    main()
