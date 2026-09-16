"""협업 필터링 정확도 검증 — 실제 평점 일부를 가리고(holdout) 예측과 비교.

대상 사용자가 **실제로 평점을 준 영화 중 일부를 무작위로 골라 "안 본 척" 가린 뒤**,
그 사용자의 나머지 평점만으로 사용자 기반 CF(user_cf_predict.predict_ratings)와
아이템 기반 CF(item_based_cf.item_based_predict) 두 방식 모두로 그 영화들의 평점을 예측하고,
실제로 그 사용자가 줬던 평점과 비교해 오차(절대오차, MAE)를 계산한다 — 두 CF 방식의
정확도를 직접 비교하기 위한 검증이다.

가림(holdout) 방식
- 대상 사용자의 평점 중 **전체 데이터에서 해당 영화의 평가 수(rating_count) ≥ 20** 인 것만
  후보로 삼는다 — 너무 소수만 평가한 영화는 어느 방식으로도 이웃/유사 영화를 안정적으로
  찾기 어려워, 가려도 의미 있는 비교가 되지 않기 때문(제외 기준을 명시).
- 그중 시드 고정 난수(--seed, 기본 42)로 --n-holdout(기본 15)편을 뽑아, **대상 사용자의
  이 영화들에 대한 평점 행만** 학습 데이터에서 제거한다(다른 사용자의 평점은 그대로 둔다 —
  이웃/유사 영화 계산에 필요한 "정답" 데이터이므로).
- 이렇게 만든 학습용 평점으로 이웃(사용자 기반)/유사 영화(아이템 기반)를 다시 계산해 예측하고,
  실제 평점(가리기 전 원본)과 비교한다.

입력 : data/processed/ratings.csv, data/processed/movies_with_ratings.csv
출력 : outputs/tables/user{U}_cf_holdout_eval.csv

사용법 (프로젝트 루트에서):
    python analysis/cf_holdout_eval.py --user 414 --min-common 30 --k 50 --n-holdout 15

필요 패키지: pandas, numpy, scipy, scikit-learn
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from user_cf_neighbors import find_neighbors_auto
from user_cf_predict import predict_ratings
from item_based_cf import item_based_predict

RATINGS_CSV = "data/processed/ratings.csv"
MOVIES_CSV = "data/processed/movies_with_ratings.csv"
OUT_DIR = Path("outputs/tables")
MIN_RATING_COUNT = 20  # 가릴 영화 후보의 최소 전체 평가 수(제외 기준)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", type=int, default=414)
    ap.add_argument("--min-common", type=int, default=30)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--n-holdout", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ratings = pd.read_csv(RATINGS_CSV)
    movies = pd.read_csv(MOVIES_CSV)

    user_ratings = ratings[ratings.userId == args.user].merge(
        movies[["movieId", "title", "rating_count"]], on="movieId", how="left"
    )
    eligible = user_ratings[user_ratings["rating_count"] >= MIN_RATING_COUNT]
    print(f"기준 사용자: {args.user}  (전체 평점 {len(user_ratings)}건 중 "
          f"전체 평가 수 {MIN_RATING_COUNT}건 이상인 영화 {len(eligible)}편이 가림 후보)")

    rng = np.random.default_rng(args.seed)
    n = min(args.n_holdout, len(eligible))
    holdout = eligible.sample(n=n, random_state=rng.integers(0, 2**31 - 1)).sort_values("movieId")
    holdout_ids = holdout["movieId"].tolist()
    print(f"가린 영화 {n}편 (시드 {args.seed}): {holdout_ids}\n")

    # 대상 사용자의 가린 영화 평점 행만 제거한 학습용 데이터
    mask = (ratings.userId == args.user) & (ratings.movieId.isin(holdout_ids))
    train_ratings = ratings[~mask]
    assert len(train_ratings) == len(ratings) - n

    # ---- 사용자 기반 CF ----
    neighbors, used_min_common = find_neighbors_auto(train_ratings, args.user, args.min_common, args.k)
    if used_min_common != args.min_common:
        print(f"[사용자 기반] 공통 평가 영화 {args.min_common}편 이상 조건에서 이웃을 찾지 못해 "
              f"{used_min_common}편으로 자동 완화")
    user_pred = predict_ratings(train_ratings, args.user, neighbors, candidate_movie_ids=holdout_ids)
    user_pred = user_pred.rename(columns={"predicted_rating": "user_cf_pred",
                                           "n_neighbor_votes": "user_cf_votes"})

    # ---- 아이템 기반 CF ----
    item_pred = item_based_predict(train_ratings, args.user, args.min_common, args.k,
                                    candidate_movie_ids=holdout_ids)
    item_pred = item_pred.rename(columns={"predicted_rating": "item_cf_pred",
                                           "n_item_votes": "item_cf_votes"})

    result = holdout[["movieId", "title", "rating"]].rename(columns={"rating": "actual_rating"})
    result = result.merge(user_pred, on="movieId", how="left")
    result = result.merge(item_pred, on="movieId", how="left")
    result["user_cf_abs_err"] = (result["user_cf_pred"] - result["actual_rating"]).abs().round(3)
    result["item_cf_abs_err"] = (result["item_cf_pred"] - result["actual_rating"]).abs().round(3)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"user{args.user}_cf_holdout_eval.csv"
    result.to_csv(out_path, index=False)

    print("[가린 영화별 실제 평점 vs 예측 평점]")
    print(result[["movieId", "title", "actual_rating",
                   "user_cf_pred", "user_cf_votes", "user_cf_abs_err",
                   "item_cf_pred", "item_cf_votes", "item_cf_abs_err"]]
          .to_string(index=False))

    n_user_valid = result["user_cf_pred"].notna().sum()
    n_item_valid = result["item_cf_pred"].notna().sum()
    user_mae = result["user_cf_abs_err"].mean()
    item_mae = result["item_cf_abs_err"].mean()
    print(f"\n[요약] 가린 영화 {n}편 중 예측 성공: 사용자 기반 {n_user_valid}편, 아이템 기반 {n_item_valid}편")
    print(f"MAE(평균 절대오차, 각자 예측에 성공한 영화 기준): 사용자 기반 {user_mae:.3f} · "
          f"아이템 기반 {item_mae:.3f}  ({'아이템' if item_mae < user_mae else '사용자'} 기반이 더 정확)")

    both = result[result["user_cf_pred"].notna() & result["item_cf_pred"].notna()]
    if len(both) > 0:
        print(f"\n[공정 비교] 두 방식 모두 예측에 성공한 {len(both)}편만 기준으로 다시 비교"
              f"(min_common={args.min_common} 조건이 아이템 기반에서 더 엄격해 예측 성공 영화 수가"
              f" 서로 달라, 전체 평균만 보면 오해할 수 있어 동일 표본으로 다시 계산)")
        print(f"MAE: 사용자 기반 {both['user_cf_abs_err'].mean():.3f} · "
              f"아이템 기반 {both['item_cf_abs_err'].mean():.3f}")

    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
