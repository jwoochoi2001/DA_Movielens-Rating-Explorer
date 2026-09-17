"""행렬분해(편향 보정) 모델로 특정 사용자에게 아직 안 본 영화 추천 — 장르 필터 지원.

mf_train.py 의 train_mf() 를 그대로 재사용해 지정한 k·epochs·lr·seed 로 학습한 뒤,
pred(u,i) = μ + b_u[u] + b_i[i] + P[u]·Q[i] (0.5~5.0으로 clip) 로 대상 사용자가 아직 평가하지
않은 영화의 평점을 전부 예측하고, 장르 필터(옵션) 후 예측 평점 상위 N편을 추천한다.

- 후보 영화는 **학습(train) 데이터에 실제로 등장한 영화**로 제한한다 — val·test에만 있는 영화는
  그 Q 벡터가 한 번도 갱신되지 않아(무작위 초기값 그대로) 예측이 무의미하기 때문이다.
- 대상 사용자가 이미 평가한 영화(학습·검증·평가 어느 묶음이든)는 후보에서 제외한다.

입력 : data/processed/ratings_split.csv, data/processed/movies_with_ratings.csv
출력 : outputs/tables/user{U}_mf_recommendations[_{장르}].csv

사용법 (프로젝트 루트에서):
    python analysis/mf_recommend.py --user 414 --k 2 --epochs 30 --genre Action --top 10

필요 패키지: pandas, numpy
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mf_train import train_mf

SPLIT_CSV = "data/processed/ratings_split.csv"
MOVIES_CSV = "data/processed/movies_with_ratings.csv"
OUT_DIR = Path("outputs/tables")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", type=int, default=414)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--genre", type=str, default=None, help="장르 부분일치 필터(예: Action)")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--no-bias", dest="use_bias", action="store_false", default=True,
                     help="편향(μ+b_u+b_i) 없이 순수 내적만 쓰려면 지정")
    args = ap.parse_args()

    split_path = Path(SPLIT_CSV)
    if not split_path.exists():
        raise SystemExit(f"{SPLIT_CSV} 가 없습니다. 먼저 python analysis/mf_data_split.py 를 실행하세요.")

    data = pd.read_csv(split_path)
    movies = pd.read_csv(MOVIES_CSV)
    train_df = data[data["split"] == "train"].reset_index(drop=True)

    user_ids = np.sort(data["userId"].unique())
    movie_ids = np.sort(data["movieId"].unique())
    user_idx = {u: i for i, u in enumerate(user_ids)}
    movie_idx = {m: i for i, m in enumerate(movie_ids)}

    if args.user not in user_idx:
        raise SystemExit(f"사용자 {args.user} 가 데이터에 없습니다.")

    print(f"학습: k={args.k} · 에포크 {args.epochs}회 · 학습률 {args.lr} · 시드 {args.seed} · "
          f"편향(μ+b_u+b_i) {'사용' if args.use_bias else '미사용'}")
    P, Q, b_u, b_i, mu, history = train_mf(
        train_df, user_idx, movie_idx, args.k, args.epochs, args.lr, args.seed, args.use_bias
    )
    print(f"학습 완료 — 최종 학습 데이터 RMSE {history['train_rmse'].iloc[-1]:.4f}\n")

    train_movie_ids = set(train_df["movieId"].unique())
    seen_ids = set(data.loc[data["userId"] == args.user, "movieId"])
    candidate_ids = np.array(sorted(train_movie_ids - seen_ids))

    u_i = user_idx[args.user]
    cand_cols = np.array([movie_idx[m] for m in candidate_ids])
    interaction = Q[cand_cols] @ P[u_i]
    pred_raw = mu + b_u[u_i] + b_i[cand_cols] + interaction
    pred_clipped = np.clip(pred_raw, 0.5, 5.0)

    rec = pd.DataFrame({
        "movieId": candidate_ids,
        "predicted_rating_raw": pred_raw.round(3),
        "predicted_rating": pred_clipped.round(3),
    })
    rec = rec.merge(
        movies[["movieId", "title", "genres", "release_year", "bayesian_rating", "rating_count"]],
        on="movieId", how="left",
    )

    suffix = ""
    if args.genre:
        before = len(rec)
        rec = rec[rec["genres"].str.contains(args.genre, case=False, na=False)]
        suffix = f"_{args.genre.lower()}"
        print(f"장르 필터 '{args.genre}': {before}편 -> {len(rec)}편")

    rec = rec.sort_values("predicted_rating", ascending=False)
    rec_top = rec.head(args.top)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"user{args.user}_mf_recommendations{suffix}.csv"
    rec_top.to_csv(out_path, index=False)

    print(f"[사용자 {args.user} 추천 상위 {len(rec_top)}편 (행렬분해, k={args.k}, epochs={args.epochs}, "
          f"편향 {'적용' if args.use_bias else '미적용'})]")
    print(rec_top[["movieId", "title", "release_year", "predicted_rating",
                    "bayesian_rating", "rating_count"]].to_string(index=False))
    print(f"\n저장: {out_path}")

    assert not (set(rec_top["movieId"]) & seen_ids), "이미 본 영화가 추천에 섞임"
    assert (rec_top["predicted_rating"].between(0.5, 5.0)).all()
    print("\n검증 통과: 추천 목록에 이미 본 영화 없음, 예측 평점 0.5~5.0 범위")


if __name__ == "__main__":
    main()
