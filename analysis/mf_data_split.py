"""행렬분해 협업 필터링 — 학습/검증/평가 데이터 분할.

앞으로 만들 행렬분해(matrix factorization) 협업 필터링 모델의 첫 단계 — 사용자마다 자신의
평점 기록을 무작위로 나눠 학습(train)/검증(val)/평가(test) 세 묶음으로 쪼갠다. 전체 사용자를
세 그룹으로 나누는 게 아니라 **각 사용자의 평점들을** 나누는 것이므로, 사용자당 최소 평가 수가
20건인 이 데이터셋(ml-latest-small)에서는 거의 모든 사용자가 세 묶음 모두에 고르게 등장한다.

- 분할 비율: 사용자별 평점 수의 10%는 검증(val), 10%는 평가(test), 나머지 80%는 학습(train).
  각 건수는 반올림(round)으로 정한다(예: 21건이면 val 2 · test 2 · train 17).
- 시드: 42 — numpy Generator 하나를 전역으로 써서 사용자 순서대로 순차 셔플하므로,
  같은 시드면 항상 같은 분할이 재현된다.
- **단일 묶음 전용 영화 제외**: train/val/test 중 **딱 한 묶음에서만** 관측되는 영화는 세 묶음
  전부에서 제외한다 — val·test에만 있는 영화는 학습이 전혀 안 돼 예측이 불가능하고(콜드스타트),
  train에만 있는 영화도 검증·평가로는 한 번도 쓰이지 않아 이 실습에서는 의미가 없기 때문이다.
  둘 이상의 묶음에 걸쳐 있는 영화는 그대로 둔다. 제외 기준과 건수는 실행 시 출력한다.
- 검증: 분할·제외 후 평점 총 건수가 앞뒤로 맞는지, 제외된 영화가 실제로 한 묶음에만 있었는지,
  남은 영화가 모두 2묶음 이상에 걸쳐 있는지 확인한다.

입력 : data/processed/ratings.csv
출력 : data/processed/ratings_split.csv (컬럼: userId, movieId, rating, split — 단일 묶음
       전용 영화 제외 후 최종본)

사용법 (프로젝트 루트에서):
    python analysis/mf_data_split.py --val-frac 0.1 --test-frac 0.1 --seed 42

필요 패키지: pandas, numpy
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RATINGS_CSV = "data/processed/ratings.csv"
OUT_CSV = Path("data/processed/ratings_split.csv")


def split_ratings(ratings: pd.DataFrame, val_frac: float = 0.1, test_frac: float = 0.1,
                   seed: int = 42) -> pd.DataFrame:
    """사용자마다 무작위로 train/val/test 를 배정한 'split' 열을 추가해 반환한다."""
    rng = np.random.default_rng(seed)
    out = ratings.reset_index(drop=True).copy()
    split = np.full(len(out), "train", dtype=object)

    for uid, group in out.groupby("userId"):
        idx = group.index.to_numpy()
        n = len(idx)
        shuffled = idx[rng.permutation(n)]
        n_val = round(n * val_frac)
        n_test = round(n * test_frac)
        split[shuffled[:n_val]] = "val"
        split[shuffled[n_val:n_val + n_test]] = "test"

    out["split"] = split
    return out


def filter_single_split_movies(out: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """train/val/test 중 단 한 묶음에서만 관측되는 영화를 전체 데이터(세 묶음 모두)에서 제외한다.
    반환: (필터링된 df, 제외된 영화의 평점 행 df — 어느 묶음에 있었는지 확인용)."""
    n_splits_per_movie = out.groupby("movieId")["split"].nunique()
    single_split_movie_ids = n_splits_per_movie[n_splits_per_movie == 1].index
    excluded = out[out["movieId"].isin(single_split_movie_ids)]
    filtered = out[~out["movieId"].isin(single_split_movie_ids)].reset_index(drop=True)
    return filtered, excluded


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ratings = pd.read_csv(RATINGS_CSV)
    n_users_total = ratings["userId"].nunique()

    n_movies_total = ratings["movieId"].nunique()
    split_raw = split_ratings(ratings, args.val_frac, args.test_frac, args.seed)

    print(f"전체 평점 {len(split_raw)}건 · 사용자 {n_users_total}명 · 영화 {n_movies_total}편")
    print(f"분할 비율: val {args.val_frac:.0%} · test {args.test_frac:.0%} · train 나머지 "
          f"(사용자별로 반올림해서 배정, 시드 {args.seed})\n")

    out, excluded = filter_single_split_movies(split_raw)
    excluded_movie_ids = excluded["movieId"].unique()
    excluded_by_split = excluded.groupby("movieId")["split"].first().value_counts().reindex(
        ["train", "val", "test"], fill_value=0
    )
    print(f"[단일 묶음 전용 영화 제외] 영화 {len(excluded_movie_ids)}편, 평점 {len(excluded)}건 제외")
    print(f"  제외된 영화가 있던 묶음별 개수: train만 {excluded_by_split['train']}편 · "
          f"val만 {excluded_by_split['val']}편 · test만 {excluded_by_split['test']}편")
    print(f"  제외 후 영화 {out['movieId'].nunique()}편, 평점 {len(out)}건 남음\n")

    summary = out.groupby("split").agg(
        n_ratings=("rating", "size"),
        n_users=("userId", "nunique"),
        n_movies=("movieId", "nunique"),
    ).reindex(["train", "val", "test"])
    summary["ratings_pct"] = (summary["n_ratings"] / len(out) * 100).round(2)
    print("[제외 후 묶음별 사용자 수 · 영화 수 · 평점 수]")
    print(summary.to_string())

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\n저장: {OUT_CSV}")

    # 검증
    assert len(out) + len(excluded) == len(ratings), "분할·제외 전후 총 평점 수가 안 맞음"
    assert set(out["split"].unique()) <= {"train", "val", "test"}
    dup_check = out.groupby(["userId", "movieId"]).size()
    assert (dup_check == 1).all(), "동일 (userId, movieId) 평점이 여러 묶음에 중복 배정됨"
    remaining_splits_per_movie = out.groupby("movieId")["split"].nunique()
    assert (remaining_splits_per_movie >= 2).all(), "제외 후에도 단일 묶음 전용 영화가 남아있음"
    users_missing = summary["n_users"][summary["n_users"] < n_users_total]
    if len(users_missing):
        print(f"\n참고: 영화 제외로 일부 사용자가 특정 묶음에서 전부 빠졌습니다 — "
              f"{users_missing.to_dict()} (전체 {n_users_total}명 기준)")
    print(f"\n검증 통과: 분할·제외 전후 총 평점 수 일치, (userId, movieId) 쌍 중복 없음, "
          f"남은 영화는 모두 2묶음 이상에 걸쳐 있음")


if __name__ == "__main__":
    main()
