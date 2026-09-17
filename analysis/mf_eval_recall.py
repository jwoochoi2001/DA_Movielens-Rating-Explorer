"""행렬분해(편향 보정) 모델 추천 성능 평가 — Recall@N (평점 4점 이상 영화 재현율).

mf_recommend.py 와 같은 학습 방식(biased MF, mf_train.train_mf)으로 모델을 **한 번만** 학습한
뒤, 무작위로 뽑은 사용자 --n-users 명에 대해 각자 이렇게 계산한다.
  1. **정답**: 평가(test) 세트에서 그 사용자가 4점 이상 준 영화 = "실제로 좋아한 영화"
  2. **추천**: 모델이 만든 예측 평점 상위 --top 개 목록(그 사용자가 이미 평가한 영화는 전부 제외)
  3. **재현율(recall)**: 정답 중 추천 목록에 실제로 들어간 비율 = (정답 ∩ 추천) / 정답 개수

- 정답을 test(평가용) 세트로 삼는 이유: train 은 모델이 이미 학습에 써서 "찾아내는 게 당연"하고,
  val(검증용)은 하이퍼파라미터 튜닝에 남겨두는 게 관례라서, 순수 미지 데이터인 test 로 평가한다.
- 사용자 표본은 시드로 고정해 재현 가능하다(기본 표본 시드는 학습 시드와 같은 42).
- test 세트에 4점 이상 준 영화가 하나도 없는 사용자는 재현율을 정의할 수 없어(분모 0) 표에는
  "-"로 표시하고, 전체 평균(micro/macro) 계산에서는 제외한다.
- **micro 재현율** = (20명 전체 적중 수 합) / (20명 전체 정답 수 합) — 정답이 많은 사용자에게
  자연히 더 큰 비중이 실린다.
- **macro 재현율** = 사용자별 재현율(%)의 단순 평균 — 정답 수와 무관하게 사용자 한 명 한 명을
  동등하게 취급한다.

입력 : data/processed/ratings_split.csv, data/processed/movies_with_ratings.csv
출력 : outputs/tables/mf_recall_by_user.csv

사용법 (프로젝트 루트에서):
    python analysis/mf_eval_recall.py --k 2 --epochs 30 --lr 0.005 --seed 42 --n-users 20 --top 10

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
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-users", type=int, default=20)
    ap.add_argument("--top", type=int, default=10, help="사용자당 추천 목록 크기(Recall@N 의 N)")
    ap.add_argument("--like-threshold", type=float, default=4.0)
    ap.add_argument("--sample-seed", type=int, default=None, help="사용자 표본 시드(기본: --seed 와 동일)")
    ap.add_argument("--no-bias", dest="use_bias", action="store_false", default=True)
    args = ap.parse_args()
    sample_seed = args.sample_seed if args.sample_seed is not None else args.seed

    split_path = Path(SPLIT_CSV)
    if not split_path.exists():
        raise SystemExit(f"{SPLIT_CSV} 가 없습니다. 먼저 python analysis/mf_data_split.py 를 실행하세요.")

    data = pd.read_csv(split_path)
    train_df = data[data["split"] == "train"].reset_index(drop=True)
    test_df = data[data["split"] == "test"]

    user_ids = np.sort(data["userId"].unique())
    movie_ids = np.sort(data["movieId"].unique())
    user_idx = {u: i for i, u in enumerate(user_ids)}
    movie_idx = {m: i for i, m in enumerate(movie_ids)}

    print(f"학습: k={args.k} · 에포크 {args.epochs}회 · 학습률 {args.lr} · 시드 {args.seed} · "
          f"편향(μ+b_u+b_i) {'사용' if args.use_bias else '미사용'}")
    P, Q, b_u, b_i, mu, history = train_mf(
        train_df, user_idx, movie_idx, args.k, args.epochs, args.lr, args.seed, args.use_bias
    )
    print(f"학습 완료 — 최종 학습 데이터 RMSE {history['train_rmse'].iloc[-1]:.4f}\n")

    rng = np.random.default_rng(sample_seed)
    sampled_users = np.sort(rng.choice(user_ids, size=args.n_users, replace=False))
    print(f"무작위 표본(시드 {sample_seed}) 사용자 {args.n_users}명: {list(sampled_users)}\n")

    train_movie_ids_arr = np.array(sorted(train_df["movieId"].unique()))
    train_movie_cols = np.array([movie_idx[m] for m in train_movie_ids_arr])

    rows = []
    for uid in sampled_users:
        u_i = user_idx[uid]
        seen_ids = set(data.loc[data["userId"] == uid, "movieId"])
        cand_mask = ~np.isin(train_movie_ids_arr, np.fromiter(seen_ids, dtype=int, count=len(seen_ids)))
        cand_ids = train_movie_ids_arr[cand_mask]
        cand_cols = train_movie_cols[cand_mask]

        interaction = Q[cand_cols] @ P[u_i]
        pred_raw = mu + b_u[u_i] + b_i[cand_cols] + interaction
        pred = np.clip(pred_raw, 0.5, 5.0)

        top_order = np.argsort(-pred)[:args.top]
        rec_ids = set(cand_ids[top_order])

        liked_test = set(test_df.loc[(test_df["userId"] == uid) &
                                      (test_df["rating"] >= args.like_threshold), "movieId"])
        n_liked = len(liked_test)
        n_hits = len(liked_test & rec_ids)
        recall_pct = round(n_hits / n_liked * 100, 1) if n_liked > 0 else np.nan

        rows.append({
            "userId": uid, "n_liked_test": n_liked, "n_hits": n_hits, "recall_pct": recall_pct,
        })

    result = pd.DataFrame(rows)

    total_liked = int(result["n_liked_test"].sum())
    total_hits = int(result["n_hits"].sum())
    micro_recall = total_hits / total_liked * 100 if total_liked > 0 else float("nan")
    valid = result[result["n_liked_test"] > 0]
    macro_recall = float(valid["recall_pct"].mean()) if len(valid) else float("nan")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "mf_recall_by_user.csv"
    result.to_csv(out_path, index=False)

    print(f"[사용자별 Recall@{args.top} — test 세트에서 {args.like_threshold}점 이상 준 영화 기준]")
    disp = result.copy()
    disp["recall_pct"] = disp["recall_pct"].map(lambda v: "-" if pd.isna(v) else f"{v:.1f}%")
    print(disp.to_string(index=False))

    print(f"\n[전체 요약] 표본 {args.n_users}명 중 test에 {args.like_threshold}점↑ 영화가 있는 "
          f"사용자 {len(valid)}명(나머지 {args.n_users - len(valid)}명은 정답이 없어 계산 제외)")
    print(f"micro 재현율(전체 적중 {total_hits} / 전체 정답 {total_liked}) = {micro_recall:.1f}%")
    print(f"macro 재현율(사용자별 비율의 평균) = {macro_recall:.1f}%")
    print(f"\n저장: {out_path}")

    # 검증
    assert len(result) == args.n_users
    assert (result["n_hits"] <= result["n_liked_test"]).all()
    assert (result["n_hits"] <= args.top).all()
    print("\n검증 통과: 사용자 수 일치, 적중 수는 정답 수·추천 목록 크기 이하")


if __name__ == "__main__":
    main()
