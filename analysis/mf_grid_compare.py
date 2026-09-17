"""행렬분해 — 잠재요인 수(k) x 학습 횟수(epochs) 그리드 비교.

mf_train.py 와 같은 SGD 학습 방식(순수 내적 예측, 관측된 평점만 학습, 편향항 없음)을 쓰되,
k ∈ {2, 10, 20} x epochs ∈ {10, 30, 50} 9가지 조합의 최종 학습 RMSE를 한 번에 비교한다.

- **같은 k는 한 번만 연속으로 학습**한다(최대 50에포크까지) — 10/30/50 에포크 지점마다 그 시점의
  RMSE와 P·Q 스냅샷을 기록만 할 뿐, 매번 처음부터 다시 학습하지 않는다. 따라서 같은 k열의
  10/30/50 결과는 서로 다른 초기화가 아니라 **하나의 학습 궤적 위 세 지점**이다(초기화·셔플
  순서 모두 같은 시드 42).
- k(잠재요인 수)가 다르면 초기화 크기(P, Q 의 열 수)가 다르므로 각 k는 독립적으로 시드 42부터
  다시 초기화해 학습한다.

입력 : data/processed/ratings_split.csv (analysis/mf_data_split.py 산출물, split 열 필요)
출력 : outputs/tables/mf_grid_train_rmse.csv           (k x epochs 학습 RMSE 표)
       data/processed/mf_user_factors_k{k}_e{epoch}.csv, mf_item_factors_k{k}_e{epoch}.csv
       (--save-checkpoint 로 지정한 (k, epoch) 조합만 잠재요인 스냅샷 저장)

사용법 (프로젝트 루트에서):
    python analysis/mf_grid_compare.py --ks 2 10 20 --checkpoints 10 30 50 --lr 0.005 --seed 42 \
        --save-checkpoint 2 30

필요 패키지: pandas, numpy
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT_CSV = "data/processed/ratings_split.csv"
PROC_DIR = Path("data/processed")
OUT_DIR = Path("outputs/tables")


def train_mf_with_checkpoints(train_df: pd.DataFrame, user_idx: dict, movie_idx: dict,
                               k: int, checkpoints: list[int], lr: float = 0.005, seed: int = 42):
    """checkpoints 에 담긴 각 에포크 시점의 (P, Q) 스냅샷과 학습 RMSE 를 기록하며 max(checkpoints)
    에포크까지 한 번만 연속 학습한다. 반환: (snapshots: {epoch: (P, Q)}, history_df)."""
    rng = np.random.default_rng(seed)
    n_users = len(user_idx)
    n_movies = len(movie_idx)
    P = rng.normal(0.0, 0.1, size=(n_users, k))
    Q = rng.normal(0.0, 0.1, size=(n_movies, k))

    u_arr = train_df["userId"].map(user_idx).to_numpy()
    i_arr = train_df["movieId"].map(movie_idx).to_numpy()
    r_arr = train_df["rating"].to_numpy(dtype=float)
    n = len(train_df)

    checkpoint_set = set(checkpoints)
    max_epoch = max(checkpoints)
    snapshots = {}
    history = []
    for epoch in range(1, max_epoch + 1):
        order = rng.permutation(n)
        sq_err_sum = 0.0
        for idx in order:
            u = u_arr[idx]
            i = i_arr[idx]
            r = r_arr[idx]
            Pu = P[u]
            Qi = Q[i]
            pred = float(Pu @ Qi)
            err = r - pred
            sq_err_sum += err * err
            Pu_old = Pu.copy()
            Pu += lr * err * Qi
            Qi += lr * err * Pu_old
        rmse = float(np.sqrt(sq_err_sum / n))
        history.append({"k": k, "epoch": epoch, "train_rmse": round(rmse, 5)})
        if epoch in checkpoint_set:
            snapshots[epoch] = (P.copy(), Q.copy())
            print(f"  k={k:>2} epoch {epoch:2d}: train RMSE = {rmse:.4f}  (체크포인트 기록)")

    return snapshots, pd.DataFrame(history)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 10, 20])
    ap.add_argument("--checkpoints", type=int, nargs="+", default=[10, 30, 50])
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-checkpoint", type=int, nargs=2, metavar=("K", "EPOCH"), default=[2, 30],
                     help="잠재요인 행렬 스냅샷을 파일로 저장할 (k, epoch) 조합 (기본 2 30)")
    args = ap.parse_args()

    split_path = Path(SPLIT_CSV)
    if not split_path.exists():
        raise SystemExit(f"{SPLIT_CSV} 가 없습니다. 먼저 python analysis/mf_data_split.py 를 실행하세요.")

    data = pd.read_csv(split_path)
    train_df = data[data["split"] == "train"].reset_index(drop=True)

    user_ids = np.sort(data["userId"].unique())
    movie_ids = np.sort(data["movieId"].unique())
    user_idx = {u: i for i, u in enumerate(user_ids)}
    movie_idx = {m: i for i, m in enumerate(movie_ids)}

    print(f"학습 데이터: 평점 {len(train_df)}건 · 사용자 {len(user_ids)}명 · 영화 {len(movie_ids)}편")
    print(f"비교할 k: {args.ks} · 체크포인트(에포크): {args.checkpoints} · 학습률 {args.lr} · 시드 {args.seed}\n")

    save_k, save_epoch = args.save_checkpoint
    grid_rows = []
    all_history = []
    saved_snapshot = None

    for k in args.ks:
        snapshots, history = train_mf_with_checkpoints(
            train_df, user_idx, movie_idx, k, args.checkpoints, args.lr, args.seed
        )
        all_history.append(history)
        for ep in args.checkpoints:
            rmse = history.loc[history["epoch"] == ep, "train_rmse"].iloc[0]
            grid_rows.append({"k": k, "epochs": ep, "train_rmse": rmse})
        if k == save_k and save_epoch in snapshots:
            saved_snapshot = snapshots[save_epoch]

    grid = pd.DataFrame(grid_rows)
    grid_wide = grid.pivot(index="k", columns="epochs", values="train_rmse")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid_path = OUT_DIR / "mf_grid_train_rmse.csv"
    grid_wide.to_csv(grid_path)

    print("\n[k x 에포크 최종 학습 RMSE]")
    print(grid_wide.to_string())
    print(f"\n저장: {grid_path}")

    if saved_snapshot is not None:
        P, Q = saved_snapshot
        PROC_DIR.mkdir(parents=True, exist_ok=True)
        p_cols = [f"f{i}" for i in range(save_k)]
        p_df = pd.DataFrame(P, columns=p_cols)
        p_df.insert(0, "userId", user_ids)
        q_df = pd.DataFrame(Q, columns=p_cols)
        q_df.insert(0, "movieId", movie_ids)

        p_path = PROC_DIR / f"mf_user_factors_k{save_k}_e{save_epoch}.csv"
        q_path = PROC_DIR / f"mf_item_factors_k{save_k}_e{save_epoch}.csv"
        p_df.to_csv(p_path, index=False)
        q_df.to_csv(q_path, index=False)
        print(f"\n저장(체크포인트 k={save_k}, epoch={save_epoch}): {p_path}, {q_path}")

    # 검증
    assert grid_wide.shape == (len(args.ks), len(args.checkpoints))
    for k in args.ks:
        row = grid_wide.loc[k]
        assert row.iloc[-1] <= row.iloc[0], f"k={k}: 에포크가 늘어도 RMSE가 줄지 않음"
    print("\n검증 통과: 그리드 크기 일치, 모든 k에서 에포크가 늘수록 학습 RMSE 감소(또는 동일)")


if __name__ == "__main__":
    main()
