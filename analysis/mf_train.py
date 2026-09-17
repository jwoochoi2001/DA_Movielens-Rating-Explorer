"""행렬분해(matrix factorization) 협업 필터링 — 경사하강법(SGD)으로 잠재요인 + 편향 학습.

analysis/mf_data_split.py 가 만든 data/processed/ratings_split.csv 의 **학습(train) 데이터만**
사용해, 사용자 잠재요인 행렬 P(사용자 수 x k)와 영화 잠재요인 행렬 Q(영화 수 x k), 그리고
**전역평균(μ) + 사용자 편향(b_u) + 영화 편향(b_i)** 을 확률적 경사하강법으로 함께 학습한다.

    pred(u, i) = μ + b_u[u] + b_i[i] + P[u]·Q[i]

쉬운 말로 풀면, 평점 하나를 네 조각으로 나눠 설명하는 것과 같다.
1) **μ (전역평균)**: "평점이란 게 원래 대략 이 정도다"라는 전체 기준선(학습 데이터 평점의 평균,
   고정값 — 학습 중 갱신하지 않는다).
2) **b_u[u] (사용자 편향)**: "이 사람은 평소에 다른 사람보다 평점을 후하게/박하게 준다"는
   그 사용자만의 습관 보정치. 양수면 평균보다 후한 사람, 음수면 박한 사람.
3) **b_i[i] (영화 편향)**: "이 영화는 원래 다들 대체로 높게/낮게 평가한다"는 그 영화만의
   인기·평판 보정치. 양수면 대체로 호평받는 영화, 음수면 대체로 박한 평을 받는 영화.
4) **P[u]·Q[i] (잠재요인 내적)**: 위 세 가지로 설명 안 되는 **"이 사람과 이 영화 사이의 궁합"**
   — 예를 들어 이 사람이 유독 이 영화 장르/스타일을 좋아하는지 같은, 사람·영화 조합에서만
   나오는 나머지 효과.
   즉 "평균 + 이 사람 성향 + 이 영화 평판 + 이 사람과 이 영화의 케미"를 다 더한 게 예측 평점이다.

- 잠재요인 수(k): 2 (기본값, --k 로 조절 가능)
- 학습 대상: **실제로 평가가 관측된 (u,i,r) 쌍에서만** 오차를 계산·역전파한다 — 평가하지 않은
  항목은 0으로 채운 밀집 행렬을 만들지 않고, 아예 학습 루프(=SGD가 도는 (u,i,r) 목록)에서 제외한다.
- 업데이트 규칙(관측된 (u,i,r) 하나마다, 갱신 전 P[u] 값을 Q 업데이트에 그대로 사용):
    pred = μ + b_u[u] + b_i[i] + P[u]·Q[i]
    err  = r - pred
    b_u[u] <- b_u[u] + lr * err        # "이 사람 성향" 보정
    b_i[i] <- b_i[i] + lr * err        # "이 영화 평판" 보정
    P[u]   <- P[u] + lr * err * Q[i]
    Q[i]   <- Q[i] + lr * err * P[u](갱신 전 값)
  네 항 모두 같은 오차(err)로 동시에(같은 스텝에서) 갱신한다 — μ만 고정, 나머지 셋은 매번 조금씩
  더 정확해지도록 움직인다.
- --no-bias 를 주면 μ·b_u·b_i 를 전부 0으로 고정해(펼치면 pred = P[u]·Q[i]) 편향 없는
  이전 버전과 동일하게 동작한다 — 편향 유무 비교용.
- 하이퍼파라미터: 학습률(lr) 0.005, 에포크 10, 시드 42 — 이 시드 하나로 P·Q 초기화와 매 에포크
  샘플 순서 셔플을 전부 재현 가능하게 한다.
- 초기화: P, Q 는 N(0, 0.1) 작은 난수, b_u·b_i 는 0에서 시작, μ 는 학습 데이터 평점의 평균으로
  고정(전부 0으로 시작하면 그래디언트도 0이라 전혀 학습되지 않는다).
- 정규화(L2)는 이번 버전에도 없다 — 편향항만 추가 요청받았기 때문.
- **평점 범위 조정(clip)**: 위 합산값은 범위 제약이 없어 0.5~5.0 을 벗어난 값이 나올 수 있다.
  학습(SGD 오차·그래디언트 계산)에는 원본 값을 그대로 쓰고, 최종적으로 저장/출력하는
  predicted_rating 열만 `np.clip(pred, 0.5, 5.0)` 으로 잘라낸다 — predicted_rating_raw 열에
  조정 전 원본값도 함께 남겨 비교할 수 있게 한다.

입력 : data/processed/ratings_split.csv (split 열 필요 — 먼저 analysis/mf_data_split.py 실행)
출력 : data/processed/mf_user_factors.csv   (userId, f0, f1, ..., bias)
       data/processed/mf_item_factors.csv   (movieId, f0, f1, ..., bias)
       outputs/tables/mf_train_loss_by_epoch.csv   (epoch, train_rmse)
       outputs/tables/mf_train_predictions_sample.csv  (샘플 예측 vs 실제)

사용법 (프로젝트 루트에서):
    python analysis/mf_train.py --k 2 --epochs 10 --lr 0.005 --seed 42
    python analysis/mf_train.py --no-bias   # 편향 없이(이전 버전과 동일) 비교용

필요 패키지: pandas, numpy
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT_CSV = "data/processed/ratings_split.csv"
PROC_DIR = Path("data/processed")
OUT_DIR = Path("outputs/tables")


def train_mf(train_df: pd.DataFrame, user_idx: dict, movie_idx: dict,
             k: int = 2, epochs: int = 10, lr: float = 0.005, seed: int = 42, use_bias: bool = True):
    """SGD로 P, Q(와 use_bias=True 면 μ, b_u, b_i)를 학습한다.
    반환: (P, Q, b_u, b_i, mu, history_df[epoch, train_rmse])."""
    rng = np.random.default_rng(seed)
    n_users = len(user_idx)
    n_movies = len(movie_idx)
    P = rng.normal(0.0, 0.1, size=(n_users, k))
    Q = rng.normal(0.0, 0.1, size=(n_movies, k))
    b_u = np.zeros(n_users)
    b_i = np.zeros(n_movies)
    mu = float(train_df["rating"].mean()) if use_bias else 0.0

    u_arr = train_df["userId"].map(user_idx).to_numpy()
    i_arr = train_df["movieId"].map(movie_idx).to_numpy()
    r_arr = train_df["rating"].to_numpy(dtype=float)
    n = len(train_df)

    history = []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        sq_err_sum = 0.0
        for idx in order:
            u = u_arr[idx]
            i = i_arr[idx]
            r = r_arr[idx]
            Pu = P[u]
            Qi = Q[i]
            pred = mu + b_u[u] + b_i[i] + float(Pu @ Qi)
            err = r - pred
            sq_err_sum += err * err
            if use_bias:
                b_u[u] += lr * err
                b_i[i] += lr * err
            Pu_old = Pu.copy()
            Pu += lr * err * Qi
            Qi += lr * err * Pu_old
        rmse = float(np.sqrt(sq_err_sum / n))
        history.append({"epoch": epoch, "train_rmse": round(rmse, 5)})
        print(f"  epoch {epoch:2d}/{epochs}: train RMSE = {rmse:.4f}")

    return P, Q, b_u, b_i, mu, pd.DataFrame(history)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=2, help="잠재요인 수(기본 2)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-size", type=int, default=20, help="예측 샘플로 보여줄 행 수")
    ap.add_argument("--no-bias", dest="use_bias", action="store_false", default=True,
                     help="전역평균+사용자편향+영화편향 없이 순수 내적만 쓰려면 지정(이전 버전과 동일)")
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

    print(f"학습 데이터: 평점 {len(train_df)}건 · 사용자 {len(user_ids)}명 · 영화 {len(movie_ids)}편 "
          f"(평가하지 않은 항목은 학습 루프에 아예 없음)")
    print(f"잠재요인 수 k={args.k} · 에포크 {args.epochs}회 · 학습률 {args.lr} · 시드 {args.seed} · "
          f"편향(μ+b_u+b_i) {'사용' if args.use_bias else '미사용'}\n")

    P, Q, b_u, b_i, mu, history = train_mf(
        train_df, user_idx, movie_idx, args.k, args.epochs, args.lr, args.seed, args.use_bias
    )
    if args.use_bias:
        print(f"\n전역평균 μ = {mu:.4f} (학습 데이터 평점의 평균, 학습 중 고정)")
        print(f"사용자 편향 b_u: 평균 {b_u.mean():+.4f} · 범위 [{b_u.min():+.4f}, {b_u.max():+.4f}]")
        print(f"영화 편향 b_i: 평균 {b_i.mean():+.4f} · 범위 [{b_i.min():+.4f}, {b_i.max():+.4f}]")

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    p_cols = [f"f{i}" for i in range(args.k)]
    p_df = pd.DataFrame(P, columns=p_cols)
    p_df.insert(0, "userId", user_ids)
    p_df["bias"] = b_u
    p_path = PROC_DIR / "mf_user_factors.csv"
    p_df.to_csv(p_path, index=False)

    q_df = pd.DataFrame(Q, columns=p_cols)
    q_df.insert(0, "movieId", movie_ids)
    q_df["bias"] = b_i
    q_path = PROC_DIR / "mf_item_factors.csv"
    q_df.to_csv(q_path, index=False)

    loss_path = OUT_DIR / "mf_train_loss_by_epoch.csv"
    history.to_csv(loss_path, index=False)

    # 학습 데이터 전체에 대한 예측 = μ + b_u[u] + b_i[i] + P[u]·Q[i]
    u_pos = train_df["userId"].map(user_idx).to_numpy()
    i_pos = train_df["movieId"].map(movie_idx).to_numpy()
    interaction = np.einsum("nk,nk->n", P[u_pos], Q[i_pos])
    pred_all = mu + b_u[u_pos] + b_i[i_pos] + interaction
    # P·Q 내적은 범위 제약이 없어 0.5~5.0 을 벗어날 수 있다 — 실제 평점 스케일을 벗어난 값은
    # 의미가 없으므로, 사용자에게 보여주는/저장하는 예측값만 이 범위로 잘라낸다(clip). 학습 자체
    # (SGD 오차·그래디언트)에는 원래 값을 그대로 쓰고, 이 클리핑은 순전히 표시·평가용이다.
    n_out_of_range = int(((pred_all < 0.5) | (pred_all > 5.0)).sum())
    pred_clipped = np.clip(pred_all, 0.5, 5.0)
    train_df = train_df.assign(
        predicted_rating_raw=np.round(pred_all, 3),
        predicted_rating=np.round(pred_clipped, 3),
    )
    raw_rmse = float(np.sqrt(np.mean((train_df["rating"] - train_df["predicted_rating_raw"]) ** 2)))
    final_rmse = float(np.sqrt(np.mean((train_df["rating"] - train_df["predicted_rating"]) ** 2)))

    sample = train_df.sample(n=min(args.sample_size, len(train_df)), random_state=args.seed) \
        .sort_values(["userId", "movieId"])
    sample_path = OUT_DIR / "mf_train_predictions_sample.csv"
    sample[["userId", "movieId", "rating", "predicted_rating_raw", "predicted_rating"]] \
        .to_csv(sample_path, index=False)

    print(f"\n[학습 완료] 0.5~5.0 범위를 벗어난 원본 예측 {n_out_of_range}건"
          f"({n_out_of_range / len(train_df) * 100:.2f}%) 을 clip으로 조정")
    print(f"최종 학습 데이터 RMSE: 조정 전 {raw_rmse:.4f} · 조정 후(0.5~5.0) {final_rmse:.4f}")
    print(f"저장: {p_path} (사용자 잠재요인) · {q_path} (영화 잠재요인)")
    print(f"저장: {loss_path} (에포크별 학습 손실)")
    print(f"저장: {sample_path} (예측 샘플 {len(sample)}건)\n")

    print(f"[학습 데이터 예측 샘플 {len(sample)}건 — 실제 평점 vs 예상 평점(조정 전/후)]")
    print(sample[["userId", "movieId", "rating", "predicted_rating_raw", "predicted_rating"]]
          .to_string(index=False))

    # 검증
    assert P.shape == (len(user_ids), args.k)
    assert Q.shape == (len(movie_ids), args.k)
    assert history["train_rmse"].is_monotonic_decreasing or history["train_rmse"].iloc[-1] < history["train_rmse"].iloc[0], \
        "10 에포크 동안 학습 손실이 줄지 않음 — 학습률/초기화를 확인하세요"
    print("\n검증 통과: P·Q 크기 일치, 학습 손실이 첫 에포크보다 마지막 에포크에서 더 낮음")


if __name__ == "__main__":
    main()
