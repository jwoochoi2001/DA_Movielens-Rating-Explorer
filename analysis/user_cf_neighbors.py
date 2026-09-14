"""사용자 기반 협업 필터링(user-based CF) — 이웃(neighbor) 탐색.

지정한 사용자와 평점 패턴이 비슷한 다른 사용자(이웃)를 코사인 유사도로 찾는다.

사양
- 평점 행렬: userId × movieId **희소행렬**(scipy.sparse.csr_matrix), 값 = 평점(0.5~5.0),
  안 본 영화는 0. (610명 × 9,719편, nnz 100,832 — 대부분 0인 전형적 희소 행렬)
- 유사도: 코사인 유사도(전체 벡터 기준, 미평가는 0으로 취급 — 평점을 조정/중심화하지 않은
  일반 코사인).
- 최소 공통 평가 영화 수: **5편 미만**인 사용자는 후보에서 제외한다 — 우연히 겹친 한두 편만으로
  유사도가 높게 나오는 걸 막기 위한 최소 지지도(support) 조건. 공통 영화 수는 평점 유무를
  1/0으로 바꾼 이진 행렬의 내적으로 구한다.
- 이웃 수: 유사도 상위 **최대 10명**까지만 사용(동점이면 공통 영화 수 많은 쪽 우선).
- 자동 완화(`find_neighbors_auto`): 요청한 최소 공통 평가 영화 수 조건에서 이웃이 하나도 없으면
  30 -> 15 -> 5 순으로(요청값보다 작은 것만) 자동으로 낮춰 재시도한다. CLI 는 이 함수를 쓴다.

입력 : data/processed/ratings.csv, data/processed/movies_enriched.csv(제목 표시용)
출력 : outputs/tables/user{U}_neighbors.csv                 (이웃 요약: userId, cosine_sim, n_common)
       outputs/tables/user{U}_neighbor_common_ratings.csv    (이웃별 공통 영화 평점 전체)

사용법 (프로젝트 루트에서):
    python analysis/user_cf_neighbors.py                       # 기본값: userId 414
    python analysis/user_cf_neighbors.py --user 414 --min-common 5 --k 10

필요 패키지: pandas, numpy, scipy, scikit-learn
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

RATINGS_CSV = "data/processed/ratings.csv"
MOVIES_CSV = "data/processed/movies_enriched.csv"
OUT_DIR = Path("outputs/tables")


def build_rating_matrix(ratings: pd.DataFrame):
    """userId x movieId 희소 평점 행렬 + 인덱스 매핑."""
    user_ids = np.sort(ratings["userId"].unique())
    movie_ids = np.sort(ratings["movieId"].unique())
    user_idx = {u: i for i, u in enumerate(user_ids)}
    movie_idx = {m: i for i, m in enumerate(movie_ids)}

    rows = ratings["userId"].map(user_idx).to_numpy()
    cols = ratings["movieId"].map(movie_idx).to_numpy()
    data = ratings["rating"].to_numpy(dtype=float)

    mat = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(movie_ids)))
    return mat, user_idx, user_ids


def find_neighbors(ratings: pd.DataFrame, target_user: int, min_common: int = 5, k: int = 10) -> pd.DataFrame:
    """target_user 의 이웃을 코사인 유사도로 찾는다 (다른 스크립트에서도 재사용).
    반환: DataFrame[neighbor_userId, cosine_sim, n_common], 유사도 내림차순 상위 k행."""
    mat, user_idx, user_ids = build_rating_matrix(ratings)
    if target_user not in user_idx:
        raise SystemExit(f"사용자 {target_user} 가 데이터에 없습니다.")

    u_i = user_idx[target_user]
    target_vec = mat[u_i]

    sims = cosine_similarity(target_vec, mat).ravel()

    binary = mat.copy()
    binary.data = np.ones_like(binary.data)
    common_counts = np.asarray((binary[u_i] @ binary.T).todense()).ravel()

    cand = pd.DataFrame({"neighbor_userId": user_ids, "cosine_sim": sims, "n_common": common_counts})
    cand = cand[cand["neighbor_userId"] != target_user]
    cand = cand[cand["n_common"] >= min_common]
    cand = cand.sort_values(["cosine_sim", "n_common"], ascending=[False, False]).head(k)
    return cand.reset_index(drop=True)


def find_neighbors_auto(ratings: pd.DataFrame, target_user: int, min_common: int = 5, k: int = 10,
                         fallback=(30, 15, 5)) -> tuple[pd.DataFrame, int]:
    """find_neighbors() 를 쓰되, 요청한 min_common 조건에서 이웃이 하나도 없으면 min_common 을
    자동으로 낮춰가며 재시도한다 — 기본 재시도 순서 30 -> 15 -> 5 중 요청값보다 작은 것만 시도한다
    (예: min_common=30 요청 시 30 -> 15 -> 5, min_common=5면 재시도 없음).
    반환: (neighbors, 실제로 이웃을 찾은 min_common 값)."""
    cand = find_neighbors(ratings, target_user, min_common, k)
    used = min_common
    if cand.empty:
        for mc in sorted({v for v in fallback if v < min_common}, reverse=True):
            cand = find_neighbors(ratings, target_user, mc, k)
            if not cand.empty:
                used = mc
                break
    return cand, used


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", type=int, default=414)
    ap.add_argument("--min-common", type=int, default=5)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    ratings = pd.read_csv(RATINGS_CSV)
    movies = pd.read_csv(MOVIES_CSV)[["movieId", "title"]]

    n_before = ratings["userId"].nunique() - 1
    cand_raw, used_min_common = find_neighbors_auto(ratings, args.user, args.min_common, args.k)
    cand = cand_raw.rename(columns={"neighbor_userId": "userId"})
    n_ratings_target = int((ratings.userId == args.user).sum())

    print(f"기준 사용자: {args.user}  (평점 {n_ratings_target}건)")
    print(f"최소 공통 평가 영화 수: {args.min_common}편 이상 · 이웃 최대 {args.k}명")
    if used_min_common != args.min_common:
        print(f"-> {args.min_common}편 이상 조건에서는 이웃을 찾지 못해, 자동으로 "
              f"{used_min_common}편 이상으로 낮춰 재시도했습니다.")
    print(f"전체 다른 사용자 {n_before}명 중 조건 만족 후보에서 유사도 상위 {len(cand)}명 선택\n")
    print("[이웃 목록: userId, 코사인 유사도, 공통 평가 영화 수]")
    print(cand.rename(columns={"userId": "neighbor_userId"}).to_string(index=False))

    # ---- 이웃별 공통 영화 평점 상세 ----
    target_ratings = ratings[ratings.userId == args.user].set_index("movieId")["rating"]
    detail_rows = []
    for _, row in cand.iterrows():
        nb = int(row["userId"])
        nb_ratings = ratings[ratings.userId == nb].set_index("movieId")["rating"]
        for mid in target_ratings.index.intersection(nb_ratings.index):
            detail_rows.append({
                "neighbor_userId": nb,
                "cosine_sim": row["cosine_sim"],
                "movieId": mid,
                f"user{args.user}_rating": target_ratings[mid],
                "neighbor_rating": nb_ratings[mid],
            })
    detail = pd.DataFrame(detail_rows).merge(movies, on="movieId", how="left")
    detail = detail[["neighbor_userId", "cosine_sim", "movieId", "title",
                      f"user{args.user}_rating", "neighbor_rating"]]
    detail = detail.sort_values(["cosine_sim", "neighbor_userId"], ascending=[False, True])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_path = OUT_DIR / f"user{args.user}_neighbors.csv"
    c_path = OUT_DIR / f"user{args.user}_neighbor_common_ratings.csv"
    cand.rename(columns={"userId": "neighbor_userId"}).to_csv(n_path, index=False)
    detail.to_csv(c_path, index=False)

    print(f"\n저장: {n_path}")
    print(f"저장: {c_path}  ({len(detail)}행 = 이웃 {len(cand)}명 x 공통 영화)")

    print("\n[이웃별 공통 영화 수]")
    print(detail.groupby("neighbor_userId").size().rename("n_common_movies").to_string())

    if not cand.empty:
        top1 = int(cand.iloc[0]["userId"])
        print(f"\n[예시: 유사도 1위 이웃 userId={top1} 의 공통 영화 평점 (상위 10개)]")
        print(detail[detail.neighbor_userId == top1].head(10).to_string(index=False))

    # 검증
    assert (cand["n_common"] >= used_min_common).all()
    assert len(cand) <= args.k
    assert args.user not in cand["userId"].to_numpy()
    print("\n검증 통과: 모든 이웃이 최소 공통 영화 수 조건을 만족, 이웃 수는 k 이하, 본인 제외")


if __name__ == "__main__":
    main()
