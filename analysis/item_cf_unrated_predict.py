"""아이템 기반 협업 필터링 — 미평가 영화 예측 + 근거(비교 영화 목록) 출력.

rating_pattern_similarity.py 와 **같은 영화 비교 방법**(사용자 × 영화 평점 행렬의 열끼리
코사인 유사도, 공통 평가자 수 최소 기준 없이 유사도 자체만 사용)을 그대로 써서,
대상 사용자가 **아직 평가하지 않은 영화**의 평점을 예측한다.

방식
- 대상 사용자가 아직 평가하지 않은 영화 중, 전체 평가 수가 --min-ratings(기본 5) 이상인 영화만
  후보로 삼는다(평가자가 너무 적어 우연히 튀는 영화를 걸러낸다 — rating_pattern_similarity.py
  와 동일한 기준).
- 각 후보 영화에 대해, **대상 사용자가 실제로 평가한 영화들** 중 코사인 유사도가 높은 순으로
  최대 --k(기본 40)편을 고른다(공통 평가자 수 하한 없음 — item_based_cf.py 의 min_common 조건보다
  느슨하다).
- 그 최대 k편에 사용자가 실제로 준 평점을 유사도로 가중 평균해 예측 평점을 만든다.
    pred(u, i) = Σ_j sim(i, j) · rating(u, j)  /  Σ_j |sim(i, j)|   (j: 상위 k개 비교 영화)
- 예측 평점 상위 --show-detail(기본 3)편은, 실제로 가중평균에 쓰인 비교 영화 전체 목록(제목·
  유사도·사용자가 그 영화에 준 평점)을 함께 출력한다.

평점 편향 보정 (--center)
- 사용자마다 평점을 후하게/박하게 주는 성향(편향)이 다르다 — 이를 보정하려면 절대 평점 대신
  **"이 사용자의 평균 평점 대비 이 영화를 얼마나 더/덜 좋아했는가"(편차)** 를 쓴다.
- 유사도 자체도 원점수가 아니라 **사용자별 평균을 뺀 편차**로 계산한다(adjusted cosine similarity
  — Sarwar et al. 2001의 아이템 기반 CF 표준 보정 방식). 단, 평가하지 않은 항목은 그대로 0(=결측)
  으로 남겨두고, 실제로 평가한 항목만 "평점 - 그 사용자의 평균"으로 바꾼 뒤 코사인 유사도를 잰다
  — 그래야 "후한 사용자 A와 박한 사용자 B가 둘 다 이 두 영화를 상대적으로 좋아했다"는 패턴이
  "둘 다 절대점수가 높았다"는 것과 구분된다.
- 예측도 편차 기준으로 가중평균한 뒤, 대상 사용자의 평균을 다시 더해 원래 평점 스케일로 되돌린다.
    pred(u, i) = mean(u) + Σ_j sim(i, j) · (rating(u, j) - mean(u))  /  Σ_j |sim(i, j)|

입력 : data/processed/ratings.csv, data/processed/movies_with_ratings.csv
출력 : outputs/tables/user{U}_itemcf_unrated_top{N}[_centered].csv
       outputs/tables/user{U}_itemcf_unrated_top{N}[_centered]_detail.csv (상위 --show-detail편의 비교 영화 목록)

사용법 (프로젝트 루트에서):
    python analysis/item_cf_unrated_predict.py --user 414 --min-ratings 5 --k 40 --top 10 --show-detail 3
    python analysis/item_cf_unrated_predict.py --user 414 --center   # 평점 편향 보정(adjusted cosine) 버전

필요 패키지: pandas, numpy, scipy, scikit-learn
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from user_cf_neighbors import build_rating_matrix

RATINGS_CSV = "data/processed/ratings.csv"
MOVIES_CSV = "data/processed/movies_with_ratings.csv"
OUT_DIR = Path("outputs/tables")


def predict_with_detail(ratings: pd.DataFrame, movies: pd.DataFrame, target_user: int,
                         min_ratings: int = 5, k: int = 40, center: bool = False):
    """반환: (pred_df[movieId, title, predicted_rating, n_compared], detail: {movieId: DataFrame[title, cosine_sim, user_rating, (center=True 면 deviation도)]})
    center=True 면 평점 편향 보정(adjusted cosine similarity) — 유사도·가중평균 모두 "평점 - 그
    사용자의 평균 평점"(편차)으로 계산하고, 최종 예측에 대상 사용자의 평균을 다시 더한다."""
    mat, user_idx, user_ids = build_rating_matrix(ratings)
    movie_ids = np.sort(ratings["movieId"].unique())
    movie_idx = {m: i for i, m in enumerate(movie_ids)}
    if target_user not in user_idx:
        raise SystemExit(f"사용자 {target_user} 가 데이터에 없습니다.")

    mat_csc = mat.tocsc()

    if center:
        # 사용자별 평균 평점(실제로 평가한 항목만) — 평가하지 않은 항목(구조적 0)은 건드리지 않는다.
        user_means = ratings.groupby("userId")["rating"].mean().reindex(user_ids).to_numpy()
        mat_coo = mat.tocoo()
        centered_data = mat_coo.data - user_means[mat_coo.row]
        sim_mat = csr_matrix((centered_data, (mat_coo.row, mat_coo.col)), shape=mat.shape).tocsc()
    else:
        user_means = None
        sim_mat = mat_csc

    u_i = user_idx[target_user]
    target_mean = float(user_means[u_i]) if center else 0.0
    user_row = mat_csc[u_i, :].tocoo()   # 원래(절대) 평점 — 표시·최종 가중평균 대상
    rated_cols = user_row.col
    rated_vals = user_row.data
    rated_movie_ids = movie_ids[rated_cols]
    rated_titles = movies.set_index("movieId")["title"].reindex(rated_movie_ids).to_numpy()

    seen = set(rated_movie_ids)
    cand_mask = (movies["movieId"].isin(movie_ids)) & (~movies["movieId"].isin(seen)) \
        & (movies["rating_count"] >= min_ratings)
    candidate_ids = movies.loc[cand_mask, "movieId"].to_numpy()
    candidate_cols = np.array([movie_idx[m] for m in candidate_ids])

    rated_sub = sim_mat[:, rated_cols]      # n_users x n_rated (유사도는 원점수 or 편차 행렬 기준)
    cand_sub = sim_mat[:, candidate_cols]   # n_users x n_candidates

    sims = cosine_similarity(cand_sub.T, rated_sub.T)   # (n_cand, n_rated) — 공통 평가자 하한 없음

    rows_out = []
    detail = {}
    for row_i, (col, mid) in enumerate(zip(candidate_cols, candidate_ids)):
        s = sims[row_i]
        if s.size > k:
            top = np.argpartition(-s, k - 1)[:k]
        else:
            top = np.arange(s.size)
        top = top[np.argsort(-s[top])]
        s_top = s[top]
        r_top = rated_vals[top]          # 대상 사용자가 그 영화에 준 실제(절대) 평점
        denom = np.abs(s_top).sum()
        if denom == 0:
            continue
        if center:
            dev_top = r_top - target_mean
            pred = target_mean + float(np.dot(s_top, dev_top) / denom)
            pred = min(5.0, max(0.5, pred))
        else:
            pred = float(np.dot(s_top, r_top) / denom)
        rows_out.append({"movieId": mid, "predicted_rating": round(pred, 3), "n_compared": int(s_top.size)})
        detail_cols = {
            "title": rated_titles[top],
            "cosine_sim": s_top.round(4),
            "user_rating": r_top,
        }
        if center:
            detail_cols["deviation"] = (r_top - target_mean).round(3)
        detail[mid] = pd.DataFrame(detail_cols)

    pred_df = pd.DataFrame(rows_out).merge(
        movies[["movieId", "title", "release_year", "rating_count", "bayesian_rating"]],
        on="movieId", how="left",
    )
    return pred_df, detail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", type=int, default=414)
    ap.add_argument("--min-ratings", type=int, default=5, help="후보 영화의 최소 전체 평가 수(기본 5)")
    ap.add_argument("--k", type=int, default=40, help="비교에 쓸 최대 비교 영화 수(기본 40)")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--show-detail", type=int, default=3, help="비교 영화 목록을 함께 보여줄 상위 N편")
    ap.add_argument("--center", action="store_true",
                     help="평점 편향 보정(adjusted cosine) — 사용자 평균을 뺀 편차로 유사도·예측 계산")
    args = ap.parse_args()

    ratings = pd.read_csv(RATINGS_CSV)
    movies = pd.read_csv(MOVIES_CSV)

    print(f"기준 사용자: {args.user}  ·  아직 평가하지 않은 영화 중 전체 평가 수 {args.min_ratings}건 이상만 후보")
    print(f"영화별로, 사용자가 실제 평가한 영화 중 코사인 유사도 상위 최대 {args.k}편의 평점을 "
          f"유사도로 가중평균 (공통 평가자 수 하한 없음)")
    if args.center:
        user_mean = ratings.loc[ratings.userId == args.user, "rating"].mean()
        print(f"[편향 보정 ON] 유사도·가중평균 모두 \"평점 - 사용자 평균\"(편차) 기준으로 계산 "
              f"— 사용자 {args.user} 평균 평점 {user_mean:.3f}점\n")
    else:
        print()

    pred_df, detail = predict_with_detail(ratings, movies, args.user, args.min_ratings, args.k,
                                           center=args.center)
    print(f"예측 성공 후보 {len(pred_df)}편")

    top_df = pred_df.sort_values(["predicted_rating", "n_compared"], ascending=[False, False]).head(args.top)

    suffix = "_centered" if args.center else ""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    top_path = OUT_DIR / f"user{args.user}_itemcf_unrated_top{args.top}{suffix}.csv"
    top_df.to_csv(top_path, index=False)

    print(f"\n[사용자 {args.user} 예상 평점 상위 {len(top_df)}편{' (편향 보정)' if args.center else ''}]")
    print(top_df[["movieId", "title", "release_year", "predicted_rating", "n_compared",
                  "bayesian_rating", "rating_count"]].to_string(index=False))
    print(f"\n저장: {top_path}")

    detail_frames = []
    for rank, (_, row) in enumerate(top_df.head(args.show_detail).iterrows(), start=1):
        mid = int(row["movieId"])
        print(f"\n=== {rank}위: {row['title']} (예상 평점 {row['predicted_rating']:.3f}, "
              f"비교 영화 {row['n_compared']}편) — 비교에 쓰인 영화 목록 ===")
        d = detail[mid].copy()
        print(d.to_string(index=False))
        d.insert(0, "target_movieId", mid)
        d.insert(1, "target_title", row["title"])
        detail_frames.append(d)

    if detail_frames:
        detail_path = OUT_DIR / f"user{args.user}_itemcf_unrated_top{args.top}{suffix}_detail.csv"
        pd.concat(detail_frames, ignore_index=True).to_csv(detail_path, index=False)
        print(f"\n저장: {detail_path}")

    seen = set(ratings.loc[ratings.userId == args.user, "movieId"])
    assert not set(top_df["movieId"]) & seen, "이미 본 영화가 추천에 섞임"
    assert (top_df["predicted_rating"].between(0.5, 5.0)).all()
    assert (top_df["n_compared"] <= args.k).all()
    assert (top_df["rating_count"] >= args.min_ratings).all()
    print("\n검증 통과: 추천 목록에 이미 본 영화 없음, 예측 평점 0.5~5.0 범위, "
          "비교 영화 수는 k 이하, 후보는 최소 평가 수 조건 충족")


if __name__ == "__main__":
    main()
