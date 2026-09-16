"""아이템 기반 협업 필터링(item-based CF) — 평점 예측 · 추천.

user_cf_*.py(사용자 기반)가 "나랑 비슷한 사람들"을 찾는 것과 반대로, 이 스크립트는
"내가 준 평점과 비슷한 패턴으로 평가된 영화들"을 찾는다.

유사 아이템(영화) 기준
- 영화 × 영화 유사도가 아니라, **영화 × 사용자 평점 벡터**(그 영화에 모든 사용자가 준 평점,
  안 본 사람은 0)끼리 **코사인 유사도**를 계산한다 — "이 영화에 평점을 준 사람들의 평가 패턴이
  저 영화와 얼마나 비슷한가"를 재는 것.
- 최소 공통 평가자 수: 두 영화를 **모두 평가한 사용자 수**가 min_common명 미만이면 후보에서
  제외한다(사용자 기반의 최소 공통 평가 "영화" 수와 정확히 대응되는 개념 — 여기선 최소 공통
  평가 "사용자" 수).
- 유사 아이템 수: 대상 사용자가 실제로 평가한 영화들 중, 코사인 유사도 상위 최대 k개만 사용.

평가 점수 반영 기준 (예측 공식 — 유사도 가중 평균)
    pred(u, i) = Σ_j sim(i, j) · rating(u, j)  /  Σ_j |sim(i, j)|
    (j 는 후보 영화 i 와 유사도 상위 k개 안에 든, 대상 사용자 u 가 실제로 평가한 영화만)

- 사용자 기반과 달리 "다른 사람의 평점"이 아니라 **대상 사용자 자신이 이미 준 평점**을
  가중 평균하므로, 그 사용자의 평가 성향(후하다/박하다)이 예측에 자연스럽게 반영된다.
- 이미 평가한 영화는 candidate_movie_ids 를 지정하지 않는 한 후보에서 제외한다.
- --genre 로 장르 부분일치 필터(예: --genre Action) — 예측 자체가 아니라 결과 순위만 추린다.

입력 : data/processed/ratings.csv, data/processed/movies_with_ratings.csv
출력 : outputs/tables/user{U}_itemcf_candidates[_{장르}].csv
       outputs/tables/user{U}_itemcf_recommendations[_{장르}].csv

사용법 (프로젝트 루트에서):
    python analysis/item_based_cf.py --user 414 --min-common 30 --k 50 --top 10

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


def item_based_predict(ratings: pd.DataFrame, target_user: int, min_common: int = 30, k: int = 50,
                        candidate_movie_ids=None) -> pd.DataFrame:
    """대상 사용자가 실제로 평가한 영화들과의 아이템(영화) 코사인 유사도 가중 평균으로
    평점을 예측한다. candidate_movie_ids 를 주면 그 영화들만(이미 평가한 영화가 섞여도 그대로
    계산 — 평점 은닉 검증용), 안 주면 아직 평가하지 않은 영화 전부를 예측한다.
    반환: DataFrame[movieId, predicted_rating, n_item_votes]."""
    mat, user_idx, user_ids = build_rating_matrix(ratings)
    movie_ids = np.sort(ratings["movieId"].unique())
    movie_idx = {m: i for i, m in enumerate(movie_ids)}
    if target_user not in user_idx:
        raise SystemExit(f"사용자 {target_user} 가 데이터에 없습니다.")

    mat_csc = mat.tocsc()
    u_i = user_idx[target_user]
    user_row = mat_csc[u_i, :].tocoo()
    rated_cols = user_row.col
    rated_vals = user_row.data
    if rated_cols.size == 0:
        return pd.DataFrame(columns=["movieId", "predicted_rating", "n_item_votes"])

    if candidate_movie_ids is None:
        candidate_cols = np.setdiff1d(np.arange(len(movie_ids)), rated_cols)
    else:
        candidate_cols = np.array(sorted({
            movie_idx[m] for m in candidate_movie_ids if m in movie_idx
        }), dtype=int)
    if candidate_cols.size == 0:
        return pd.DataFrame(columns=["movieId", "predicted_rating", "n_item_votes"])

    rated_sub = mat_csc[:, rated_cols]      # n_users x n_rated (대상 사용자가 평가한 영화들)
    cand_sub = mat_csc[:, candidate_cols]   # n_users x n_candidates (예측할 영화들)

    # 후보영화 x 평가영화 코사인 유사도 (전체를 한 번에 계산 — 영화별로 반복 호출하지 않는다)
    sims = cosine_similarity(cand_sub.T, rated_sub.T).astype(np.float32)  # (n_cand, n_rated)

    binary = mat_csc.copy()
    binary.data = np.ones_like(binary.data)
    common = np.asarray(
        (binary[:, candidate_cols].T @ binary[:, rated_cols]).todense()
    )  # (n_cand, n_rated) — 두 영화를 모두 평가한 사용자 수

    eligible = common >= min_common

    rows_out = []
    for row_i, col in enumerate(candidate_cols):
        elig_idx = np.nonzero(eligible[row_i])[0]
        if elig_idx.size == 0:
            continue
        s = sims[row_i, elig_idx]
        r = rated_vals[elig_idx]
        if s.size > k:
            top = np.argpartition(-s, k - 1)[:k]
            s = s[top]
            r = r[top]
        denom = np.abs(s).sum()
        if denom == 0:
            continue
        pred = float(np.dot(s, r) / denom)
        rows_out.append({
            "movieId": movie_ids[col],
            "predicted_rating": round(pred, 3),
            "n_item_votes": int(s.size),
        })
    return pd.DataFrame(rows_out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", type=int, default=414)
    ap.add_argument("--min-common", type=int, default=30,
                     help="두 영화를 모두 평가한 사용자 수 최소 기준(기본 30명)")
    ap.add_argument("--k", type=int, default=50, help="유사 영화 최대 개수(기본 50)")
    ap.add_argument("--min-votes", type=int, default=2,
                     help="추천 목록에 넣기 위한 최소 유사 영화 수(기본 2)")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--genre", type=str, default=None, help="장르 부분일치 필터(예: Action)")
    args = ap.parse_args()

    ratings = pd.read_csv(RATINGS_CSV)
    movies = pd.read_csv(MOVIES_CSV)

    n_rated = int((ratings.userId == args.user).sum())
    print(f"기준 사용자: {args.user}  (평점 {n_rated}건)")
    print(f"아이템 기반 CF: 두 영화를 모두 평가한 사용자 {args.min_common}명 이상 · "
          f"영화별 유사 영화 최대 {args.k}개")

    pred = item_based_predict(ratings, args.user, args.min_common, args.k)
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
    all_path = OUT_DIR / f"user{args.user}_itemcf_candidates{suffix}.csv"
    pred.sort_values(["predicted_rating", "n_item_votes"], ascending=[False, False]) \
        .to_csv(all_path, index=False)
    print(f"미평가 후보(유사 영화 1개 이상) {len(pred)}편 -> 저장: {all_path}")

    rec = pred[pred["n_item_votes"] >= args.min_votes]
    rec = rec.sort_values(["predicted_rating", "n_item_votes"], ascending=[False, False])
    rec_top = rec.head(args.top)

    rec_path = OUT_DIR / f"user{args.user}_itemcf_recommendations{suffix}.csv"
    rec_top.to_csv(rec_path, index=False)

    print(f"유사 영화 {args.min_votes}개 이상인 후보 {len(rec)}편 중 예측 평점 상위 "
          f"{len(rec_top)}편 -> 저장: {rec_path}\n")
    print(f"[사용자 {args.user} 추천 상위 {len(rec_top)}편 (아이템 기반)]")
    print(rec_top[["movieId", "title", "release_year", "predicted_rating",
                    "n_item_votes", "bayesian_rating", "rating_count"]]
          .to_string(index=False))

    target_seen = set(ratings.loc[ratings.userId == args.user, "movieId"])
    assert not set(rec_top["movieId"]) & target_seen, "이미 본 영화가 추천에 섞임"
    assert (rec_top["predicted_rating"].between(0.5, 5.0)).all()
    assert (rec_top["n_item_votes"] >= args.min_votes).all()
    assert (rec_top["n_item_votes"] <= args.k).all()
    print("\n검증 통과: 추천 목록에 이미 본 영화 없음, 예측 평점 0.5~5.0 범위, "
          "유사 영화 수는 최소 기준 이상·k 이하")


if __name__ == "__main__":
    main()
