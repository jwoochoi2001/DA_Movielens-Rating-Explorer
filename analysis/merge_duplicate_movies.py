"""동일 제목(동일 연도) 중복 영화 병합.

규칙
- data/raw/ 는 읽기 전용. 결과는 data/processed/ 에 저장.
- 중복 제목 쌍에서 genres 개수('|' 분리)가 많은 movieId 를 대표로 유지한다.
  (개수가 같으면 더 작은 movieId 를 유지 — 본 데이터에는 동점 없음)
- 나머지 movieId 는 대표 movieId 로 치환한다. movies, ratings 양쪽에 적용.
- 치환 후 같은 (userId, movieId) 평가가 중복되면 timestamp 가 가장 최근인 1건만 남긴다.

실행: 프로젝트 루트에서
    python analysis/merge_duplicate_movies.py
필요 패키지: pandas
"""
import pandas as pd

RAW_MOVIES = "data/raw/movies.csv"
RAW_RATINGS = "data/raw/ratings.csv"
OUT_MOVIES = "data/processed/movies.csv"
OUT_RATINGS = "data/processed/ratings.csv"


def build_merge_map(movies: pd.DataFrame) -> dict:
    dup = movies[movies["title"].duplicated(keep=False)].copy()
    dup["n_genres"] = dup["genres"].str.split("|").apply(len)
    merge_map = {}
    for _, grp in dup.groupby("title"):
        grp = grp.sort_values(["n_genres", "movieId"], ascending=[False, True])
        keep_id = grp.iloc[0]["movieId"]
        for drop_id in grp.iloc[1:]["movieId"]:
            merge_map[int(drop_id)] = int(keep_id)
    return merge_map


def main() -> None:
    movies = pd.read_csv(RAW_MOVIES)
    ratings = pd.read_csv(RAW_RATINGS)

    merge_map = build_merge_map(movies)
    print("병합 매핑 (drop movieId -> keep movieId):")
    for d, k in merge_map.items():
        print(f"  {d} -> {k}")

    # movies: 치환 대상 행 제거 (대표 행만 유지)
    movies_out = movies[~movies["movieId"].isin(merge_map)].reset_index(drop=True)
    print(f"\nmovies: {len(movies)} -> {len(movies_out)} (제거 {len(movies) - len(movies_out)}행)")

    # ratings: movieId 치환
    ratings_out = ratings.copy()
    ratings_out["movieId"] = ratings_out["movieId"].replace(merge_map)
    remapped = ratings["movieId"].isin(merge_map).sum()
    print(f"ratings: movieId 치환 {remapped}건")

    # 치환으로 생긴 (userId, movieId) 중복 -> 최신 timestamp 1건 유지
    before = len(ratings_out)
    dup_mask = ratings_out.duplicated(subset=["userId", "movieId"], keep=False)
    print(f"치환 후 (userId, movieId) 중복: {dup_mask.sum()}행")
    ratings_out = (
        ratings_out.sort_values("timestamp")
        .drop_duplicates(subset=["userId", "movieId"], keep="last")
        .sort_values(["userId", "movieId"])
        .reset_index(drop=True)
    )
    print(f"ratings: {before} -> {len(ratings_out)} (제거 {before - len(ratings_out)}행)")

    movies_out.to_csv(OUT_MOVIES, index=False)
    ratings_out.to_csv(OUT_RATINGS, index=False)
    print(f"\n저장:\n  {OUT_MOVIES}\n  {OUT_RATINGS}")

    # 검증
    assert movies_out["movieId"].duplicated().sum() == 0
    assert movies_out["title"].duplicated().sum() == 0
    assert ratings_out.duplicated(subset=["userId", "movieId"]).sum() == 0
    assert ratings_out["movieId"].isin(movies_out["movieId"]).all()
    print("검증 통과: movieId/제목 중복 없음, 평가 중복 없음, 모든 평가가 movies 에 매칭")


if __name__ == "__main__":
    main()
