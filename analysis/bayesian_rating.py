"""영화별 베이지안 가중평균 평점(bayesian_rating) 파생.

입력 : data/processed/ratings.csv          (중복 병합 완료)
       data/processed/movies_enriched.csv  (연도/장르 보강 완료)
출력 : data/processed/movie_scores.csv
       컬럼: movieId, title, release_year, genres, n_ratings, mean_rating, rating_std, bayesian_rating

rating_std : 영화별 평점의 표본표준편차(ddof=1). 값이 클수록 호불호가 갈리는(양극화) 영화,
             작을수록 평가가 일치. 평점이 1건뿐인 영화는 정의되지 않아 NA.

공식 (베이지안 축소평균 / IMDb weighted rating):
    bayesian_rating = (v * R + m * C) / (v + m)
      R = 영화별 평균 평점 (mean_rating)
      v = 영화별 평점 수  (n_ratings)
      C = 전역 평균 평점  (전체 ratings 의 평균)
      m = 10  (평활 상수, 신뢰할 만한 최소 표본 수)

- 평점이 하나도 없는 영화(v=0)는 bayesian_rating = C, mean_rating = NA.
- data/raw 는 읽지 않는다(가공본만 사용). 재실행 가능.

실행: 프로젝트 루트에서
    python analysis/bayesian_rating.py
필요 패키지: pandas
"""
import pandas as pd

RATINGS = "data/processed/ratings.csv"
MOVIES = "data/processed/movies_enriched.csv"
OUT = "data/processed/movie_scores.csv"
M = 10  # 평활 상수


def main() -> None:
    ratings = pd.read_csv(RATINGS)
    movies = pd.read_csv(MOVIES)

    C = ratings["rating"].mean()

    agg = (
        ratings.groupby("movieId")["rating"]
        .agg(n_ratings="count", mean_rating="mean", rating_std="std")  # std: 표본(ddof=1)
    )

    df = movies[["movieId", "title", "release_year", "genres"]].merge(
        agg, on="movieId", how="left"
    )
    df["n_ratings"] = df["n_ratings"].fillna(0).astype(int)

    v = df["n_ratings"]
    r = df["mean_rating"].fillna(C)  # v=0 이면 어차피 (0*C + m*C)/m = C
    df["bayesian_rating"] = (v * r + M * C) / (v + M)

    df["mean_rating"] = df["mean_rating"].round(4)
    df["rating_std"] = df["rating_std"].round(4)  # n_ratings<=1 이면 NA
    df["bayesian_rating"] = df["bayesian_rating"].round(4)
    df = df.sort_values("bayesian_rating", ascending=False).reset_index(drop=True)
    df.to_csv(OUT, index=False)

    print(f"C(전역 평균) = {C:.4f},  m = {M}")
    print(f"영화 수: {len(df)}  (평점 없음 {int((df['n_ratings'] == 0).sum())}건 → bayesian_rating = C)")
    print(f"저장: {OUT}")

    # 검증
    assert df["bayesian_rating"].between(0.5, 5.0).all()
    lo, hi = df["bayesian_rating"].min(), df["bayesian_rating"].max()
    print(f"bayesian_rating 범위: {lo:.4f} ~ {hi:.4f}")
    print("\n[bayesian_rating 상위 10]")
    print(df.head(10)[["title", "release_year", "n_ratings", "mean_rating", "bayesian_rating"]]
          .to_string(index=False))
    print("\n[mean_rating=5.0 이지만 표본이 적어 강하게 보정된 예]")
    ex = df[(df["mean_rating"] == 5.0)].head(5)
    print(ex[["title", "release_year", "n_ratings", "mean_rating", "bayesian_rating"]]
          .to_string(index=False))

    # rating_std: 호불호 지표
    n1 = int((df["n_ratings"] <= 1).sum())
    print(f"\nrating_std: 정의됨 {int(df['rating_std'].notna().sum())}건 / NA {n1}건(n_ratings<=1)")
    reliable = df[df["n_ratings"] >= 20]
    print(f"rating_std describe (n_ratings>=20, {len(reliable)}편):")
    print(reliable["rating_std"].describe().round(3).to_string())
    print("\n[호불호 큰 영화 top5  (n_ratings>=50, rating_std 내림차순)]")
    pol = df[df["n_ratings"] >= 50].nlargest(5, "rating_std")
    print(pol[["title", "release_year", "n_ratings", "mean_rating", "rating_std"]].to_string(index=False))
    print("\n[호평 일치 영화 top5  (n_ratings>=50, mean>=4, rating_std 오름차순)]")
    con = df[(df["n_ratings"] >= 50) & (df["mean_rating"] >= 4)].nsmallest(5, "rating_std")
    print(con[["title", "release_year", "n_ratings", "mean_rating", "rating_std"]].to_string(index=False))


if __name__ == "__main__":
    main()
