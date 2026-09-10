"""최종 분석 테이블 생성 (평점 1건 = 1행).

입력 : data/processed/ratings.csv        (중복 병합 완료)
       data/processed/movie_scores.csv   (연도/장르 보강 + 베이지안 평점까지 반영된 영화 단위 파생표)
출력 : data/processed/analysis_table.csv

병합 : ratings  LEFT JOIN  movie_scores  ON movieId
       movie_scores 가 movies_enriched(제목/연도/장르 보강) + 영화별 평점 통계를 모두 담고 있으므로
       이 하나와만 조인하면 movies 정보가 함께 붙는다.

파생 열
- rating_dt        : timestamp(Unix 초) → datetime
- rating_year      : 평점이 매겨진 연도
- movie_age        : rating_year - release_year  (개봉 후 몇 년 뒤에 평가되었나)
- n_genres         : 장르 개수
- primary_genre    : '|' 로 나눈 첫 번째 장르

실행: 프로젝트 루트에서
    python analysis/build_analysis_table.py
필요 패키지: pandas
"""
import pandas as pd

RATINGS = "data/processed/ratings.csv"
SCORES = "data/processed/movie_scores.csv"
OUT = "data/processed/analysis_table.csv"


def main() -> None:
    ratings = pd.read_csv(RATINGS)
    scores = pd.read_csv(SCORES)

    df = ratings.merge(scores, on="movieId", how="left", validate="many_to_one")

    # 조인 누락 점검
    missing = df["title"].isna().sum()
    if missing:
        raise ValueError(f"movie_scores 에 없는 movieId 를 가진 평점 {missing}건")

    # 시간 파생
    df["rating_dt"] = pd.to_datetime(df["timestamp"], unit="s")
    df["rating_year"] = df["rating_dt"].dt.year
    df["movie_age"] = df["rating_year"] - df["release_year"]

    # 장르 파생
    df["n_genres"] = df["genres"].str.split("|").apply(len)
    df["primary_genre"] = df["genres"].str.split("|").str[0]

    df = df[[
        "userId", "movieId", "title", "release_year", "genres",
        "primary_genre", "n_genres",
        "rating", "timestamp", "rating_dt", "rating_year", "movie_age",
        "n_ratings", "mean_rating", "rating_std", "bayesian_rating",
    ]]

    df.to_csv(OUT, index=False)
    print(f"저장: {OUT}   shape={df.shape}")
    print("\n[dtypes]")
    print(df.dtypes.to_string())
    print("\n[상위 5개 행]")
    print(df.head().to_string(index=False))
    print("\n[수치형 describe]")
    print(df[["rating", "release_year", "rating_year", "movie_age",
              "n_genres", "n_ratings", "mean_rating", "rating_std", "bayesian_rating"]]
          .describe().round(3).to_string())
    print("\n[primary_genre 상위 10]")
    print(df["primary_genre"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
