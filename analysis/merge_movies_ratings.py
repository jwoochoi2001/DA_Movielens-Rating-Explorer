"""movies + ratings 를 movieId 기준으로 합친 영화 단위 테이블.

입력 : data/processed/movies_enriched.csv , data/processed/ratings.csv
       (1~2단계 산출물: 동일 제목 중복 영화 5쌍 병합 + release_year 분리 + 결측 연도/장르 보강)
출력 : data/processed/movies_with_ratings.csv
       data/processed/label_text.txt      (rating_label 숫자 → 라벨 이름)

사양
- movies LEFT JOIN ratings (movieId) — 영화 정보의 행을 모두 유지.
- 제목 중복으로 병합했던 5쌍은 대표 movieId 로 이미 합쳐진 상태이고,
  ratings 의 movieId 도 치환돼 있으므로 mean_rating 이 병합 기준으로 계산된다.
- release_year   : movies_enriched 의 값(제목에서 추출 + 표기 없던 12건은 검색 추정치).
- mean_rating    : 영화별 평균 평점. 평가가 없으면 비움(NA).
- rating_count   : 영화별 평점 수. 평가가 없으면 0.
- rating_std     : 영화별 평점의 표본표준편차(ddof=1). 평점 1개 이하면 비움(NA).
- bayesian_rating: 베이지안 가중평균 (v*R + m*C)/(v+m), m=10,
                   C = 전체 ratings 의 전역 평균. 평가가 없으면(v=0) C.
- pos_ratio      : 4점 이상 평점의 비율(0~1). 평가가 없으면 비움(NA).
- rating_label   : 추천지수 (숫자 범주형 0~5). 위에서부터 순서대로 검사하고
                   처음 해당하는 값으로 확정한다. 사용 지표: rating_count, pos_ratio, rating_std
      0  평가 없음               rating_count == 0
      1  평가 부족               rating_count < 30
      2  대다수가 좋아하는 영화  rating_count >= 30 & pos_ratio >= 0.65 & rating_std < 0.95
      3  호불호가 갈리는 영화    rating_count >= 30 & rating_std >= 1.05
      4  무난하게 좋은 영화      rating_count >= 30 & pos_ratio >= 0.45
      5  대다수가 아쉬워한 영화  rating_count >= 30 & 나머지 전부 (pos_ratio < 0.45)

실행: 프로젝트 루트에서 (먼저 merge_duplicate_movies.py, enrich_movies.py 실행 필요 — run_all.py 참고)
    python analysis/merge_movies_ratings.py
필요 패키지: pandas, numpy
"""
import re
import numpy as np
import pandas as pd

MOVIES = "data/processed/movies_enriched.csv"
RATINGS = "data/processed/ratings.csv"
OUT = "data/processed/movies_with_ratings.csv"
LABEL_TXT = "data/processed/label_text.txt"

YEAR_RE = re.compile(r"^(?P<title>.*?)\s*\((?P<year>\d{4})(?:[-–]\d{4})?\)\s*$")

MIN_N = 30  # 추천지수 산정에 필요한 최소 평점 수

RATING_LABELS = {
    0: "평가 없음",
    1: "평가 부족",
    2: "대다수가 좋아하는 영화",
    3: "호불호가 갈리는 영화",
    4: "무난하게 좋은 영화",
    5: "대다수가 아쉬워한 영화",
}


def _rating_label(df):
    """기준표를 위에서부터 검사 — 먼저 맞는 값으로 확정.
    (아래 코드는 우선순위가 낮은 값부터 덮어써서 같은 결과를 만든다.)"""
    rc = df["rating_count"]
    p4, sd = df["pos_ratio"], df["rating_std"]
    enough = rc >= MIN_N

    lab = np.full(len(df), 5, dtype=int)      # rc>=30 기본값: 5 (대다수가 아쉬워한 영화)
    lab[enough & (p4 >= 0.45)] = 4
    lab[enough & (sd >= 1.05)] = 3
    lab[enough & (p4 >= 0.65) & (sd < 0.95)] = 2
    lab[rc < MIN_N] = 1
    lab[rc == 0] = 0
    return lab


def split_year(title: str):
    m = YEAR_RE.match(title)
    if m:
        return m.group("title").strip(), int(m.group("year"))
    return title.strip(), pd.NA


def main() -> None:
    movies = pd.read_csv(MOVIES)
    ratings = pd.read_csv(RATINGS)

    if "release_year" in movies.columns:
        movies["release_year"] = movies["release_year"].astype("Int64")
    else:  # 원본 movies 를 직접 넣은 경우: 제목에서 추출
        titles, years = zip(*movies["title"].map(split_year))
        movies["title"] = list(titles)
        movies["release_year"] = pd.array(years, dtype="Int64")

    M = 10
    C = ratings["rating"].mean()

    ratings = ratings.assign(_pos=(ratings["rating"] >= 4.0))
    agg = ratings.groupby("movieId").agg(
        mean_rating=("rating", "mean"),
        rating_count=("rating", "count"),
        rating_std=("rating", "std"),   # 표본(ddof=1)
        pos_ratio=("_pos", "mean"),     # 4점 이상 비율
    )

    df = movies.merge(agg, on="movieId", how="left")
    df["rating_count"] = df["rating_count"].fillna(0).astype(int)

    v = df["rating_count"]
    r = df["mean_rating"].fillna(C)  # v=0 이면 (0*C + M*C)/M = C
    df["bayesian_rating"] = ((v * r + M * C) / (v + M)).round(4)
    df["mean_rating"] = df["mean_rating"].round(4)  # 평가 없으면 NaN → CSV 공란
    df["rating_std"] = df["rating_std"].round(4)    # rating_count<=1 이면 NaN → CSV 공란
    df["pos_ratio"] = df["pos_ratio"].round(4)

    df["rating_label"] = _rating_label(df)

    df = df[["movieId", "title", "genres", "release_year",
             "mean_rating", "rating_count", "rating_std", "bayesian_rating",
             "pos_ratio", "rating_label"]]
    df.to_csv(OUT, index=False)

    with open(LABEL_TXT, "w", encoding="utf-8") as f:
        for k, name in RATING_LABELS.items():
            f.write(f"{k}\t{name}\n")

    print(f"저장: {OUT}   shape={df.shape}")
    print(f"저장: {LABEL_TXT}")
    print(f"movies 행: {len(movies)}  / 병합 후: {len(df)}  (일치해야 정상)")
    print(f"베이지안: m={M}, C={C:.4f}")
    print("\n[dtypes]")
    print(df.dtypes.to_string())
    print("\n[rating_label 분포]")
    for k, n in df["rating_label"].value_counts().sort_index().items():
        print(f"  {k} {RATING_LABELS[k]:<22} {n:5d}  ({n / len(df) * 100:4.1f}%)")
    print("\n[상위 5행]")
    print(df.head().to_string(index=False))
    for lab in (2, 3):
        ex = (df[df["rating_label"] == lab]
              .sort_values("rating_count", ascending=False).head(5))
        print(f"\n[rating_label={lab} {RATING_LABELS[lab]} 예시]")
        print(ex[["title", "release_year", "rating_count", "mean_rating",
                  "pos_ratio", "rating_std"]].to_string(index=False))

    # 검증
    assert len(df) == len(movies)
    assert df["rating_label"].between(0, 5).all()
    assert (df.loc[df["rating_count"] == 0, "rating_label"] == 0).all()
    assert (df.loc[df["rating_count"].between(1, 29), "rating_label"] == 1).all()
    assert df.loc[df["rating_label"] >= 2, "rating_count"].ge(MIN_N).all()


if __name__ == "__main__":
    main()
