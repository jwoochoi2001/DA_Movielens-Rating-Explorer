"""장르 원-핫 인코딩 — 장르 유사도 기반 추천의 기준 테이블.

입력 : data/processed/movies_enriched.csv
       (1~2단계 산출물: 중복 병합 + release_year 분리 + 결측 연도/장르 보강 완료)
출력 : data/processed/movies_genre_onehot.csv

사양
- 영화(행) × 장르(열) 매트릭스. 해당 장르면 1, 아니면 0.
- 장르 열 순서는 전 행에 걸쳐 동일 — 알파벳순으로 고정한다(GENRES 리스트).
- movieId, title, release_year, genres(원본 문자열)를 식별용으로 앞에 둔다.

실행: 프로젝트 루트에서
    python analysis/genre_encode.py
필요 패키지: pandas
"""
import pandas as pd

MOVIES = "data/processed/movies_enriched.csv"
OUT = "data/processed/movies_genre_onehot.csv"


def main() -> None:
    movies = pd.read_csv(MOVIES)

    genres = sorted({g for row in movies["genres"] for g in row.split("|")})
    print(f"장르 {len(genres)}종 (알파벳순 고정): {genres}")

    genre_set = movies["genres"].str.split("|").apply(set)
    onehot = pd.DataFrame(
        {g: genre_set.apply(lambda s: int(g in s)) for g in genres},
        index=movies.index,
    )

    df = pd.concat(
        [movies[["movieId", "title", "release_year", "genres"]], onehot], axis=1
    )
    df.to_csv(OUT, index=False)

    print(f"저장: {OUT}   shape={df.shape}")
    print("\n[장르별 영화 수]")
    print(onehot.sum().sort_values(ascending=False).to_string())
    print("\n[상위 5행]")
    print(df.head().to_string(index=False))

    # 검증: 원본 genres 와 원-핫 결과 일치
    check = onehot.apply(lambda r: "|".join(sorted(g for g in genres if r[g])), axis=1)
    orig = movies["genres"].apply(lambda s: "|".join(sorted(s.split("|"))))
    assert (check == orig).all(), "원-핫 인코딩이 원본 genres 와 불일치"
    assert list(onehot.columns) == genres, "장르 열 순서가 GENRES 와 다름"
    assert len(df) == len(movies)
    print("\n검증 통과: 원-핫 인코딩이 원본 genres 와 일치, 열 순서 고정, 행 수 일치")


if __name__ == "__main__":
    main()
