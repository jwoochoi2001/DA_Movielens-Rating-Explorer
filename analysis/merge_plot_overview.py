"""줄거리(overview) 열을 기준 테이블(movies_with_ratings.csv)에 movieId 기준으로 병합.

입력 : data/processed/movies_with_ratings.csv (기준 테이블)
       data/raw/movie_text_metadata.csv        (TMDB 메타데이터, 3,537편만 커버)
출력 : data/processed/movies_with_ratings.csv  (같은 파일에 overview 열 추가)

- LEFT JOIN(movieId) — 기준 테이블의 행/순서를 그대로 유지한다.
  커버되지 않는 나머지(약 6,200편)는 overview 공란(NA).
- movie_text_metadata.csv 에는 movieId 6003(중복 병합으로 이미 제거된 영화)과
  144606(그 대표 movieId)이 완전히 같은 내용(tmdbId 4912)으로 중복 존재하지만,
  기준 테이블에는 144606만 남아 있으므로 LEFT JOIN 특성상 6003 쪽은 애초에
  매칭 대상이 없어 자동으로 무시된다 — 별도 remap 없이도 안전하다(assert로 재확인).

실행: 프로젝트 루트에서 (merge_movies_ratings.py 이후에 실행 — run_all.py 참고)
    python analysis/merge_plot_overview.py
필요 패키지: pandas
"""
import pandas as pd

BASE = "data/processed/movies_with_ratings.csv"
META = "data/raw/movie_text_metadata.csv"
OUT = BASE  # 기준 테이블에 그대로 병합(같은 파일에 덮어쓰기)


def main() -> None:
    base = pd.read_csv(BASE)
    if "overview" in base.columns:  # 재실행 대비: 이미 붙어 있으면 지우고 다시 병합
        base = base.drop(columns=["overview"])
    meta = pd.read_csv(META, usecols=["movieId", "overview"])

    before_cols = list(base.columns)
    assert meta["movieId"].is_unique, "movie_text_metadata.csv 의 movieId 가 유일하지 않음"

    df = base.merge(meta, on="movieId", how="left")

    n_have = df["overview"].notna().sum()
    print(f"기준 테이블 행: {len(base)}  / 병합 후 행: {len(df)}  (일치해야 정상)")
    print(f"overview 있는 영화: {n_have}건 / {len(df)}건 ({n_have / len(df) * 100:.1f}%)")
    print(f"overview 결측(정보 없음): {df['overview'].isna().sum()}건")

    df.to_csv(OUT, index=False)
    print(f"저장: {OUT}   shape={df.shape}")

    # 검증
    assert len(df) == len(base), "병합 후 행 수 변화 — movieId 중복 매치 의심"
    assert list(df.columns) == before_cols + ["overview"]
    assert df["movieId"].is_unique

    print("\n[overview 있는 영화 예시 3편]")
    print(df[df["overview"].notna()].head(3)[["movieId", "title", "overview"]]
          .to_string(index=False))
    print("\n[overview 없는 영화 예시 3편]")
    print(df[df["overview"].isna()].head(3)[["movieId", "title"]].to_string(index=False))


if __name__ == "__main__":
    main()
