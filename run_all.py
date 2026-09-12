"""전체 파이프라인 순차 실행.

프로젝트 루트에서:
    python run_all.py

data/raw/{movies,ratings}.csv 를 입력으로 다음 순서를 실행한다.
각 단계는 앞 단계의 출력을 입력으로 받으며, 하나라도 실패하면 즉시 중단한다.

  1) analysis/merge_duplicate_movies.py
        raw            -> data/processed/movies.csv, data/processed/ratings.csv
        (동일 제목 중복 영화 병합 + 치환 후 중복 평가 정리)
  2) analysis/enrich_movies.py
        processed/movies.csv -> data/processed/movies_enriched.csv
        (제목에서 release_year 분리, 결측 연도/장르 보강)
  3) analysis/genre_encode.py
        processed/movies_enriched.csv -> data/processed/movies_genre_onehot.csv
        (장르 원-핫 인코딩 — 열 순서 고정, 장르 유사도 추천의 기준 테이블)
  4) analysis/merge_movies_ratings.py
        processed/movies_enriched.csv + processed/ratings.csv
        -> data/processed/movies_with_ratings.csv , data/processed/label_text.txt
        (영화 단위: mean_rating, rating_count, rating_std, bayesian_rating(m=10),
         pos_ratio(4점 이상 비율), rating_label(추천지수 0~5))
  5) analysis/merge_plot_overview.py
        processed/movies_with_ratings.csv + raw/movie_text_metadata.csv
        -> data/processed/movies_with_ratings.csv (같은 파일에 overview 열 추가)
        (TMDB 줄거리 병합, movieId 기준 LEFT JOIN — 3,537편만 존재, 나머지는 결측)
  6) analysis/bayesian_rating.py
        processed/ratings.csv + movies_enriched.csv -> data/processed/movie_scores.csv
        (영화별 n_ratings, mean_rating, rating_std, bayesian_rating(m=10))
  7) analysis/build_analysis_table.py
        processed/ratings.csv + movie_scores.csv -> data/processed/analysis_table.csv
        (평점 1건 = 1행 최종 분석 테이블 + 시간/장르 파생)
  8) analysis/eda_figures.py
        analysis_table.csv + movie_scores.csv -> outputs/figures/*.png, outputs/tables/*.csv

필요 패키지: requirements.txt 참조 (pandas, numpy, matplotlib)
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    "analysis/merge_duplicate_movies.py",
    "analysis/enrich_movies.py",
    "analysis/genre_encode.py",
    "analysis/merge_movies_ratings.py",
    "analysis/merge_plot_overview.py",
    "analysis/bayesian_rating.py",
    "analysis/build_analysis_table.py",
    "analysis/eda_figures.py",
]
REQUIRED_INPUTS = ["data/raw/movies.csv", "data/raw/ratings.csv"]


def main() -> int:
    for rel in REQUIRED_INPUTS:
        if not (ROOT / rel).exists():
            print(f"[중단] 입력 파일 없음: {rel}")
            return 1

    for i, step in enumerate(STEPS, 1):
        print(f"\n{'=' * 70}\n[{i}/{len(STEPS)}] {step}\n{'=' * 70}")
        t0 = time.time()
        proc = subprocess.run([sys.executable, step], cwd=ROOT)
        if proc.returncode != 0:
            print(f"\n[중단] {step} 실패 (exit {proc.returncode})")
            return proc.returncode
        print(f"[완료] {step}  ({time.time() - t0:.1f}s)")

    print(f"\n{'=' * 70}\n전체 파이프라인 완료. 산출물:")
    for p in sorted((ROOT / "data/processed").glob("*.csv")):
        print("  ", p.relative_to(ROOT))
    for p in sorted((ROOT / "outputs").rglob("*")):
        if p.is_file() and p.suffix in {".png", ".csv"}:
            print("  ", p.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
