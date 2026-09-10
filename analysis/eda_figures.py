"""EDA 그림/표 생성.

입력 : data/processed/analysis_table.csv, data/processed/movie_scores.csv
출력 : outputs/figures/*.png , outputs/tables/*.csv

라벨은 폰트 문제를 피하려고 영어로 표기한다.

실행: 프로젝트 루트에서
    python analysis/eda_figures.py
필요 패키지: pandas, numpy, matplotlib
"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

AT = "data/processed/analysis_table.csv"
MS = "data/processed/movie_scores.csv"
FIG = "outputs/figures"
TAB = "outputs/tables"

plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight", "font.size": 10})


def save(fig, name):
    path = f"{FIG}/{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print("  ", path)


def main() -> None:
    at = pd.read_csv(AT, parse_dates=["rating_dt"])
    ms = pd.read_csv(MS)
    rated = ms[ms["n_ratings"] > 0].copy()
    C = at["rating"].mean()

    print("figures:")

    # 1. 평점 값 분포
    fig, ax = plt.subplots(figsize=(6, 3.6))
    vc = at["rating"].value_counts().sort_index()
    ax.bar(vc.index, vc.values, width=0.4, color="#4C72B0")
    ax.axvline(C, color="crimson", ls="--", lw=1, label=f"mean {C:.2f}")
    ax.set(title="Rating value distribution", xlabel="rating", ylabel="count")
    ax.legend()
    save(fig, "01_rating_distribution")

    # 2. 연도별 평점 수 (활동량)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    by_year = at.groupby(at["rating_dt"].dt.year).size()
    ax.bar(by_year.index, by_year.values, color="#55A868")
    ax.set(title="Ratings per calendar year", xlabel="year rated", ylabel="ratings")
    save(fig, "02_ratings_per_year")

    # 3. 개봉연도(10년 단위) 분포 — 평점 기준
    fig, ax = plt.subplots(figsize=(7, 3.6))
    dec = (at["release_year"] // 10 * 10).value_counts().sort_index()
    ax.bar(dec.index.astype(str), dec.values, color="#C44E52")
    ax.set(title="Ratings by movie release decade", xlabel="release decade", ylabel="ratings")
    ax.tick_params(axis="x", rotation=45)
    save(fig, "03_release_decade")

    # 4. primary_genre 분포
    fig, ax = plt.subplots(figsize=(7, 3.6))
    pg = at["primary_genre"].value_counts().head(12)[::-1]
    ax.barh(pg.index, pg.values, color="#8172B2")
    ax.set(title="Ratings by primary genre (top 12)", xlabel="ratings")
    save(fig, "04_primary_genre")

    # 5. 축소(shrinkage) 효과: n_ratings vs mean / bayesian
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.scatter(rated["n_ratings"], rated["mean_rating"], s=8, alpha=0.25,
               label="mean_rating", color="#999999")
    ax.scatter(rated["n_ratings"], rated["bayesian_rating"], s=8, alpha=0.35,
               label="bayesian_rating", color="#4C72B0")
    ax.axhline(C, color="crimson", ls="--", lw=1, label=f"global mean {C:.2f}")
    ax.set(title="Shrinkage effect (m=10)", xlabel="n_ratings (log)", ylabel="score",
           xscale="log")
    ax.legend()
    save(fig, "05_shrinkage_effect")

    # 6. 호불호 지도: mean vs std (표본 20+)
    rel = rated[rated["n_ratings"] >= 20]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sc = ax.scatter(rel["mean_rating"], rel["rating_std"], s=12, alpha=0.5,
                    c=rel["n_ratings"], cmap="viridis")
    ax.axhline(rel["rating_std"].median(), color="grey", ls=":", lw=1)
    ax.set(title="Polarization map (n_ratings >= 20)",
           xlabel="mean_rating", ylabel="rating_std (higher = divisive)")
    fig.colorbar(sc, label="n_ratings")
    save(fig, "06_polarization_map")

    # 7. 단순평균 vs 보정평균 Top 15 비교 (표본 50+)
    top = rated[rated["n_ratings"] >= 50].nlargest(15, "bayesian_rating").iloc[::-1]
    y = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hlines(y, top["bayesian_rating"], top["mean_rating"], color="#cccccc", lw=2)
    ax.scatter(top["mean_rating"], y, label="mean_rating", color="#999999", zorder=3)
    ax.scatter(top["bayesian_rating"], y, label="bayesian_rating", color="#4C72B0", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(top["title"].str.slice(0, 32))
    ax.set(title="Top 15 by bayesian_rating (n>=50): mean vs corrected", xlabel="score")
    ax.legend()
    save(fig, "07_mean_vs_bayesian_top15")

    # 8. 장르별 평점량 vs 보정평균
    ex_g = (rated.assign(genre=rated["genres"].str.split("|")).explode("genre")
            .groupby("genre")
            .agg(total_ratings=("n_ratings", "sum"), mean_bayesian=("bayesian_rating", "mean"))
            .sort_values("total_ratings"))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(ex_g.index, ex_g["total_ratings"], color="#4C72B0")
    ax.set_xlabel("total ratings", color="#4C72B0")
    ax2 = ax.twiny()
    ax2.plot(ex_g["mean_bayesian"], ex_g.index, "o-", color="#C44E52")
    ax2.set_xlabel("mean bayesian_rating (dots)", color="#C44E52")
    ax.set_title("Genre: rating volume vs corrected score")
    save(fig, "08_genre_summary")

    # 9. 추천 사분면: 보정평균 높고 표준편차 낮은 영화 (n>=50) + 상위 10 라벨
    cand = rated[rated["n_ratings"] >= 50].copy()
    zb = (cand["bayesian_rating"] - cand["bayesian_rating"].mean()) / cand["bayesian_rating"].std()
    zs = (cand["rating_std"] - cand["rating_std"].mean()) / cand["rating_std"].std()
    cand["pick_score"] = zb - zs
    top10 = cand.nlargest(10, "pick_score")
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(cand["mean_rating"], cand["rating_std"], s=14, alpha=0.35, color="#999999")
    ax.scatter(top10["mean_rating"], top10["rating_std"], s=45, color="#C44E52", zorder=3)
    for _, r in top10.iterrows():
        ax.annotate(r["title"][:24], (r["mean_rating"], r["rating_std"]),
                    fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set(title="Recommendation picks: high bayesian_rating + low rating_std (n>=50)",
           xlabel="mean_rating", ylabel="rating_std (lower = more consensus)")
    save(fig, "09_recommendation_picks")

    # ---- 표 ----
    print("tables:")
    tabs = {
        "top20_by_bayesian": rated.nlargest(20, "bayesian_rating"),
        "most_rated_top20": rated.nlargest(20, "n_ratings"),
        "most_polarizing_n50": rated[rated["n_ratings"] >= 50].nlargest(20, "rating_std"),
        "consensus_praised_n50": rated[(rated["n_ratings"] >= 50) & (rated["mean_rating"] >= 4)]
        .nsmallest(20, "rating_std"),
    }
    keep = ["movieId", "title", "release_year", "genres", "n_ratings",
            "mean_rating", "rating_std", "bayesian_rating"]
    for name, t in tabs.items():
        p = f"{TAB}/{name}.csv"
        t[keep].to_csv(p, index=False)
        print("  ", p)

    # 장르별 요약 (다중 장르는 explode)
    ex = rated.assign(genre=rated["genres"].str.split("|")).explode("genre")
    gsum = (ex.groupby("genre")
            .agg(n_movies=("movieId", "nunique"),
                 total_ratings=("n_ratings", "sum"),
                 mean_of_mean=("mean_rating", "mean"),
                 mean_bayesian=("bayesian_rating", "mean"),
                 mean_std=("rating_std", "mean"))
            .sort_values("total_ratings", ascending=False).round(3))
    gsum.to_csv(f"{TAB}/genre_summary.csv")
    print("  ", f"{TAB}/genre_summary.csv")


if __name__ == "__main__":
    main()
