"""영화 추천 탐색기 (Streamlit).

data/processed/ 산출물을 바탕으로 화면을 **개인화 / 비개인화 추천 영역**으로 나눠 보여준다.

  🔸 개인화 추천 영역 (화면 최상단, 어떤 영화를 보고 있든 항상 표시 — 선호 영화 기반이라
     사용자마다 결과가 다르다)
    - 내 선호 장르 프로필 (10점 만점): 선호작 장르 비율 × 10 을 막대그래프로
    - 장르 벡터 기반 추천: 선호작들의 장르 원-핫 벡터를 (합산 ÷ 선호 영화 수)한 프로필로
      전체 영화와 코사인 유사도 계산
    - 비슷한 줄거리의 영화(TF-IDF 기반): 선호작들의 TF-IDF 벡터를 평균 낸 프로필로
      전체 영화와 코사인 유사도 계산

  🔹 비개인화 추천 영역 (영화를 선택했을 때만, 그 영화를 기준으로 — 누가 봐도 같은 결과)
    - 이 영화와 같은 장르에서 추천할 만한 영화: 장르 겹침 + 보정 평점(bayesian_rating) 순
    - 비슷한 장르의 영화: 장르 원-핫 벡터 코사인 유사도 순(동점은 보정 평점 순)

  🎬 감독·출연진 기반 추천 (영화를 선택했을 때만, 검색 결과 하단) — 위 두 영역과는 완전히
     별개로 장르/평점/선호 목록을 전혀 쓰지 않는다. data/raw/movie_text_metadata.csv(TMDB) 기준.
    - 같은 감독의 다른 영화: 감독이 한 명이라도 겹치는 영화, 보정 평점 순
    - 출연진이 겹치는 영화: 영화 x 배우 희소행렬(scipy.sparse, 원-핫)을 만들어
      코사인 유사도가 높은 순으로 추천(동점이면 보정 평점 순), 겹치는 배우가
      하나도 없는 영화는 제외

그 외
  - 제목·장르 검색 (평가 수와 무관하게 전부 검색), 영화 상세(평점 분포·평균·추천 라벨)
  - 하트 버튼으로 "선호 영화"에 담기 — data/processed/favorites.json 에 저장되어
    새로고침/재시작해도 유지된다 (로컬 실행 전제, git에는 올리지 않는다)
  - TF-IDF 는 전체 영화 overview 로 **앱 실행 중 단 한 번만** 계산해 캐시한다
    (build_tfidf, @st.cache_data). 목록 필터링·추천 목록 갱신으로는 재계산되지 않고,
    movies_with_ratings.csv 가 실제로 바뀌었을 때(= 새 영화/줄거리 추가, 앱 재시작)만
    다시 계산된다.

실행: 프로젝트 루트에서
    streamlit run streamlit_app.py

먼저 `python run_all.py` 로 data/processed/ 를 생성해야 한다.
필요 패키지: streamlit, pandas, numpy, scikit-learn
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
MOVIES_CSV = PROC / "movies_with_ratings.csv"
GENRE_ONEHOT_CSV = PROC / "movies_genre_onehot.csv"
RATINGS_CSV = PROC / "ratings.csv"
RAW_RATINGS_CSV = ROOT / "data" / "raw" / "ratings.csv"
LABEL_TXT = PROC / "label_text.txt"
FAVORITES_JSON = PROC / "favorites.json"
CREDITS_CSV = ROOT / "data" / "raw" / "movie_text_metadata.csv"  # TMDB 감독/출연진(directors, cast)

RATING_BINS = [x / 2 for x in range(1, 11)]  # 0.5, 1.0, ... 5.0

st.set_page_config(page_title="영화 추천 탐색기", page_icon="🎬", layout="wide")


# ----------------------------- 데이터 로드 -----------------------------
@st.cache_data
def load_movies() -> pd.DataFrame:
    df = pd.read_csv(MOVIES_CSV)
    df["release_year"] = df["release_year"].astype("Int64")
    df["genre_list"] = df["genres"].fillna("").str.split("|")
    return df


@st.cache_data
def load_rating_hist() -> pd.DataFrame:
    """movieId × 평점값 카운트 피벗 (막대그래프용)."""
    path = RATINGS_CSV if RATINGS_CSV.exists() else RAW_RATINGS_CSV
    r = pd.read_csv(path, usecols=["movieId", "rating"])
    piv = (r.groupby(["movieId", "rating"]).size()
           .unstack(fill_value=0)
           .reindex(columns=RATING_BINS, fill_value=0))
    return piv


@st.cache_data
def load_genre_matrix() -> pd.DataFrame:
    """movieId 를 인덱스로 하는 장르 원-핫 매트릭스 (열 순서 고정, 값 0/1)."""
    df = pd.read_csv(GENRE_ONEHOT_CSV)
    meta_cols = {"title", "release_year", "genres"}
    genre_cols = [c for c in df.columns if c not in meta_cols and c != "movieId"]
    return df.set_index("movieId")[genre_cols]


@st.cache_data
def build_tfidf(movies_df: pd.DataFrame):
    """전체 영화 overview 로 TF-IDF 행렬을 **딱 한 번** 계산해 캐시한다.

    - 영어 불용어 제외(stop_words="english").
    - 동일한 overview 텍스트가 여러 movieId 에 걸쳐 있으면 한 번만 코퍼스에 넣는다
      (IDF 왜곡 방지) — movie_to_row 로 movieId -> 행 인덱스를 매핑해 준다.
    - @st.cache_data 는 인자로 받은 movies_df 의 내용을 해시해 캐시 키로 쓰므로,
      overview 데이터가 실제로 바뀔 때(=새 영화/줄거리 추가)만 재계산되고, 화면에서
      필터링하거나 추천 목록을 다시 그릴 때는 절대 재계산되지 않는다.
    """
    have = movies_df.loc[movies_df["overview"].notna(), ["movieId", "overview"]]
    if have.empty:
        return None, {}

    corpus = have.drop_duplicates(subset="overview", keep="first").reset_index(drop=True)
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(corpus["overview"])  # 행마다 L2 정규화됨(단위벡터)

    text_to_row = {text: i for i, text in enumerate(corpus["overview"])}
    movie_to_row = {int(mid): text_to_row[txt] for mid, txt in zip(have["movieId"], have["overview"])}
    return tfidf, movie_to_row


def cosine_similarity_vec(vec, matrix: pd.DataFrame) -> pd.Series | None:
    """vec(장르 원-핫/프로필 벡터) 와 matrix 의 모든 행 사이 코사인 유사도.
    vec 이 전부 0(장르 정보 없음)이면 None."""
    if vec is None or np.sum(vec) == 0:
        return None
    mat = matrix.to_numpy(dtype=float)
    dot = mat @ vec
    norms = np.linalg.norm(mat, axis=1) * np.linalg.norm(vec)
    sim = np.divide(dot, norms, out=np.zeros_like(dot, dtype=float), where=norms > 0)
    return pd.Series(sim, index=matrix.index, name="cosine_sim")


def cosine_similarity_to(mid: int, matrix: pd.DataFrame) -> pd.Series | None:
    """mid 영화의 장르 벡터 기준 코사인 유사도. mid 가 없으면 None."""
    if mid not in matrix.index:
        return None
    return cosine_similarity_vec(matrix.loc[mid].to_numpy(dtype=float), matrix)


def build_favorite_profile(fav_ids, matrix: pd.DataFrame) -> np.ndarray | None:
    """선호 영화들의 장르 원-핫 벡터를 합산한 뒤 선호 영화 수로 나눈 평균 벡터.
    (열별로 '이 장르를 가진 선호작의 비율'이 된다.) 유효한 선호작이 없으면 None."""
    ids = [i for i in fav_ids if i in matrix.index]
    if not ids:
        return None
    vecs = matrix.loc[ids].to_numpy(dtype=float)
    return vecs.sum(axis=0) / len(ids)


def build_favorite_plot_profile(fav_ids, movie_to_row: dict, tfidf):
    """선호 영화들의 TF-IDF 벡터를 합산한 뒤 선호 영화 수로 나눈 평균 벡터.
    (장르 프로필과 같은 방식 — 평균이 곧 합산÷개수다.) 반환: (벡터 또는 None, 사용된 편수)."""
    rows = [movie_to_row[i] for i in fav_ids if i in movie_to_row]
    if not rows:
        return None, 0
    vec = np.asarray(tfidf[rows].mean(axis=0)).ravel()
    return vec, len(rows)


def cosine_sim_tfidf(vec, tfidf) -> np.ndarray | None:
    """vec 과 tfidf 의 모든 행 사이 코사인 유사도. tfidf 행은 이미 단위벡터(L2 정규화)라
    분모는 vec 의 노름만 계산하면 된다."""
    if vec is None or not np.any(vec):
        return None
    vec_norm = np.linalg.norm(vec)
    if vec_norm == 0:
        return None
    dot = np.asarray(tfidf @ vec).ravel()
    return dot / vec_norm


def load_favorites_from_disk() -> set:
    """favorites.json 에서 선호 movieId 목록을 읽는다. 없거나 손상됐으면 빈 집합."""
    if not FAVORITES_JSON.exists():
        return set()
    try:
        data = json.loads(FAVORITES_JSON.read_text(encoding="utf-8"))
        return {int(row["movieId"]) for row in data.get("favorites", [])}
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return set()


def save_favorites_to_disk(fav_ids: set) -> None:
    """선호 movieId 집합을 favorites.json 에 저장 (제목은 가독성을 위해 함께 기록)."""
    rows = []
    for fid in sorted(fav_ids):
        hit = movies.loc[movies["movieId"] == fid, "title"]
        rows.append({"movieId": int(fid), "title": hit.iloc[0] if not hit.empty else ""})
    payload = {"updated_at": datetime.now().isoformat(timespec="seconds"), "favorites": rows}
    FAVORITES_JSON.parent.mkdir(parents=True, exist_ok=True)
    FAVORITES_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


@st.cache_data
def load_credits():
    """movieId -> 감독 이름 리스트 / 출연진 이름 리스트(TMDB 표기 순서 그대로).
    다른 추천 영역(장르·TF-IDF)과는 완전히 별개로, 감독·출연진 섹션에서만 쓴다."""
    if not CREDITS_CSV.exists():
        return {}, {}
    df = pd.read_csv(CREDITS_CSV, usecols=["movieId", "directors", "cast"])

    def parse_names(cell) -> list:
        try:
            return [p["name"] for p in json.loads(cell)]
        except (json.JSONDecodeError, TypeError):
            return []

    directors_by_movie = {int(mid): parse_names(d) for mid, d in zip(df["movieId"], df["directors"])}
    cast_by_movie = {int(mid): parse_names(c) for mid, c in zip(df["movieId"], df["cast"])}
    return directors_by_movie, cast_by_movie


@st.cache_data
def build_cast_matrix(cast_dict: dict):
    """movieId × 배우 **희소행렬**(scipy.sparse.csr_matrix, 원-핫 0/1).
    배우 수(열)가 매우 많고 영화 하나당 배우는 몇 명뿐이라 대부분 0인 희소 행렬로 구성한다
    — "출연진이 겹치는 영화" 추천에서 코사인 유사도 계산에 쓴다."""
    if not cast_dict:
        return None, {}
    movie_ids = sorted(cast_dict.keys())
    vocab: dict[str, int] = {}
    rows, cols = [], []
    for i, mid in enumerate(movie_ids):
        for actor in cast_dict[mid]:
            j = vocab.setdefault(actor, len(vocab))
            rows.append(i)
            cols.append(j)
    data = np.ones(len(rows), dtype=float)
    mat = csr_matrix((data, (rows, cols)), shape=(len(movie_ids), len(vocab)))
    movie_to_row = {mid: i for i, mid in enumerate(movie_ids)}
    return mat, movie_to_row


@st.cache_data
def load_labels() -> dict:
    labels = {}
    if LABEL_TXT.exists():
        for line in LABEL_TXT.read_text(encoding="utf-8").splitlines():
            if "\t" in line:
                k, name = line.split("\t", 1)
                labels[int(k)] = name
    return labels


if not MOVIES_CSV.exists():
    st.error(
        f"데이터가 없습니다: `{MOVIES_CSV.relative_to(ROOT)}`\n\n"
        "프로젝트 루트에서 먼저 실행하세요:\n\n```\npython run_all.py\n```"
    )
    st.stop()

movies = load_movies()
hist = load_rating_hist()
LABELS = load_labels()
genre_matrix = load_genre_matrix() if GENRE_ONEHOT_CSV.exists() else None
# 캐시 키 해시가 단순하도록 movieId·overview 두 열만 넘긴다(다른 열엔 리스트형 genre_list 가 있어 해시 불가).
tfidf_matrix, movie_to_row = build_tfidf(movies[["movieId", "overview"]])  # 전체 영화 기준 1회 계산(캐시)
directors_by_movie, cast_by_movie = load_credits()  # 감독·출연진 (다른 추천 영역과는 별개로 사용)
cast_matrix, cast_movie_to_row = build_cast_matrix(cast_by_movie)  # 배우 희소행렬, 1회 계산(캐시)

LABEL_HELP = {
    0: "평가가 하나도 없는 영화",
    1: "평점 수 30건 미만 — 신뢰하기엔 표본이 부족",
    2: "30건 이상 · 4점 이상 비율 65%↑ · 평점이 고르게 일치",
    3: "30건 이상 · 평점 표준편차 1.05↑ — 사람마다 평이 갈림",
    4: "30건 이상 · 4점 이상 비율 45%↑ — 무난하게 볼 만함",
    5: "30건 이상 · 4점 이상 비율 45% 미만 — 다수가 아쉬워함",
}
LABEL_COLOR = {0: "gray", 1: "gray", 2: "green", 3: "orange", 4: "blue", 5: "red"}


# ----------------------------- 상태 -----------------------------
if "movie_id" not in st.session_state:
    st.session_state.movie_id = None
if "favorites" not in st.session_state:
    st.session_state.favorites = load_favorites_from_disk()  # 파일에서 복원


def select_movie(mid: int):
    st.session_state.movie_id = int(mid)


def toggle_favorite(mid: int):
    mid = int(mid)
    favs = st.session_state.favorites
    if mid in favs:
        favs.discard(mid)
    else:
        favs.add(mid)
    save_favorites_to_disk(favs)


def render_movie_row(row: pd.Series, key_prefix: str, shared_label: str = "공통 장르") -> None:
    """하트 버튼 + 이동 버튼 + 요약 정보 한 줄. 여러 추천 리스트에서 공통으로 쓴다.
    row 에 'shared'(리스트) 열이 있으면 shared_label 이름으로 덧붙여 보여준다
    (장르 목록/감독 이름/배우 이름 등 섹션에 맞게 바꿔 쓸 수 있다)."""
    yr = "" if pd.isna(row["release_year"]) else f" ({int(row['release_year'])})"
    lab = int(row["rating_label"])
    rid = row["movieId"]
    is_fav = rid in st.session_state.favorites

    col_fav, col_btn, col_info = st.columns([1, 3, 2])
    col_fav.button(
        "❤️" if is_fav else "🤍",
        key=f"fav_{key_prefix}_{rid}",
        use_container_width=True,
        on_click=toggle_favorite, args=(rid,),
        help="선호 영화에서 제거" if is_fav else "선호 영화에 추가",
    )
    col_btn.button(
        f"▶  {row['title']}{yr}",
        key=f"go_{key_prefix}_{rid}",
        use_container_width=True,
        on_click=select_movie,
        args=(rid,),
    )

    bits = []
    if "cosine_sim" in row.index and pd.notna(row["cosine_sim"]):
        bits.append(f"유사도 **{row['cosine_sim']:.3f}**")
    bits.append(f"보정 {row['bayesian_rating']:.2f}")
    bits.append(f"평균 {'-' if pd.isna(row['mean_rating']) else format(row['mean_rating'], '.2f')}")
    bits.append(f"평가 {int(row['rating_count'])}")
    bits.append(f":{LABEL_COLOR.get(lab, 'gray')}[{LABELS.get(lab, lab)}]")
    line = " · ".join(bits)
    if "shared" in row.index and row["shared"]:
        line += f"  \n<sub>{shared_label}: {', '.join(row['shared'])}</sub>"
    col_info.markdown(line, unsafe_allow_html=True)


def render_profile_score() -> None:
    """선호 장르 프로필을 10점 만점 점수로 바꿔 시각화."""
    st.subheader("🎯 내 선호 장르 프로필 (10점 만점)")
    fav_ids = st.session_state.favorites

    if not fav_ids:
        st.caption("❤️ 선호 영화를 담으면 장르별 선호 점수(10점 만점)를 여기에 보여줍니다.")
        return
    if genre_matrix is None:
        st.info(f"`{GENRE_ONEHOT_CSV.relative_to(ROOT)}` 가 없습니다. `python run_all.py` 로 생성하세요.")
        return

    profile = build_favorite_profile(fav_ids, genre_matrix)
    if profile is None:
        st.info("선호한 영화들에 장르 정보가 없어 프로필을 만들 수 없습니다.")
        return

    # 프로필 값(0~1, 그 장르를 가진 선호작의 비율)을 10점 만점으로 환산
    scores = (pd.Series(profile, index=genre_matrix.columns) * 10).round(1)
    scores = scores[scores > 0].sort_values(ascending=False)
    if scores.empty:
        st.info("선호한 영화들에 장르 정보가 없어 점수를 계산할 수 없습니다.")
        return

    chart_df = pd.DataFrame({"점수": scores.to_numpy()}, index=pd.Index(scores.index, name="장르"))
    st.bar_chart(chart_df, y="점수", color="#E4572E", height=260)
    st.caption(
        f"선호 영화 {len(fav_ids)}편 기준 · 장르별 (선호작 중 그 장르 비율) × 10점 "
        f"— 10.0 = 선호작 전부가 이 장르, 5.0 = 절반. 최고점: "
        + ", ".join(f"{g} {v:.1f}점" for g, v in scores.head(3).items())
    )


def render_genre_profile_recs() -> None:
    """[개인화] 선호작들의 평균 장르 벡터(프로필)로 만드는 추천."""
    fav_ids = st.session_state.favorites
    st.subheader("장르 벡터 기반 추천")

    if not fav_ids:
        st.caption(
            "❤️ 선호 영화를 1편 이상 담으면, 선호작들의 장르를 (합산 ÷ 선호 영화 수)로 "
            "평균 낸 '선호 장르 프로필' 벡터로 추천을 만들어 보여줍니다."
        )
        return
    if genre_matrix is None:
        st.info(f"`{GENRE_ONEHOT_CSV.relative_to(ROOT)}` 가 없습니다. `python run_all.py` 로 생성하세요.")
        return

    profile = build_favorite_profile(fav_ids, genre_matrix)
    sims = cosine_similarity_vec(profile, genre_matrix)
    if sims is None:
        st.info("선호한 영화들에 장르 정보가 없어 프로필을 만들 수 없습니다.")
        return

    sims = sims.drop(index=[i for i in fav_ids if i in sims.index], errors="ignore")
    sims = sims[sims > 0]  # 겹치는 장르가 하나도 없거나 장르 정보 없는 영화 제외

    prof_df = pd.DataFrame({"movieId": sims.index, "cosine_sim": sims.to_numpy()})
    prof_df = prof_df.merge(movies, on="movieId", how="left")
    prof_df = prof_df.sort_values(
        ["cosine_sim", "bayesian_rating"], ascending=[False, False]
    )

    topn_prof = st.slider("표시 개수", 5, 30, 10, key="profile_topn")
    st.caption(f"프로필과 겹치는 영화 {len(prof_df)}편 중 상위 {min(topn_prof, len(prof_df))}편 "
               "(선호 영화 자신은 제외)")

    for _, row in prof_df.head(topn_prof).iterrows():
        render_movie_row(row, "genreprofile")


def render_plot_profile_recs() -> None:
    """[개인화] 선호작들의 평균 TF-IDF 벡터(줄거리 프로필)로 만드는 추천.
    TF-IDF 행렬 자체는 build_tfidf() 캐시를 그대로 재사용 — 여기서 다시 계산하지 않는다."""
    fav_ids = st.session_state.favorites
    st.subheader("비슷한 줄거리의 영화 (TF-IDF 기반)")

    if not fav_ids:
        st.caption(
            "❤️ 선호 영화를 1편 이상 담으면, 선호작 줄거리들의 TF-IDF 벡터를 "
            "(합산 ÷ 선호 영화 수)로 평균 낸 '선호 줄거리 프로필'로 추천을 보여줍니다."
        )
        return
    if tfidf_matrix is None:
        st.info("overview(줄거리) 데이터가 없어 줄거리 기반 추천을 만들 수 없습니다.")
        return

    vec, n_used = build_favorite_plot_profile(fav_ids, movie_to_row, tfidf_matrix)
    if vec is None:
        st.info("선호한 영화들에 줄거리(overview) 정보가 없어 프로필을 만들 수 없습니다.")
        return

    sims = cosine_sim_tfidf(vec, tfidf_matrix)
    sims_by_movie = {mid_: sims[row] for mid_, row in movie_to_row.items()}
    sim_series = pd.Series(sims_by_movie, name="cosine_sim")
    sim_series = sim_series.drop(index=[i for i in fav_ids if i in sim_series.index], errors="ignore")
    sim_series = sim_series[sim_series > 0]  # 줄거리가 하나도 안 겹치거나 정보 없는 영화 제외

    plot_df = pd.DataFrame({"movieId": sim_series.index, "cosine_sim": sim_series.to_numpy()})
    plot_df = plot_df.merge(movies, on="movieId", how="left")
    plot_df = plot_df.sort_values(["cosine_sim", "bayesian_rating"], ascending=[False, False])

    topn_plot = st.slider("표시 개수", 5, 30, 10, key="plot_topn")
    st.caption(
        f"선호 영화 중 줄거리 있는 {n_used}편 기준 · 줄거리 겹치는 영화 {len(plot_df)}편 중 "
        f"상위 {min(topn_plot, len(plot_df))}편 (선호 영화 자신은 제외)"
    )

    for _, row in plot_df.head(topn_plot).iterrows():
        render_movie_row(row, "plotprofile")


def render_same_director_section(mid: int) -> None:
    """[감독·출연진 — 다른 추천 영역과 무관] 감독이 한 명이라도 겹치는 다른 영화."""
    st.subheader("같은 감독의 다른 영화")
    directors = directors_by_movie.get(mid, [])

    if not directors_by_movie:
        st.info(f"`{CREDITS_CSV.relative_to(ROOT)}` 가 없습니다.")
        return
    if not directors:
        st.caption("이 영화는 감독 정보(TMDB)가 없습니다.")
        return

    target_set = set(directors)
    rows = []
    for other_mid, other_directors in directors_by_movie.items():
        if other_mid == mid:
            continue
        shared = target_set & set(other_directors)
        if shared:
            rows.append({"movieId": other_mid, "shared": sorted(shared)})

    if not rows:
        st.caption(f"감독: {', '.join(directors)} — 겹치는 다른 영화를 찾지 못했습니다.")
        return

    df = pd.DataFrame(rows).merge(movies, on="movieId", how="inner")  # 병합으로 제거된 movieId 는 자동 제외
    df = df.sort_values("bayesian_rating", ascending=False)

    topn = st.slider("표시 개수", 5, 30, 10, key="director_topn")
    st.caption(f"감독: {', '.join(directors)} · {len(df)}편 중 상위 {min(topn, len(df))}편"
               " (정렬: 보정 평점)")

    for _, row in df.head(topn).iterrows():
        render_movie_row(row, "director", shared_label="공통 감독")


def render_cast_overlap_section(mid: int) -> None:
    """[감독·출연진 — 다른 추천 영역과 무관] 출연진이 겹치는 영화.
    영화 × 배우 희소행렬(build_cast_matrix, 앱 실행 중 1회만 계산)에서 대상 영화의
    배우 원-핫 벡터와 다른 모든 영화의 코사인 유사도를 계산해 높은 순으로 추천한다.
    겹치는 배우가 하나도 없는 영화(유사도 0)는 제외하고, 동점이면 보정 평점 순."""
    st.subheader("출연진이 겹치는 영화")

    if cast_matrix is None or mid not in cast_movie_to_row:
        st.caption("이 영화는 출연진 정보(TMDB)가 없어 이 섹션을 만들 수 없습니다.")
        return

    row_idx = cast_movie_to_row[mid]
    sims = sk_cosine_similarity(cast_matrix[row_idx], cast_matrix).ravel()
    target_cast = set(cast_by_movie.get(mid, []))

    rows = []
    for other_mid, other_row in cast_movie_to_row.items():
        if other_mid == mid:
            continue
        sim = sims[other_row]
        if sim <= 0:
            continue  # 겹치는 배우가 하나도 없으면 제외
        shared = target_cast & set(cast_by_movie.get(other_mid, []))
        rows.append({"movieId": other_mid, "cosine_sim": float(sim), "shared": sorted(shared)})

    if not rows:
        st.caption("출연진이 겹치는 영화를 찾지 못했습니다.")
        return

    df = pd.DataFrame(rows).merge(movies, on="movieId", how="inner")
    df = df.sort_values(["cosine_sim", "bayesian_rating"], ascending=[False, False])

    topn = st.slider("표시 개수", 5, 30, 10, key="cast_topn")
    st.caption(f"배우 희소행렬 코사인 유사도 기준 · 겹치는 영화 {len(df)}편 중 상위 {min(topn, len(df))}편"
               " (동점은 보정 평점 순)")

    for _, row in df.head(topn).iterrows():
        render_movie_row(row, "cast", shared_label="공통 배우")


# ----------------------------- 사이드바: 검색 -----------------------------
st.sidebar.title("🎬 영화 검색")
query = st.sidebar.text_input(
    "제목 또는 장르", placeholder="예: matrix, Comedy, 로마…",
    help="제목과 장르(genres) 양쪽에서 부분일치로 찾습니다. 평가 수와 무관하게 모든 영화가 검색됩니다.",
)

if query:
    q = query.strip().lower()
    hit = movies[
        movies["title"].str.lower().str.contains(q, na=False)
        | movies["genres"].str.lower().str.contains(q, na=False)
    ].copy()
    hit = hit.sort_values(["rating_count", "bayesian_rating"], ascending=False)

    st.sidebar.caption(f"검색 결과 {len(hit)}건" + (" (상위 50건 표시)" if len(hit) > 50 else ""))
    for _, row in hit.head(50).iterrows():
        yr = "" if pd.isna(row["release_year"]) else f" ({int(row['release_year'])})"
        st.sidebar.button(
            f"{row['title']}{yr}",
            key=f"s_{row['movieId']}",
            use_container_width=True,
            on_click=select_movie,
            args=(row["movieId"],),
        )
else:
    st.sidebar.info("검색어를 입력하세요.")

st.sidebar.markdown("---")

# ----------------------------- 사이드바: 내가 선호한 영화 -----------------------------
fav_ids = st.session_state.favorites
st.sidebar.subheader(f"❤️ 내가 선호한 영화 ({len(fav_ids)})")

if not fav_ids:
    st.sidebar.caption("추천 목록이나 상세 화면에서 🤍 버튼을 눌러 담아보세요.")
else:
    fav_df = movies[movies["movieId"].isin(fav_ids)].sort_values("title")
    for _, row in fav_df.iterrows():
        yr = "" if pd.isna(row["release_year"]) else f" ({int(row['release_year'])})"
        c_title, c_remove = st.sidebar.columns([4, 1])
        c_title.button(
            f"{row['title']}{yr}",
            key=f"fav_go_{row['movieId']}",
            use_container_width=True,
            on_click=select_movie,
            args=(row["movieId"],),
        )
        c_remove.button(
            "✕", key=f"fav_rm_{row['movieId']}",
            on_click=toggle_favorite, args=(row["movieId"],),
            help="선호 목록에서 제거",
        )


# ============================= 🔸 개인화 추천 영역 =============================
# 선호 영화(favorites)를 기반으로 계산 — 화면 최상단, 어떤 영화를 보든 항상 표시.
# 사용자(선호 목록)에 따라 결과가 달라진다.
st.header("🔸 개인화 추천")
st.caption("내가 담은 선호 영화들을 바탕으로 계산 — 사람마다 결과가 다릅니다.")

render_profile_score()
render_genre_profile_recs()
render_plot_profile_recs()

st.markdown("---")


# ============================= 본문: 영화 상세 =============================
mid = st.session_state.movie_id

if mid is None or mid not in set(movies["movieId"]):
    st.title("영화 추천 탐색기")
    st.markdown(
        "왼쪽에서 영화를 검색하고 선택하세요.\n\n"
        "- **검색**: 제목·장르로 찾기 (평가 수 무관)\n"
        "- **🔸 개인화 추천**(위): 선호 영화를 담을수록 정교해지는 추천\n"
        "- **🔹 비개인화 추천**(영화 선택 시): 평점·인기도·보정 평점 기반 추천\n"
        "- **상세**: 제목·장르·개봉년도·평점 분포·평균 평점·추천 라벨"
    )
    st.stop()

m = movies[movies["movieId"] == mid].iloc[0]
target_genres = set(g for g in m["genre_list"] if g and g != "(no genres listed)")

left, right = st.columns([3, 2])

with left:
    yr = "연도 미상" if pd.isna(m["release_year"]) else int(m["release_year"])
    t_col, fav_col = st.columns([5, 1])
    t_col.title(m["title"])
    is_fav = mid in st.session_state.favorites
    fav_col.button(
        "❤️ 선호함" if is_fav else "🤍 선호 추가",
        key=f"fav_detail_{mid}",
        on_click=toggle_favorite, args=(mid,),
    )
    st.markdown(f"**개봉년도** {yr}  ·  **장르** {m['genres']}")

    lab = int(m["rating_label"])
    lab_name = LABELS.get(lab, str(lab))
    c1, c2, c3 = st.columns(3)
    c1.metric("평균 평점", "-" if pd.isna(m["mean_rating"]) else f"{m['mean_rating']:.2f}")
    c2.metric("평점 수", f"{int(m['rating_count'])}")
    c3.metric("보정 평점", f"{m['bayesian_rating']:.2f}")
    st.markdown(
        f"#### 추천 라벨: :{LABEL_COLOR.get(lab, 'gray')}[{lab} · {lab_name}]"
    )
    st.caption(LABEL_HELP.get(lab, ""))

with right:
    counts = (hist.loc[mid].reindex(RATING_BINS, fill_value=0)
              if mid in hist.index else pd.Series(0, index=RATING_BINS))
    st.markdown("**평점 분포 (0.5 ~ 5.0)**")
    dist_df = pd.DataFrame(
        {"응답 수": counts.values},
        index=pd.Index([f"{b:g}" for b in RATING_BINS], name="평점"),
    )
    st.bar_chart(dist_df, y="응답 수", color="#4C72B0", height=260)
    if counts.sum() == 0:
        st.caption("이 영화에는 평점 기록이 없습니다.")

st.markdown("---")


# ============================= 🎬 감독·출연진 기반 추천 =============================
# 검색해서 선택한 영화 기준. 장르/평점/선호 목록을 전혀 쓰지 않는, 위 두 영역과는
# 완전히 별개인 섹션이다. data/raw/movie_text_metadata.csv(TMDB) 정보가 있는 영화만 대상.
st.header("🎬 감독·출연진 기반 추천")
st.caption(f"「{m['title']}」 검색 결과 하단 — 감독·출연진(TMDB)만 사용, 장르·평점·선호 목록은 쓰지 않습니다.")

render_same_director_section(mid)
render_cast_overlap_section(mid)

st.markdown("---")


# ============================= 🔹 비개인화 추천 영역 =============================
# 지금 보고 있는 영화 하나만 기준 — 평점/인기도/보정 평점 기반. 누가 봐도 같은 결과.
st.header("🔹 비개인화 추천")
st.caption(f"「{m['title']}」 기준 — 평점·인기도(평가 수)·보정 평점만으로 계산, 선호 목록과 무관합니다.")

# ---- 같은 장르 추천 (평점/인기도/보정 평점 기반) ----
st.subheader("이 영화와 같은 장르에서 추천할 만한 영화")

if not target_genres:
    st.info("이 영화는 장르 정보가 없어 장르 기반 추천을 만들 수 없습니다.")
else:
    only_enough = st.checkbox("평가 30건 이상만 보기", value=True)
    topn = st.slider("표시 개수", 5, 30, 10)

    same = movies[movies["movieId"] != mid].copy()
    same["shared"] = same["genre_list"].apply(
        lambda gs: sorted(target_genres.intersection(gs))
    )
    same = same[same["shared"].apply(len) > 0]
    if only_enough:
        same = same[same["rating_count"] >= 30]
    same = same.sort_values("bayesian_rating", ascending=False).head(topn)

    st.caption(f"기준 장르: {', '.join(sorted(target_genres))}  ·  {len(same)}편  ·  "
               "정렬: 보정 평점(bayesian_rating) 내림차순")

    for _, row in same.iterrows():
        render_movie_row(row, "genre")

st.markdown("---")

# ---- 비슷한 장르의 영화 (코사인 유사도, 동점은 보정 평점) ----
st.subheader("비슷한 장르의 영화")
st.caption(
    "장르를 원-핫 벡터로 바꿔 코사인 유사도를 계산합니다. "
    "겹치는 장르가 하나도 없거나 장르 정보가 없는 영화는 제외하고, "
    "유사도가 높은 순으로 나열합니다(동점이면 보정 평점 순)."
)

if genre_matrix is None:
    st.info(f"`{GENRE_ONEHOT_CSV.relative_to(ROOT)}` 가 없습니다. `python run_all.py` 로 생성하세요.")
elif not target_genres:
    st.info("이 영화는 장르 정보가 없어 유사도 기반 추천을 만들 수 없습니다.")
else:
    sims = cosine_similarity_to(mid, genre_matrix)
    if sims is None:
        st.info("이 영화는 장르 정보가 없어 유사도 기반 추천을 만들 수 없습니다.")
    else:
        sims = sims.drop(index=mid, errors="ignore")
        sims = sims[sims > 0]  # 겹치는 장르가 하나도 없으면(장르 없음 포함) 제외

        sim_df = pd.DataFrame({"movieId": sims.index, "cosine_sim": sims.to_numpy()})
        sim_df = sim_df.merge(movies, on="movieId", how="left")
        sim_df["shared"] = sim_df["genre_list"].apply(
            lambda gs: sorted(target_genres.intersection(gs))
        )
        sim_df = sim_df.sort_values(
            ["cosine_sim", "bayesian_rating"], ascending=[False, False]
        )

        topn_sim = st.slider("표시 개수", 5, 30, 10, key="sim_topn")
        st.caption(f"장르 겹치는 영화 {len(sim_df)}편 중 상위 {min(topn_sim, len(sim_df))}편")

        for _, row in sim_df.head(topn_sim).iterrows():
            render_movie_row(row, "sim")
