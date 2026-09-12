"""영화 추천 탐색기 (Streamlit).

data/processed/ 산출물을 바탕으로
  1) 제목·장르로 영화 검색 (평가 수와 무관하게 전부 검색)
  2) 검색한 영화와 같은 장르에서 보정 평점(bayesian_rating) 높은 영화 추천
  3) 장르 원-핫 벡터의 코사인 유사도로 "비슷한 장르의 영화" 순위 (유사도 높은 순,
     동점이면 보정 평점 순 / 겹치는 장르가 없거나 장르 정보가 없는 영화는 제외)
  4) 영화 선택 시 상세: 제목·장르·개봉년도·평점 분포 막대그래프·평균 평점·추천 라벨
  5) 추천 리스트(및 상세)에서 하트 버튼으로 "선호 영화"에 담고, 사이드바에서 목록 확인
     (세션에만 저장 — 브라우저 새로고침/재시작 시 초기화됨)
  6) 선호 영화가 여러 편이면, 그 영화들의 장르 원-핫 벡터를 합산한 뒤 선호 영화 수로
     나눈 "선호 장르 프로필" 벡터를 만들어 별도로 추천 ("내 선호 영화 프로필 기반 추천")

실행: 프로젝트 루트에서
    streamlit run streamlit_app.py

먼저 `python run_all.py` 로 data/processed/ 를 생성해야 한다.
필요 패키지: streamlit, pandas, numpy
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
MOVIES_CSV = PROC / "movies_with_ratings.csv"
GENRE_ONEHOT_CSV = PROC / "movies_genre_onehot.csv"
RATINGS_CSV = PROC / "ratings.csv"
RAW_RATINGS_CSV = ROOT / "data" / "raw" / "ratings.csv"
LABEL_TXT = PROC / "label_text.txt"

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
    st.session_state.favorites = set()  # 선호 영화 movieId 집합 (세션 한정)


def select_movie(mid: int):
    st.session_state.movie_id = int(mid)


def toggle_favorite(mid: int):
    mid = int(mid)
    favs = st.session_state.favorites
    if mid in favs:
        favs.discard(mid)
    else:
        favs.add(mid)


def render_movie_row(row: pd.Series, key_prefix: str) -> None:
    """하트 버튼 + 이동 버튼 + 요약 정보 한 줄. 여러 추천 리스트에서 공통으로 쓴다."""
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
        line += f"  \n<sub>공통 장르: {', '.join(row['shared'])}</sub>"
    col_info.markdown(line, unsafe_allow_html=True)


def render_profile_section() -> None:
    """선호 영화들의 평균 장르 벡터(프로필)로 만드는 추천 — 어느 화면에서든 표시."""
    fav_ids = st.session_state.favorites
    st.subheader("내 선호 영화 프로필 기반 추천")

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

    prof = pd.Series(profile, index=genre_matrix.columns)
    prof = prof[prof > 0].sort_values(ascending=False)
    st.caption(
        f"선호 영화 {len(fav_ids)}편의 장르 평균(비율): "
        + ", ".join(f"{g} {v:.0%}" for g, v in prof.items())
    )

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
        render_movie_row(row, "profile")


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


# ----------------------------- 본문 -----------------------------
mid = st.session_state.movie_id

if mid is None or mid not in set(movies["movieId"]):
    st.title("영화 추천 탐색기")
    st.markdown(
        "왼쪽에서 영화를 검색하고 선택하세요.\n\n"
        "- **검색**: 제목·장르로 찾기 (평가 수 무관)\n"
        "- **추천**: 선택한 영화와 같은 장르에서 보정 평점이 높은 영화\n"
        "- **상세**: 제목·장르·개봉년도·평점 분포·평균 평점·추천 라벨\n"
        "- **프로필 추천**: 선호 영화를 담을수록 아래 프로필 추천이 정교해집니다."
    )
    st.markdown("---")
    render_profile_section()
    st.stop()

m = movies[movies["movieId"] == mid].iloc[0]
target_genres = set(g for g in m["genre_list"] if g and g != "(no genres listed)")

# ---- 상세 정보 ----
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

# ---- 같은 장르 추천 ----
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

    st.caption(f"기준 장르: {', '.join(sorted(target_genres))}  ·  {len(same)}편")

    for _, row in same.iterrows():
        render_movie_row(row, "genre")

st.markdown("---")

# ---- 비슷한 장르의 영화 (코사인 유사도) ----
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

st.markdown("---")

# ---- 내 선호 영화 프로필 기반 추천 ----
render_profile_section()
