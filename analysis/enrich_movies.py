"""제목에서 개봉연도 분리 + 결측 연도/장르 보강.

입력 : data/processed/movies.csv   (중복 병합까지 끝난 파일)
출력 : data/processed/movies_enriched.csv
        컬럼: movieId, title(연도 제거), release_year, genres, year_estimated, genres_filled

규칙
- 제목 끝 '(YYYY)' 를 떼어 release_year(정수) 로 분리한다.
- 제목 끝이 '(YYYY-YYYY)' 처럼 기간이면 앞선 연도를 반영한다.
- 연도가 없는 13건은 웹 검색으로 확인한 추정 개봉연도를 넣고 year_estimated=True.
- genres 가 '(no genres listed)' 인 34건은 웹 검색으로 확인한 장르를
  기존 movies 데이터에 이미 쓰인 19개 장르 범주 안에서만 골라 채우고 genres_filled=True.
- data/raw 는 수정하지 않는다. ratings.csv 는 이 단계에서 바뀌지 않는다(제목/장르 미포함).

실행: 프로젝트 루트에서
    python analysis/enrich_movies.py
필요 패키지: pandas
"""
import re
import pandas as pd

SRC = "data/processed/movies.csv"
OUT = "data/processed/movies_enriched.csv"

# --- 웹 검색으로 확인한 추정 개봉연도 (제목에 연도 표기가 없는 13건) ---
# 출처: IMDb / Wikipedia (아래 대화 로그의 검색 결과 참조)
ESTIMATED_YEAR = {
    40697: 1993,   # Babylon 5 (TV 시리즈, 1993 파일럿/시리즈 시작)
    140956: 2018,  # Ready Player One
    143410: 2015,  # Hyena Road
    147250: 1980,  # The Adventures of Sherlock Holmes and Doctor Watson (소련 TV, 1979~1986; 대표 1980)
    149334: 2016,  # Nocturnal Animals
    156605: 2016,  # Paterson
    162414: 2016,  # Moonlight
    167570: 2016,  # The OA
    171495: 2015,  # Cosmos (Andrzej Żuławski)
    171631: 2017,  # Maria Bamford: Old Baby
    171891: 2017,  # Generation Iron 2
    176601: 2011,  # Black Mirror (TV 시리즈 시작연도)
}

# --- 웹 검색으로 확인한 장르 (genres 가 '(no genres listed)' 인 34건) ---
# 19개 허용 장르: Action, Adventure, Animation, Children, Comedy, Crime, Documentary,
#   Drama, Fantasy, Film-Noir, Horror, IMAX, Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western
GENRE_FILL = {
    114335: "Comedy|Fantasy",                      # La cravate (1957) 초현실 무언극 단편
    122888: "Action|Adventure|Drama",              # Ben-Hur (2016)
    122896: "Action|Adventure|Comedy|Fantasy",     # Pirates of the Caribbean: Dead Men Tell No Tales
    129250: "Action|Comedy|Crime",                 # Superfast! (2015) 패러디
    132084: "Drama|Musical|Romance",               # Let It Be Me (1995)
    134861: "Comedy|Documentary",                  # Trevor Noah: African American (2013) 스탠드업
    141131: "Action|Adventure|Sci-Fi",             # Guardians / Zaschitniki
    141866: "Crime|Horror|Thriller",               # Green Room (2015)
    142456: "Comedy|Drama|Fantasy",                # The Brand New Testament (2015)
    149330: "Animation|Children",                  # A Cosmic Christmas (1977) 애니 특집
    152037: "Comedy|Musical|Romance",              # Grease Live! (2016)
    155589: "Adventure|Comedy",                    # Noin 7 veljestä (1968)
    159161: "Comedy|Documentary",                  # Ali Wong: Baby Cobra (2016) 스탠드업
    159779: "Comedy|Drama|Fantasy|Romance",        # A Midsummer Night's Dream (2016)
    161008: "Drama|Musical|Romance",               # The Forbidden Dance (1990)
    165489: "Animation|Drama",                     # Ethel & Ernest (2016)
    166024: "Drama",                               # Whiplash (2013) 단편
    169034: "Musical",                             # Lemonade (2016) 비주얼 앨범
    171495: "Comedy|Drama|Mystery",                # Cosmos (2015)
    171631: "Comedy|Documentary",                  # Maria Bamford: Old Baby (2017)
    171749: "Crime|Drama|Mystery|Thriller",        # Death Note: Desu nôto (2006)
    171891: "Documentary",                         # Generation Iron 2 (2017)
    172497: "Action|Adventure|Sci-Fi",             # T2 3-D: Battle Across Time (1996)
    172591: "Crime|Drama",                         # The Godfather Trilogy: 1972-1990 (1992)
    173535: "Crime|Drama|Mystery",                 # Sherlock Holmes...: The Hunt for the Tiger (1980)
    174403: "Documentary",                         # The Putin Interviews (2017)
    176601: "Drama|Sci-Fi|Thriller",               # Black Mirror (2011)
    181413: "Comedy|Documentary",                  # Too Funny to Fail (2017)
    181719: "Drama",                               # Serving in Silence (1995) 전기 드라마
    182727: "Comedy|Musical",                      # A Christmas Story Live! (2017)
    143410: "Drama|War",                           # Hyena Road (2015)
    147250: "Crime|Drama|Mystery",                 # The Adventures of Sherlock Holmes and Doctor Watson
    156605: "Comedy|Drama|Romance",                # Paterson (2016)
    167570: "Drama|Mystery|Sci-Fi",                # The OA (2016)
}

YEAR_RE = re.compile(r"^(?P<title>.*?)\s*\((?P<year>\d{4})\)\s*$")
RANGE_RE = re.compile(r"^(?P<title>.*?)\s*\((?P<start>\d{4})[-–]\d{4}\)\s*$")


def split_year(title: str):
    m = YEAR_RE.match(title)
    if m:
        return m.group("title").strip(), int(m.group("year"))
    m = RANGE_RE.match(title)
    if m:  # 기간 표기 → 앞선 연도 반영
        return m.group("title").strip(), int(m.group("start"))
    return title.strip(), None


def main() -> None:
    m = pd.read_csv(SRC)
    allowed = {
        g for row in m["genres"] for g in row.split("|") if g != "(no genres listed)"
    }

    titles, years = zip(*m["title"].map(split_year))
    m["title"] = list(titles)
    m["release_year"] = list(years)

    # 연도 보강
    m["year_estimated"] = False
    miss_year = m["release_year"].isna()
    for mid, y in ESTIMATED_YEAR.items():
        mask = m["movieId"].eq(mid)
        m.loc[mask, "release_year"] = y
        m.loc[mask, "year_estimated"] = True
    still_missing = m["release_year"].isna()
    print(f"연도 없음 {miss_year.sum()}건 → 보강 {miss_year.sum() - still_missing.sum()}건, "
          f"남은 결측 {still_missing.sum()}건")

    # 장르 보강
    m["genres_filled"] = False
    no_genre = m["genres"].eq("(no genres listed)")
    for mid, g in GENRE_FILL.items():
        bad = set(g.split("|")) - allowed
        if bad:
            raise ValueError(f"movieId {mid}: 허용되지 않은 장르 {bad}")
        mask = m["movieId"].eq(mid)
        m.loc[mask, "genres"] = g
        m.loc[mask, "genres_filled"] = True
    left = m["genres"].eq("(no genres listed)")
    print(f"장르 없음 {no_genre.sum()}건 → 보강 {no_genre.sum() - left.sum()}건, "
          f"남은 '(no genres listed)' {left.sum()}건")

    m["release_year"] = m["release_year"].astype("Int64")
    m = m[["movieId", "title", "release_year", "genres", "year_estimated", "genres_filled"]]
    m.to_csv(OUT, index=False)
    print(f"\n저장: {OUT}  ({len(m)}행)")

    # 검증
    assert m["release_year"].notna().all(), "연도 결측 남음"
    assert m["release_year"].between(1870, 2030).all(), "연도 범위 이상"
    assert not m["genres"].eq("(no genres listed)").any(), "장르 결측 남음"
    assert m["movieId"].is_unique
    print("검증 통과: 모든 행에 연도 존재, '(no genres listed)' 없음, movieId 유일")

    print(f"\n[연도 추정 {int(m['year_estimated'].sum())}건] (기간표기 1건은 앞 연도로 규칙 파생)")
    print(m[m["year_estimated"]][["movieId", "title", "release_year"]].to_string(index=False))
    print(f"\n[장르 보강 {int(m['genres_filled'].sum())}건]")
    print(m[m["genres_filled"]][["movieId", "title", "release_year", "genres"]].to_string(index=False))


if __name__ == "__main__":
    main()
