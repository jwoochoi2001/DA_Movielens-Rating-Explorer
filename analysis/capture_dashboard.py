"""streamlit_app.py 를 띄워 대시보드 스크린샷을 outputs/screenshots/ 에 저장.

streamlit 서버를 직접 실행하고, playwright(chromium)로 화면을 캡처한 뒤 종료한다.
🔸 개인화 추천 / 🔹 비개인화 추천 두 영역을 모두 보여주도록 순서를 짰다.

실행: 프로젝트 루트에서
    python analysis/capture_dashboard.py
필요 패키지: streamlit, playwright (+ `playwright install chromium`)
"""
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "screenshots"
PORT = 8577
URL = f"http://localhost:{PORT}"


def wait_idle(page, ms=1500):
    page.wait_for_timeout(ms)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    srv = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
         "--server.port", str(PORT), "--server.headless", "true",
         "--browser.gatherUsageStats", "false"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(9)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.goto(URL, wait_until="networkidle")
            wait_idle(page, 2500)

            search = page.get_by_role("textbox", name="제목 또는 장르")
            search.click()
            search.fill("matrix")
            page.keyboard.press("Enter")
            wait_idle(page, 2500)
            page.screenshot(path=OUT / "app_01_search.png")
            print("  ", OUT / "app_01_search.png")

            page.get_by_role("button", name="Matrix, The (1999)").click()
            wait_idle(page, 2500)
            page.screenshot(path=OUT / "app_02_detail.png", full_page=True)
            print("  ", OUT / "app_02_detail.png")

            search.click()
            search.fill("Film-Noir")
            page.keyboard.press("Enter")
            wait_idle(page, 2000)
            page.get_by_role("button", name="Chinatown (1974)").click()
            wait_idle(page, 2500)
            page.screenshot(path=OUT / "app_03_genre_search.png", full_page=True)
            print("  ", OUT / "app_03_genre_search.png")

            # 선호 영화 담기: 검색창을 비워 사이드바 여백을 확보한 뒤
            # Chinatown(상세) + 첫 추천작 하트를 클릭 -> 이후로는 🔸 개인화 추천 영역이
            # 항상 이 2편(Chinatown, Shawshank Redemption) 기준으로 채워져 보인다.
            search.fill("")
            page.keyboard.press("Enter")
            wait_idle(page, 1200)
            page.get_by_role("button", name="🤍 선호 추가").click()
            wait_idle(page, 1200)
            page.locator("button:visible").filter(has_text="🤍").first.click()
            wait_idle(page, 1200)
            page.screenshot(path=OUT / "app_04_favorites.png", full_page=True)
            print("  ", OUT / "app_04_favorites.png")

            # Matrix 로 다시 이동 -> 🔹 비개인화 추천의 "비슷한 장르의 영화" 섹션까지 스크롤
            # (Streamlit 앱 컨테이너가 자체 스크롤이라 full_page 캡처가 뷰포트로 제한됨)
            search.click()
            search.fill("matrix")
            page.keyboard.press("Enter")
            wait_idle(page, 1500)
            page.get_by_role("button", name="Matrix, The (1999)").click()
            wait_idle(page, 1500)
            page.get_by_text("비슷한 장르의 영화", exact=True).scroll_into_view_if_needed()
            wait_idle(page, 1000)
            page.screenshot(path=OUT / "app_05_similar_genre.png")
            print("  ", OUT / "app_05_similar_genre.png")

            # 🔸 개인화: 장르 벡터 기반 추천 (Chinatown + Shawshank Redemption 2편 기준)
            page.get_by_text("장르 벡터 기반 추천", exact=True).scroll_into_view_if_needed()
            wait_idle(page, 1000)
            page.screenshot(path=OUT / "app_06_genre_profile.png")
            print("  ", OUT / "app_06_genre_profile.png")

            # 🔸 개인화: 비슷한 줄거리의 영화 (TF-IDF 기반)
            page.get_by_text("비슷한 줄거리의 영화", exact=False).first.scroll_into_view_if_needed()
            wait_idle(page, 1000)
            page.screenshot(path=OUT / "app_07_plot_profile.png")
            print("  ", OUT / "app_07_plot_profile.png")

            # 🔸 개인화: 내 선호 장르 프로필 (10점 만점) — 화면 최상단
            page.get_by_text("내 선호 장르 프로필 (10점 만점)").scroll_into_view_if_needed()
            wait_idle(page, 1000)
            page.screenshot(path=OUT / "app_08_profile_score.png")
            print("  ", OUT / "app_08_profile_score.png")

            # 🎬 감독·출연진 기반 추천 (Matrix 기준 — 같은 감독의 다른 영화 + 출연진 겹치는 영화)
            page.get_by_text("감독·출연진 기반 추천", exact=False).scroll_into_view_if_needed()
            wait_idle(page, 1000)
            page.screenshot(path=OUT / "app_09_director_cast.png")
            print("  ", OUT / "app_09_director_cast.png")

            # 배우 버튼 클릭 -> 그 배우의 출연작 목록
            page.get_by_role("button", name="Keanu Reeves", exact=True).click()
            wait_idle(page, 1500)
            page.get_by_text("Keanu Reeves 출연작", exact=False).first.scroll_into_view_if_needed()
            wait_idle(page, 1000)
            page.screenshot(path=OUT / "app_10_actor_filmography.png")
            print("  ", OUT / "app_10_actor_filmography.png")

            # 🔸 개인화: 나랑 비슷한 사람들이 좋아하는 영화 (사용자 기반 협업 필터링, userId 414)
            # exact=True — 아이템 기반 CF 섹션의 안내문이 이 제목을 인용해서, 부분일치로는 두 곳이 걸린다.
            page.get_by_text("나랑 비슷한 사람들이 좋아하는 영화", exact=True).scroll_into_view_if_needed()
            wait_idle(page, 800)
            uid_input = page.get_by_role("spinbutton")
            uid_input.click()
            uid_input.fill("414")
            uid_input.press("Tab")
            wait_idle(page, 3000)
            page.get_by_text("나랑 비슷한 사람들이 좋아하는 영화", exact=True).scroll_into_view_if_needed()
            wait_idle(page, 1000)
            page.screenshot(path=OUT / "app_11_user_cf.png")
            print("  ", OUT / "app_11_user_cf.png")

            # 🔸 개인화: 내가 좋아했던 영화와 비슷한 영화 (아이템 기반 협업 필터링, 같은 userId 414)
            wait_idle(page, 1500)
            page.get_by_text("내가 좋아했던 영화와 비슷한 영화", exact=False).scroll_into_view_if_needed()
            wait_idle(page, 1000)
            page.screenshot(path=OUT / "app_12_item_cf.png")
            print("  ", OUT / "app_12_item_cf.png")

            browser.close()
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=10)
        except subprocess.TimeoutExpired:
            srv.kill()


if __name__ == "__main__":
    print("screenshots:")
    main()
