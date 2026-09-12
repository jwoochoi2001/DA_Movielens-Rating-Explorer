"""streamlit_app.py 를 띄워 대시보드 스크린샷을 outputs/screenshots/ 에 저장.

streamlit 서버를 직접 실행하고, playwright(chromium)로 화면을 캡처한 뒤 종료한다.

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
            # Chinatown(상세) + 첫 추천작 하트를 클릭
            search.fill("")
            page.keyboard.press("Enter")
            wait_idle(page, 1200)
            page.get_by_role("button", name="🤍 선호 추가").click()
            wait_idle(page, 1200)
            page.locator("button:visible").filter(has_text="🤍").first.click()
            wait_idle(page, 1200)
            page.screenshot(path=OUT / "app_04_favorites.png", full_page=True)
            print("  ", OUT / "app_04_favorites.png")

            # 비슷한 장르의 영화 (코사인 유사도) 섹션까지 스크롤해서 캡처
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

            # 내 선호 영화 프로필 기반 추천 (Chinatown + Shawshank Redemption 2편 기준)
            page.get_by_text("내 선호 영화 프로필 기반 추천", exact=True).scroll_into_view_if_needed()
            wait_idle(page, 1000)
            page.screenshot(path=OUT / "app_06_profile.png")
            print("  ", OUT / "app_06_profile.png")

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
