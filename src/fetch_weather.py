# -*- coding: utf-8 -*-
"""Tier B — TB2: 기상청 API허브 ASOS 일자료 수집 (곤지암 인근: 이천 관측소 203).

`kma_sfcdd.php`는 날짜 범위가 아니라 **하루(tm=YYYYMMDD) 단위**로만 조회된다(공식 스펙:
`?tm=20100715&stn=0&help=1`). 그래서 2023-01-01~2025-06-10(테스트 마지막 창+호라이즌 여유분)을
하루씩 순회해서 모은다. 인증키는 .env의 KMA_API_KEY (기상청 API허브 authKey — data.go.kr의
ServiceKey와는 다른 별도 시스템이다).

출력: data/tierb/weather_icheon.csv (date, ta_avg, ta_max, ta_min, rn_day, sd_new, sd_max)
"""
import os
import time
import urllib.request

import pandas as pd

STN = "203"          # 이천 — 곤지암(경기 광주) 인근 최근접 ASOS
START, END = "2023-01-01", "2025-06-10"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "tierb", "weather_icheon.csv")
BASE = "https://apihub.kma.go.kr/api/typ01/url/kma_sfcdd.php"


def _load_key():
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("KMA_API_KEY="):
                return line.strip().split("=", 1)[1]
    raise RuntimeError("KMA_API_KEY가 .env에 없음")


def fetch_one_day(key, tm):
    url = f"{BASE}?tm={tm}&stn={STN}&authKey={key}"
    with urllib.request.urlopen(url, timeout=10) as r:
        text = r.read().decode("utf-8", errors="replace")
    lines = [l for l in text.splitlines() if l and not l.startswith("#")]
    if not lines:
        return None
    parts = lines[0].split(",")
    if len(parts) < 15:
        return None
    # 헤더 순서(help=1 기준): TM,STN,WS_AVG,WR_DAY,WD_MAX,WS_MAX,WS_MAX_TM,WD_INS,WS_INS,
    #   WS_INS_TM,TA_AVG,TA_MAX,TA_MAX_TM,TA_MIN,TA_MIN_TM,...RN_DAY(idx39)...SD_NEW(idx48)...SD_MAX(idx50)
    def f(i, default=-9.0):
        try:
            v = float(parts[i])
        except (IndexError, ValueError):
            return default
        return default if v in (-9.0, -99.0) else v
    return {
        "ta_avg": f(10), "ta_max": f(11), "ta_min": f(13),
        "rn_day": f(38, 0.0), "sd_new": f(47, 0.0), "sd_max": f(49, 0.0),
    }


def main():
    key = _load_key()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    dates = pd.date_range(START, END, freq="D")
    rows = []
    t0 = time.time()
    for i, d in enumerate(dates):
        tm = d.strftime("%Y%m%d")
        try:
            row = fetch_one_day(key, tm)
        except Exception as e:
            print(f"  {tm} 실패: {e}", flush=True)
            row = None
        if row is None:
            row = {"ta_avg": -9.0, "ta_max": -9.0, "ta_min": -9.0,
                   "rn_day": 0.0, "sd_new": 0.0, "sd_max": 0.0}
        row["date"] = d.strftime("%Y-%m-%d")
        rows.append(row)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(dates)} ({time.time()-t0:.0f}s)", flush=True)
        time.sleep(0.05)
    df = pd.DataFrame(rows)[["date", "ta_avg", "ta_max", "ta_min", "rn_day", "sd_new", "sd_max"]]
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    n_missing = (df["ta_avg"] == -9.0).sum()
    print(f"저장: {OUT} · {len(df)}행 · 결측(-9.0) {n_missing}행 · 총 {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
