# -*- coding: utf-8 -*-
"""Tier B — TB3: 한국관광공사 지역별 방문자수(기초 지자체) 수집 — 경기 광주시(signguCode 41610).

⚠️ 이 API의 `locgoRegnVisitrDDList`는 지역을 요청 파라미터로 거르지 않는다(공식 가이드
`TourAPI_Guide_(관광빅데이터)v4.1` 확인 — 요청 파라미터는 serviceKey·MobileOS·MobileApp·
startYmd·endYmd·numOfRows·pageNo뿐이고 `signguCode`는 **응답 필드**다). 그래서 전국 데이터를
한 번에 받아서 41610(광주시)만 클라이언트에서 걸러낸다. 하루치가 약 792행(시군구 264개 ×
관광객구분 3종)이라 한 달씩 끊어서(약 24,000행/월) 받는다.

touDivCd: 1=현지인 2=외지인 3=외국인. 곤지암리조트는 외지인(당일/숙박) 비중이 핵심일 가능성이 높다.

출력: data/tierb/visitors_gwangju.csv (date, local, outside, foreign)
"""
import json
import os
import time
import urllib.request

import pandas as pd

SIGNGU = "41610"      # 경기도 광주시
START, END = "2023-01-01", "2025-06-10"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "tierb", "visitors_gwangju.csv")
URL = "http://apis.data.go.kr/B551011/DataLabService/locgoRegnVisitrDDList"
TOU_NAME = {"1": "local", "2": "outside", "3": "foreign"}


def _load_key():
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("DATA_GO_KR_KEY="):
                return line.strip().split("=", 1)[1]
    raise RuntimeError("DATA_GO_KR_KEY가 .env에 없음")


def _chunks(start, end, months=1):
    cur = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    out = []
    while cur <= end_ts:
        nxt = min(cur + pd.DateOffset(months=months) - pd.Timedelta(days=1), end_ts)
        out.append((cur.strftime("%Y%m%d"), nxt.strftime("%Y%m%d")))
        cur = nxt + pd.Timedelta(days=1)
    return out


def fetch_chunk(key, d0, d1):
    # 인증키는 이미 %-인코딩된 값이므로 quote_via로 재인코딩하지 않는다.
    url = (f"{URL}?serviceKey={key}&pageNo=1&numOfRows=30000"
           f"&MobileOS=ETC&MobileApp=AppTest&startYmd={d0}&endYmd={d1}&_type=json")
    with urllib.request.urlopen(url, timeout=20) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    key = _load_key()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = {}
    t0 = time.time()
    for d0, d1 in _chunks(START, END):
        resp = fetch_chunk(key, d0, d1)
        body = resp.get("response", {}).get("body", {})
        items = (body.get("items") or {}).get("item") or []
        if isinstance(items, dict):
            items = [items]
        n_total = body.get("totalCount", 0)
        if n_total > 30000:
            print(f"  ⚠️ {d0}~{d1} totalCount={n_total} > 30000 — 유실 가능", flush=True)
        got = 0
        for it in items:
            if it.get("signguCode") != SIGNGU:
                continue
            d = it["baseYmd"]
            col = TOU_NAME.get(it["touDivCd"])
            rows.setdefault(d, {})[col] = float(it["touNum"])
            got += 1
        print(f"  {d0}~{d1}: 전체 {len(items)}행 · 광주시 {got}행 ({time.time()-t0:.0f}s)", flush=True)
        time.sleep(0.2)
    df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    df.index.name = "date"
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    for c in ("local", "outside", "foreign"):
        if c not in df.columns:
            df[c] = 0.0
    df = df[["date", "local", "outside", "foreign"]]
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"저장: {OUT} · {len(df)}행 · 총 {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
