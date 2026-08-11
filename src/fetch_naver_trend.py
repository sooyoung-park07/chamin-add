# -*- coding: utf-8 -*-
"""Tier B — TB4: 네이버 데이터랩 검색어트렌드 API 수집.

"곤지암리조트"/"화담숲" 등 키워드의 일별 상대 검색량(ratio, 0~100)을 받는다. 절대 검색량이
아니라 기간 내 최고치 대비 비율이므로, 그 자체를 피처로 쓰기보다는 창(28일) 내에서의
상대적 모멘텀(예: 최근 7일 평균 대비)으로 가공해서 쓴다.

인증: .env의 NAVER_CLIENT_ID / NAVER_CLIENT_SECRET (헤더로 전달, ServiceKey 아님).
출력: data/tierb/naver_trend.csv (date, gonjiam, hwadam)
"""
import json
import os
import time
import urllib.request
import urllib.error

import pandas as pd

START, END = "2023-01-01", "2025-06-10"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "tierb", "naver_trend.csv")
URL = "https://naverapihub.apigw.ntruss.com/search-trend/v1/search"
# 데이터랩은 한 번 호출에 최대 1년(구간)까지만 허용 — 기간을 나눠서 호출한다.
CHUNK_MONTHS = 12
KEYWORD_GROUPS = [
    {"groupName": "gonjiam", "keywords": ["곤지암리조트"]},
    {"groupName": "hwadam", "keywords": ["화담숲"]},
]


def _load_env():
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    kv = {}
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                kv[k] = v
    return kv["NAVER_CLIENT_ID"], kv["NAVER_CLIENT_SECRET"]


def _chunks(start, end, months=CHUNK_MONTHS):
    cur = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    out = []
    while cur <= end_ts:
        nxt = min(cur + pd.DateOffset(months=months) - pd.Timedelta(days=1), end_ts)
        out.append((cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
        cur = nxt + pd.Timedelta(days=1)
    return out


def fetch_chunk(cid, secret, d0, d1):
    body = json.dumps({
        "startDate": d0, "endDate": d1, "timeUnit": "date",
        "keywordGroups": KEYWORD_GROUPS, "device": "", "ages": [], "gender": "",
    }, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(URL, data=body, method="POST")
    req.add_header("X-NCP-APIGW-API-KEY-ID", cid)
    req.add_header("X-NCP-APIGW-API-KEY", secret)
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    cid, secret = _load_env()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = {}
    t0 = time.time()
    for d0, d1 in _chunks(START, END):
        try:
            resp = fetch_chunk(cid, secret, d0, d1)
        except urllib.error.HTTPError as e:
            print(f"  {d0}~{d1} 실패: {e.code} {e.read().decode('utf-8', 'replace')}", flush=True)
            continue
        for series in resp.get("results", []):
            name = series["title"]
            for pt in series["data"]:
                rows.setdefault(pt["period"], {})[name] = pt["ratio"]
        print(f"  {d0}~{d1} 완료 ({time.time()-t0:.0f}s)", flush=True)
        time.sleep(0.2)
    df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    df.index.name = "date"
    df = df.reset_index().rename(columns={"gonjiam": "gonjiam", "hwadam": "hwadam"})
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"저장: {OUT} · {len(df)}행 · 총 {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
