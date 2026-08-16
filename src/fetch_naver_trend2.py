# -*- coding: utf-8 -*-
"""Tier B — TB7 신규 검색어 후보 스크리닝용 원본 수집.

fetch_naver_trend.py(TB4, "곤지암리조트"/"화담숲")와 별개 파일로 둔다 — 실패한 TB4
계열을 건드리지 않고, 완전히 새로운 키워드 후보를 원본만 먼저 받아 저비용 EDA로
거른 뒤 유망한 것만 4폴드 검증으로 넘기기 위함(experiments/tb7_naver_keywords_brief.md
사전등록 참고).

키워드 그룹은 한 번 호출에 5개까지만 허용되어 두 번에 나눠 받는다.
출력: data/tierb/naver_trend2.csv (date, 각 groupName 열)
"""
import json
import os
import time
import urllib.request
import urllib.error

import pandas as pd

START, END = "2023-01-01", "2025-06-10"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "tierb", "naver_trend2.csv")
URL = "https://naverapihub.apigw.ntruss.com/search-trend/v1/search"
CHUNK_MONTHS = 12

BATCH1 = [
    {"groupName": "hwadam_foliage", "keywords": ["화담숲 단풍"]},
    {"groupName": "ski_field", "keywords": ["곤지암 스키장"]},
    {"groupName": "mirasia", "keywords": ["미라시아"]},
    {"groupName": "mirasia_brunch", "keywords": ["미라시아 브런치"]},
    {"groupName": "damha", "keywords": ["담하"]},
]
BATCH2 = [
    {"groupName": "sled", "keywords": ["곤지암리조트 눈썰매장"]},
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


def fetch_chunk(cid, secret, d0, d1, groups):
    body = json.dumps({
        "startDate": d0, "endDate": d1, "timeUnit": "date",
        "keywordGroups": groups, "device": "", "ages": [], "gender": "",
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
    for groups in (BATCH1, BATCH2):
        for d0, d1 in _chunks(START, END):
            try:
                resp = fetch_chunk(cid, secret, d0, d1, groups)
            except urllib.error.HTTPError as e:
                print(f"  {d0}~{d1} 실패: {e.code} {e.read().decode('utf-8', 'replace')}", flush=True)
                continue
            for series in resp.get("results", []):
                name = series["title"]
                for pt in series["data"]:
                    rows.setdefault(pt["period"], {})[name] = pt["ratio"]
            print(f"  [{groups[0]['groupName']}군] {d0}~{d1} 완료 ({time.time()-t0:.0f}s)", flush=True)
            time.sleep(0.2)
    df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    df.index.name = "date"
    df = df.reset_index()
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"저장: {OUT} · {len(df)}행 · 총 {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
