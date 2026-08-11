# -*- coding: utf-8 -*-
"""Tier B — TB4: 네이버 검색어트렌드 원본 → 모멘텀 피처 가공.

fetch_naver_trend.py가 받아온 원본 ratio(기간 내 최고치 대비 0~100)는 청크(연도)별로
기준점이 달라 그대로 쓰면 안 된다. 최근 7일 이동평균 대비 비율(모멘텀)로 바꿔서
청크 간 기준점 차이를 상쇄하고, "평소보다 지금 얼마나 뜨거운가"라는 캘린더 피처가
못 만드는 정보로 바꾼다.

⚠️ rolling(7)은 예측 대상일(td) 당일을 포함한다 — TB2·TB3와 같은 성격의
   "실측치 상한선" 실험이다. 실전 배포라면 미래 검색량은 알 수 없다.

입력: data/tierb/naver_trend.csv   (date, gonjiam, hwadam)
출력: data/tierb/naver_trend_mom.csv (date, gonjiam_mom, hwadam_mom)
"""
import os

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "data", "tierb", "naver_trend.csv")
OUT = os.path.join(BASE, "data", "tierb", "naver_trend_mom.csv")
COLS = ["gonjiam", "hwadam"]
WINDOW = 7


def main():
    df = pd.read_csv(SRC, parse_dates=["date"]).sort_values("date")
    for c in COLS:
        ma = df[c].rolling(WINDOW, min_periods=1).mean()
        df[f"{c}_mom"] = df[c] / ma.replace(0, 1)

    out_cols = ["date"] + [f"{c}_mom" for c in COLS]
    df[out_cols].to_csv(OUT, index=False, encoding="utf-8-sig")

    print(f"저장: {OUT}")
    print(f"  {len(df)}행 · 기간 {df['date'].min().date()} ~ {df['date'].max().date()}")
    print(df[["date"] + COLS + out_cols[1:]].tail())


if __name__ == "__main__":
    main()