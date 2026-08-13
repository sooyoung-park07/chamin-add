# -*- coding: utf-8 -*-
"""Tier B — TB4c: 네이버 검색 모멘텀의 18~33일 전 구간 평균.

EDA(eda_naver_lag.py) 결과: 곤지암 검색 모멘텀은 lag 18~33일 구간에서 상관이 완만하게
이어짐(고립 봉우리 아님, r 0.05~0.19). 화담숲은 전 구간 상관 없어 제외.

입력: data/tierb/naver_trend_mom.csv (date, gonjiam_mom, hwadam_mom)
출력: data/tierb/naver_trend_lag.csv (date, gonjiam_lag18_33)
"""
import os

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "data", "tierb", "naver_trend_mom.csv")
OUT = os.path.join(BASE, "data", "tierb", "naver_trend_lag.csv")
LAG_LO, LAG_HI = 18, 33  # EDA에서 확인된 넓은 상관 구간


def main():
    df = pd.read_csv(SRC, parse_dates=["date"]).sort_values("date")
    s = df.set_index("date")["gonjiam_mom"].asfreq("D")
    # d 기준 (d-33)~(d-18) 16일 평균 = shift(18) 후 16일 rolling
    lag_avg = s.shift(LAG_LO).rolling(LAG_HI - LAG_LO + 1, min_periods=1).mean().fillna(0.0)
    out = lag_avg.reset_index()
    out.columns = ["date", "gonjiam_lag18_33"]
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"저장: {OUT} · {len(out)}행")
    print(out.tail())


if __name__ == "__main__":
    main()