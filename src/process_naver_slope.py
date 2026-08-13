# -*- coding: utf-8 -*-
"""Tier B — TB4d: 곤지암 검색량 7일 기울기(slope7)의 lag 7~29일 구간 평균.

EDA(eda_naver_residual.py → eda_naver_residual_curve.py) 근거: 베이스 모델 잔차와
slope7의 상관이 lag 1~29 구간에서 넓게 양수(r 0.14~0.28), 고립 봉우리 아님.
lag 하한을 7로 잡은 이유: HORIZON=7이라 h=1~7 중 어떤 예측이든 td-7 ≤ origin이
항상 성립해야 실전 배포 가능(미래 정보 누출 방지).

입력: data/tierb/naver_trend.csv (date, gonjiam, hwadam)
출력: data/tierb/naver_slope_lag.csv (date, gonjiam_slope_lag7_29)
"""
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "data", "tierb", "naver_trend.csv")
OUT = os.path.join(BASE, "data", "tierb", "naver_slope_lag.csv")
LAG_LO, LAG_HI = 7, 29


def _slope(w):
    return np.polyfit(range(len(w)), w, 1)[0]


def main():
    df = pd.read_csv(SRC, parse_dates=["date"]).set_index("date").sort_index().asfreq("D")
    raw = df["gonjiam"]
    slope7 = raw.rolling(7, min_periods=2).apply(_slope, raw=True)
    avg = (slope7.shift(LAG_LO)
                 .rolling(LAG_HI - LAG_LO + 1, min_periods=1).mean()
                 .fillna(0.0))
    out = avg.reset_index()
    out.columns = ["date", "gonjiam_slope_lag7_29"]
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"저장: {OUT} · {len(out)}행")
    print(out.tail())


if __name__ == "__main__":
    main()
