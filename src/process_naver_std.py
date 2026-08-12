# -*- coding: utf-8 -*-
"""Tier B — TB4e: 화담숲 검색량 7일 변동성(std7)의 lag 20~35일 구간 평균.

EDA(eda_naver_residual_curve.py) 근거: 베이스 모델 잔차와 std7의 상관이 lag 16부터
서서히 올라 22~33 구간에서 고원(r 0.17~0.22), 34 이후 서서히 하강. 원본 수요 기준
EDA(eda_naver_lag.py)에서는 화담숲이 전 구간 무신호였던 것과 대비되는 지점.

입력: data/tierb/naver_trend.csv (date, gonjiam, hwadam)
출력: data/tierb/naver_std_lag.csv (date, hwadam_std_lag20_35)
"""
import os

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "data", "tierb", "naver_trend.csv")
OUT = os.path.join(BASE, "data", "tierb", "naver_std_lag.csv")
LAG_LO, LAG_HI = 20, 35


def main():
    df = pd.read_csv(SRC, parse_dates=["date"]).set_index("date").sort_index().asfreq("D")
    raw = df["hwadam"]
    std7 = raw.rolling(7, min_periods=2).std()
    avg = (std7.shift(LAG_LO)
                .rolling(LAG_HI - LAG_LO + 1, min_periods=1).mean()
                .fillna(0.0))
    out = avg.reset_index()
    out.columns = ["date", "hwadam_std_lag20_35"]
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"저장: {OUT} · {len(out)}행")
    print(out.tail())


if __name__ == "__main__":
    main()