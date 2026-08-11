# -*- coding: utf-8 -*-
"""EDA — 네이버 검색 모멘텀과 리조트 전체 수요 간 시차(lag) 상관관계 탐색.

TB4/TB4b는 "어느 날짜의 검색량을 쓸지"를 근거 없이 골랐다(td 당일 또는 origin 당일).
여기서는 검색일과 수요일 사이 간격(lag)을 1~35일로 바꿔가며 상관계수를 재서,
실제로 상관이 센 지점이 어딘지 데이터로 확인한다.
"""
import numpy as np
import pandas as pd

import dataio as D
import features as F

LAGS = range(1, 36)


def main():
    tr = D.load_train()
    mat, dates = D.to_matrix(tr, D.item_order())
    demand = pd.Series(mat.sum(0), index=pd.DatetimeIndex(dates))  # 리조트 전체 일별 수요

    naver = F._load_tierb_csv("naver_trend_mom", ["gonjiam_mom", "hwadam_mom"])
    naver_df = pd.DataFrame.from_dict(naver, orient="index",
                                       columns=["gonjiam_mom", "hwadam_mom"])
    naver_df.index = pd.DatetimeIndex(naver_df.index)
    naver_df = naver_df.sort_index().asfreq("D")

    print(f"{'lag':>4} | {'gonjiam_mom':>12} | {'hwadam_mom':>12}")
    print("-" * 36)
    rows = []
    for lag in LAGS:
        shifted = naver_df.shift(lag)
        cg = demand.corr(shifted["gonjiam_mom"])
        ch = demand.corr(shifted["hwadam_mom"])
        rows.append((lag, cg, ch))
        print(f"{lag:>4} | {cg:>12.4f} | {ch:>12.4f}")

    best_g = max(rows, key=lambda r: abs(r[1]))
    best_h = max(rows, key=lambda r: abs(r[2]))
    print(f"\n최고 상관 — 곤지암: lag={best_g[0]}일 전, r={best_g[1]:.4f}")
    print(f"최고 상관 — 화담숲: lag={best_h[0]}일 전, r={best_h[1]:.4f}")


if __name__ == "__main__":
    main()