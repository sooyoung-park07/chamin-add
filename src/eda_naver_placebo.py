# -*- coding: utf-8 -*-
"""EDA 방법론 자체의 다중비교 위험도 측정 — 검색량을 무작위로 섞은 가짜 데이터로
같은 스캔(5변형×35lag)을 반복해서, 우연히 나올 수 있는 최대 상관값의 분포를 본다.
관측한 TB4d(0.28)·TB4e(0.22) 후보가 이 분포 안에 흔하게 들어가면 우연일 가능성이 크다.
"""
import numpy as np
import pandas as pd

LAGS = range(1, 36)
RESID = "experiments/tierb_residual_daily.csv"
RAW = "data/tierb/naver_trend.csv"
N_SHUFFLE = 20


def variants(raw):
    return {
        "level": raw,
        "mom7": raw / raw.rolling(7, min_periods=1).mean().replace(0, 1),
        "diff1": raw.diff(1),
        "slope7": raw.rolling(7, min_periods=2).apply(
            lambda w: np.polyfit(range(len(w)), w, 1)[0], raw=True),
        "std7": raw.rolling(7, min_periods=2).std(),
    }


def main():
    resid = pd.read_csv(RESID, parse_dates=["td"]).set_index("td")["residual"]
    df = pd.read_csv(RAW, parse_dates=["date"]).set_index("date").sort_index().asfreq("D")

    rng = np.random.default_rng(0)
    maxrs = []
    for i in range(N_SHUFFLE):
        shuffled = pd.Series(rng.permutation(df["gonjiam"].ffill().bfill().values),
                              index=df.index)
        best = 0.0
        for series in variants(shuffled).values():
            for lag in LAGS:
                c = resid.corr(series.shift(lag).reindex(resid.index))
                if not np.isnan(c) and abs(c) > abs(best):
                    best = c
        maxrs.append(abs(best))
        print(f"  섞기 {i+1}/{N_SHUFFLE}: 이번 회차 최대 |r| = {abs(best):.4f}")

    maxrs = np.array(maxrs)
    print(f"\n무작위(가짜) 데이터에서의 최대 |r| — 평균 {maxrs.mean():.4f} · "
          f"중앙값 {np.median(maxrs):.4f} · 최댓값 {maxrs.max():.4f}")
    print(f"참고: 관측된 TB4d(slope7) r=0.279, TB4e(std7) r=0.222")


if __name__ == "__main__":
    main()