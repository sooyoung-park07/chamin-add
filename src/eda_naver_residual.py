# -*- coding: utf-8 -*-
"""EDA — 네이버 검색량의 여러 변형 × td 기준 정확한 lag 정렬 × 베이스 모델 잔차 상관.

이전 EDA(eda_naver_lag.py)의 두 가지 한계를 고친다:
  1. 원본 수요가 아니라 build_residual_baseline.py의 잔차와 비교 (재인코딩 여부 검증)
  2. origin이 아니라 td(=origin+h) 기준으로 lag를 맞춤 (h별로 값이 달라짐, TB4c의 정렬 오류 수정)

원본 ratio(naver_trend.csv)부터 다시 읽어 레벨·모멘텀·변화율·추세·변동성 5가지 변형을 만든다.
"""
import numpy as np
import pandas as pd

LAGS = range(1, 36)
RESID = "experiments/tierb_residual_daily.csv"
RAW = "data/tierb/naver_trend.csv"


def variants(raw):
    """raw: date 인덱스 Series(0~100 ratio). 여러 변형을 dict로 반환."""
    return {
        "level": raw,
        "mom7": raw / raw.rolling(7, min_periods=1).mean().replace(0, 1),
        "diff1": raw.diff(1),
        "slope7": raw.rolling(7, min_periods=2).apply(
            lambda w: np.polyfit(range(len(w)), w, 1)[0], raw=True),
        "std7": raw.rolling(7, min_periods=2).std(),
    }


def main():
    resid = pd.read_csv(RESID, parse_dates=["td"]).set_index("td")
    df = pd.read_csv(RAW, parse_dates=["date"]).set_index("date").sort_index().asfreq("D")

    for label, col in [("곤지암", "gonjiam"), ("화담숲", "hwadam")]:
        print(f"\n{'='*60}\n{label}\n{'='*60}")
        raw = df[col]
        for vname, series in variants(raw).items():
            best = (None, 0.0)
            for lag in LAGS:
                shifted = series.shift(lag)
                aligned = shifted.reindex(resid.index)
                c = resid["residual"].corr(aligned)
                if not np.isnan(c) and abs(c) > abs(best[1]):
                    best = (lag, c)
            print(f"  {vname:>8s} | 최고 lag={best[0]:>3} | r={best[1]:+.4f}")


if __name__ == "__main__":
    main()