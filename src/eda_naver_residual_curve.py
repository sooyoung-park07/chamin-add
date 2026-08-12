# -*- coding: utf-8 -*-
"""후보로 좁힌 변형들의 lag 1~50 전체 곡선 출력 — 고립 봉우리 vs 고원 판정용."""
import numpy as np
import pandas as pd

RESID = "experiments/tierb_residual_daily.csv"
RAW = "data/tierb/naver_trend.csv"
LAGS = range(1, 51)

TARGETS = [("곤지암", "gonjiam", "level"), ("곤지암", "gonjiam", "slope7"),
           ("곤지암", "gonjiam", "mom7"), ("화담숲", "hwadam", "std7")]


def make_variant(raw, vname):
    if vname == "level":
        return raw
    if vname == "mom7":
        return raw / raw.rolling(7, min_periods=1).mean().replace(0, 1)
    if vname == "slope7":
        return raw.rolling(7, min_periods=2).apply(
            lambda w: np.polyfit(range(len(w)), w, 1)[0], raw=True)
    if vname == "std7":
        return raw.rolling(7, min_periods=2).std()


def main():
    resid = pd.read_csv(RESID, parse_dates=["td"]).set_index("td")
    df = pd.read_csv(RAW, parse_dates=["date"]).set_index("date").sort_index().asfreq("D")

    for label, col, vname in TARGETS:
        series = make_variant(df[col], vname)
        print(f"\n{label} · {vname}")
        for lag in LAGS:
            c = resid["residual"].corr(series.shift(lag).reindex(resid.index))
            bar = "#" * int(abs(c) * 40)
            print(f"  lag={lag:>3} r={c:+.3f} {bar}")


if __name__ == "__main__":
    main()