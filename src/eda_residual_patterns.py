# -*- coding: utf-8 -*-
"""베이스 모델(v24) 잔차를 깊이 파본다 — 요일/월별 패턴, 가장 크게 틀린 날짜들,
잔차 자체의 자기상관(어제 오차가 오늘 오차를 예측하는가). 여기서 나온 패턴이
어떤 외부 정보가 필요한지에 대한 직접적인 단서가 된다.
"""
import numpy as np
import pandas as pd

RESID = "experiments/tierb_residual_daily.csv"


def main():
    df = pd.read_csv(RESID, parse_dates=["td"]).sort_values("td")
    df["dow"] = df["td"].dt.day_name()
    df["month"] = df["td"].dt.month

    print("=" * 60)
    print("1. 요일별 잔차 패턴 (양수=과소예측, 음수=과대예측)")
    print("=" * 60)
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                 "Saturday", "Sunday"]
    g = df.groupby("dow")["residual"].agg(["mean", "std", "count"]).reindex(dow_order)
    print(g.round(1))

    print("\n" + "=" * 60)
    print("2. 월별 잔차 패턴")
    print("=" * 60)
    g2 = df.groupby("month")["residual"].agg(["mean", "std", "count"])
    print(g2.round(1))

    print("\n" + "=" * 60)
    print("3. 가장 크게 틀린 날짜 TOP 15 (절대값 기준)")
    print("=" * 60)
    top = df.reindex(df["residual"].abs().sort_values(ascending=False).index).head(15)
    print(top[["td", "dow", "actual", "pred", "residual"]].to_string(index=False))

    print("\n" + "=" * 60)
    print("4. 잔차 자체의 자기상관 (어제 오차가 오늘 오차를 설명하는가)")
    print("=" * 60)
    s = df.set_index("td")["residual"].asfreq("D")
    for lag in range(1, 8):
        c = s.corr(s.shift(lag))
        print(f"  lag={lag}: r={c:+.3f}")


if __name__ == "__main__":
    main()