# -*- coding: utf-8 -*-
"""업장/군집별 잔차를 월별로 뜯어본다 — 새 네이버 검색어 후보를 어느 업장·계절에
겨냥해야 할지 정량적으로 잡기 위한 EDA. build_residual_baseline_store.py의 출력을 읽는다.
"""
import pandas as pd

CLUSTER = "experiments/tierb_residual_cluster.csv"
STORE = "experiments/tierb_residual_store.csv"


def show(path, unit):
    df = pd.read_csv(path, parse_dates=["td"])
    df["month"] = df["td"].dt.month
    print("=" * 70)
    print(f"{unit}별 월별 resid_ratio(=actual/pred, <1=과대예측 · >1=과소예측) 중앙값")
    print("=" * 70)
    piv = df.pivot_table(index=unit, columns="month", values="resid_ratio", aggfunc="median")
    piv = piv.reindex(columns=sorted(piv.columns))
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(piv.round(2))

    print(f"\n{unit}별 전체 표본수(=검증기간 내 origin×horizon 관측 일수)")
    print(df.groupby(unit).size())

    print(f"\n{unit}별 잔차 절대합 상위 (어디서 크게 틀렸나, 부호 포함 평균 resid_ratio)")
    g = df.groupby(unit).agg(n=("residual", "size"),
                              resid_ratio_median=("resid_ratio", "median"),
                              resid_abs_mean=("residual", lambda s: s.abs().mean()))
    print(g.sort_values("resid_abs_mean", ascending=False).round(3))


if __name__ == "__main__":
    show(CLUSTER, "cluster")
    print("\n\n")
    show(STORE, "store")
