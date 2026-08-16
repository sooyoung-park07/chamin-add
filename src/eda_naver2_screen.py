# -*- coding: utf-8 -*-
"""TB7 신규 검색어 후보 저비용 스크리닝.

TB4d의 교훈(규칙: lag 1~35 전체를 스캔해 최댓값만 고르면 다중비교 함정에 걸린다)을
피하기 위해, 여기서는 사전에 도메인 근거로 정한 소수의 lag(0=당일 상한선, 7, 14일)만
확인하고, 값을 무작위 순열(shuffle) 분포와 비교해 "우연히 나올 수 있는 크기인가"를
같이 본다 — 이 스크린을 통과한 후보만 4폴드 검증으로 넘긴다.

td(예측 대상일) 기준으로 정렬한다(TB4c가 origin 기준으로 잘못 정렬했던 실수를 반복하지
않기 위해 build_residual_baseline_store.py의 td 인덱스를 그대로 쓴다).
"""
import numpy as np
import pandas as pd

RAW = "data/tierb/naver_trend2.csv"
CLUSTER_RESID = "experiments/tierb_residual_cluster.csv"
LAGS = (0, 7, 14)
N_PLACEBO = 200
RNG = np.random.default_rng(42)

PAIRS = [
    ("hwadam_foliage", "hwadam"),
    ("ski_field", "ski"),
]


def corr_at_lag(series, resid, lag):
    shifted = series.shift(lag).reindex(resid.index)
    return resid.corr(shifted)


def main():
    raw = pd.read_csv(RAW, parse_dates=["date"]).set_index("date").sort_index().asfreq("D")
    cres = pd.read_csv(CLUSTER_RESID, parse_dates=["td"])

    for kw, cluster in PAIRS:
        sub = cres[cres["cluster"] == cluster].set_index("td")["residual"].sort_index()
        series = raw[kw]
        print("=" * 60)
        print(f"{kw}  ↔  {cluster} 군집 잔차 (n={len(sub)}일)")
        print("=" * 60)
        for lag in LAGS:
            r = corr_at_lag(series, sub, lag)
            # 플라시보: 같은 시리즈를 무작위로 굴려(circular shift) 우연 분포 추정
            placebo = []
            vals = series.reindex(sub.index.union(series.index)).values
            n = len(series)
            for _ in range(N_PLACEBO):
                shift = RNG.integers(30, n - 30)
                rolled = pd.Series(np.roll(series.values, shift), index=series.index)
                placebo.append(corr_at_lag(rolled, sub, lag))
            placebo = np.array([p for p in placebo if not np.isnan(p)])
            pctile = (np.abs(placebo) < abs(r)).mean() * 100 if len(placebo) else float("nan")
            print(f"  lag={lag:>2}일 | r={r:+.4f} | 플라시보 |r| 평균={np.abs(placebo).mean():.4f} "
                  f"· 95th={np.percentile(np.abs(placebo), 95):.4f} | 관측치 백분위={pctile:.0f}%")
        print()


if __name__ == "__main__":
    main()
