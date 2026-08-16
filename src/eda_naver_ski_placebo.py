# -*- coding: utf-8 -*-
"""TB7 설계 근거 보강 — eda_naver2_placebo.py(TB8용)와 동일한 방법(무작위 순열×5변형×
lag1~35 스캔)을 "곤지암 스키장" vs ski 업장군 잔차에 적용. 최초 TB7 설계(origin 기준
최근 7일 평균)는 이 스캔 없이 TB2c의 설계 패턴만 빌려 정했었는데, 사용자 질문
("언제 전 검색량이 가장 영향을 주는지 EDA로 확인했나")에 답하기 위해 뒤늦게 실행.

결과 요약(2026-08-15, 척도 수정된 naver_trend3_solo.csv 기준): level 변형이 lag=1일에
r=+0.5157로 최댓값, 이후 lag이 늘수록 매끄럽게 단조 감소해 lag 30+에서는 거의 0 —
화담숲 단풍(고원형)과 달리 **뾰족하게 최근일에 몰린 감쇠 곡선**. 플라시보(무작위 순열
20회) 대비 백분위 100%로 우연 범위 밖. 시사점: origin 기준 짧은 창(예: 3일)이 7일
평균보다 신호를 덜 희석시킬 가능성 — TB7-corrected 결과가 약하면 다음 후보로 검토.
"""
import numpy as np
import pandas as pd

LAGS = range(1, 36)
RAW = "data/tierb/naver_trend3_solo.csv"
CLUSTER_RESID = "experiments/tierb_residual_cluster.csv"
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


def best_over_scan(series, resid, return_all=False):
    best = 0.0
    rows = []
    for vname, v in variants(series).items():
        for lag in LAGS:
            c = resid.corr(v.shift(lag).reindex(resid.index))
            if not np.isnan(c):
                rows.append((vname, lag, c))
                if abs(c) > abs(best):
                    best = c
    return (best, rows) if return_all else best


def main():
    cres = pd.read_csv(CLUSTER_RESID, parse_dates=["td"])
    resid = cres[cres["cluster"] == "ski"].set_index("td")["residual"].sort_index()
    df = pd.read_csv(RAW, parse_dates=["date"]).set_index("date").sort_index().asfreq("D")
    raw = df["ski_field"].fillna(0.0)

    obs, rows = best_over_scan(raw, resid, return_all=True)
    print(f"관측 - ski_field vs ski잔차, 5변형×lag1~35 스캔 최대: r={obs:+.4f}")
    print("\nlevel 변형, lag 1~35 곡선:")
    for l, c in sorted((l, c) for vn, l, c in rows if vn == "level"):
        print(f"  lag={l:>2} r={c:+.4f}")

    rng = np.random.default_rng(0)
    maxrs = np.array([abs(best_over_scan(
        pd.Series(rng.permutation(raw.values), index=raw.index), resid))
        for _ in range(N_SHUFFLE)])
    pctile = (maxrs < abs(obs)).mean() * 100
    print(f"\n플라시보 {N_SHUFFLE}회 - 평균 {maxrs.mean():.4f} · 95th "
          f"{np.percentile(maxrs, 95):.4f} · 최댓값 {maxrs.max():.4f}")
    print(f"관측치 |r|={abs(obs):.4f} 백분위: {pctile:.0f}%")


if __name__ == "__main__":
    main()
