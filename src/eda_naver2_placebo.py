# -*- coding: utf-8 -*-
"""TB8 선행조건 — "화담숲 단풍"이 eda_naver2_screen.py의 순환이동 플라시보를 명확히
못 넘었으므로(플라시보 68~76%ile), TB4d가 썼던 정식 플라시보 방법(eda_naver_placebo.py —
전체 무작위 순열 × 5변형 × lag 1~35 스캔)을 hwadam 업장군 잔차 기준으로 그대로 재현한다.

여기서도 관측치가 우연 분포 안에 들어가면 TB8은 4폴드 검증 없이 폐기한다
(experiments/tb7_naver_keyword_brief.md 2순위 조건).
"""
import numpy as np
import pandas as pd

LAGS = range(1, 36)
RAW = "data/tierb/naver_trend2.csv"
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


def best_over_scan(series, resid):
    best = 0.0
    for v in variants(series).values():
        for lag in LAGS:
            c = resid.corr(v.shift(lag).reindex(resid.index))
            if not np.isnan(c) and abs(c) > abs(best):
                best = c
    return best


def main():
    cres = pd.read_csv(CLUSTER_RESID, parse_dates=["td"])
    resid = cres[cres["cluster"] == "hwadam"].set_index("td")["residual"].sort_index()
    df = pd.read_csv(RAW, parse_dates=["date"]).set_index("date").sort_index().asfreq("D")
    raw = df["hwadam_foliage"].fillna(0.0)

    obs = best_over_scan(raw, resid)
    print(f"관측 — 실제 '화담숲 단풍' 데이터, 5변형×lag1~35 스캔 최대: r={obs:+.4f}")

    rng = np.random.default_rng(0)
    maxrs = []
    for i in range(N_SHUFFLE):
        shuffled = pd.Series(rng.permutation(raw.values), index=raw.index)
        b = best_over_scan(shuffled, resid)
        maxrs.append(abs(b))
        print(f"  섞기 {i+1}/{N_SHUFFLE}: 이번 회차 최대 |r| = {abs(b):.4f}")

    maxrs = np.array(maxrs)
    pctile = (maxrs < abs(obs)).mean() * 100
    print(f"\n무작위(가짜) 데이터 최대 |r| — 평균 {maxrs.mean():.4f} · 중앙값 "
          f"{np.median(maxrs):.4f} · 95th {np.percentile(maxrs, 95):.4f} · 최댓값 {maxrs.max():.4f}")
    print(f"관측치 |r|={abs(obs):.4f} 의 플라시보 분포 내 백분위: {pctile:.0f}%")
    print("→ 95% 이상이어야 '우연이 아니다'로 판단 (TB4d 관례).")


if __name__ == "__main__":
    main()
