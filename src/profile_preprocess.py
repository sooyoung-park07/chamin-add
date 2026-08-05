# -*- coding: utf-8 -*-
"""Phase 2 사전 프로파일링 — 전처리 결정에 필요한 실태 파악.

무엇을 결정해야 하는가:
  ① 음수(환불) : 얼마나 있고, 어디에 몰려 있고, 채점에 얼마나 영향 주나
  ② 대형 스파이크 : 얼마나 극단적이고 몇 %인가
  ③ 휴점일 : 영업장별로 며칠이고, 품목의 0과 어떻게 구분되나
"""
import numpy as np
import pandas as pd

import config as C
import dataio as D

pd.set_option("display.width", 200)


def main():
    tr = D.load_train()
    items = D.item_order()
    mat, dates = D.to_matrix(tr, items)
    store_of = np.array([k.split("_")[0] for k in items])

    print("=" * 78)
    print("① 음수(환불) 실태")
    print("=" * 78)
    neg = tr[tr["qty"] < 0]
    print(f"  음수 행: {len(neg):,}개 / 전체 {len(tr):,} ({100*len(neg)/len(tr):.3f}%)")
    print(f"  값 범위: {neg['qty'].min():.0f} ~ {neg['qty'].max():.0f}, 중앙값 {neg['qty'].median():.0f}")
    print(f"  0이 아닌 행 중 음수 비중: {100*len(neg)/int((tr['qty']!=0).sum()):.3f}%")
    print("\n  영업장별 음수 행 수:")
    for s, n in neg["store"].value_counts().items():
        tot = int((tr[tr["store"] == s]["qty"] != 0).sum())
        print(f"    {s:<14s} {n:>4d}개  (그 영업장 유효행의 {100*n/tot:.2f}%)")
    print("\n  음수가 가장 많은 품목 5개:")
    for k, n in neg["key"].value_counts().head(5).items():
        print(f"    {k:<40s} {n}개")
    print("\n  → 판단 근거: 음수 행은 채점 대상(A≠0)이지만 모델은 양수만 예측 가능.")
    print("     이 행들은 사실상 최대벌점(2.0)을 먹는다. 비중이 작으면 무시해도 됨.")
    worst = 2.0 * len(neg) / int((tr["qty"] != 0).sum())
    print(f"     전부 최대벌점이라 가정한 점수 기여 상한: 약 {worst:.4f}")

    print()
    print("=" * 78)
    print("② 대형 스파이크 실태")
    print("=" * 78)
    pos = tr[tr["qty"] > 0]["qty"]
    print(f"  양수 행 {len(pos):,}개 · 중앙값 {pos.median():.0f} · 평균 {pos.mean():.1f} · 최대 {pos.max():.0f}")
    for q in [0.9, 0.95, 0.99, 0.995, 0.999]:
        print(f"    {q*100:>5.1f}분위 = {pos.quantile(q):>7.0f}")
    big = tr[tr["qty"] > 200]
    print(f"\n  200 초과 행: {len(big):,}개 ({100*len(big)/len(pos):.2f}% of 양수)")
    print("  200 초과가 많은 품목 5개:")
    for k, n in big["key"].value_counts().head(5).items():
        sub = tr[tr["key"] == k]["qty"]
        print(f"    {k:<40s} {n:>4d}일  (그 품목 중앙값 {sub[sub>0].median():.0f}, 최대 {sub.max():.0f})")
    print("\n  → 판단 근거: SMAPE는 비율오차라 큰 값의 절대오차가 곧바로 벌점이 되진 않음.")
    print("     다만 학습 시 소수 극단값이 트리 분할을 흔들 수 있어 캡핑이 도움될 수 있다.")

    print()
    print("=" * 78)
    print("③ 휴점일 실태 (영업장 합계 = 0 인 날)")
    print("=" * 78)
    day = tr.groupby(["store", "date"])["qty"].sum().reset_index()
    print(f"  {'영업장':<14s} {'휴점일':>6s} {'전체':>6s} {'비율':>7s}   비고")
    for s in sorted(day["store"].unique()):
        d = day[day["store"] == s]
        closed = int((d["qty"] == 0).sum())
        note = ""
        if closed > 0:
            months = sorted(d[d["qty"] == 0]["date"].dt.strftime("%Y-%m").unique())
            note = f"{months[0]}~{months[-1]} 등 {len(months)}개월"
        print(f"  {s:<14s} {closed:>6d} {len(d):>6d} {100*closed/len(d):>6.1f}%   {note}")

    print("\n  [품목의 0 = 휴점 때문인가, 그냥 안 팔린 건가]")
    tot = 0
    by_closed = 0
    for si, s in enumerate(sorted(day["store"].unique())):
        m = store_of == s
        sd = day[day["store"] == s].set_index("date")["qty"]
        closed_mask = np.array([sd.get(d, 0) == 0 for d in dates])
        sub = mat[m]
        zeros = (sub == 0)
        tot += zeros.sum()
        by_closed += zeros[:, closed_mask].sum()
    print(f"    전체 0 셀 {tot:,}개 중 휴점일에 속한 것 {by_closed:,}개 "
          f"({100*by_closed/tot:.1f}%)")
    print(f"    나머지 {100*(1-by_closed/tot):.1f}%는 '영업했지만 그 메뉴는 안 팔림'")
    print("\n  → 판단 근거: 휴점일 0과 영업일 0은 성격이 완전히 다르다.")
    print("     창 안에 휴점일이 섞여 있으면 w_mean/w_nzratio 같은 통계가 왜곡된다.")

    print()
    print("=" * 78)
    print("④ price 피처 점검 (범주화 검토용)")
    print("=" * 78)
    lab = D.load_labels()
    prices = np.array([D.label_of(lab, k, "price_krw", 0) or 0 for k in items])
    print(f"  고유 가격값 {len(set(prices.tolist()))}개 / 품목 {len(items)}개")
    print(f"  → 가격이 사실상 품목 식별자 역할을 하는지 확인 필요 (item_id와 중복)")
    print(f"  범위 {prices.min():,} ~ {prices.max():,}원, 중앙값 {np.median(prices):,.0f}원")
    edges = [0, 3000, 8000, 20000, 50000, 10**9]
    names = ["저가(~3천)", "중저가(~8천)", "중가(~2만)", "고가(~5만)", "초고가(5만~)"]
    band = np.digitize(prices, edges[1:-1])
    print("  5등급 범주화 시 분포:")
    for i, nm in enumerate(names):
        print(f"    {nm:<14s} {int((band == i).sum()):>4d}개")


if __name__ == "__main__":
    main()
