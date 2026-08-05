# -*- coding: utf-8 -*-
"""Phase 11-a — 피처 57개의 **중복 구조**를 본다. 학습도 제출도 없다.

목적 (사용자 제안 (1) "같은 정보를 담거나 쓸모없어 보이는 피처 제거")
    제거 후보를 **점수로** 고르면 선택 편향에 걸린다(9-e 에서 실력차 0인 설계 11개 중
    최댓값이 자동으로 +1.6σ 나왔다). 그래서 먼저 **점수를 보지 않고** 구조만 본다:
      ① 완전 동일 열      — 논쟁의 여지가 없다. 그냥 잉여
      ② 상관 0.99+ 짝     — 사실상 같은 열
      ③ 상수/거의 상수 열 — 정보 0
      ④ 결측 많은 열
    여기서 나온 것만 나중에 사다리에 올린다.

⚠️ permutation importance 를 쓰지 않는 이유
    `w_mean·w_posmean·w_median·w_last7·w_last14` 처럼 강상관 열이 뭉쳐 있으면
    하나만 섞어도 나머지가 대신해서 **전부 "안 중요"로 나온다.** 쓰려면 상관 그룹 단위로
    함께 섞어야 하는데, 그 그룹을 정하는 게 바로 이 스크립트가 하는 일이다.

⚠️ gain 순위로 자르지 않는 이유 — 규칙 ⑥. `item_id` 는 gain 40.9% 1위인데 저데이터에선 해롭고,
    9-d 는 레벨↑ → gain↑ → 성적↓ 였다. **gain 은 '암기 통로의 크기'를 재기도 한다.**
"""
import os

import numpy as np
import pandas as pd

import config as C
import dataio as D
import features as F

# 20시드 학습이 코어를 다 쓰고 있을 수 있으므로 가볍게 돈다
ORIGIN_STRIDE = 7


def main():
    ctx = F.Context()
    tr = D.load_train()
    mat, dates = D.to_matrix(tr, ctx.items)
    nd = mat.shape[1]
    ctx.set_proxy(F.pick_proxy_items(mat, dates, nd, ctx.store_codes))

    origins = list(range(C.WINDOW - 1, nd - C.HORIZON, ORIGIN_STRIDE))
    X, y, meta = F.build_samples(mat, dates, origins, ctx)
    keep, names = F.active_columns(), F.active_names()
    X = X[:, keep]
    print("=" * 96)
    print(f"Phase 11-a — 피처 중복 구조 (표본 {X.shape[0]:,}행 × {X.shape[1]}열, "
          f"origin {ORIGIN_STRIDE}일 간격)")
    print("=" * 96)

    # ── ① 완전 동일 열 ────────────────────────────────────────────────
    print("\n[1] 완전히 같은 열 (비트 단위)")
    dup = []
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            ca, cb = X[:, a], X[:, b]
            m = ~(np.isnan(ca) | np.isnan(cb))
            if m.sum() and np.array_equal(ca[m], cb[m]) and \
               np.array_equal(np.isnan(ca), np.isnan(cb)):
                dup.append((names[a], names[b]))
    if dup:
        for a, b in dup:
            print(f"  ⚠️ {a}  ==  {b}   ← 하나는 순수 잉여")
    else:
        print("  없음")

    # ── ② 상관 0.99+ ─────────────────────────────────────────────────
    print("\n[2] 상관 |r| >= 0.99 인 짝 (사실상 같은 열)")
    Xf = np.where(np.isnan(X), np.nan, X)
    with np.errstate(invalid="ignore"):
        R = pd.DataFrame(Xf, columns=names).corr().values
    hi = []
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            r = R[a, b]
            if np.isfinite(r) and abs(r) >= 0.99:
                hi.append((abs(r), names[a], names[b]))
    for r, a, b in sorted(hi, reverse=True):
        print(f"  r={r:.5f}  {a:<20s} ~ {b}")
    if not hi:
        print("  없음")

    print("\n[2b] 상관 0.95~0.99 (중복 의심)")
    mid = [(abs(R[a, b]), names[a], names[b])
           for a in range(len(names)) for b in range(a + 1, len(names))
           if np.isfinite(R[a, b]) and 0.95 <= abs(R[a, b]) < 0.99]
    for r, a, b in sorted(mid, reverse=True)[:15]:
        print(f"  r={r:.4f}  {a:<20s} ~ {b}")
    print(f"  (총 {len(mid)}쌍)")

    # ── ③ 정보량이 없는 열 ────────────────────────────────────────────
    print("\n[3] 거의 상수인 열 / 결측 많은 열")
    for i, n in enumerate(names):
        c = X[:, i]
        nan = float(np.isnan(c).mean())
        v = c[~np.isnan(c)]
        uniq = len(np.unique(v)) if len(v) else 0
        top = float(pd.Series(v).value_counts(normalize=True).iloc[0]) if len(v) else 1.0
        if uniq <= 2 or top > 0.97 or nan > 0.3:
            print(f"  {n:<22s} 고유값 {uniq:>6d} · 최빈값비중 {top:6.1%} · 결측 {nan:6.1%}")

    # ── ④ 상관 그룹 (덩어리 단위로 자를 후보) ─────────────────────────
    print("\n[4] 상관 0.9 이상으로 이어지는 덩어리 (permutation 을 쓴다면 이 단위로)")
    n = len(names)
    par = list(range(n))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x

    for a in range(n):
        for b in range(a + 1, n):
            if np.isfinite(R[a, b]) and abs(R[a, b]) >= 0.90:
                par[find(a)] = find(b)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(names[i])
    for g in sorted(groups.values(), key=len, reverse=True):
        if len(g) > 1:
            print(f"  ({len(g)}) " + " · ".join(g))

    # ── ⑤ 절대시간 vs 상대시간 (자를 후보의 '종류') ────────────────────
    print("\n[5] cal 13개를 '낡는 열 / 안 낡는 열' 로 가른다")
    ABS = ["month", "doy_sin", "doy_cos", "day"]
    REL = ["dow", "is_weekend", "is_holiday", "is_holiday_eve", "is_dayoff"]
    DOM = ["hwadam_open", "ski_season", "ski_peak", "foliage"]
    print("  절대 시간(학습기간에 맞춰 외울 수 있음 → 낡음) :",
          [x for x in ABS if x in names])
    print("  상대 시간(창과 무관하게 항상 유효 → 안 낡음)   :",
          [x for x in REL if x in names])
    print("  도메인 전환점(창이 미리 알 수 없는 정보)        :",
          [x for x in DOM if x in names])
    print("  그 외 cal :", [x for x in F.CAL_KEYS if x not in ABS + REL + DOM])
    print("\n  → cal 은 지금까지 **통째로만** 잘라봤다(+0.0173 → 유지 확정).")
    print("     그 안에서 절대/상대를 가른 적이 없다. 미측정 명제.")


if __name__ == "__main__":
    main()
