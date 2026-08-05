# -*- coding: utf-8 -*-
"""Phase 11-c — 왜곡 교정 피처를 **만들기 전에 전제부터 잰다** (규칙 ⑤).

두 후보의 전제
  (가) 창 안 공휴일 오염 : 28일 창에 공휴일이 끼면 창 통계(w_mean 등)가 왜곡되는데
       모델은 그걸 "수요 변화"로 오해한다. → Phase 2 휴점보정(+0.0036)과 **같은 족보**
  (나) 영업일 기준 store 집계 : `x_store_*` 5개가 휴점일을 포함해 계산된다
       (화담숲은 46%가 휴점) → 창 절반이 휴점이면 평균이 반토막

재기 전에 죽일 수 있으면 죽인다. 물어야 할 것:
  ① 공휴일이 창에 얼마나 자주 끼는가? 드물면 볼 것도 없다
  ② 공휴일에 실제로 매출이 다른가? 안 다르면 오염 자체가 없다
  ③ 그 차이가 창 통계를 얼마나 흔드는가?
  ④ store 집계에서 휴점일이 차지하는 비중은?

**Phase 2 가 성공한 조건과 비교해야 한다** — 그때는 화담숲 46%·포레스트릿 28%가 휴점이라
창의 절반이 죽어 있었다. 그 정도 크기가 아니면 기대를 낮춰야 한다.
"""
import os

import numpy as np
import pandas as pd

import config as C
import dataio as D
import features as F


def main():
    ctx = F.Context()
    tr = D.load_train()
    mat, dates = D.to_matrix(tr, ctx.items)
    dates = pd.DatetimeIndex(dates)
    nd = mat.shape[1]

    hol = np.array([F.is_holiday(d) if hasattr(F, "is_holiday")
                    else (d.normalize() in set(pd.to_datetime(C.HOLIDAYS)))
                    for d in dates])
    print("=" * 96)
    print("Phase 11-c — 왜곡 교정 후보의 전제 측정")
    print("=" * 96)
    print(f"  학습 구간 {dates[0].date()} ~ {dates[-1].date()} ({nd}일) · "
          f"품목 {ctx.n}개")

    # ── ① 공휴일이 창에 얼마나 끼는가 ────────────────────────────────
    print("\n[1] 28일 창 안의 공휴일 개수 분포")
    cnt = np.array([hol[o - C.WINDOW + 1:o + 1].sum()
                    for o in range(C.WINDOW - 1, nd - C.HORIZON)])
    print(f"  전체 공휴일 {hol.sum()}일 / {nd}일 = {100*hol.mean():.1f}%")
    for k in range(int(cnt.max()) + 1):
        f = (cnt == k).mean()
        if f > 0:
            print(f"    창 안 공휴일 {k}일 : {100*f:5.1f}%  {'█'*int(f*50)}")
    print(f"  → 창 하나당 평균 {cnt.mean():.2f}일 (28일 중 {100*cnt.mean()/28:.1f}%)")

    # ── ② 공휴일에 매출이 실제로 다른가 ──────────────────────────────
    print("\n[2] 공휴일 vs 평일/주말 — 실제로 다른가 (양수 매출만, 품목별 정규화)")
    dow = dates.dayofweek.values
    grp = np.where(hol, "공휴일", np.where(dow >= 5, "주말", "평일"))
    rows = []
    for i in range(ctx.n):
        v = mat[i]
        m = v > 0
        if m.sum() < 30:
            continue
        base = np.median(v[m])
        for g in ("평일", "주말", "공휴일"):
            s = m & (grp == g)
            if s.sum() >= 3:
                rows.append((ctx.store_of_item[i], g, np.median(v[s]) / base))
    df = pd.DataFrame(rows, columns=["store", "grp", "ratio"])
    piv = df.groupby("grp")["ratio"].agg(["median", "count"])
    print(f"  {'구분':>8s}{'중앙 배율':>10s}{'표본':>8s}")
    for g in ("평일", "주말", "공휴일"):
        if g in piv.index:
            print(f"  {g:>8s}{piv.loc[g,'median']:>10.3f}{int(piv.loc[g,'count']):>8d}")
    print("\n  영업장별 공휴일 배율:")
    hb = df[df.grp == "공휴일"].groupby("store")["ratio"].median().sort_values()
    for s, v in hb.items():
        print(f"    {s:<16s} {v:.3f}")

    # ── ③ 창 통계가 얼마나 흔들리는가 ────────────────────────────────
    print("\n[3] 공휴일을 빼면 창 평균이 얼마나 달라지는가 (w_mean 기준)")
    diffs = []
    for o in range(C.WINDOW - 1, nd - C.HORIZON, 7):
        w = slice(o - C.WINDOW + 1, o + 1)
        hw = hol[w]
        if hw.sum() == 0:
            continue
        sub = mat[:, w]
        a = np.nanmean(sub, 1)
        b = np.nanmean(sub[:, ~hw], 1)
        ok = (a > 0) & np.isfinite(a) & np.isfinite(b)
        if ok.sum():
            diffs.append(np.abs(b[ok] - a[ok]) / a[ok])
    if diffs:
        d = np.concatenate(diffs)
        print(f"  공휴일 포함 창 {len(diffs)}개 · 품목-창 쌍 {len(d):,}")
        for q in (50, 75, 90, 99):
            print(f"    {q}분위 상대변화 {100*np.percentile(d,q):5.2f}%")
        print(f"    평균 {100*d.mean():.2f}%")
    print("  ※ 대조: Phase 2 휴점보정은 창의 **절반이 죽는** 경우를 고쳐 +0.0036 이었다.")

    # ── ④ store 집계의 휴점일 비중 ───────────────────────────────────
    print("\n[4] x_store_* 집계에 휴점일이 섞이는 정도 (영업장별)")
    print(f"  {'영업장':<16s}{'품목수':>7s}{'전일0 비율':>12s}{'창평균 왜곡':>12s}")
    for s in sorted(set(ctx.store_of_item.tolist())):
        idx = np.where(ctx.store_of_item == s)[0]
        sub = mat[idx]
        closed = (np.nansum(np.abs(sub), 0) == 0)          # 그 날 업장 전체가 0
        tot = np.nansum(sub, 0)
        openm = ~closed
        dist = (tot.mean() / tot[openm].mean()) if openm.sum() else np.nan
        print(f"  {s:<16s}{len(idx):>7d}{100*closed.mean():>11.1f}%"
              f"{100*(1-dist):>11.1f}%")
    print("\n  → '창평균 왜곡' = 휴점일을 포함해 평균내면 실제 영업일 평균 대비 몇 % 낮게 나오는가.")
    print("     이게 크면 (나) 영업일 기준 집계가 값어치 있다.")


if __name__ == "__main__":
    main()
