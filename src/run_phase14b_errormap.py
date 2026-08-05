# -*- coding: utf-8 -*-
"""Phase 14-b — 내부 오답 현황 종합 (여러 축). 학습 0회.

축마다 세 값을 잰다:
    기여   = 그 버킷이 점수에서 차지하는 몫 (가중 SMAPE 합, %)
    편향   = 가중 평균 ln(P/A). 음수 = 과소예측, 양수 = 과대예측
    배율여지 = 그 버킷만 사후 최적 배율을 먹였을 때 이득 (수준 오차의 상한)
'많이 틀림'과 '고칠 수 있음'을 가르기 위해 품목 축은 정확일치율·item-week 오라클 여지를 병기.
"""
import os
import numpy as np
import pandas as pd

import config as C
import dataio as D
import features as F
import validate as V
from run_phase10c_thresholds import cell_weights

FOLDS = [("F2 겨울", "2023-11-24", "2023-11-24", "2024-02-22", 0),
         ("F3 봄", "2024-02-23", "2024-02-23", "2024-06-08", 1)]


def seg_base(r):
    p = np.where(r < 1.8, 0.55 * r, np.where(r < 10.0, 0.90 * r, 1.02 * r))
    return np.maximum(p, 1.0)


def gsnap(p):
    k = np.maximum(np.floor(p), 1.0)
    return np.maximum(np.where(p >= np.sqrt(k * (k + 1.0)), k + 1.0, k), 1.0)


def loss(a, p):
    a = np.abs(a); p = np.abs(p)
    den = a + p
    out = np.zeros(len(a)); m = den > 0
    out[m] = 2.0 * np.abs(a[m] - p[m]) / den[m]
    return out


MS = np.round(np.arange(0.70, 1.41, 0.01), 2)


def bucket_stats(f, mask):
    """(기여%, 편향, 최적배율, 배율이득)"""
    m = mask & f["valid"]
    if m.sum() < 30:
        return None
    contrib = float(f["cur"][m].sum())
    a = np.abs(f["y"][m]); W = f["W"][m]; base = f["base"][m]
    bias = float(np.average(np.log(np.maximum(f["ours"][m], 1.0) / np.maximum(a, 1.0)),
                            weights=W))
    pen = [float((W * loss(a, gsnap(mm * base))).sum()) for mm in MS]
    j = int(np.argmin(pen))
    return contrib, bias, float(MS[j]), contrib and float(f["cur"][m].sum() - pen[j])


def show_axis(title, fos, keyfn, labels):
    print("\n" + "=" * 108)
    print(title)
    print("=" * 108)
    print(f"  {'버킷':<14s}" + "".join(
        f"|{fn:^45s}" for fn in [f[0]["name"] for f in [[x] for x in fos]]))
    print(f"  {'':<14s}" + "|" + f"{'기여%':>8s}{'편향':>9s}{'배율':>7s}{'여지':>10s}{'':>11s}"
          + "|" + f"{'기여%':>8s}{'편향':>9s}{'배율':>7s}{'여지':>10s}")
    for lab in labels:
        row = f"  {str(lab):<14s}"
        for f in fos:
            st = bucket_stats(f, keyfn(f) == lab if not callable(lab) else lab(f))
            if st is None:
                row += "|" + f"{'—':>45s}"
            else:
                c, b, m, g = st
                tot = f["cur"].sum()
                row += ("|" + f"{100*c/tot:>7.1f}%{b:>+9.3f}{m:>7.2f}{g:>+10.5f}"
                        + ("  ◀" if g > 0.0015 else "   ") + f"{'':>8s}")
        print(row)


def main():
    ctx = F.Context()
    tr = D.load_train()
    mat, dates = D.to_matrix(tr, ctx.items)
    nd = mat.shape[1]
    ctx.set_proxy(F.pick_proxy_items(mat, dates, nd, ctx.store_codes))
    dvals = pd.DatetimeIndex(dates)
    d20 = np.load(os.path.join(C.EXPERIMENTS, "phase10e_oof20.npz"))

    fos = []
    for name, cut, v0, v1, fi in FOLDS:
        va = V.origins(dates, v0, v1, nd)
        _, yva, mva = F.build_samples(mat, dates, va, ctx)
        raw = d20[f"ps{fi}"].mean(0).astype(np.float64)
        W, valid = cell_weights(yva, mva[:, 2], ctx.store_of_item, ctx.n)
        td = dvals[mva[:, 0] + mva[:, 1]]
        base = seg_base(raw); ours = gsnap(base)
        fos.append(dict(name=name, y=yva, W=W, valid=valid, iid=mva[:, 2],
                        store=ctx.store_of_item[mva[:, 2]],
                        dow=td.dayofweek.values, month=td.month.values,
                        week=(td - pd.to_timedelta(td.weekday, unit="D")).normalize(),
                        h=mva[:, 1], base=base, ours=ours,
                        cur=W * loss(yva, ours)))
        print(f"  [{name}] 점수 {fos[-1]['cur'].sum():.5f}")

    # ── 업장 ──
    show_axis("① 업장별", fos, lambda f: f["store"],
              sorted(set(ctx.store_of_item.tolist())))

    # ── 월(시기) ──
    show_axis("② 월별 (대상일 기준)", fos, lambda f: f["month"],
              [11, 12, 1, 2, 3, 4, 5, 6])

    # ── 요일 ──
    DOW = "월화수목금토일"
    show_axis("③ 요일별 (대상일 기준)", fos,
              lambda f: np.array([DOW[d] for d in f["dow"]]), list(DOW))

    # ── horizon ──
    show_axis("④ horizon (창 끝에서 며칠 뒤)", fos, lambda f: f["h"],
              [1, 2, 3, 4, 5, 6, 7])

    # ── 실측 크기 ──
    ABINS = [(1, 2, "A=1"), (2, 3, "A=2"), (3, 5, "A 3-4"), (5, 9, "A 5-8"),
             (9, 17, "A 9-16"), (17, 33, "A 17-32"), (33, 1e9, "A 33+")]
    print("\n" + "=" * 108)
    print("⑤ 실측 크기별 — 기여 · 정확일치율 · 편향 · 배율여지")
    print("=" * 108)
    for f in fos:
        tot = f["cur"].sum()
        print(f"\n  [{f['name']}]")
        print(f"  {'구간':<9s}{'기여%':>8s}{'정확일치':>9s}{'편향':>9s}{'배율':>7s}{'여지':>10s}")
        for lo, hi, lab in ABINS:
            m = (np.abs(f["y"]) >= lo) & (np.abs(f["y"]) < hi)
            st = bucket_stats(f, m)
            if st is None:
                continue
            c, b, mm, g = st
            ex = float((f["ours"][m & f["valid"]] == np.abs(f["y"][m & f["valid"]])).mean())
            print(f"  {lab:<9s}{100*c/tot:>7.1f}%{100*ex:>8.1f}%{b:>+9.3f}{mm:>7.2f}{g:>+10.5f}"
                  + ("  ◀" if g > 0.0015 else ""))

    # ── 품목: 여지 상위 (단체 함정 방지 — 오라클 여지 기준) ──
    print("\n" + "=" * 108)
    print("⑥ 품목별 — '많이 틀림'이 아니라 '여지(우리−item·week 오라클 상수)' 상위 15")
    print("=" * 108)
    for f in fos:
        gaps = {}
        wk = f["week"]
        idx_by = pd.Series(range(len(wk))).groupby(
            [pd.Series(f["iid"]), pd.Series(wk)]).groups
        for (it, w), idx in idx_by.items():
            idx = np.asarray(idx)
            v = f["valid"][idx]
            if not v.any():
                continue
            a = np.abs(f["y"][idx][v]); Wg = f["W"][idx][v]
            cand = np.arange(1, int(a.max()) + 2)
            pen = np.array([(Wg * (2 * np.abs(a - c) / (a + c))).sum() for c in cand])
            gaps[it] = gaps.get(it, 0.0) + float(f["cur"][idx].sum() - pen.min())
        top = sorted(gaps.items(), key=lambda x: -x[1])[:15]
        tot = sum(gaps.values())
        print(f"\n  [{f['name']}]  여지 총합 {tot:+.4f}")
        for it, g in top:
            m = (f["iid"] == it) & f["valid"]
            a = np.abs(f["y"][m])
            sm = float(loss(f["y"][m], f["ours"][m]).mean())
            print(f"    {g:+.5f} ({100*g/tot:4.1f}%)  SMAPE {sm:.3f} · 평균A {a.mean():6.1f}"
                  f" · {ctx.items[it]}")


if __name__ == "__main__":
    main()
