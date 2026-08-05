# -*- coding: utf-8 -*-
"""Phase 12-b — Q6·Q7 이 함께 가리킨 '급등 창 과소예측'의 정밀 검증. 학습 0회.

Phase 12 관찰
    Q7: 추세 r=(last7+1)/(prev7+1) > 2 인 칸에서 부호편향 -0.30~-0.64 (강한 과소예측),
        사후 최적 배율이 겨울 1.58 · 봄 1.53 으로 **독립 2구간이 거의 같은 값**을 골랐다.
    Q6: 봄 여지 상위 3주 = 화담숲 재개장 직후(3/25~4/21) · 겨울 상위 3주 = 스키 개막기(12월).
        → 둘 다 '수요가 급히 올라가는 구간'. 하나의 현상으로 보인다:
        **트리는 창을 평균 내므로, 급등 중에는 창 평균이 현재 수준보다 낮다 → 과소예측.**

여기서 재는 것 (전부 사후 검증 규율 하에)
    ① 단일구간 부스트: r > 2 인 칸만 base × m 후 재스냅. m ∈ {1.3~1.7} 고원인가 (규칙 7)
    ② 양방향 전이: 겨울에서 고른 m → 봄 채점, 봄에서 고른 m → 겨울 채점 (규칙 3)
       ※ Phase 12 의 5구간 전이는 봄→겨울이 -0.0034 로 실패했다. 가운데 구간들이 잡음이었는지 확인.
    ③ 4폴드 방향 일관성 (독립 2구간 × 학습거리 2)
    ④ 이득의 출처 분해: 영업장 × 월 — 화담숲 재개장 전용인가, 12월 스키 개막에서도 나는가
    ⑤ r>2 의 성격 분해: prev7=0 (휴점 복귀) vs prev7>0 (진짜 급증) — 어느 쪽의 이득인가
    ⑥ 실제 TEST 노출: TEST 창에서 r 계산 (창만 쓰므로 규칙 위반 없음) — r>2 칸이 몇 %, 어느 시즌에
"""
import os
import numpy as np
import pandas as pd

import config as C
import dataio as D
import features as F
import validate as V
from run_phase10c_thresholds import cell_weights

FOLDS = [
    ("F2 겨울",  "2023-11-24", "2023-11-24", "2024-02-22"),
    ("F3 봄",    "2024-02-23", "2024-02-23", "2024-06-08"),
    ("FAR-봄",   "2023-11-24", "2024-02-23", "2024-06-08"),
    ("FAR-겨울", "2023-08-25", "2023-11-24", "2024-02-22"),
]


def seg_base(raw):
    p = np.where(raw < 1.8, 0.55 * raw, np.where(raw < 10.0, 0.90 * raw, 1.02 * raw))
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


def main():
    ctx = F.Context()
    tr = D.load_train()
    mat, dates = D.to_matrix(tr, ctx.items)
    dser = pd.Series(pd.DatetimeIndex(dates))
    nd = mat.shape[1]
    ctx.set_proxy(F.pick_proxy_items(mat, dates, nd, ctx.store_codes))
    d20 = np.load(os.path.join(C.EXPERIMENTS, "phase10e_oof20.npz"))

    fo = []
    for fi, (name, cut, v0, v1) in enumerate(FOLDS):
        va = V.origins(dates, v0, v1, nd)
        _, yva, mva = F.build_samples(mat, dates, va, ctx)
        assert np.array_equal(yva, d20[f"y{fi}"])
        raw = d20[f"ps{fi}"].mean(0).astype(np.float64)
        W, valid = cell_weights(yva, mva[:, 2], ctx.store_of_item, ctx.n)
        r = np.empty(len(yva)); p70 = np.empty(len(yva))
        for o in np.unique(mva[:, 0]):
            m = mva[:, 0] == o
            it = mva[m, 2]
            l7 = mat[:, o - 6:o + 1].mean(1); p7 = mat[:, o - 13:o - 6].mean(1)
            r[m] = (l7[it] + 1.0) / (p7[it] + 1.0)
            p70[m] = p7[it]
        tdate = pd.DatetimeIndex(dser.values[mva[:, 0] + mva[:, 1]])
        base = seg_base(raw)
        fo.append(dict(name=name, y=yva, W=W, iid=mva[:, 2], r=r, prev7=p70,
                       month=tdate.month, base=base, ours=gsnap(base)))

    def gain_fixed(f, m, thr=2.0):
        p = f["base"].copy()
        b = f["r"] > thr
        p[b] = m * p[b]
        return float((f["W"] * (loss(f["y"], f["ours"]) - loss(f["y"], gsnap(p)))).sum()), b

    # ── ① 고원 확인 + ③ 4폴드 일관성 ──────────────────────────────
    print("=" * 96)
    print("①③  r>2 단일구간 부스트 — m 스윕 · 4폴드")
    print("=" * 96)
    print(f"  {'m':>6s}" + "".join(f"{f['name']:>11s}" for f in fo) + f"{'평균':>10s}{'일관':>6s}")
    for m in (1.2, 1.3, 1.4, 1.5, 1.6, 1.7):
        g = [gain_fixed(f, m)[0] for f in fo]
        ok = sum(x > 0 for x in g)
        print(f"  {m:>6.2f}" + "".join(f"{x:>+11.5f}" for x in g)
              + f"{np.mean(g):>+10.5f}{ok:>4d}/4")

    # ── ② 양방향 전이 (단일 구간만) ────────────────────────────────
    print("\n" + "=" * 96)
    print("②  양방향 전이 — 한 구간에서 최적 m 을 고르고 다른 구간에서 채점")
    print("=" * 96)
    MS = np.round(np.arange(1.0, 2.01, 0.02), 2)
    best = {}
    for f in fo[:2]:
        gg = [gain_fixed(f, m)[0] for m in MS]
        best[f["name"]] = float(MS[int(np.argmax(gg))])
    for src, dst in (("F2 겨울", "F3 봄"), ("F3 봄", "F2 겨울")):
        m = best[src]
        f = [x for x in fo if x["name"] == dst][0]
        g, _ = gain_fixed(f, m)
        print(f"  {src} 최적 m={m:.2f}  →  {dst} 적용: {g:+.5f}")

    # ── ④ 이득의 출처: 영업장 × 월 ─────────────────────────────────
    print("\n" + "=" * 96)
    print("④  m=1.5 이득의 출처 (r>2 칸만) — 재개장 전용인가, 일반 현상인가")
    print("=" * 96)
    for f in fo[:2]:
        p = f["base"].copy(); b = f["r"] > 2.0
        p[b] = 1.5 * p[b]
        d = f["W"] * (loss(f["y"], f["ours"]) - loss(f["y"], gsnap(p)))
        print(f"\n  [{f['name']}]  r>2 칸 {b.sum():,}개 · 총이득 {d.sum():+.5f}")
        df = pd.DataFrame(dict(g=d[b], st=ctx.store_of_item[f["iid"][b]],
                               mo=np.asarray(f["month"])[b]))
        bs = df.groupby("st")["g"].sum().sort_values(ascending=False)
        print("   영업장:", " · ".join(f"{s} {v:+.5f}" for s, v in bs.items() if abs(v) > 2e-4))
        bm = df.groupby("mo")["g"].sum().sort_values(ascending=False)
        print("   월    :", " · ".join(f"{mo}월 {v:+.5f}" for mo, v in bm.items() if abs(v) > 2e-4))

    # ── ⑤ 휴점 복귀 vs 진짜 급증 ───────────────────────────────────
    print("\n" + "=" * 96)
    print("⑤  r>2 의 성격 — prev7=0 (휴점 복귀) vs prev7>0 (진짜 급증), m=1.5 이득")
    print("=" * 96)
    for f in fo[:2]:
        p = f["base"].copy(); b = f["r"] > 2.0
        p[b] = 1.5 * p[b]
        d = f["W"] * (loss(f["y"], f["ours"]) - loss(f["y"], gsnap(p)))
        z = b & (f["prev7"] < 1e-9); nz = b & ~ (f["prev7"] < 1e-9)
        print(f"  [{f['name']}]  복귀형 {z.sum():>5,}칸 {d[z].sum():+.5f}   "
              f"급증형 {nz.sum():>5,}칸 {d[nz].sum():+.5f}")

    # ── ⑥ 실제 TEST 노출 ───────────────────────────────────────────
    print("\n" + "=" * 96)
    print("⑥  실제 TEST 에서 r>2 노출 (각 TEST 의 28일 창만 사용 — 규칙 위반 없음)")
    print("=" * 96)
    tot = 0; per = []
    for t in range(C.N_TEST):
        te = D.load_test(t)
        tmat, tdates = D.to_matrix(te, ctx.items)
        o = tmat.shape[1] - 1
        l7 = tmat[:, o - 6:o + 1].mean(1); p7 = tmat[:, o - 13:o - 6].mean(1)
        rr = (l7 + 1.0) / (p7 + 1.0)
        b = rr > 2.0
        tot += b.sum()
        st = pd.Series(ctx.store_of_item[np.where(b)[0]]).value_counts()
        lab = str(pd.Timestamp(tdates[-1]).date())
        per.append((t, lab, int(b.sum()), " · ".join(f"{s}{n}" for s, n in st.items())))
    for t, lab, n, s in per:
        print(f"  TEST_{t:02d} (창 끝 {lab})  r>2 품목 {n:>3d}개   {s}")
    print(f"\n  합계 {tot}품목 × 7일 = {7 * tot:,}칸 / 13,510칸 = {100 * 7 * tot / 13510:.1f}%")


if __name__ == "__main__":
    main()
