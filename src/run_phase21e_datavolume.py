# -*- coding: utf-8 -*-
"""Phase 21-e — 앙상블 이득을 죽인 게 '거리'인가 '학습량'인가. **2×2 로 가른다.**

21-c 에서 ridge 단독 혼합(w=0.10)의 폴드별 이득:
    F2 겨울(300 origin) +0.0036 · F3 봄(391) +0.0045 · FAR-봄(300) +0.0040
    **FAR-겨울(209) −0.0047**  ← 유일한 반대표

FAR-겨울은 **같은 겨울 창을 검증하는데 F2 와 정반대**다. 둘의 차이는 두 개뿐:
    · 학습량   300 origin vs 209 origin (학습행 기준 실제 제출의 58% vs **37%**)
    · 거리     0개월 vs 3개월
그리고 log.md Phase 5-f 는 이미 이렇게 적어놨다 —
    "거리를 벌리려면 학습 구간을 앞으로 당겨야 하는데 데이터가 1.5년뿐이라
     당기는 만큼 학습량이 줄어든다. FAR 폴드는 F1 의 결함을 그대로 물려받았다."
즉 **FAR-겨울은 두 변수가 엉켜 있어 단독으로는 아무것도 못 말한다.**

가를 수 있는 방향은 하나뿐이다: **거리를 0 으로 고정한 채 학습량만 줄인다.**
(반대로 학습량을 고정한 채 거리를 늘리는 건 데이터가 1.5년이라 불가능하다.)

    실험      학습 cut        origin 수   거리    검증
    A        2023-11-24       300       0개월   겨울   ← F2 재현
    B        2023-11-24       209       0개월   겨울   ← A 대비 **학습량만** ↓
    B2       2023-11-24       150       0개월   겨울   ← 추세 확인용
    C        2023-08-25       209       3개월   겨울   ← B 대비 **거리만** ↑ (= FAR-겨울)

판정 (사전 고정, 결과 보기 전)
    · gain(A) > gain(B) ≈ gain(C) 이고 B2 가 더 낮으면 → **학습량이 원인.**
      FAR-겨울의 반대표는 인공물이고, 실제 제출(498 origin·320k행)은 A 보다 위쪽이다.
    · gain(A) ≈ gain(B) 인데 gain(C) 만 음수면 → **거리가 원인.** 반대표는 진짜고,
      실제 제출은 TEST 가 0~11개월 뒤이므로 혼합이 위험하다. 축을 접는다.
    · 둘 다 아니면 판정 보류.

왜 Ridge 가 데이터에 더 굶주릴 것으로 보는가 (사전 가설):
    설계행렬이 3,201 열이고 그중 `item×요일`·`item×horizon` 이 각각 1,351 수준이다.
    칸당 표본이 실제 제출 237행 → FAR-겨울 88행으로 **2.7배 얇아진다.**
    트리는 분할을 공유하지만 원핫 상호작용은 칸마다 독립이라 더 크게 무너진다.
"""
import os
import json
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

import config as C
import dataio as D
import features as F
import validate as V
import model_ridge_tm as TM
from run_phase10c_thresholds import cell_weights

NT = os.cpu_count()
PARAMS = dict(objective="regression_l1", metric="l1", learning_rate=0.05,
              num_leaves=127, min_data_in_leaf=40, feature_fraction=0.65,
              bagging_fraction=0.85, bagging_freq=1, lambda_l2=1.0,
              verbosity=-1, num_threads=NT)
ROUNDS = 1000
SEEDS = (42, 7, 2024, 913, 31)
DROP3 = ["w_posmedian", "w_last14", "w_std"]

VAL = ("2023-11-24", "2024-02-22")          # 네 실험 모두 같은 겨울 창을 검증한다
ARMS = [("A  300·0개월", "2023-11-24", 300),
        ("B  209·0개월", "2023-11-24", 209),
        ("B2 150·0개월", "2023-11-24", 150),
        ("C  209·3개월", "2023-08-25", None)]
WGRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
OUT = os.path.join(C.EXPERIMENTS, "phase21e_datavolume.json")


def seg_snap(raw):
    p = np.where(raw < 1.8, 0.55 * raw, np.where(raw < 10.0, 0.90 * raw, 1.02 * raw))
    p = np.maximum(p, 1.0)
    k = np.maximum(np.floor(p), 1.0)
    return np.maximum(np.where(p >= np.sqrt(k * (k + 1.0)), k + 1.0, k), 1.0)


def loss(a, p):
    a, p = np.abs(a), np.abs(p)
    den = a + p
    out = np.zeros(len(a))
    m = den > 0
    out[m] = 2.0 * np.abs(a[m] - p[m]) / den[m]
    return out


def main():
    t0 = time.time()
    ctx = F.Context()
    tctx = TM.TMContext(D.item_order())
    tr = D.load_train()
    mat, dates = D.to_matrix(tr, ctx.items)
    nd = mat.shape[1]
    keep0, names0 = F.active_columns(), F.active_names()
    idx = [i for i, n in enumerate(names0) if n not in DROP3]
    keep = [keep0[i] for i in idx]
    names = [names0[i] for i in idx]
    cats = [c for c in F.CATEGORICAL if c in names]

    va = V.origins(dates, VAL[0], VAL[1], nd)
    print(f"Phase 21-e — 검증 창 고정(겨울 {VAL[0]}~{VAL[1]}, origin {len(va)}개)\n",
          flush=True)

    res = {}
    for label, cut, ntrain in ARMS:
        cut_col = int(np.searchsorted(np.array(dates), pd.Timestamp(cut)))
        ctx.set_proxy(F.pick_proxy_items(mat, dates, cut_col, ctx.store_codes))
        trn = [o for o in range(C.WINDOW - 1, nd - C.HORIZON)
               if dates[o] < pd.Timestamp(cut)]
        if ntrain is not None:
            trn = trn[-ntrain:]              # 뒤쪽(최근)을 남긴다 = 거리 고정

        Xtr, ytr, _ = F.build_samples(mat, dates, trn, ctx)
        Xva, yva, mva = F.build_samples(mat, dates, va, ctx)
        m = ytr != 0
        Xt = np.ascontiguousarray(Xtr[m][:, keep])
        yt = np.log1p(np.maximum(ytr[m], 1.0))
        Xv = np.ascontiguousarray(Xva[:, keep])
        n_rows = int(m.sum())
        del Xtr, Xva
        ps = []
        for sd in SEEDS:
            ds = lgb.Dataset(Xt, label=yt, feature_name=names,
                             categorical_feature=cats, free_raw_data=False)
            ps.append(np.expm1(lgb.train(dict(PARAMS, seed=sd), ds,
                                         num_boost_round=ROUNDS).predict(Xv)))
            del ds
        ours = np.mean(ps, 0)
        del Xt, Xv, ps

        tn, tc, ty, ti = TM.build_frames(tctx, mat, dates, trn)
        vn, vc, vy, vi = TM.build_frames(tctx, mat, dates, va)
        assert np.array_equal(vi, mva[:, 2].astype(int))
        tm = TM.fit_predict(tctx, tn, tc, ty, ti, vn, vc)
        n_cols = int(tm["design_shape"][1])
        del tn, tc, vn, vc

        W, _ = cell_weights(yva, mva[:, 2], ctx.store_of_item, ctx.n)
        sw = {w: float((W * loss(yva, seg_snap(
            (1 - w) * ours + w * tm["ridge"]))).sum()) for w in WGRID}
        sc = yva != 0
        lg = lambda v: np.log1p(np.maximum(v, 0.0))
        ay = lg(np.maximum(np.abs(yva[sc]), 1.0))
        rho = float(np.corrcoef(lg(np.maximum(ours[sc], 1.0)) - ay,
                                lg(tm["ridge"][sc]) - ay)[0, 1])
        solo = float((W * loss(yva, seg_snap(tm["ridge"]))).sum())

        res[label] = dict(origins=len(trn), rows=n_rows, cols=n_cols, rho=rho,
                          sweep={str(k): v for k, v in sw.items()}, solo_tm=solo)
        print(f"  [{label}] origin {len(trn):>3d} · 학습행 {n_rows:,} · "
              f"ρ {rho:.4f} · 우리 {sw[0.0]:.5f} · 그들 {solo:.5f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    # ── 요약
    print("\n" + "=" * 92)
    print("학습량을 줄이면 앙상블 이득이 어떻게 되는가 (검증 창 동일 · ridge 단독 혼합)")
    print(f"{'실험':<14s}{'origin':>7s}{'학습행':>9s}{'행/열':>7s}{'ρ':>8s}" +
          "".join(f"{'w='+format(w,'.2f'):>10s}" for w in WGRID[1:]))
    for label, _, _ in [(a[0], a[1], a[2]) for a in ARMS]:
        r = res[label]
        g = [r["sweep"]["0.0"] - r["sweep"][str(w)] for w in WGRID[1:]]
        print(f"{label:<14s}{r['origins']:>7d}{r['rows']:>9,d}"
              f"{r['rows']/r['cols']:>7.0f}{r['rho']:>8.4f}"
              + "".join(f"{x:>+10.5f}" for x in g))
    print("\n(참고) 실제 제출 = origin 498 · 학습행 320,012 · 행/열 100 — 위 어느 실험보다 많다.")

    gA = res[ARMS[0][0]]["sweep"]["0.0"] - res[ARMS[0][0]]["sweep"]["0.1"]
    gB = res[ARMS[1][0]]["sweep"]["0.0"] - res[ARMS[1][0]]["sweep"]["0.1"]
    gB2 = res[ARMS[2][0]]["sweep"]["0.0"] - res[ARMS[2][0]]["sweep"]["0.1"]
    gC = res[ARMS[3][0]]["sweep"]["0.0"] - res[ARMS[3][0]]["sweep"]["0.1"]
    print("\n" + "=" * 92)
    print(f"w=0.10 기준:  A(300) {gA:+.5f} · B(209) {gB:+.5f} · "
          f"B2(150) {gB2:+.5f} · C(209·먼거리) {gC:+.5f}")
    drop_data = gA - gB
    drop_dist = gB - gC
    print(f"  학습량 300→209 로 잃은 이득: {drop_data:+.5f}")
    print(f"  거리 0→3개월 로 잃은 이득  : {drop_dist:+.5f}")
    if drop_data > 2 * abs(drop_dist):
        v = "✅ **학습량이 원인.** FAR-겨울 반대표는 인공물 — 실제 제출은 A 보다 위쪽이다."
    elif drop_dist > 2 * abs(drop_data):
        v = "❌ **거리가 원인.** 반대표는 진짜다. TEST 는 0~11개월 뒤라 혼합이 위험하다."
    else:
        v = "⚠️ 둘 다 기여 — 판정 보류. 어느 쪽도 단독 설명이 안 된다."
    print(f"  → {v}")
    print("=" * 92)

    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"저장: {os.path.basename(OUT)} · 총 {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
