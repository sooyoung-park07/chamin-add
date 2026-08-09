# -*- coding: utf-8 -*-
"""Phase 26 — "봄엔 섞고 겨울엔 안 섞으면 되지 않나". **제대로 시험한다.**

사용자 지적이 옳다. 계절은 창 끝 날짜만 보면 알 수 있고 채점 시점에도 알 수 있다.
자유도도 2개뿐이다. 그런데 지금까지 이걸 **직접 시험한 적이 없다.**
Phase 22 는 `month` 를 1~12 순서값으로 넣었는데 12월과 1월이 최대 거리가 되는 인코딩이라
**'겨울'을 표현할 수 없었다.** 계절을 시험한 게 아니었다.

여기서 재는 것: **k = σ그들/σ우리 가 계절에 따라 안정적으로 갈리는가.**
   (ρ 는 네 폴드에서 0.84~0.88 로 거의 고정이었다. 움직이는 건 k 하나다.
    그리고 w* 의 부호를 정하는 건 조건 ρ < 1/k 다.)

설계 — 학습량을 **209 origin 으로 고정**하고 계절만 바꾼다.
   21-e 에서 학습량이 k 를 크게 흔든다는 걸 이미 봤다(119k행 k=1.18 vs 185k행 k=1.14).
   기존 4폴드는 학습량과 계절이 엉켜 있어서 계절 효과만 뽑을 수 없다.

   P1 가을 : 학습 → 2023-08-25 의 뒤 209개 · 검증 2023-08-25~2023-11-23
   P2 겨울 : 학습 → 2023-11-24 의 뒤 209개 · 검증 2023-11-24~2024-02-22
   P3 봄   : 학습 → 2024-02-23 의 뒤 209개 · 검증 2024-02-23~2024-06-08

⚠️ **여름은 못 잰다.** 여름을 검증하려면 학습이 2023년 상반기뿐이라 120 origin 밖에 안 된다.
   그런데 TEST 10개 중 **T00(7월)·T01(8월) 두 개가 여름**이다. 이건 메울 수 없는 구멍이고,
   결과 해석에 반드시 반영해야 한다.
⚠️ 학습 구간의 계절 구성도 셋이 다르다(P1 겨울~여름 / P2 봄~가을 / P3 여름~겨울).
   학습량은 맞췄지만 이건 못 맞춘다. 1.5년 데이터의 한계다.
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
ROUNDS, SEEDS = 1000, (42, 7, 2024, 913, 31)
DROP3 = ["w_posmedian", "w_last14", "w_std"]
NTRAIN = 209

PERIODS = [("P1 가을", "2023-08-25", "2023-08-25", "2023-11-23"),
           ("P2 겨울", "2023-11-24", "2023-11-24", "2024-02-22"),
           ("P3 봄", "2024-02-23", "2024-02-23", "2024-06-08")]
WGRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
OUT = os.path.join(C.EXPERIMENTS, "phase26_season.json")


def post(raw, seg=True, snap=True):
    p = (np.where(raw < 1.8, 0.55 * raw, np.where(raw < 10.0, 0.90 * raw, 1.02 * raw))
         if seg else raw.copy())
    p = np.maximum(p, 1.0)
    if not snap:
        return p
    k = np.maximum(np.floor(p), 1.0)
    return np.maximum(np.where(p >= np.sqrt(k * (k + 1.0)), k + 1.0, k), 1.0)


def loss(a, p):
    a, p = np.abs(a), np.abs(p)
    den = a + p
    out = np.zeros(len(a))
    m = den > 0
    out[m] = 2.0 * np.abs(a[m] - p[m]) / den[m]
    return out


def wstat(w, x):
    m = np.sum(w * x) / np.sum(w)
    return m, np.sqrt(np.sum(w * (x - m) ** 2) / np.sum(w))


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

    print(f"학습량 {NTRAIN} origin 고정 · 계절만 변화\n", flush=True)
    res = {}
    for pname, cut, v0, v1 in PERIODS:
        cut_col = int(np.searchsorted(np.array(dates), pd.Timestamp(cut)))
        ctx.set_proxy(F.pick_proxy_items(mat, dates, cut_col, ctx.store_codes))
        trn = [o for o in range(C.WINDOW - 1, nd - C.HORIZON)
               if dates[o] < pd.Timestamp(cut)][-NTRAIN:]
        va = V.origins(dates, v0, v1, nd)

        Xtr, ytr, _ = F.build_samples(mat, dates, trn, ctx)
        Xva, yva, mva = F.build_samples(mat, dates, va, ctx)
        m = ytr != 0
        Xt = np.ascontiguousarray(Xtr[m][:, keep])
        yt = np.log1p(np.maximum(ytr[m], 1.0))
        Xv = np.ascontiguousarray(Xva[:, keep])
        nrow = int(m.sum())
        del Xtr, Xva
        ps = []
        for sd in SEEDS:
            ds = lgb.Dataset(Xt, label=yt, feature_name=names,
                             categorical_feature=cats, free_raw_data=False)
            ps.append(np.expm1(lgb.train(dict(PARAMS, seed=sd), ds,
                                         num_boost_round=ROUNDS).predict(Xv)))
            del ds
        A = np.mean(ps, 0)
        del Xt, Xv, ps

        tn, tc, ty, ti = TM.build_frames(tctx, mat, dates, trn)
        vn, vc, vy, vi = TM.build_frames(tctx, mat, dates, va)
        assert np.array_equal(vi, mva[:, 2].astype(int))
        B = TM.fit_predict(tctx, tn, tc, ty, ti, vn, vc)["ridge"]
        del tn, tc, vn, vc

        W, _ = cell_weights(yva, mva[:, 2], ctx.store_of_item, ctx.n)
        sc = yva != 0
        lg = lambda v: np.log1p(np.maximum(v, 0.0))
        a = lg(np.maximum(np.abs(yva[sc]), 1.0))
        r1, r2, ww = lg(np.maximum(A[sc], 1.0)) - a, lg(B[sc]) - a, W[sc]
        m1, s1 = wstat(ww, r1)
        m2, s2 = wstat(ww, r2)
        rho = float(np.sum(ww * (r1 - m1) * (r2 - m2)) / np.sum(ww) / (s1 * s2))
        k = float(s2 / s1)
        ws = (1 - rho * k) / (1 + k * k - 2 * rho * k)

        sweeps = {}
        for tag, kw in (("raw", dict(seg=False, snap=False)),
                        ("prod", dict(seg=True, snap=True))):
            sweeps[tag] = {w: float((W * loss(yva, post((1 - w) * A + w * B, **kw))).sum())
                           for w in WGRID}
        res[pname] = dict(rows=nrow, origins=len(trn), n_val=len(va),
                          sigma_A=float(s1), sigma_B=float(s2), k=k, rho=rho,
                          w_star=float(ws), sweeps={t: {str(a_): b for a_, b in v.items()}
                                                    for t, v in sweeps.items()})
        print(f"  [{pname}] 학습행 {nrow:,} · 검증 origin {len(va)} · "
              f"σA {s1:.4f} σB {s2:.4f} **k {k:.3f}** ρ {rho:.4f} w* {ws:+.3f}"
              f"  ({time.time()-t0:.0f}s)", flush=True)

    # ── 요약
    print("\n" + "=" * 96)
    print("① k 가 계절로 갈리는가 (학습량 209 origin 고정)")
    print(f"{'구간':<10s}{'학습행':>9s}{'σ우리':>8s}{'σ그들':>8s}{'k':>7s}{'ρ':>8s}"
          f"{'문턱 1/k':>9s}{'통과':>6s}{'w*':>8s}")
    for p, _, _, _ in PERIODS:
        r = res[p]
        print(f"{p:<10s}{r['rows']:>9,d}{r['sigma_A']:>8.4f}{r['sigma_B']:>8.4f}"
              f"{r['k']:>7.3f}{r['rho']:>8.4f}{1/r['k']:>9.4f}"
              f"{'✅' if r['rho'] < 1/r['k'] else '❌':>6s}{r['w_star']:>+8.3f}")

    print("\n" + "=" * 96)
    print("② 실측 w 곡선 (w=0 대비 이득) — raw 가 진실, prod 는 스냅 포함")
    for tag, nm in (("raw", "raw (하한만)"), ("prod", "prod (seg+스냅)")):
        print(f"\n  [{nm}]")
        print(f"{'w':>6s}" + "".join(f"{p:>12s}" for p, _, _, _ in PERIODS))
        for w in WGRID:
            row = [res[p]["sweeps"][tag]["0.0"] - res[p]["sweeps"][tag][str(w)]
                   for p, _, _, _ in PERIODS]
            print(f"{w:>6.2f}" + "".join(f"{v:>+12.5f}" for v in row))

    print("\n" + "=" * 96)
    print("③ 계절 게이트가 성립하려면")
    ks = {p: res[p]["k"] for p, _, _, _ in PERIODS}
    win = ks["P2 겨울"]
    others = [v for p, v in ks.items() if p != "P2 겨울"]
    print(f"   겨울 k = {win:.3f} · 가을 k = {ks['P1 가을']:.3f} · 봄 k = {ks['P3 봄']:.3f}")
    if win > max(others) + 0.03:
        print("   ✅ **겨울만 확실히 높다.** 계절 게이트('겨울엔 섞지 않는다')에 근거가 생긴다.")
        print("      → 다음 단계: 그 게이트로 제출본을 만들고 사전등록한다.")
        print(f"      ⚠️ 단 **여름(T00·T01)은 측정 구간이 없다.** 10개 중 2개가 외삽이다.")
    elif abs(ks["P1 가을"] - ks["P3 봄"]) > 0.03 or win < min(others):
        print("   ⚠️ 계절 순서가 '겨울만 나쁨'이 아니다. 단순 게이트로는 안 잡힌다.")
    else:
        print("   ❌ **k 가 계절로 안 갈린다.** 겨울이 특별하다는 것 자체가 성립하지 않는다.")
        print("      → 21-c 의 계절 차이는 계절이 아니라 **학습량** 때문이었다는 뜻이다.")
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\n저장: {os.path.basename(OUT)} · 총 {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
