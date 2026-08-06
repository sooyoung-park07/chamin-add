# -*- coding: utf-8 -*-
"""Phase 17 — 피처 선별 스크린 2종. (사전등록: 이 단계는 '후보 발굴'만 한다.
채택/기각 판정은 어느 쪽도 여기서 내리지 않고, 반드시 Phase 18 짝시드 사다리로 넘긴다.)

① 그림자 대조 (Boruta 식)
    57개 피처 각각의 행-셔플 복사본 57개를 붙여 114열로 학습.
    진짜 피처의 gain 이 **그림자 전체의 최대 gain** 을 못 넘으면 = 잡음과 구분 불가.
    판정 기준(사전 고정): 2폴드 × 시드 3 전부에서 그림자 최대를 못 넘어야 '잡음 후보'.
    ※ gain 은 가치 척도로는 금지(규칙 ⑨)지만, "잡음보다 나은가"의 귀무 대조로는 유효하다 —
      그림자가 바로 그 귀무분포이기 때문.

② 오라클 전수 스윕 (학습 0회)
    57개 피처 각각으로 검증 칸을 5구간(범주형은 레벨)으로 나누고, 구간별 사후 최적 배율의
    이득을 잰다. 이득이 크고 + 배율이 단조/일관 = 모델이 그 축을 덜 쓰고 있다 → 변환 후보.
    ※ 사후 상한이므로 실현률은 낮다(피처 실현 ~20% 전례). 순위표로만 쓴다.
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
from run_phase10c_thresholds import cell_weights

NT = os.cpu_count()
PARAMS = dict(objective="regression_l1", metric="l1", learning_rate=0.05,
              num_leaves=127, min_data_in_leaf=40, feature_fraction=0.65,
              bagging_fraction=0.85, bagging_freq=1, lambda_l2=1.0,
              verbosity=-1, num_threads=NT)
ROUNDS = 1000
SH_SEEDS = (42, 7, 2024)

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


def main():
    t0 = time.time()
    ctx = F.Context()
    tr = D.load_train()
    mat, dates = D.to_matrix(tr, ctx.items)
    nd = mat.shape[1]
    keep, names = F.active_columns(), F.active_names()
    cats = [c for c in F.CATEGORICAL if c in names]

    # ================================================================ ①
    print("=" * 100)
    print("① 그림자 대조 — 진짜 57 + 셔플 57 = 114열, 폴드 2 × 시드 3")
    print("=" * 100, flush=True)
    fails = {n: 0 for n in names}
    runs = 0
    for fname, cut, v0, v1, fi in FOLDS:
        cut_col = int(np.searchsorted(np.array(dates), pd.Timestamp(cut)))
        ctx.set_proxy(F.pick_proxy_items(mat, dates, cut_col, ctx.store_codes))
        trn = [o for o in range(C.WINDOW - 1, nd - C.HORIZON)
               if dates[o] < pd.Timestamp(cut)]
        Xtr, ytr, _ = F.build_samples(mat, dates, trn, ctx)
        m = ytr != 0
        Xt = np.ascontiguousarray(Xtr[m][:, keep])
        yt = np.log1p(np.maximum(ytr[m], 1.0))
        del Xtr
        rng = np.random.default_rng(0)
        Xsh = Xt.copy()
        for j in range(Xsh.shape[1]):
            rng.shuffle(Xsh[:, j])
        X2 = np.hstack([Xt, Xsh])
        n2 = names + [n + "_sh" for n in names]
        c2 = cats + [c + "_sh" for c in cats]
        for sd in SH_SEEDS:
            ds = lgb.Dataset(X2, label=yt, feature_name=n2,
                             categorical_feature=c2, free_raw_data=False)
            md = lgb.train(dict(PARAMS, seed=sd), ds, num_boost_round=ROUNDS)
            g = np.array(md.feature_importance("gain"))
            real, shadow = g[:len(names)], g[len(names):]
            bar = shadow.max()
            runs += 1
            for j, n in enumerate(names):
                if real[j] < bar:
                    fails[n] += 1
            del ds, md
        print(f"  [{fname}] 완료 — 그림자 최대 gain 이 귀무 문턱  ({time.time()-t0:.0f}s)",
              flush=True)

    noise = sorted([n for n, f in fails.items() if f == runs])
    borderline = sorted([n for n, f in fails.items() if 0 < f < runs])
    print(f"\n  잡음 후보 (전 {runs}회에서 그림자 최대 미달): {len(noise)}개")
    for n in noise:
        print(f"    {n}")
    print(f"  경계 (일부 회차 미달): {len(borderline)}개 — " + " · ".join(borderline))

    # ================================================================ ②
    print("\n" + "=" * 100)
    print("② 오라클 전수 스윕 — 피처별 5구간 사후 배율 이득 (학습 0회, 20시드 OOF)")
    print("=" * 100, flush=True)
    d20 = np.load(os.path.join(C.EXPERIMENTS, "phase10e_oof20.npz"))
    MS = np.round(np.arange(0.70, 1.41, 0.02), 2)
    rows = {}
    ALLF = [("F2 겨울", "2023-11-24", "2023-11-24", "2024-02-22", 0),
            ("F3 봄", "2024-02-23", "2024-02-23", "2024-06-08", 1),
            ("FAR-봄", "2023-11-24", "2024-02-23", "2024-06-08", 2),
            ("FAR-겨울", "2023-08-25", "2023-11-24", "2024-02-22", 3)]
    for fname, cut, v0, v1, fi in ALLF:
        cut_col = int(np.searchsorted(np.array(dates), pd.Timestamp(cut)))
        ctx.set_proxy(F.pick_proxy_items(mat, dates, cut_col, ctx.store_codes))
        va = V.origins(dates, v0, v1, nd)
        Xva, yva, mva = F.build_samples(mat, dates, va, ctx)
        Xv = Xva[:, keep]
        del Xva
        raw = d20[f"ps{fi}"].mean(0).astype(np.float64)
        W, valid = cell_weights(yva, mva[:, 2], ctx.store_of_item, ctx.n)
        base = seg_base(raw); ours = gsnap(base)
        cur = W * loss(yva, ours)
        for j, n in enumerate(names):
            col = Xv[:, j]
            if n in cats:
                vals = pd.Series(col).value_counts()
                lv = vals[vals >= 300].index.values[:8]
                bins = [col == v for v in lv]
            else:
                fin = np.isfinite(col)
                if fin.sum() < 500:
                    continue
                q = np.nanquantile(col, [0.2, 0.4, 0.6, 0.8])
                q = np.unique(q)
                bi = np.digitize(col, q)
                bins = [(bi == b) & fin for b in range(len(q) + 1)]
            gain = 0.0; mults = []
            for bmask in bins:
                m2 = bmask & valid
                if m2.sum() < 200:
                    continue
                a = np.abs(yva[m2]); Wb = W[m2]; bb = base[m2]
                pen = [float((Wb * (2 * np.abs(a - gsnap(mm * bb))
                                    / (a + gsnap(mm * bb)))).sum()) for mm in MS]
                k = int(np.argmin(pen))
                gain += float(cur[m2].sum() - pen[k])
                mults.append(float(MS[k]))
            r = rows.setdefault(n, dict(g=[], mono=0))
            r["g"].append(gain)
            if len(mults) >= 3:
                dm = np.diff(mults)
                if (dm >= 0).all() or (dm <= 0).all():
                    r["mono"] += 1
        print(f"  [{fname}] 스윕 완료  ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n  {'피처':<20s}{'평균이득':>10s}{'4폴드 최소':>10s}{'단조':>6s}")
    ranked = sorted(rows.items(), key=lambda kv: -np.mean(kv[1]["g"]))
    for n, r in ranked[:15]:
        print(f"  {n:<20s}{np.mean(r['g']):>+10.5f}{min(r['g']):>+10.5f}{r['mono']:>5d}/4")

    json.dump(dict(shadow_noise=noise, shadow_borderline=borderline, shadow_runs=runs,
                   oracle={n: dict(g=r["g"], mono=r["mono"]) for n, r in rows.items()}),
              open(os.path.join(C.EXPERIMENTS, "phase17_screen.json"), "w"), indent=1)
    print(f"\n총 {(time.time()-t0)/60:.0f}분")


if __name__ == "__main__":
    main()
