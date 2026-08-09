# -*- coding: utf-8 -*-
"""Phase 19 — 피처 솎기 사다리 2탄. base = v24 (54개). 목표: 1등과의 0.0006.

후보는 전부 11-a 상관 구조에서만 뽑았다 (점수를 보고 고르지 않음 — 사전등록):
    d_last     : sd_lastweek 와 비트 단위 동일 (증명된 중복. 57기반 R1에선 정확히 중립)
    w_median   : w_mean 과 r=0.965 (median 계열의 남은 중복)
    d_posmean  : 요일 군집 4개 중 d_max·d_mean_open 과 r=0.97 (Phase 2가 채택한 d_mean_open 유지)
    w_prev7    : w_mean 과 r=0.969 (⚠ w_trend 의 분모 역할 — 위험 표기)
    d_max      : 요일 군집 잔여 중복

사다리 (중첩 + 분기 1):
    L0  v24 54 (대조군)          L1  −d_last (53)
    L2  L1−w_median (52)         L3  L2−d_posmean (51)
    L4  L3−w_prev7 (50)          L5  L3−d_max (50, 분기)
    L6  L4−d_max (49, 전부)

판정 (사전 고정):
    채택 = 4폴드 평균 > +0.001 AND 4/4 방향 AND 모양이 단조/고원 (한 칸만 튀면 기각 — 규칙 ⑦).
    가장 넓은 고원의 중앙 1개만 확인 제출. 비교 대상 = v24 Private 0.4375952.
    기대: 1탄 전이율 0.63 기준, 내부 +0.001~+0.002 → 실측 +0.0006~+0.0013 (1등 추월권은 상단만).
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
SEEDS = (42, 7, 2024, 913, 31, 101, 202, 303)
PARAMS = dict(objective="regression_l1", metric="l1", learning_rate=0.05,
              num_leaves=127, min_data_in_leaf=40, feature_fraction=0.65,
              bagging_fraction=0.85, bagging_freq=1, lambda_l2=1.0,
              verbosity=-1, num_threads=NT)
ROUNDS = 1000
DROP3 = ["w_posmedian", "w_last14", "w_std"]          # v24 확정 제거분

RUNGS = [
    ("L0 v24 54",        []),
    ("L1 -d_last",       ["d_last"]),
    ("L2 L1-w_median",   ["d_last", "w_median"]),
    ("L3 L2-d_posmean",  ["d_last", "w_median", "d_posmean"]),
    ("L4 L3-w_prev7",    ["d_last", "w_median", "d_posmean", "w_prev7"]),
    ("L5 L3-d_max",      ["d_last", "w_median", "d_posmean", "d_max"]),
    ("L6 L4-d_max",      ["d_last", "w_median", "d_posmean", "w_prev7", "d_max"]),
]

FOLDS = [("F2 겨울", "2023-11-24", "2023-11-24", "2024-02-22"),
         ("F3 봄", "2024-02-23", "2024-02-23", "2024-06-08"),
         ("FAR-봄", "2023-11-24", "2024-02-23", "2024-06-08"),
         ("FAR-겨울", "2023-08-25", "2023-11-24", "2024-02-22")]


def seg_snap(raw):
    p = np.where(raw < 1.8, 0.55 * raw, np.where(raw < 10.0, 0.90 * raw, 1.02 * raw))
    p = np.maximum(p, 1.0)
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
    keep0, names0 = F.active_columns(), F.active_names()
    b_idx = [i for i, n in enumerate(names0) if n not in DROP3]
    base_keep = [keep0[i] for i in b_idx]
    base_names = [names0[i] for i in b_idx]

    print("=" * 100)
    print(f"Phase 19 — 사다리 2탄 · base 54개 · 칸 {len(RUNGS)} · 짝시드 {len(SEEDS)}")
    print("=" * 100, flush=True)

    prepared = []
    for fname, cut, v0, v1 in FOLDS:
        cut_col = int(np.searchsorted(np.array(dates), pd.Timestamp(cut)))
        ctx.set_proxy(F.pick_proxy_items(mat, dates, cut_col, ctx.store_codes))
        trn = [o for o in range(C.WINDOW - 1, nd - C.HORIZON)
               if dates[o] < pd.Timestamp(cut)]
        va = V.origins(dates, v0, v1, nd)
        Xtr, ytr, _ = F.build_samples(mat, dates, trn, ctx)
        Xva, yva, mva = F.build_samples(mat, dates, va, ctx)
        m = ytr != 0
        W, _ = cell_weights(yva, mva[:, 2], ctx.store_of_item, ctx.n)
        prepared.append(dict(name=fname,
                             Xt=np.ascontiguousarray(Xtr[m][:, base_keep]),
                             yt=np.log1p(np.maximum(ytr[m], 1.0)),
                             Xv=np.ascontiguousarray(Xva[:, base_keep]),
                             y=yva, W=W))
        del Xtr, Xva
        print(f"  준비 [{fname}]", flush=True)

    results = {}
    for lab, drop in RUNGS:
        t1 = time.time()
        sub = [i for i, n in enumerate(base_names) if n not in drop]
        names = [base_names[i] for i in sub]
        cats = [c for c in F.CATEGORICAL if c in names]
        per = []
        for f in prepared:
            Xt = np.ascontiguousarray(f["Xt"][:, sub])
            Xv = np.ascontiguousarray(f["Xv"][:, sub])
            ps = []
            for sd in SEEDS:
                ds = lgb.Dataset(Xt, label=f["yt"], feature_name=names,
                                 categorical_feature=cats, free_raw_data=False)
                ps.append(np.expm1(lgb.train(dict(PARAMS, seed=sd), ds,
                                             num_boost_round=ROUNDS).predict(Xv)))
                del ds
            per.append(float((f["W"] * loss(f["y"], seg_snap(np.mean(ps, 0)))).sum()))
            del Xt, Xv
        results[lab] = per
        b = results["L0 v24 54"]
        print(f"  [{lab:<16s}] " + " ".join(f"{x:.5f}" for x in per)
              + f"  평균 {np.mean(per):.5f}"
              + (f"  기준대비 {np.mean(b)-np.mean(per):+.5f}"
                 f"  일관 {sum(x < y for x, y in zip(per, b))}/4"
                 if lab != "L0 v24 54" else "")
              + f"   ({time.time()-t1:.0f}s)", flush=True)

    print("\n" + "=" * 100)
    b = results["L0 v24 54"]
    print(f"  {'칸':<18s}" + "".join(f"{f['name']:>11s}" for f in prepared)
          + f"{'평균':>11s}{'기준대비':>11s}{'일관':>7s}")
    for lab, _ in RUNGS:
        v = results[lab]
        print(f"  {lab:<18s}" + "".join(f"{x:>11.5f}" for x in v)
              + f"{np.mean(v):>11.5f}"
              + (f"{np.mean(b)-np.mean(v):>+11.5f}"
                 f"{sum(x < y for x, y in zip(v, b)):>5d}/4" if lab != RUNGS[0][0]
                 else f"{'—':>11s}{'—':>7s}"))
    print("\n  채택 = 평균 > +0.001 AND 4/4 AND 고원. 고원 중앙 1개만 확인 제출 (vs v24 0.4375952).")

    json.dump({k: v for k, v in results.items()},
              open(os.path.join(C.EXPERIMENTS, "phase19_ladder2.json"), "w"), indent=1)
    print(f"\n총 {(time.time()-t0)/60:.0f}분")


if __name__ == "__main__":
    main()
