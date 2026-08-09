# -*- coding: utf-8 -*-
"""Phase 19-b — L1(−d_last) 확인시드 재측정. 사다리 2탄의 유일한 재심 대상.

왜 L1 만 재심하나 (사전등록):
    · 사다리 2탄은 전 칸 3/4 로 기준(4/4) 미달 — 원칙대로 전부 기각.
    · 단 L1 은 **비트 단위 중복 제거**라 정보 손실이 원리적으로 0 인 유일한 후보.
      57기반 측정 −0.0000, 54기반 측정 +0.0022 — 진위를 가릴 값어치가 있다.
    · L4 등 다른 칸은 재심하지 않는다 (최댓값 고르기 = 선택 편향. L1 은 사전 근거가
      점수가 아니라 구조[비트 동일]에 있어 재심 자격이 다르다).

방법: 사다리 판정에 쓴 적 없는 **새 시드 8개** (505,606,707,808,909,1010,1111,1212).
판정 (사전 고정): 채택 = 새 시드에서 평균 > +0.001 AND 4/4.
    통과 → 53개 전체학습 → v25 확인 제출 (비교 = v24 0.4375952, 기대 +0.0006~+0.0014)
    탈락 → 피처 축 최종 종료. v24 가 최종.
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
SEEDS = (505, 606, 707, 808, 909, 1010, 1111, 1212)
PARAMS = dict(objective="regression_l1", metric="l1", learning_rate=0.05,
              num_leaves=127, min_data_in_leaf=40, feature_fraction=0.65,
              bagging_fraction=0.85, bagging_freq=1, lambda_l2=1.0,
              verbosity=-1, num_threads=NT)
ROUNDS = 1000
DROP3 = ["w_posmedian", "w_last14", "w_std"]

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

    print(f"Phase 19-b — L1 확인시드 재측정 · 새 시드 {SEEDS}", flush=True)
    results = {}
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
        Xt0 = np.ascontiguousarray(Xtr[m][:, base_keep])
        yt = np.log1p(np.maximum(ytr[m], 1.0))
        Xv0 = np.ascontiguousarray(Xva[:, base_keep])
        del Xtr, Xva
        for lab, drop in (("L0", []), ("L1", ["d_last"])):
            sub = [i for i, n in enumerate(base_names) if n not in drop]
            names = [base_names[i] for i in sub]
            cats = [c for c in F.CATEGORICAL if c in names]
            Xt = np.ascontiguousarray(Xt0[:, sub])
            Xv = np.ascontiguousarray(Xv0[:, sub])
            ps = []
            for sd in SEEDS:
                ds = lgb.Dataset(Xt, label=yt, feature_name=names,
                                 categorical_feature=cats, free_raw_data=False)
                ps.append(np.expm1(lgb.train(dict(PARAMS, seed=sd), ds,
                                             num_boost_round=ROUNDS).predict(Xv)))
                del ds
            results[(fname, lab)] = float(
                (W * loss(yva, seg_snap(np.mean(ps, 0)))).sum())
            del Xt, Xv
        g = results[(fname, "L0")] - results[(fname, "L1")]
        print(f"  [{fname:<8s}] L0 {results[(fname,'L0')]:.5f} · "
              f"L1 {results[(fname,'L1')]:.5f} · 개선 {g:+.5f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    gs = [results[(f[0], "L0")] - results[(f[0], "L1")] for f in FOLDS]
    ok = sum(x > 0 for x in gs)
    dec = (np.mean(gs) > 0.001) and (ok == 4)
    print(f"\n  새 시드 판정: 평균 {np.mean(gs):+.5f} · 일관 {ok}/4"
          f"  →  {'✅ 채택 (v25 진행)' if dec else '❌ 탈락 (피처 축 최종 종료, v24 확정)'}")
    json.dump({f"{k[0]}|{k[1]}": v for k, v in results.items()},
              open(os.path.join(C.EXPERIMENTS, "phase19b_confirm.json"), "w"), indent=1)
    print(f"총 {(time.time()-t0)/60:.0f}분")


if __name__ == "__main__":
    main()
