# -*- coding: utf-8 -*-
"""Phase 17-b — 그림자 대조 재실행 (문턱 수정판).

17 의 결함: 문턱을 '그림자 전체의 최대 gain'으로 잡았는데, 셔플된 고카디널리티 범주형
(item_id 193레벨 등)이 잡음인 채로 거대한 gain 을 쓸어담아 문턱이 천장이 됐다 (55/57 미달).
gain 은 학습손실 감소량이라 레벨 많은 범주형은 잡음이어도 크게 나온다 — 규칙 ⑨의 재판.

수정 (사전 고정):
    문턱 A = 자기 자신의 그림자 (같은 열을 셔플한 것) — 일대일 비교
    문턱 B = **수치형** 그림자들의 최대 — 유형이 같은 귀무분포
    '잡음 후보' = 6회(2폴드×시드3) 전부에서 A 와 B 를 **둘 다** 못 넘는 피처.
    범주형 진짜 피처는 자기 그림자(A)로만 판정 (B 는 유형이 달라 부적용).
이번엔 gain 원자료를 json 으로 저장해 사후 재분석 가능하게 한다.
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

NT = os.cpu_count()
PARAMS = dict(objective="regression_l1", metric="l1", learning_rate=0.05,
              num_leaves=127, min_data_in_leaf=40, feature_fraction=0.65,
              bagging_fraction=0.85, bagging_freq=1, lambda_l2=1.0,
              verbosity=-1, num_threads=NT)
ROUNDS = 1000
SEEDS = (42, 7, 2024)
FOLDS = [("F2 겨울", "2023-11-24"), ("F3 봄", "2024-02-23")]


def main():
    t0 = time.time()
    ctx = F.Context()
    tr = D.load_train()
    mat, dates = D.to_matrix(tr, ctx.items)
    nd = mat.shape[1]
    keep, names = F.active_columns(), F.active_names()
    cats = set(c for c in F.CATEGORICAL if c in names)

    all_gains = []
    failA = {n: 0 for n in names}
    failB = {n: 0 for n in names}
    runs = 0
    for fname, cut in FOLDS:
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
        c2 = [c for c in names if c in cats] + [c + "_sh" for c in names if c in cats]
        for sd in SEEDS:
            ds = lgb.Dataset(X2, label=yt, feature_name=n2,
                             categorical_feature=c2, free_raw_data=False)
            md = lgb.train(dict(PARAMS, seed=sd), ds, num_boost_round=ROUNDS)
            g = np.array(md.feature_importance("gain"))
            real, shadow = g[:len(names)], g[len(names):]
            barB = max(shadow[j] for j, n in enumerate(names) if n not in cats)
            runs += 1
            for j, n in enumerate(names):
                if real[j] < shadow[j]:
                    failA[n] += 1
                if n not in cats and real[j] < barB:
                    failB[n] += 1
            all_gains.append(dict(fold=fname, seed=sd,
                                  real=real.tolist(), shadow=shadow.tolist()))
            del ds, md
        print(f"  [{fname}] 완료 ({time.time()-t0:.0f}s)", flush=True)

    print("\n" + "=" * 92)
    print(f"판정 (총 {runs}회) — A: 자기 그림자 미달 횟수 / B: 수치형 그림자최대 미달 횟수")
    print("=" * 92)
    noise = []
    for n in sorted(names):
        a, b = failA[n], failB[n]
        is_cat = n in cats
        bad = (a == runs) if is_cat else (a == runs and b == runs)
        if bad:
            noise.append(n)
        if a > 0 or b > 0:
            print(f"  {n:<20s} A {a}/{runs}" + ("" if is_cat else f" · B {b}/{runs}")
                  + ("   ← 잡음 후보" if bad else ""))
    print(f"\n  잡음 후보 (전 회차 미달): {len(noise)}개 → {noise}")

    json.dump(dict(noise=noise, failA=failA, failB=failB, runs=runs, gains=all_gains),
              open(os.path.join(C.EXPERIMENTS, "phase17b_shadow.json"), "w"), indent=1)
    print(f"\n총 {(time.time()-t0)/60:.0f}분")


if __name__ == "__main__":
    main()
