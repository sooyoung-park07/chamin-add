# -*- coding: utf-8 -*-
"""Phase 10-e — 시드 5 → 20. **점수보다 저울을 깎는 게 목적이다.**

동기
    Phase 10-d 에서 σ(시드5 평균) ≈ 0.0014(내부) 로 추정됐다 → 실제 2σ̂ ≈ 0.002~0.003.
    그런데 남은 피처 후보들의 기대가 +0.001~0.004 라 **문턱과 겹친다.**
    시드를 20으로 올리면 σ 가 √(20/5)=2배 줄어 문턱이 0.001~0.0015 로 내려가고,
    **그때부터 피처 실험이 판정 가능해진다.**
    점수 자체의 이득은 1/n 외삽으로 +0.0004 정도로 작다 — 목적은 계측기지 점수가 아니다.

저장 방식 — **시드별로 따로 저장한다** (평균 내지 않는다)
    나중에 어떤 시드 부분집합의 평균이든 재학습 없이 즉시 계산할 수 있다.
    σ(n) 곡선을 그려 "몇 개면 충분한가"를 데이터로 답할 수 있고,
    짝시드 비교에도 그대로 쓴다.

⚠️ 이 스크립트는 2026-08-04 에 고친 두 결함을 전제로 한다
    · `model_lgbm.py:49`  범주형 목록을 실제 열로 필터
    · `run_phase6a_calibration.py:70`  `F.active_columns()` 사용
    직접 `PROF_KEYS` 로 거르지 말 것 — 새 그룹이 기각될 때마다 조용히 낡는다.
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

NT = os.cpu_count()

# 앞 5개는 기존 v8/v17 과 동일 — 부분집합으로 기존 결과를 재현해 대조할 수 있다.
SEEDS = (42, 7, 2024, 913, 31,
         101, 202, 303, 404, 505,
         606, 707, 808, 909, 1010,
         1111, 1212, 1313, 1414, 1515)

# v8 설정 (현 챔피언 v17 의 LightGBM 쪽)
PARAMS = dict(objective="regression_l1", metric="l1", learning_rate=0.05,
              num_leaves=127, min_data_in_leaf=40, feature_fraction=0.65,
              bagging_fraction=0.85, bagging_freq=1, lambda_l2=1.0,
              verbosity=-1, num_threads=NT)
ROUNDS = 1000

FOLDS = [
    ("F2 겨울",  "2023-11-24", "2023-11-24", "2024-02-22", "가까움"),
    ("F3 봄",    "2024-02-23", "2024-02-23", "2024-06-08", "가까움"),
    ("FAR-봄",   "2023-11-24", "2024-02-23", "2024-06-08", "멂"),
    ("FAR-겨울", "2023-08-25", "2023-11-24", "2024-02-22", "멂"),
]

OOF_NPZ = os.path.join(C.EXPERIMENTS, "phase10e_oof20.npz")
TEST_NPY = os.path.join(C.EXPERIMENTS, "phase10e_test20.npy")


def main():
    t0 = time.time()
    ctx = F.Context()
    tr = D.load_train()
    mat, dates = D.to_matrix(tr, ctx.items)
    nd = mat.shape[1]
    keep, names = F.active_columns(), F.active_names()
    cats = [c for c in F.CATEGORICAL if c in names]

    print("=" * 98)
    print(f"Phase 10-e — 시드 {len(SEEDS)}개 · v8 설정 · {ROUNDS}라운드 · 피처 {len(names)}개")
    print("=" * 98, flush=True)

    # ============================================================ PART 1 : OOF
    store = {}
    for i, (fname, cut, v0, v1_, kind) in enumerate(FOLDS):
        t1 = time.time()
        cut_col = int(np.searchsorted(np.array(dates), pd.Timestamp(cut)))
        ctx.set_proxy(F.pick_proxy_items(mat, dates, cut_col, ctx.store_codes))
        trn = [o for o in range(C.WINDOW - 1, nd - C.HORIZON)
               if dates[o] < pd.Timestamp(cut)]
        va = V.origins(dates, v0, v1_, nd)
        Xtr, ytr, _ = F.build_samples(mat, dates, trn, ctx)
        Xva, yva, mva = F.build_samples(mat, dates, va, ctx)
        m = ytr != 0
        Xt = np.ascontiguousarray(Xtr[m][:, keep])
        yt = np.log1p(np.maximum(ytr[m], 1.0))
        Xv = np.ascontiguousarray(Xva[:, keep])
        del Xtr, Xva

        ps = np.empty((len(SEEDS), len(yva)), dtype=np.float32)
        for j, sd in enumerate(SEEDS):
            ds = lgb.Dataset(Xt, label=yt, feature_name=names,
                             categorical_feature=cats, free_raw_data=False)
            ps[j] = np.expm1(lgb.train(dict(PARAMS, seed=sd), ds,
                                       num_boost_round=ROUNDS).predict(Xv))
            if (j + 1) % 5 == 0:
                print(f"    [{fname}] 시드 {j+1}/{len(SEEDS)} "
                      f"({time.time()-t1:.0f}s)", flush=True)
            del ds
        store[f"ps{i}"] = ps
        store[f"y{i}"] = yva
        store[f"i{i}"] = mva[:, 2]
        store[f"h{i}"] = mva[:, 1]
        print(f"  [{fname:<8s}] {kind:<4s} 학습 {Xt.shape[0]:>7,}행 · "
              f"검증 {len(yva):>6,}칸 · {time.time()-t1:.0f}s", flush=True)
        del Xt, Xv

    np.savez_compressed(OOF_NPZ, **store)
    print(f"\n저장: {os.path.basename(OOF_NPZ)}  (시드별 pre-floor 예측)")

    # ============================================================ PART 2 : TEST
    print("\n" + "=" * 98)
    print("전체학습 → 테스트 예측 (시드별로 저장)")
    print("=" * 98, flush=True)
    ctx.set_proxy(F.pick_proxy_items(mat, dates, nd, ctx.store_codes))
    Xa, ya, _ = F.build_samples(mat, dates,
                                list(range(C.WINDOW - 1, nd - C.HORIZON)), ctx)
    m = ya != 0
    Xt = np.ascontiguousarray(Xa[m][:, keep])
    yt = np.log1p(np.maximum(ya[m], 1.0))
    del Xa
    tests = []
    for t in range(C.N_TEST):
        te = D.load_test(t)
        tmat, tdates = D.to_matrix(te, ctx.items)
        X, _, _ = F.build_samples(tmat, tdates, [C.WINDOW - 1], ctx,
                                  with_target=False)
        tests.append(X[:, keep])
    Xte = np.concatenate(tests, 0)

    t1 = time.time()
    P = np.empty((len(SEEDS), Xte.shape[0]), dtype=np.float32)
    for j, sd in enumerate(SEEDS):
        ds = lgb.Dataset(Xt, label=yt, feature_name=names,
                         categorical_feature=cats, free_raw_data=False)
        P[j] = np.expm1(lgb.train(dict(PARAMS, seed=sd), ds,
                                  num_boost_round=ROUNDS).predict(Xte))
        print(f"    시드 {j+1}/{len(SEEDS)} ({time.time()-t1:.0f}s)", flush=True)
        del ds
    np.save(TEST_NPY, P)
    print(f"\n저장: {os.path.basename(TEST_NPY)}  shape {P.shape}")

    # 기존 5시드 재현 대조 — 앞 5개가 phase6a_test_raw.npy 와 같아야 한다
    old = os.path.join(C.EXPERIMENTS, "phase6a_test_raw.npy")
    if os.path.exists(old):
        a, b = P[:5].mean(0), np.load(old)
        rel = float(np.abs(a - b).mean() / b.mean())
        print(f"\n  [재현 대조] 앞 5시드 평균 vs phase6a_test_raw.npy")
        print(f"    평균 상대차 {100*rel:.4f}%  · 로그공간 상관 "
              f"{np.corrcoef(np.log1p(a), np.log1p(b))[0,1]:.6f}")
        print("    → 거의 0 이어야 정상(같은 설정·같은 시드). 크게 다르면 파이프라인이 바뀐 것.")

    for n in (1, 2, 5, 10, 15, 20):
        print(f"    시드{n:>2d} 평균 → 1.818 미만 비율 "
              f"{100*(P[:n].mean(0) < 1.818).mean():.2f}%")

    json.dump(dict(seeds=list(SEEDS), params=PARAMS, rounds=ROUNDS,
                   n_features=len(names), minutes=(time.time() - t0) / 60),
              open(os.path.join(C.EXPERIMENTS, "phase10e_seeds20.json"),
                   "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n총 {(time.time()-t0)/60:.0f}분")


if __name__ == "__main__":
    main()
