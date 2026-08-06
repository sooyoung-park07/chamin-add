# -*- coding: utf-8 -*-
"""Phase 18 — 피처 선별 프로그램의 확인 제출본: R5 (창 중복 3개 제거) 전체학습.

경위 (선택은 전부 내부에서 완료 — 확인 제출 1회만 남음):
    · 그림자 대조(17-b): 잡음 피처 0개 — 모든 피처가 자기 귀무를 이김
    · 오라클 스윕(17): 단조 4/4 인 '덜 쓰인 축' 0개
    · 남은 유일한 통과 후보 = Phase 11-b 사다리의 R5:
      `w_posmedian · w_last14 · w_std` 제거 (상관 0.95~0.98 중복) → 내부 +0.0023 · 4/4
    ⇒ 그림자(전부 유의)와 사다리(빼면 이득)가 모순이 아님을 주의 —
      그림자는 "신호가 있는가", 사다리는 "형제 피처가 이미 가진 신호인가"를 잰다.
      중복의 비용(선택 확률 희석·분기 제비뽑기)은 후자만 본다.

사전등록:
    비교 대상 = v20 (같은 파이프라인·같은 시드5·LGBM 단독·같은 후처리) Private 0.4390529.
    예상: 내부 +0.0023 의 전이율 0~1.0 → Private 0.4368 ~ 0.4391. 단 R5 이득이
    겨울 편중(F2 +0.0030 / FAR-겨울 +0.0062 / 봄 ~0)이라 규칙 ③ 전례상 하단 확률이 높다.
    판정: v20 보다 좋으면 채택, 아니면 피처 축 완전 종료.
"""
import os
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

import config as C
import dataio as D
import features as F

NT = os.cpu_count()
DROP = ["w_posmedian", "w_last14", "w_std"]
SEEDS = (42, 7, 2024, 913, 31)
PARAMS = dict(objective="regression_l1", metric="l1", learning_rate=0.05,
              num_leaves=127, min_data_in_leaf=40, feature_fraction=0.65,
              bagging_fraction=0.85, bagging_freq=1, lambda_l2=1.0,
              verbosity=-1, num_threads=NT)
ROUNDS = 1000
OUT = os.path.join(C.EXPERIMENTS, "phase18_prune3_raw.npy")


def main():
    t0 = time.time()
    ctx = F.Context()
    tr = D.load_train()
    mat, dates = D.to_matrix(tr, ctx.items)
    nd = mat.shape[1]
    base_keep, base_names = F.active_columns(), F.active_names()
    sub = [i for i, n in enumerate(base_names) if n not in DROP]
    keep = [base_keep[i] for i in sub]
    names = [base_names[i] for i in sub]
    cats = [c for c in F.CATEGORICAL if c in names]
    print(f"피처 {len(base_names)} → {len(names)} (제거: {', '.join(DROP)})", flush=True)

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
        X, _, _ = F.build_samples(tmat, tdates, [C.WINDOW - 1], ctx, with_target=False)
        tests.append(X[:, keep])
    Xte = np.concatenate(tests, 0)
    print(f"학습 {Xt.shape[0]:,}행 · 테스트 {Xte.shape[0]:,}칸", flush=True)

    P = []
    for sd in SEEDS:
        ds = lgb.Dataset(Xt, label=yt, feature_name=names,
                         categorical_feature=cats, free_raw_data=False)
        P.append(np.expm1(lgb.train(dict(PARAMS, seed=sd), ds,
                                    num_boost_round=ROUNDS).predict(Xte)))
        print(f"  시드 {sd} 완료 ({time.time()-t0:.0f}s)", flush=True)
        del ds
    raw = np.mean(P, 0)
    np.save(OUT, raw)
    print(f"저장: {os.path.basename(OUT)} · min {raw.min():.3f} / 중앙 {np.median(raw):.2f}"
          f" / max {raw.max():.1f} · 1.818미만 {100*(raw < 1.818).mean():.2f}%")
    print("다음: python make_submission.py v24_prune3 --raw "
          f"{OUT} --seg 0.55,0.90,1.02 --t 1.8,10 --snap geom")
    print(f"총 {(time.time()-t0)/60:.0f}분")


if __name__ == "__main__":
    main()
