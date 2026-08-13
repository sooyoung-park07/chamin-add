# -*- coding: utf-8 -*-
"""Tier B — TB4 채택 확인 제출본: v24(54피처) + 네이버 검색어트렌드(tb_naver) = 56피처 전체학습.

`run_tb3_submit.py`와 동일한 절차(시드5·라운드1000·L1)에 tb_naver만 더한다.
⚠️ Tier B — 대회 규정 밖(외부데이터). v24 계열 번호와 분리, `x1`은 TB3가 이미 썼으니 `x2`를 쓴다.
⚠️ 예측 대상일(td) 당일의 실측 검색량을 쓰는 상한선 실험 — TB3(x1_visit)와 같은 성격.
   테스트 대상일이 naver_trend_mom.csv 수집 범위(2023-01-01~2025-06-10) 안에 있어야 함.
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
OUT = os.path.join(C.EXPERIMENTS, "phase_tb4_naver_x2_raw.npy")


def main():
    t0 = time.time()
    ctx = F.Context()
    tr = D.load_train()
    mat, dates = D.to_matrix(tr, ctx.items)
    nd = mat.shape[1]
    base_keep = F.active_columns(include=("tb_naver",))
    base_names = F.active_names(include=("tb_naver",))
    sub = [i for i, n in enumerate(base_names) if n not in DROP]
    keep = [base_keep[i] for i in sub]
    names = [base_names[i] for i in sub]
    cats = [c for c in F.CATEGORICAL if c in names]
    print(f"피처 {len(names)}개 (v24 54 + tb_naver 2): {', '.join(F.TB_NAVER_KEYS)}", flush=True)

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
          f" / max {raw.max():.1f} · 1.8미만 {100*(raw < 1.8).mean():.2f}%")
    print("다음: python make_submission.py x2_naver --raw "
          f"{OUT} --seg 0.55,0.90,1.02 --t 1.8,10 --snap geom")
    print(f"총 {(time.time()-t0)/60:.0f}분")


if __name__ == "__main__":
    main()