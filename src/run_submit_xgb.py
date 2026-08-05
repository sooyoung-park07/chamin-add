# -*- coding: utf-8 -*-
"""Phase 5-b 후속 — 튜닝된 XGBoost로 제출본 생성.

설정 근거: `run_phase5b_xgb_search.py` 40회 탐색의 상위1.
  확인 시드(2024,913,31) 기준 F2+F3 **0.4680** (같은 시드 무튜닝 기준선 0.4790 대비 +0.0110).
  참고로 v1 LightGBM 최종본은 0.4725.

규약 (log.md 확정 사항):
  - 피처 57개 (`prof` 제외) · 프록시는 학습 데이터 전체로 선정
  - y != 0 행만 학습, 음수는 1로 · 타깃 log1p · 목적함수 L1 · 예측은 expm1
  - **하한 1.0은 시드 평균을 낸 뒤 마지막에 한 번만** (max 는 비선형)
  - 대회 규칙 준수: 각 TEST 파일은 **자기 28일 창만** 보고 독립 추론

⚠️ CV 이득이 F2(겨울)에 몰려 있어 LB에서 그대로 안 나올 수 있다.
   이번 제출의 목적 절반은 **CV↔LB 관측치를 2건으로 늘리는 것**이다(현재 1건).
"""
import os
import json
import time

import numpy as np
import pandas as pd
import xgboost as xgb

import config as C
import dataio as D
import features as F

STAMP = "v3_xgb"
OUT_CSV = os.path.join(C.SUBMISSIONS, f"submission_{STAMP}.csv")

# ---- Phase 5-b 탐색 상위1 (확인 시드에서도 1위 유지) ----
CFG = dict(objective="reg:absoluteerror", tree_method="hist",
           grow_policy="lossguide", max_leaves=255, max_depth=0,
           eta=0.05, min_child_weight=40, subsample=1.0,
           colsample_bytree=0.6, reg_lambda=3.0, reg_alpha=2.0,
           max_cat_to_onehot=1,          # 193레벨 item_id 를 원-핫으로 터뜨리지 않기
           nthread=os.cpu_count())
ROUNDS = 900
SEEDS = (42, 7, 2024, 913, 31)
FLOOR = 1.0
CV_REF = 0.4680                          # 확인 시드 F2+F3 (기록용)


def main():
    t0 = time.time()
    ctx = F.Context()
    tr = D.load_train()
    mat, dates = D.to_matrix(tr, ctx.items)
    nd = mat.shape[1]

    fn_all = F.feature_names()
    keep = [i for i, k in enumerate(fn_all) if k not in set(F.PROF_KEYS)]
    names = [fn_all[i] for i in keep]
    cats = [c for c in F.CATEGORICAL if c in names]

    # ---------- 제출 형식 스모크 테스트 (15분짜리 학습 전에 먼저) ----------
    tpl = D.load_submission_template()
    rk = tpl.columns[0]
    assert list(tpl.columns[1:]) == ctx.items, "제출 템플릿 열 순서 불일치"
    chk = tpl.copy()
    chk[ctx.items] = chk[ctx.items].astype(float)
    for t in range(C.N_TEST):
        for h in range(1, C.HORIZON + 1):
            ridx = chk.index[chk[rk] == f"TEST_{t:02d}+{h}일"]
            assert len(ridx) == 1, f"TEST_{t:02d}+{h}일 행을 못 찾음"
            chk.loc[ridx[0], ctx.items] = 1.23
    assert (chk[ctx.items].values == 1.23).all()
    print(f"제출 형식 스모크 테스트 통과 {chk.shape}")

    # ---------- 학습 행렬 ----------
    # 최종 모델의 프록시는 학습 데이터 전체로 고른다 (검증 구간이 없으므로 누수 아님).
    ctx.set_proxy(F.pick_proxy_items(mat, dates, nd, ctx.store_codes))
    all_origins = list(range(C.WINDOW - 1, nd - C.HORIZON))
    Xa, ya, _ = F.build_samples(mat, dates, all_origins, ctx)
    m = ya != 0
    Xt = Xa[m][:, keep]
    yt = np.log1p(np.maximum(ya[m], 1.0))
    del Xa
    print(f"학습 origin {len(all_origins)} · 유효행 {Xt.shape[0]:,} "
          f"(전체 {len(ya):,} 중) · 피처 {len(names)}")

    # ---------- 테스트 행렬 (범주 레벨을 학습과 합쳐 고정하려면 먼저 다 만든다) ----------
    tests = []
    for t in range(C.N_TEST):
        te = D.load_test(t)
        tmat, tdates = D.to_matrix(te, ctx.items)
        assert tmat.shape == (ctx.n, C.WINDOW), f"TEST_{t:02d} 창 길이 이상"
        X, _, _ = F.build_samples(tmat, tdates, [C.WINDOW - 1], ctx,
                                  with_target=False)
        tests.append(dict(t=t, X=X[:, keep], d0=tdates[0], d1=tdates[-1]))
    Xte_all = np.concatenate([d["X"] for d in tests], 0)

    # ---------- DMatrix ----------
    dft = pd.DataFrame(Xt, columns=names)
    for c in cats:
        j = names.index(c)
        lv = np.unique(np.concatenate([Xt[:, j], Xte_all[:, j]])).astype(int)
        dft[c] = pd.Categorical(dft[c].astype(int), categories=lv)
    lvs = {c: dft[c].cat.categories for c in cats}
    dtr = xgb.DMatrix(dft, label=yt, enable_categorical=True)
    del dft, Xt

    for d in tests:
        df = pd.DataFrame(d["X"], columns=names)
        for c in cats:
            df[c] = pd.Categorical(df[c].astype(int), categories=lvs[c])
        d["dm"] = xgb.DMatrix(df, enable_categorical=True)
        del df

    # ---------- 학습 ----------
    print(f"\n시드 {len(SEEDS)}개 × {ROUNDS}라운드 학습 시작 "
          f"(leaves{CFG['max_leaves']}·{CFG['grow_policy']})")
    models = []
    for i, sd in enumerate(SEEDS):
        t1 = time.time()
        models.append(xgb.train(dict(CFG, seed=sd), dtr, num_boost_round=ROUNDS))
        print(f"  시드 {sd:<5d} {time.time()-t1:5.0f}s", flush=True)

    # ---------- 예측 ----------
    out = tpl.copy()
    out[ctx.items] = out[ctx.items].astype(float)   # pandas 3.0: int열에 float 대입 금지
    print()
    for d in tests:
        # 시드 평균을 먼저, 하한은 그 다음. (max 는 비선형이라 순서가 결과를 바꾼다)
        p = np.mean([np.expm1(mdl.predict(d["dm"])) for mdl in models], axis=0)
        p = np.maximum(p, FLOOR).reshape(C.HORIZON, ctx.n)
        for h in range(1, C.HORIZON + 1):
            ridx = out.index[out[rk] == f"TEST_{d['t']:02d}+{h}일"]
            out.loc[ridx[0], ctx.items] = p[h - 1]
        print(f"  TEST_{d['t']:02d}  창 {d['d0'].date()}~{d['d1'].date()}  →  "
              f"예측 {(d['d1']+pd.Timedelta(days=1)).date()}~"
              f"{(d['d1']+pd.Timedelta(days=7)).date()}")

    out[ctx.items] = out[ctx.items].round(2)
    os.makedirs(C.SUBMISSIONS, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    v = out[ctx.items].values.astype(float)
    assert not np.isnan(v).any(), "NaN 이 들어갔다"
    assert (v >= FLOOR - 1e-9).all(), "하한 미달 값이 있다"
    print(f"\n저장: {OUT_CSV}  {out.shape}")
    print(f"  예측값 min {v.min():.2f} / 중앙값 {np.median(v):.2f} / "
          f"평균 {v.mean():.2f} / max {v.max():.2f}")

    # v1(LightGBM) 제출본과 얼마나 다른지 — 앙상블 여지 가늠용
    v1 = os.path.join(C.SUBMISSIONS, "submission_v1.csv")
    if os.path.exists(v1):
        o1 = pd.read_csv(v1)[ctx.items].values.astype(float)
        if o1.shape == v.shape:
            r = np.corrcoef(np.log1p(o1).ravel(), np.log1p(v).ravel())[0, 1]
            print(f"  v1(LightGBM) 제출본과의 상관 (로그공간) {r:.4f} · "
                  f"평균 절대차 {np.abs(o1-v).mean():.2f}")

    json.dump(dict(stamp=STAMP, model="xgboost", cfg=CFG, rounds=ROUNDS,
                   seeds=list(SEEDS), floor=FLOOR, n_features=len(names),
                   cv_f2f3_confirm_seeds=CV_REF, source="phase5b top1"),
              open(os.path.join(C.EXPERIMENTS, f"config_{STAMP}.json"),
                   "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nCV(F2+F3, 확인시드) = {CV_REF}   [통과선 Public {C.PASS_PUBLIC}]")
    print(f"총 {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
