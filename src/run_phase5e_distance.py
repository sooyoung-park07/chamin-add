# -*- coding: utf-8 -*-
"""Phase 5-e — 저울 고치기: **예측 거리**가 모형 용량 선택을 뒤집는가.

배경(log.md Phase 5-e): 제출 4건에서 CV 순위와 LB 순위가 거의 뒤집혔고,
CV↔LB 격차가 **모형 용량과 함께 단조 증가**했다(+0.009 → +0.023).
가설: **CV는 학습 직후 0~3개월을 재는데 실제 채점은 0~11개월 뒤로 이뤄진다.**
용량이 큰 모델일수록 "창통계→다음주 판매량" 관계를 학습기간에 세밀히 맞추므로
멀어질수록 더 크게 무너진다. 지금 저울로는 이게 안 보인다.

━━━ 설계에서 가장 중요한 것: 교란 요인 차단 ━━━
단순히 "가까운 검증 / 먼 검증"으로 나누면 **거리와 계절이 섞인다**(가까움=겨울, 멂=봄).
그건 F2/F3가 이미 가진 결함이다. 그래서:

    검증 구간을 **봄으로 고정** (2024-02-23 ~ 2024-06-08, F3와 동일)
    학습 origin 수를 **300개로 동일**하게 맞춤
    학습 창만 밀어서 **거리만** 바꾼다

      NEAR : 학습 origin 마지막 300개, 2024-02-22 까지  → 검증까지 거리 0
      FAR  : 학습 origin 앞쪽  300개, 2023-11-23 까지  → 검증까지 거리 ~3개월

학습량이 같고 검증 구간이 같으므로, 차이는 **거리 하나**다.
(잔여 교란: NEAR 학습구간은 겨울을 포함하고 FAR는 아니다. 완전 분리는 데이터 1.5년으로 불가능.)

측정: 용량 사다리(leaves 63→511)에서 **FAR − NEAR 열화폭**이 용량과 함께 커지는가.
커지면 가설 확정 → 앞으로 판정 지표에 '먼 검증'을 포함시킨다.
"""
import os
import json
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

import config as C
import dataio as D
import features as F
import validate as V
from metrics import competition_score, make_weights

SEEDS = (42, 7)
FLOOR = 1.0
NT = os.cpu_count()

VAL_D0, VAL_D1 = "2024-02-23", "2024-06-08"      # F3 검증 구간 = 봄, 고정
NEAR_CUT = "2024-02-23"                          # 이 날짜 **미만** origin 으로 학습
FAR_CUT = "2023-11-24"

# 용량 사다리 — v1 의 나머지 파라미터를 고정하고 num_leaves 만 바꾼다 (용량 축 분리)
LGB_COMMON = dict(objective="regression_l1", metric="l1", learning_rate=0.05,
                  min_data_in_leaf=40, feature_fraction=0.85,
                  bagging_fraction=0.85, bagging_freq=1, lambda_l2=1.0,
                  verbosity=-1, num_threads=NT)
CONFIGS = [
    ("LGB leaves63",       dict(LGB_COMMON, num_leaves=63), 1000, "lgb"),
    ("LGB leaves127",      dict(LGB_COMMON, num_leaves=127), 1000, "lgb"),
    ("LGB leaves255 (=v1)", dict(LGB_COMMON, num_leaves=255), 1000, "lgb"),
    ("LGB leaves511",      dict(LGB_COMMON, num_leaves=511), 1000, "lgb"),
    # 실제 제출에 쓴 두 설정
    ("v4 재튜닝(공격적)", dict(objective="regression_l1", metric="l1",
                            learning_rate=0.05, num_leaves=511,
                            min_data_in_leaf=10, feature_fraction=0.65,
                            bagging_fraction=1.0, bagging_freq=0,
                            lambda_l2=3.0, lambda_l1=2.0,
                            verbosity=-1, num_threads=NT), 1000, "lgb"),
    ("v3 XGBoost 튜닝", dict(objective="reg:absoluteerror", tree_method="hist",
                            grow_policy="lossguide", max_leaves=255, max_depth=0,
                            eta=0.05, min_child_weight=40, subsample=1.0,
                            colsample_bytree=0.6, reg_lambda=3.0, reg_alpha=2.0,
                            max_cat_to_onehot=1, nthread=NT), 900, "xgb"),
]


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

    print("=" * 100)
    print("Phase 5-e — 예측 거리가 모형 용량 선택을 뒤집는가")
    print("=" * 100)

    # 검증 구간 고정. 프록시는 **FAR 학습 끝**까지로 통일 — 두 설정에 같은 프록시를 준다
    # (프록시가 달라지면 거리 외의 차이가 생긴다).
    far_cut_col = int(np.searchsorted(np.array(dates), pd.Timestamp(FAR_CUT)))
    ctx.set_proxy(F.pick_proxy_items(mat, dates, far_cut_col, ctx.store_codes))

    va = V.origins(dates, VAL_D0, VAL_D1, nd)
    Xva, yva, mva = F.build_samples(mat, dates, va, ctx)
    Xv = np.ascontiguousarray(Xva[:, keep])
    iids = mva[:, 2]
    del Xva

    allo = [o for o in range(C.WINDOW - 1, nd - C.HORIZON)]
    far_tr = [o for o in allo if dates[o] < pd.Timestamp(FAR_CUT)]
    near_all = [o for o in allo if dates[o] < pd.Timestamp(NEAR_CUT)]
    n = len(far_tr)
    near_tr = near_all[-n:]                      # 크기를 FAR 와 동일하게 맞춘다

    print(f"  검증 구간(고정)  {VAL_D0} ~ {VAL_D1} · origin {len(va)}개 · "
          f"검증행 {Xv.shape[0]:,}")
    print(f"  NEAR 학습  {dates[near_tr[0]].date()} ~ {dates[near_tr[-1]].date()}"
          f"  origin {len(near_tr)}  · 검증까지 거리 0개월")
    print(f"  FAR  학습  {dates[far_tr[0]].date()} ~ {dates[far_tr[-1]].date()}"
          f"  origin {len(far_tr)}  · 검증까지 거리 ~3개월")
    assert len(near_tr) == len(far_tr), "학습량이 달라지면 실험이 무의미하다"

    raw = {}
    for tag, trn in [("NEAR", near_tr), ("FAR", far_tr)]:
        Xtr, ytr, _ = F.build_samples(mat, dates, trn, ctx)
        m = ytr != 0
        raw[tag] = (Xtr[m][:, keep], np.log1p(np.maximum(ytr[m], 1.0)))
        del Xtr
        print(f"  {tag} 학습행(원본) {raw[tag][0].shape[0]:,}")

    # ⚠️ origin 수를 맞춰도 **y!=0 필터 후 행 수는 다르다** (NEAR 쪽이 성수기를 더 많이 품음).
    #   행이 15% 많으면 NEAR 가 그 이유만으로 유리해져 거리 효과가 부풀려진다.
    #   → 적은 쪽에 맞춰 무작위 다운샘플. 트리는 행을 독립으로 보므로 분포가 보존된다.
    k = min(raw["NEAR"][0].shape[0], raw["FAR"][0].shape[0])
    rng = np.random.default_rng(0)
    sets = {}
    for tag in ("NEAR", "FAR"):
        X, y = raw[tag]
        if X.shape[0] > k:
            sel = np.sort(rng.choice(X.shape[0], k, replace=False))
            X, y = X[sel], y[sel]
        sets[tag] = (np.ascontiguousarray(X), y)
    del raw
    print(f"  → 두 설정 모두 {k:,} 행으로 맞춤 (학습량 동일 · 차이는 거리뿐)\n")

    def score(p):
        return competition_score(yva, np.maximum(p, FLOOR), iids,
                                 ctx.store_of_item, make_weights(1.0), ctx.n)

    def run(kind, params, rounds, Xt, yt):
        if kind == "lgb":
            ds = lgb.Dataset(Xt, label=yt, feature_name=names,
                             categorical_feature=cats, free_raw_data=False)
            return np.mean([np.expm1(lgb.train(dict(params, seed=sd), ds,
                                               num_boost_round=rounds).predict(Xv))
                            for sd in SEEDS], 0)
        lv = {c: np.unique(np.concatenate([Xt[:, names.index(c)],
                                           Xv[:, names.index(c)]])).astype(int)
              for c in cats}

        def frame(X):
            df = pd.DataFrame(X, columns=names)
            for c in cats:
                df[c] = pd.Categorical(df[c].astype(int), categories=lv[c])
            return df
        dtr = xgb.DMatrix(frame(Xt), label=yt, enable_categorical=True)
        dva = xgb.DMatrix(frame(Xv), enable_categorical=True)
        return np.mean([np.expm1(xgb.train(dict(params, seed=sd), dtr,
                                           num_boost_round=rounds).predict(dva))
                        for sd in SEEDS], 0)

    rows = []
    print(f"  {'설정':<22s} {'NEAR':>9s} {'FAR':>9s} {'열화 (FAR−NEAR)':>17s} {'시간':>7s}")
    for label, params, rounds, kind in CONFIGS:
        t1 = time.time()
        sn = score(run(kind, params, rounds, *sets["NEAR"]))
        sf = score(run(kind, params, rounds, *sets["FAR"]))
        rows.append(dict(label=label, near=sn, far=sf, decay=sf - sn))
        print(f"  {label:<22s} {sn:>9.4f} {sf:>9.4f} {sf-sn:>+17.4f} "
              f"{time.time()-t1:>6.0f}s", flush=True)

    print("\n" + "=" * 100)
    print("판정 — 용량이 클수록 멀어질 때 더 무너지는가")
    print("=" * 100)
    lad = [r for r in rows if r["label"].startswith("LGB leaves")]
    print("\n  [용량 사다리 — num_leaves 만 바꿈]")
    print(f"  {'':<22s} {'NEAR 순위':>10s} {'FAR 순위':>10s}")
    rn = {r["label"]: i + 1 for i, r in enumerate(sorted(lad, key=lambda r: r["near"]))}
    rf = {r["label"]: i + 1 for i, r in enumerate(sorted(lad, key=lambda r: r["far"]))}
    for r in lad:
        print(f"  {r['label']:<22s} {rn[r['label']]:>10d} {rf[r['label']]:>10d}")

    dec = [r["decay"] for r in lad]
    lv_ = [63, 127, 255, 511]
    corr = float(np.corrcoef(np.log2(lv_), dec)[0, 1])
    print(f"\n  log2(잎 개수) 와 열화폭의 상관 = {corr:+.3f}")
    print(f"  최소 용량(63) 열화 {dec[0]:+.4f}  →  최대 용량(511) 열화 {dec[-1]:+.4f}")
    if corr > 0.7 and dec[-1] > dec[0]:
        print("  → ★ 가설 확정: **용량이 클수록 멀어질 때 더 무너진다.**")
        print("     지금 CV(거리 0~3개월)는 이 열화를 못 본다. 판정 지표에 FAR 를 넣어야 한다.")
    elif corr < -0.7:
        print("  → 반대 방향. 가설 기각 — 용량은 거리 열화의 원인이 아니다.")
    else:
        print("  → 판정 불가. 열화폭이 용량과 뚜렷한 관계를 안 보인다. 다른 원인을 찾아야 한다.")

    print("\n  [실제 제출 설정 비교]")
    v1 = [r for r in rows if "leaves255" in r["label"]][0]
    v4 = [r for r in rows if r["label"].startswith("v4")][0]
    v3 = [r for r in rows if r["label"].startswith("v3")][0]
    print(f"  {'':<22s} {'NEAR':>9s} {'FAR':>9s}")
    for r in (v1, v4, v3):
        print(f"  {r['label']:<22s} {r['near']:>9.4f} {r['far']:>9.4f}")
    print(f"\n  v4 − v1 :  NEAR {v4['near']-v1['near']:+.4f}  →  FAR {v4['far']-v1['far']:+.4f}")
    if v4["near"] < v1["near"] and v4["far"] > v1["far"]:
        print("  → ★★ **부호가 뒤집힌다.** 가까이선 v4 가 이기고 멀리선 v1 이 이긴다.")
        print("     LB에서 v4(0.487)가 v1(0.4818)보다 나빴던 것과 정확히 일치한다.")
    print(f"  v3 − v1 :  NEAR {v3['near']-v1['near']:+.4f}  →  FAR {v3['far']-v1['far']:+.4f}")

    json.dump(dict(val=[VAL_D0, VAL_D1], n_train=len(far_tr),
                   near_cut=NEAR_CUT, far_cut=FAR_CUT, seeds=list(SEEDS),
                   rows=rows, capacity_corr=corr),
              open(os.path.join(C.EXPERIMENTS, "phase5e_distance.json"),
                   "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n저장: experiments/phase5e_distance.json")
    print(f"총 {(time.time()-t0)/60:.0f}분")


if __name__ == "__main__":
    main()
