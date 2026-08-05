# -*- coding: utf-8 -*-
"""Phase 2 — 전처리 A/B. 모든 판단을 공식 지표 CV로.

설계:
  · 스크리닝은 가벼운 설정(leaves127·round600·시드2)으로. 상대 비교가 목적.
  · 피처 행렬은 변형별로 1번만 만들고, 열 마스킹으로 되는 실험은 재사용.
  · 판단 지표 = 공식지표 3폴드 평균(상대비교) + F2+F3 평균(LB 대리).
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
import preprocess as P
import validate as V
from metrics import competition_score, make_weights

SCREEN = dict(objective="regression_l1", metric="l1", learning_rate=0.05,
              num_leaves=127, min_data_in_leaf=40, feature_fraction=0.85,
              bagging_fraction=0.85, bagging_freq=1, lambda_l2=1.0,
              verbosity=-1, num_threads=os.cpu_count())
ROUNDS = 600
SEEDS = (42, 7)
FLOOR = 1.0

RESULTS = []


def evaluate(tag, fold_data, drop_cols=(), transform=None, note=""):
    """fold_data: 폴드별 (Xtr,ytr,Xva,yva,iids). 열 제거/변환 후 학습·평가."""
    fn = F.feature_names()
    keep = [i for i, k in enumerate(fn) if k not in drop_cols]
    names = [fn[i] for i in keep]
    cats = [c for c in F.CATEGORICAL if c in names]
    scores = []
    t0 = time.time()
    for fd in fold_data:
        Xt, Xv = fd["Xtr"][:, keep], fd["Xva"][:, keep]
        if transform is not None:
            Xt, Xv = transform(Xt.copy(), names), transform(Xv.copy(), names)
        m = fd["ytr"] > 0
        preds = []
        for sd in SEEDS:
            ds = lgb.Dataset(Xt[m], label=np.log1p(fd["ytr"][m]),
                             feature_name=names, categorical_feature=cats,
                             free_raw_data=False)
            mdl = lgb.train(dict(SCREEN, seed=sd), ds, num_boost_round=ROUNDS)
            preds.append(np.expm1(mdl.predict(Xv)))
        p = np.maximum(np.mean(preds, 0), FLOOR)
        scores.append(competition_score(fd["yva"], p, fd["iids"],
                                        fd["store_of"], make_weights(1.0), fd["n"]))
    m3, m23 = float(np.mean(scores)), float(np.mean(scores[1:]))
    RESULTS.append(dict(tag=tag, cv3=m3, cv23=m23,
                        folds=[float(s) for s in scores], note=note))
    print(f"  {tag:<34s} 3폴드 {m3:.4f} | F2+F3 {m23:.4f}  "
          f"({' '.join(f'{s:.4f}' for s in scores)})  [{time.time()-t0:.0f}s]")
    return m3


def build_folds(mat, dates, ctx, cap_q=None, cap_target=False):
    """폴드별 학습/검증 행렬 생성. cap_q가 있으면 폴드 학습구간에서 캡 산출."""
    nd = mat.shape[1]
    out = []
    for name, d0, d1 in V.FOLDS:
        va = V.origins(dates, d0, d1, nd)
        trn = V.train_origins(dates, d0, nd)
        fmat, tmat = mat, None
        if cap_q is not None:
            cut = int(np.searchsorted(np.array(dates), pd.Timestamp(d0)))
            caps = P.item_caps(mat, cut, q=cap_q)
            fmat = P.apply_caps(mat, caps)
            tmat = fmat if cap_target else mat      # 타깃도 캡할지 여부
        Xtr, ytr, _ = F.build_samples(fmat, dates, trn, ctx, target_mat=tmat)
        Xva, yva, mva = F.build_samples(fmat, dates, va, ctx, target_mat=mat)
        out.append(dict(name=name, Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva,
                        iids=mva[:, 2], store_of=ctx.store_of_item, n=ctx.n))
    return out


def main():
    ctx = F.Context()
    tr = D.load_train()
    mat, dates = D.to_matrix(tr, ctx.items)
    fn = F.feature_names()
    print(f"피처 {len(fn)}개 (휴점 그룹 {len(F.CLOSED_KEYS)}개 신규 포함)\n")

    # ---------------- 기본 행렬 (캡 없음) ----------------
    t0 = time.time()
    base = build_folds(mat, dates, ctx)
    print(f"기본 피처 행렬 생성 완료 [{time.time()-t0:.0f}s]\n")

    print("=" * 84)
    print("A. 휴점일 보정 피처")
    print("=" * 84)
    evaluate("A0 없음(기존)", base, drop_cols=set(F.CLOSED_KEYS),
             note="휴점 피처 제외 = v1과 동일 조건")
    evaluate("A1 휴점 보정 추가", base, note="st_closed_ratio 등 6개 추가")

    print()
    print("=" * 84)
    print("C. price 피처 (범주화 검토)")
    print("=" * 84)

    def to_band(X, names):
        j = names.index("price")
        edges = [3000, 8000, 20000, 50000]
        X[:, j] = np.digitize(X[:, j], edges).astype(np.float32)
        return X

    def to_log(X, names):
        j = names.index("price")
        X[:, j] = np.log1p(X[:, j])
        return X

    evaluate("C0 원본 가격(숫자)", base)
    evaluate("C1 가격 제외", base, drop_cols={"price"},
             note="외부데이터 리스크 회피안")
    evaluate("C2 log 가격", base, transform=to_log)
    evaluate("C3 5등급 범주화", base, transform=to_band,
             note="인간 심리에 가까운 형태")

    print()
    print("=" * 84)
    print("B. 대형 스파이크 캡핑 (품목별 분위, 폴드 학습구간에서만 산출)")
    print("=" * 84)
    for q, ct, tag in [(0.99, False, "B1 99분위·피처만"),
                       (0.99, True, "B2 99분위·피처+타깃"),
                       (0.995, False, "B3 99.5분위·피처만")]:
        t0 = time.time()
        fd = build_folds(mat, dates, ctx, cap_q=q, cap_target=ct)
        print(f"  (행렬 생성 {time.time()-t0:.0f}s)")
        evaluate(tag, fd)
        del fd

    # ---------------- 정리 ----------------
    print()
    print("=" * 84)
    print("Phase 2 결과 요약 (공식지표, 낮을수록 좋음)")
    print("=" * 84)
    for r in sorted(RESULTS, key=lambda x: x["cv3"]):
        print(f"  {r['tag']:<34s} 3폴드 {r['cv3']:.4f} | F2+F3 {r['cv23']:.4f}")
    json.dump(RESULTS, open(os.path.join(C.EXPERIMENTS, "phase2_results.json"), "w"),
              ensure_ascii=False, indent=1)
    print("\n저장: experiments/phase2_results.json")


if __name__ == "__main__":
    main()
