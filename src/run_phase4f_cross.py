# -*- coding: utf-8 -*-
"""Phase 4-f — 교차 품목(인원수 프록시 · 침투율) 피처 검증.

4-e(요일 프로파일)가 실패한 이유: 모델이 item_id × dow 로 **이미 알 수 있는** 정보였다.
이번 것은 다르다 — 학습 행 하나에는 자기 품목 통계만 들어 있어서,
**같은 영업장 다른 품목의 판매량은 구조적으로 접근 불가**다. 모델이 스스로 못 만든다.

prof 그룹은 4-e에서 기각됐으므로 이번 비교에서는 제외한다.
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
from metrics import competition_score, make_weights

SCREEN = dict(objective="regression_l1", metric="l1", learning_rate=0.05,
              num_leaves=127, min_data_in_leaf=40, feature_fraction=0.85,
              bagging_fraction=0.85, bagging_freq=1, lambda_l2=1.0,
              verbosity=-1, num_threads=os.cpu_count())
ROUNDS, FLOOR = 600, 1.0
SEED_SETS = [(42, 7), (2024, 913), (31, 11), (13, 101), (103, 999)]


def score(folds, keep_names, seeds):
    fn = F.feature_names()
    keep = [i for i, k in enumerate(fn) if k in keep_names]
    names = [fn[i] for i in keep]
    cats = [c for c in F.CATEGORICAL if c in names]
    s = []
    for fd in folds:
        m = fd["ytr"] > 0
        preds = []
        for sd in seeds:
            ds = lgb.Dataset(fd["Xtr"][m][:, keep], label=np.log1p(fd["ytr"][m]),
                             feature_name=names, categorical_feature=cats,
                             free_raw_data=False)
            mdl = lgb.train(dict(SCREEN, seed=sd), ds, num_boost_round=ROUNDS)
            preds.append(np.expm1(mdl.predict(fd["Xva"][:, keep])))
        p = np.maximum(np.mean(preds, 0), FLOOR)
        s.append(competition_score(fd["yva"], p, fd["iids"],
                                   fd["store_of"], make_weights(1.0), fd["n"]))
    return float(np.mean(s)), float(np.mean(s[1:]))


def main():
    ctx = F.Context()
    tr = D.load_train()
    mat, dates = D.to_matrix(tr, ctx.items)
    nd = mat.shape[1]

    print("=" * 92)
    print("먼저 — 데이터가 고른 '인원수 온도계' 품목은 무엇인가")
    print("=" * 92)
    px = F.pick_proxy_items(mat, dates, nd, ctx.store_codes)
    ctx.set_proxy(px)
    day = tr.groupby(["store", "date"])["qty"].sum().unstack(0).fillna(0)
    rank = {}
    for sc, sname in enumerate(ctx.store_names):
        m = np.where(ctx.store_codes == sc)[0]
        order = m[np.argsort(-mat[m].sum(1))]
        rank[sc] = {int(i): r + 1 for r, i in enumerate(order)}
    for sc, sname in enumerate(ctx.store_names):
        i = px[sc]
        st = day[sname]
        openi = st.values > 0
        v = mat[i][openi]
        rest = st.values[openi] - v
        c = np.corrcoef(v, rest)[0, 1] if rest.std() > 0 else np.nan
        n_it = int((ctx.store_codes == sc).sum())
        print(f"  {sname:<14s} → {ctx.items[i].split('_',1)[1]:<26s} "
              f"(나머지와 상관 {c:.3f}, 판매순위 {rank[sc][int(i)]}/{n_it})")

    print()
    print("=" * 92)
    print("검증 — 시드 조합 5개 반복 (prof 그룹은 4-e에서 기각되어 제외)")
    print("=" * 92)
    ALL = set(F.feature_names()) - set(F.PROF_KEYS)
    NOCROSS = ALL - set(F.CROSS_KEYS)

    t0 = time.time()
    folds = []
    for name, d0, d1 in V.FOLDS:
        cut = int(np.searchsorted(np.array(dates), pd.Timestamp(d0)))
        ctx.set_proxy(F.pick_proxy_items(mat, dates, cut, ctx.store_codes))
        va = V.origins(dates, d0, d1, nd)
        trn = V.train_origins(dates, d0, nd)
        Xtr, ytr, _ = F.build_samples(mat, dates, trn, ctx)
        Xva, yva, mva = F.build_samples(mat, dates, va, ctx)
        folds.append(dict(Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva,
                          iids=mva[:, 2], store_of=ctx.store_of_item, n=ctx.n))
    print(f"  (피처 행렬 {time.time()-t0:.0f}s)\n")

    rows = []
    print(f"  {'시드':<16s} {'교차 없음':>11s} {'교차 추가':>11s} {'차이':>9s}")
    for sds in SEED_SETS:
        _, a = score(folds, NOCROSS, sds)
        _, b = score(folds, ALL, sds)
        rows.append((a, b))
        print(f"  {str(sds):<16s} {a:>11.4f} {b:>11.4f} {b-a:>+9.4f}")
    a = np.array([r[0] for r in rows])
    b = np.array([r[1] for r in rows])
    d = b - a
    print(f"\n  평균   없음 {a.mean():.4f} · 추가 {b.mean():.4f} · 차이 {d.mean():+.4f}")
    print(f"  차이의 σ {d.std():.4f} · 부호 일관성 {int((d < 0).sum())}/5 개선")
    if d.mean() < -0.002 and (d < 0).sum() >= 4:
        print("  → ★ 채택 (평균 개선 + 부호 일관)")
    elif d.mean() > 0.002 and (d > 0).sum() >= 4:
        print("  → 기각")
    else:
        print("  → 판정 불가 (측정 한계 안)")

    json.dump({"nocross": a.tolist(), "cross": b.tolist(), "diff": d.tolist(),
               "proxy": {ctx.store_names[sc]: ctx.items[px[sc]]
                         for sc in range(len(px))}},
              open(os.path.join(C.EXPERIMENTS, "phase4f_cross.json"),
                   "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n저장: experiments/phase4f_cross.json")


if __name__ == "__main__":
    main()
