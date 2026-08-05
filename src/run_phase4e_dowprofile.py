# -*- coding: utf-8 -*-
"""Phase 4-e — 품목별 요일 프로파일 피처 검증.

관찰: '단체' 성격 메뉴는 목·금에 몰리고(1.7~1.8배) 일요일은 0.17배인데,
일반 메뉴는 토·일에 몰린다(1.5배). **요일의 의미가 품목마다 정반대.**
창(28일) 안의 같은 요일 표본은 4개뿐이라 이 패턴이 잘 안 잡힌다.
→ 학습기간 전체로 안정적인 요일 프로파일을 만들어 정적 피처로 준다.

측정 한계가 2σ≈0.003이므로 **시드 조합 5개로 반복해 부호 일관성까지** 본다.
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


def build(mat, dates, ctx):
    """폴드별 데이터. 요일 프로파일은 폴드 학습구간까지만으로 계산(누수 차단)."""
    nd = mat.shape[1]
    out = []
    for name, d0, d1 in V.FOLDS:
        cut = int(np.searchsorted(np.array(dates), pd.Timestamp(d0)))
        ctx.set_profile(F.item_dow_profile(mat, dates, cut, ctx.store_codes))
        va = V.origins(dates, d0, d1, nd)
        trn = V.train_origins(dates, d0, nd)
        Xtr, ytr, _ = F.build_samples(mat, dates, trn, ctx)
        Xva, yva, mva = F.build_samples(mat, dates, va, ctx)
        out.append(dict(name=name, Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva,
                        iids=mva[:, 2], store_of=ctx.store_of_item, n=ctx.n))
    return out


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
    ALL = set(F.feature_names())
    NOPROF = ALL - set(F.PROF_KEYS)

    print("=" * 88)
    print("먼저 — 만들어진 요일 프로파일이 관찰과 맞는지 확인")
    print("=" * 88)
    prof = F.item_dow_profile(mat, dates, mat.shape[1], ctx.store_codes)
    DOW = ["월", "화", "수", "목", "금", "토", "일"]
    g = ctx.is_group == 1
    print(f"  단체성으로 판별된 품목 {int(g.sum())}개 / 전체 {ctx.n}개")
    print(f"  {'구분':<10s} " + "  ".join(f"{d:>5s}" for d in DOW))
    print(f"  {'단체성':<10s} " + "  ".join(f"{v:5.2f}" for v in prof[g].mean(0)))
    print(f"  {'일반':<10s} " + "  ".join(f"{v:5.2f}" for v in prof[~g].mean(0)))
    print("\n  주말/평일 비율이 가장 낮은(=평일형) 품목 6개:")
    wk, we = prof[:, :4].mean(1), prof[:, 5:].mean(1)
    ratio = np.divide(we, wk, out=np.full_like(we, np.nan), where=wk > 0)
    for i in np.argsort(ratio)[:6]:
        print(f"    {ctx.items[i]:<40s} {ratio[i]:.2f}  "
              f"({' '.join(f'{DOW[w]}{prof[i,w]:.1f}' for w in range(7))})")
    print("\n  주말/평일 비율이 가장 높은(=주말형) 품목 4개:")
    for i in np.argsort(-np.nan_to_num(ratio))[:4]:
        print(f"    {ctx.items[i]:<40s} {ratio[i]:.2f}")

    print()
    print("=" * 88)
    print("검증 — 시드 조합 5개로 반복 (측정 한계 2σ≈0.003)")
    print("=" * 88)
    t0 = time.time()
    folds = build(mat, dates, ctx)
    print(f"  (피처 행렬 {time.time()-t0:.0f}s, 피처 {len(F.feature_names())}개)\n")

    rows = []
    print(f"  {'시드':<16s} {'프로파일 없음':>13s} {'프로파일 추가':>13s} {'차이':>9s}")
    for sds in SEED_SETS:
        a3, a23 = score(folds, NOPROF, sds)
        b3, b23 = score(folds, ALL, sds)
        rows.append((a23, b23))
        print(f"  {str(sds):<16s} {a23:>13.4f} {b23:>13.4f} {b23-a23:>+9.4f}")
    a = np.array([r[0] for r in rows])
    b = np.array([r[1] for r in rows])
    d = b - a
    print(f"\n  평균   없음 {a.mean():.4f} · 추가 {b.mean():.4f} · 차이 {d.mean():+.4f}")
    print(f"  차이의 σ {d.std():.4f} · 부호 일관성 {int((d < 0).sum())}/5 개선")
    if d.mean() < -0.002 and (d < 0).sum() >= 4:
        print("  → ★ 채택 (평균 개선 + 부호 일관)")
    elif d.mean() > 0.002 and (d > 0).sum() >= 4:
        print("  → 기각 (일관되게 악화)")
    else:
        print("  → 판정 불가 (측정 한계 안)")

    json.dump({"noprof": a.tolist(), "prof": b.tolist(), "diff": d.tolist()},
              open(os.path.join(C.EXPERIMENTS, "phase4e_dowprofile.json"),
                   "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n저장: experiments/phase4e_dowprofile.json")


if __name__ == "__main__":
    main()
