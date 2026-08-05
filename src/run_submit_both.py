# -*- coding: utf-8 -*-
"""제출본 2개를 한 번에 만든다 — 단일(LightGBM) 과 앙상블(LightGBM+XGBoost).

  submission_v4_lgbm.csv  : 재튜닝 LightGBM 단독 (Phase 5-c 확정, 현재 최고 단일)
  submission_v5_ens.csv   : 위 + 튜닝 XGBoost(Phase 5-b) 를 **등가중 평균**

왜 등가중인가: Phase 5-a에서 가중치 최적화가 등가중보다 나을 게 없음을 확인했다
(최적가중 +0.0016 ≤ 등가중 +0.0017, 폴드마다 가중치가 뒤집힘). 재시도 금지 목록에 있음.

⚠️ 하한 1.0은 **평균을 낸 뒤 마지막에 한 번만** 적용한다. max 는 비선형이라
   '각자 하한 → 평균' ≠ '평균 → 하한' 이다.

PART 1 에서 먼저 F2+F3(확인 시드)로 둘을 채점해 **어느 쪽을 올릴지** 판단하고,
PART 2 에서 전체 데이터로 학습해 제출본을 만든다.
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

SEEDS = (42, 7, 2024, 913, 31)
OOF_SEEDS = (2024, 913, 31)      # Phase 5-b/5-c 확인 시드 — 두 모델 모두 이걸로 판정했음
FLOOR = 1.0
SIGMA2 = 0.003
NT = os.cpu_count()

# ---- Phase 5-c 확정 (model_lgbm.PARAMS 와 동일) ----
LGB_P = dict(objective="regression_l1", metric="l1", learning_rate=0.05,
             num_leaves=511, min_data_in_leaf=10, feature_fraction=0.65,
             bagging_fraction=1.0, bagging_freq=0, lambda_l2=3.0, lambda_l1=2.0,
             verbosity=-1, num_threads=NT)
LGB_ROUNDS = 1000

# ---- Phase 5-b 확정 ----
XGB_P = dict(objective="reg:absoluteerror", tree_method="hist",
             grow_policy="lossguide", max_leaves=255, max_depth=0,
             eta=0.05, min_child_weight=40, subsample=1.0,
             colsample_bytree=0.6, reg_lambda=3.0, reg_alpha=2.0,
             max_cat_to_onehot=1, nthread=NT)
XGB_ROUNDS = 900


def cat_frame(X, names, cats, levels=None):
    df = pd.DataFrame(X, columns=names)
    lv = {}
    for c in cats:
        j = names.index(c)
        cur = levels[c] if levels else np.unique(X[:, j]).astype(int)
        df[c] = pd.Categorical(df[c].astype(int), categories=cur)
        lv[c] = cur
    return df, lv


def fit_lgb(Xt, yt, names, cats, seeds):
    ds = lgb.Dataset(Xt, label=yt, feature_name=names,
                     categorical_feature=cats, free_raw_data=False)
    return [lgb.train(dict(LGB_P, seed=sd), ds, num_boost_round=LGB_ROUNDS)
            for sd in seeds]


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

    tpl = D.load_submission_template()
    rk = tpl.columns[0]
    assert list(tpl.columns[1:]) == ctx.items
    for t in range(C.N_TEST):
        for h in range(1, C.HORIZON + 1):
            assert (tpl[rk] == f"TEST_{t:02d}+{h}일").sum() == 1
    print(f"제출 형식 확인 통과 {tpl.shape}\n")

    # ================================================================ PART 1
    print("=" * 92)
    print(f"PART 1 — F2+F3 (확인 시드 {OOF_SEEDS}) 로 단일 vs 앙상블 판정")
    print("=" * 92)
    oof = {}
    for fi in (1, 2):
        fname, d0, d1 = V.FOLDS[fi]
        cut = int(np.searchsorted(np.array(dates), pd.Timestamp(d0)))
        ctx.set_proxy(F.pick_proxy_items(mat, dates, cut, ctx.store_codes))
        va = V.origins(dates, d0, d1, nd)
        trn = V.train_origins(dates, d0, nd)
        Xtr, ytr, _ = F.build_samples(mat, dates, trn, ctx)
        Xva, yva, mva = F.build_samples(mat, dates, va, ctx)
        m = ytr != 0
        Xt = np.ascontiguousarray(Xtr[m][:, keep])
        yt = np.log1p(np.maximum(ytr[m], 1.0))
        Xv = np.ascontiguousarray(Xva[:, keep])
        del Xtr, Xva

        t1 = time.time()
        pl = np.mean([np.expm1(mdl.predict(Xv))
                      for mdl in fit_lgb(Xt, yt, names, cats, OOF_SEEDS)], 0)
        dft, lv = cat_frame(Xt, names, cats)
        dfv, _ = cat_frame(Xv, names, cats, lv)
        dtr = xgb.DMatrix(dft, label=yt, enable_categorical=True)
        dva = xgb.DMatrix(dfv, enable_categorical=True)
        del dft, dfv
        px = np.mean([np.expm1(xgb.train(dict(XGB_P, seed=sd), dtr,
                                         num_boost_round=XGB_ROUNDS).predict(dva))
                      for sd in OOF_SEEDS], 0)
        oof[fi] = dict(name=fname, y=yva, iids=mva[:, 2], lgb=pl, xgb=px)
        print(f"  [{fname}] 완료 {time.time()-t1:.0f}s", flush=True)
        del Xt, Xv, dtr, dva

    def sc(fi, p):
        f = oof[fi]
        return competition_score(f["y"], np.maximum(p, FLOOR), f["iids"],
                                 ctx.store_of_item, make_weights(1.0), ctx.n)

    cand = {
        "LightGBM 단독": lambda f: oof[f]["lgb"],
        "XGBoost 단독": lambda f: oof[f]["xgb"],
        "등가중 앙상블": lambda f: 0.5 * (oof[f]["lgb"] + oof[f]["xgb"]),
    }
    res = {}
    print(f"\n  {'':<16s} {'F2 겨울':>9s} {'F3 봄':>9s} {'F2+F3':>9s}")
    for k, g in cand.items():
        s = [sc(1, g(1)), sc(2, g(2))]
        res[k] = dict(f2=s[0], f3=s[1], f2f3=float(np.mean(s)))
        print(f"  {k:<16s} {s[0]:>9.4f} {s[1]:>9.4f} {res[k]['f2f3']:>9.4f}")

    # 잔차 상관 (로그 공간, 유효행만)
    rs = {}
    for k in ("lgb", "xgb"):
        rs[k] = np.concatenate([
            np.log1p(np.maximum(oof[fi][k][oof[fi]["y"] > 0], 0))
            - np.log1p(oof[fi]["y"][oof[fi]["y"] > 0]) for fi in (1, 2)])
    rho = float(np.corrcoef(rs["lgb"], rs["xgb"])[0, 1])

    best_single = min(res["LightGBM 단독"]["f2f3"], res["XGBoost 단독"]["f2f3"])
    gain = best_single - res["등가중 앙상블"]["f2f3"]
    print(f"\n  잔차 상관 ρ = {rho:.4f}")
    print(f"  앙상블이 최고 단일보다 {gain:+.4f}  (문턱 2σ = {SIGMA2})")
    pick = "앙상블" if gain > SIGMA2 else "단일(LightGBM)"
    print(f"  → **먼저 올려볼 것: {pick}**")
    if 0 < gain <= SIGMA2:
        print("     (개선은 있으나 측정 한계 안 — 둘 다 만들어두고 LB로 확인)")

    # ================================================================ PART 2
    print("\n" + "=" * 92)
    print("PART 2 — 전체 데이터 학습 → 제출본 2개")
    print("=" * 92)
    ctx.set_proxy(F.pick_proxy_items(mat, dates, nd, ctx.store_codes))
    Xa, ya, _ = F.build_samples(mat, dates,
                                list(range(C.WINDOW - 1, nd - C.HORIZON)), ctx)
    m = ya != 0
    Xt = np.ascontiguousarray(Xa[m][:, keep])
    yt = np.log1p(np.maximum(ya[m], 1.0))
    del Xa
    print(f"  학습행 {Xt.shape[0]:,} · 피처 {len(names)}")

    tests = []
    for t in range(C.N_TEST):
        te = D.load_test(t)
        tmat, tdates = D.to_matrix(te, ctx.items)
        X, _, _ = F.build_samples(tmat, tdates, [C.WINDOW - 1], ctx,
                                  with_target=False)
        tests.append(X[:, keep])
    Xte = np.concatenate(tests, 0)

    t1 = time.time()
    models = fit_lgb(Xt, yt, names, cats, SEEDS)
    PL = np.mean([np.expm1(mdl.predict(Xte)) for mdl in models], 0)
    print(f"  LightGBM 시드 {len(SEEDS)}개 {time.time()-t1:.0f}s", flush=True)
    del models

    t1 = time.time()
    lvs = {c: np.unique(np.concatenate([Xt[:, names.index(c)],
                                        Xte[:, names.index(c)]])).astype(int)
           for c in cats}
    dft, _ = cat_frame(Xt, names, cats, lvs)
    dfe, _ = cat_frame(Xte, names, cats, lvs)
    dtr = xgb.DMatrix(dft, label=yt, enable_categorical=True)
    dte = xgb.DMatrix(dfe, enable_categorical=True)
    del dft, dfe, Xt
    PX = np.mean([np.expm1(xgb.train(dict(XGB_P, seed=sd), dtr,
                                     num_boost_round=XGB_ROUNDS).predict(dte))
                  for sd in SEEDS], 0)
    print(f"  XGBoost  시드 {len(SEEDS)}개 {time.time()-t1:.0f}s", flush=True)

    def write(pred, stamp, note):
        out = tpl.copy()
        out[ctx.items] = out[ctx.items].astype(float)
        p = np.maximum(pred, FLOOR)          # ← 평균 낸 뒤 마지막에 한 번만
        off = 0
        for t in range(C.N_TEST):
            blk = p[off:off + C.HORIZON * ctx.n].reshape(C.HORIZON, ctx.n)
            off += C.HORIZON * ctx.n
            for h in range(1, C.HORIZON + 1):
                ridx = out.index[out[rk] == f"TEST_{t:02d}+{h}일"][0]
                out.loc[ridx, ctx.items] = blk[h - 1]
        out[ctx.items] = out[ctx.items].round(2)
        path = os.path.join(C.SUBMISSIONS, f"submission_{stamp}.csv")
        out.to_csv(path, index=False, encoding="utf-8-sig")
        v = out[ctx.items].values.astype(float)
        assert not np.isnan(v).any() and (v >= FLOOR - 1e-9).all()
        print(f"\n  저장: submission_{stamp}.csv   ({note})")
        print(f"    min {v.min():.2f} / 중앙값 {np.median(v):.2f} / "
              f"평균 {v.mean():.2f} / max {v.max():.2f}")
        return path

    p4 = write(PL, "v4_lgbm", f"재튜닝 LightGBM 단독 · CV {res['LightGBM 단독']['f2f3']:.4f}")
    p5 = write(0.5 * (PL + PX), "v5_ens",
               f"LGBM+XGB 등가중 · CV {res['등가중 앙상블']['f2f3']:.4f}")

    r = np.corrcoef(np.log1p(PL).ravel(), np.log1p(PX).ravel())[0, 1]
    print(f"\n  두 모델 제출값의 로그공간 상관 {r:.4f} · 평균절대차 {np.abs(PL-PX).mean():.2f}")

    json.dump(dict(oof=res, rho=float(rho), ens_gain=float(gain),
                   recommend=pick, lgb=dict(params=LGB_P, rounds=LGB_ROUNDS),
                   xgbp=dict(params=XGB_P, rounds=XGB_ROUNDS), seeds=list(SEEDS)),
              open(os.path.join(C.EXPERIMENTS, "phase5d_submit_both.json"),
                   "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print("\n" + "=" * 92)
    print(f"예상 LB  (보정식: F2+F3 + 0.011)")
    for k, v in res.items():
        print(f"   {k:<16s} CV {v['f2f3']:.4f}  →  예상 LB {v['f2f3']+0.011:.4f}")
    print(f"\n   먼저 올릴 것: **{pick}**")
    print(f"총 {(time.time()-t0)/60:.0f}분")


if __name__ == "__main__":
    main()
