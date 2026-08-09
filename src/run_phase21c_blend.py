# -*- coding: utf-8 -*-
"""Phase 21-c — 팀원 Ridge × 우리 LGBM 앙상블: **전제부터 잰다** (제출 0회)

배경. 알고리즘 앙상블 축은 Phase 5-a/10 에서 닫혔다 — LGBM+XGB 등가중이 스냅 위에
+0.0005 뿐이었고, 약한 v6(leaves15) 를 넣으면 −0.0011 이었다. 다만 그건 전부
**트리끼리**였다. 팀원 모델은 Ridge(선형) 라 함수 형태가 다르므로 축이 새로 열린다.
단 피처는 우리 것을 거의 그대로 쓴다(이름까지 동일) — 재료가 같고 조리법만 다르다.

이득 조건 (산수). 두 모델을 (1−w):w 로 섞을 때
    σ² = (1−w)²σ₁² + w²σ₂² + 2w(1−w)·ρ·σ₁σ₂
w>0 이 이득이 되려면  **ρ < σ₁/σ₂**.  약한 모델일수록 문턱이 낮다.
팀원 Private 0.4957 / 우리 v24 0.4376 이라 σ₂/σ₁ ≈ 1.13 → **ρ < 0.885** 가 필요하다.
참고: 트리끼리는 ρ = 0.93~0.96 이었다(Phase 5-a). 그래서 안 통했다.

여기서 재는 것
  ① **잔차 상관 ρ** — 예측 상관이 아니라 잔차 상관이다(규칙 ⑬).
     예측 상관은 공통 신호 때문에 부풀려진다 (v8↔v3: 예측 0.9968 vs 잔차 0.9783).
  ② w 격자 0.00~0.50 에서 **후처리까지 얹은 실제 점수** — 우리 v17 구성으로.
     섞는 지점은 **raw** 다. 후처리는 섞은 뒤 한 번만 건다.
  ③ 폴드 4개 일관성 — 개선이 한 폴드에만 몰리면 안 믿는다(규칙 ③).

판정 (사전 고정, 결과 보기 전에 적는다)
  채택 후보 = 4폴드 평균 이득 > +0.003 (2σ) AND 4/4 같은 부호 AND
              최적 w 가 고립된 봉우리가 아닐 것(규칙 ⑦ — 이웃 격자도 같이 좋을 것)
  그 밖 → 축 기각. v24 (Private 0.4375952) 가 최종으로 남는다.
  ※ 내부 수치는 그대로 실전 이득이 아니다. 후처리 축은 ×0.8~1.1 이었지만
    **재학습 동반 축은 배율이 다르고 CV 자체가 거꾸로 간 전력이 있다**(Phase 5-e).
    그래서 여기 통과는 '제출해볼 자격'이지 채택이 아니다.
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
import model_ridge_tm as TM
from run_phase10c_thresholds import cell_weights

NT = os.cpu_count()
PARAMS = dict(objective="regression_l1", metric="l1", learning_rate=0.05,
              num_leaves=127, min_data_in_leaf=40, feature_fraction=0.65,
              bagging_fraction=0.85, bagging_freq=1, lambda_l2=1.0,
              verbosity=-1, num_threads=NT)
ROUNDS = 1000
SEEDS = (42, 7, 2024, 913, 31)
DROP3 = ["w_posmedian", "w_last14", "w_std"]          # v24 구성 (57 − 3 = 54)

FOLDS = [("F2 겨울", "2023-11-24", "2023-11-24", "2024-02-22"),
         ("F3 봄", "2024-02-23", "2024-02-23", "2024-06-08"),
         ("FAR-봄", "2023-11-24", "2024-02-23", "2024-06-08"),
         ("FAR-겨울", "2023-08-25", "2023-11-24", "2024-02-22")]

WGRID = np.round(np.arange(0.0, 0.55, 0.05), 3)
OUT_JSON = os.path.join(C.EXPERIMENTS, "phase21c_blend.json")
OUT_NPZ = os.path.join(C.EXPERIMENTS, "phase21c_oof.npz")


def seg_snap(raw):
    """v17 후처리 — 3구간 배율 → 하한 1.0 → 정수 격자 스냅(기하 경계)."""
    p = np.where(raw < 1.8, 0.55 * raw, np.where(raw < 10.0, 0.90 * raw, 1.02 * raw))
    p = np.maximum(p, 1.0)
    k = np.maximum(np.floor(p), 1.0)
    return np.maximum(np.where(p >= np.sqrt(k * (k + 1.0)), k + 1.0, k), 1.0)


def loss(a, p):
    a, p = np.abs(a), np.abs(p)
    den = a + p
    out = np.zeros(len(a))
    m = den > 0
    out[m] = 2.0 * np.abs(a[m] - p[m]) / den[m]
    return out


def main():
    t0 = time.time()
    items = D.item_order()
    ctx = F.Context()
    tctx = TM.TMContext(items)
    tr = D.load_train()
    mat, dates = D.to_matrix(tr, ctx.items)
    nd = mat.shape[1]
    assert list(ctx.items) == items, "품목 순서 불일치"

    keep0, names0 = F.active_columns(), F.active_names()
    idx = [i for i, n in enumerate(names0) if n not in DROP3]
    keep = [keep0[i] for i in idx]
    names = [names0[i] for i in idx]
    cats = [c for c in F.CATEGORICAL if c in names]
    print(f"Phase 21-c — LGBM({len(names)}개 피처) × 팀원 Ridge · 시드 {SEEDS}", flush=True)

    store = {}
    rows = []
    for fname, cut, v0, v1 in FOLDS:
        cut_col = int(np.searchsorted(np.array(dates), pd.Timestamp(cut)))
        ctx.set_proxy(F.pick_proxy_items(mat, dates, cut_col, ctx.store_codes))
        trn = [o for o in range(C.WINDOW - 1, nd - C.HORIZON)
               if dates[o] < pd.Timestamp(cut)]
        va = V.origins(dates, v0, v1, nd)

        # ── 우리 LGBM
        Xtr, ytr, _ = F.build_samples(mat, dates, trn, ctx)
        Xva, yva, mva = F.build_samples(mat, dates, va, ctx)
        m = ytr != 0
        Xt = np.ascontiguousarray(Xtr[m][:, keep])
        yt = np.log1p(np.maximum(ytr[m], 1.0))
        Xv = np.ascontiguousarray(Xva[:, keep])
        del Xtr, Xva
        ps = []
        for sd in SEEDS:
            ds = lgb.Dataset(Xt, label=yt, feature_name=names,
                             categorical_feature=cats, free_raw_data=False)
            ps.append(np.expm1(lgb.train(dict(PARAMS, seed=sd), ds,
                                         num_boost_round=ROUNDS).predict(Xv)))
            del ds
        ours = np.mean(ps, 0)
        del Xt, Xv, ps
        print(f"  [{fname:<8s}] LGBM 완료 ({time.time()-t0:.0f}s)", flush=True)

        # ── 팀원 Ridge (같은 origin · 같은 행 순서)
        tn, tc, ty, ti = TM.build_frames(tctx, mat, dates, trn)
        vn, vc, vy, vi = TM.build_frames(tctx, mat, dates, va)
        assert np.array_equal(vi, mva[:, 2].astype(int)), "행 순서 불일치 — ρ 계산 전제 붕괴"
        assert np.allclose(vy, yva), "검증 타깃 불일치"
        tm = TM.fit_predict(tctx, tn, tc, ty, ti, vn, vc)
        del tn, tc, vn, vc
        print(f"  [{fname:<8s}] Ridge 완료 ({time.time()-t0:.0f}s)", flush=True)

        # ── ① 잔차 상관 (채점 대상 칸에서만, 로그공간)
        sc = yva != 0
        lg = lambda v: np.log1p(np.maximum(v, 0.0))
        ay = lg(np.maximum(np.abs(yva[sc]), 1.0))
        r_ours = lg(np.maximum(ours[sc], 1.0)) - ay
        r_tm = lg(tm["blend"][sc]) - ay
        r_rg = lg(tm["ridge"][sc]) - ay
        rho_b = float(np.corrcoef(r_ours, r_tm)[0, 1])
        rho_r = float(np.corrcoef(r_ours, r_rg)[0, 1])
        rho_pred = float(np.corrcoef(lg(np.maximum(ours[sc], 1.0)),
                                     lg(tm["blend"][sc]))[0, 1])

        # ── ② w 격자 (섞은 뒤 후처리 1회)
        W, _ = cell_weights(yva, mva[:, 2], ctx.store_of_item, ctx.n)
        sweep_b = {float(w): float((W * loss(yva, seg_snap(
            (1 - w) * ours + w * tm["blend"]))).sum()) for w in WGRID}
        sweep_r = {float(w): float((W * loss(yva, seg_snap(
            (1 - w) * ours + w * tm["ridge"]))).sum()) for w in WGRID}
        solo_tm = float((W * loss(yva, seg_snap(tm["blend"]))).sum())

        store[fname] = dict(rho_blend=rho_b, rho_ridge=rho_r, rho_pred=rho_pred,
                            sweep_blend=sweep_b, sweep_ridge=sweep_r,
                            solo_ours=sweep_b[0.0], solo_tm=solo_tm)
        rows.append((fname, ours, tm["blend"], tm["ridge"], yva, mva[:, 2]))
        print(f"  [{fname:<8s}] ρ(잔차) blend {rho_b:.4f} · ridge {rho_r:.4f}"
              f"  (예측상관 {rho_pred:.4f})", flush=True)
        print(f"           단독: 우리 {sweep_b[0.0]:.5f} · 팀원 {solo_tm:.5f}"
              f" · 팀원단독 열세 {solo_tm - sweep_b[0.0]:+.5f}", flush=True)

    # ── 요약
    print("\n" + "=" * 78)
    print("① 잔차 상관 — 이득 조건은 ρ < σ_ours/σ_theirs")
    print(f"{'폴드':<10s} {'ρ(잔차,blend)':>14s} {'ρ(잔차,ridge)':>14s} "
          f"{'ρ(예측)':>10s} {'σ비(추정)':>10s} {'문턱통과':>8s}")
    for fname, _, _, _, _, _ in rows:
        s = store[fname]
        sig = s["solo_tm"] / s["solo_ours"]
        print(f"{fname:<10s} {s['rho_blend']:>14.4f} {s['rho_ridge']:>14.4f} "
              f"{s['rho_pred']:>10.4f} {sig:>10.3f} "
              f"{'✅' if s['rho_blend'] < 1/sig else '❌':>8s}")

    print("\n② w 격자 — 4폴드 평균 점수 (낮을수록 좋음) · 괄호는 w=0 대비 개선")
    print(f"{'w':>6s} {'blend 평균':>12s} {'개선':>10s} {'일관':>6s} | "
          f"{'ridge 평균':>12s} {'개선':>10s} {'일관':>6s}")
    best = {}
    for w in WGRID:
        w = float(w)
        for tag, key in (("blend", "sweep_blend"), ("ridge", "sweep_ridge")):
            g = [store[f[0]][key][0.0] - store[f[0]][key][w] for f in FOLDS]
            best.setdefault(tag, {})[w] = (float(np.mean(g)), sum(x > 0 for x in g))
        b, rr = best["blend"][w], best["ridge"][w]
        mb = np.mean([store[f[0]]["sweep_blend"][w] for f in FOLDS])
        mr = np.mean([store[f[0]]["sweep_ridge"][w] for f in FOLDS])
        star = " ←" if w > 0 and b[0] == max(v[0] for v in best["blend"].values()) else ""
        print(f"{w:>6.2f} {mb:>12.5f} {b[0]:>+10.5f} {b[1]:>4d}/4 | "
              f"{mr:>12.5f} {rr[0]:>+10.5f} {rr[1]:>4d}/4{star}")

    # ── 판정
    wb = max((w for w in best["blend"] if w > 0), key=lambda w: best["blend"][w][0])
    gain, cons = best["blend"][wb]
    nb = [w for w in best["blend"] if w > 0 and abs(w - wb) < 0.051 and w != wb]
    ridge_ok = all(best["blend"][w][0] > 0 for w in nb)
    ok = (gain > 0.003) and (cons == 4) and ridge_ok
    print("\n" + "=" * 78)
    print(f"③ 최적 w = {wb:.2f} · 평균 이득 {gain:+.5f} · 일관 {cons}/4 · "
          f"이웃격자 동반상승 {'예' if ridge_ok else '아니오(고립 봉우리 — 규칙 ⑦)'}")
    print(f"   판정 기준: 이득 > +0.003 AND 4/4 AND 봉우리 아님")
    print(f"   → {'✅ 제출해볼 자격 획득' if ok else '❌ 축 기각. v24 가 최종으로 남는다'}")
    print("=" * 78)

    json.dump(store, open(OUT_JSON, "w"), indent=1, ensure_ascii=False)
    np.savez_compressed(OUT_NPZ, **{
        f"{k}|{f[0]}": v for f in rows
        for k, v in (("ours", f[1]), ("tm_blend", f[2]), ("tm_ridge", f[3]),
                     ("y", f[4]), ("iid", f[5]))})
    print(f"저장: {os.path.basename(OUT_JSON)} · {os.path.basename(OUT_NPZ)}")
    print(f"총 {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
