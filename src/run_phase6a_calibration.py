# -*- coding: utf-8 -*-
"""Phase 6-a — 예측 후보정(하한 · 전역배율 · 구간별) 탐색.

동기 두 갈래가 **서로 반대 방향**이라 실측이 필요하다:
  ① 하한을 1보다 올린다 (작은 예측 ↑)
     근거: 채점 대상의 34%가 y=1(21.2%) 또는 y=2(13.1%). 1.0으로 눌러두면 y=2에서 벌점 0.667.
  ② 전체를 낮춘다 (모든 예측 ↓)
     근거: SMAPE 최적 예측은 **중앙값보다 아래**다(상수 최적 3 vs 중앙값 4).
     그런데 우리 모델은 log1p+L1이라 조건부 **중앙값**을 뱉는다 → 구조적으로 살짝 높을 수 있다.
     수식: 최적점은 가중치 y/(y+p)² 로 잰 가중중앙값인데, 이 가중치가 큰 y에서 1/y로 줄어
     오른쪽 꼬리를 깎는다. 수요가 우편향(평균 22.8 ≫ 중앙값 4)이라 효과가 크다.

**전부 후처리다 — 모델을 다시 만들 필요가 없다.** 예측을 한 번 저장해두면 어떤 보정이든 몇 초.

━━ 설계 ━━
PART 1  OOF 예측 생성 (4폴드 × 확인시드 3개) — **하한 적용 전 원본**으로 보관
PART 2  스윕: 하한 단독 / 배율 단독 / 2차원 격자 / 구간별 진단
        ※ **4개 폴드에서 최적점이 일치하는가**를 반드시 본다. 한 폴드에서만 좋으면 노이즈다.
PART 3  전체학습 → **테스트 예측도 원본으로 .npy 저장**
        → 이후 어떤 보정이든 `make_submission.py` 로 즉시 생성 (재학습 0초)

설정은 현재 1위 v8 (leaves127 + ff0.65, 합산 0.46883).
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

SEEDS = (2024, 913, 31)
SUBMIT_SEEDS = (42, 7, 2024, 913, 31)
NT = os.cpu_count()

# 현재 1위 = v8
PARAMS = dict(objective="regression_l1", metric="l1", learning_rate=0.05,
              num_leaves=127, min_data_in_leaf=40, feature_fraction=0.65,
              bagging_fraction=0.85, bagging_freq=1, lambda_l2=1.0,
              verbosity=-1, num_threads=NT)
ROUNDS = 1000

# (이름, 학습 종료, 검증 시작, 검증 끝, 저울)
FOLDS = [
    ("F2 겨울",  "2023-11-24", "2023-11-24", "2024-02-22", "가까움"),
    ("F3 봄",    "2024-02-23", "2024-02-23", "2024-06-08", "가까움"),
    ("FAR-봄",   "2023-11-24", "2024-02-23", "2024-06-08", "멂"),
    ("FAR-겨울", "2023-08-25", "2023-11-24", "2024-02-22", "멂"),
]

FLOORS = np.round(np.arange(0.80, 2.51, 0.05), 2)
MULTS = np.round(np.arange(0.80, 1.161, 0.01), 3)
OOF_NPZ = os.path.join(C.EXPERIMENTS, "phase6a_oof.npz")
RAW_NPY = os.path.join(C.EXPERIMENTS, "phase6a_test_raw.npy")


def main():
    t0 = time.time()
    ctx = F.Context()
    tr = D.load_train()
    mat, dates = D.to_matrix(tr, ctx.items)
    nd = mat.shape[1]
    # ⚠️ 2026-08-04 수정: 예전엔 `k not in set(F.PROF_KEYS)` 로 **PROF 만** 걸렀다.
    #   그 뒤 Phase 9-b(resort 6개)·9-c/d/e(name 1개)가 기각되면서 이 줄이 조용히 낡았고,
    #   지금 그대로 재실행하면 기각된 7열이 섞여 **64열**로 학습된다(v9 원본이 재현되지 않음).
    #   → 반드시 F.active_columns() 를 쓴다.
    keep = F.active_columns()
    names = F.active_names()
    cats = [c for c in F.CATEGORICAL if c in names]

    print("=" * 98)
    print("Phase 6-a — 예측 후보정 탐색 (하한 · 전역배율 · 구간별)")
    print("=" * 98)
    print(f"  설정: v8 leaves127 · ff0.65 · {ROUNDS}라운드 · 시드 {SEEDS}\n")

    # ============================================================ PART 1
    oof = []
    for fname, cut, v0, v1_, kind in FOLDS:
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
        ds = lgb.Dataset(Xt, label=yt, feature_name=names,
                         categorical_feature=cats, free_raw_data=False)
        # ★ 하한을 적용하지 않은 원본 예측을 보관한다
        p = np.mean([np.expm1(lgb.train(dict(PARAMS, seed=sd), ds,
                                        num_boost_round=ROUNDS).predict(Xv))
                     for sd in SEEDS], 0)
        oof.append(dict(name=fname, kind=kind, p=p, y=yva, iids=mva[:, 2]))
        print(f"  [{fname:<8s}] {kind:<4s} 학습 {Xt.shape[0]:>7,}행 · "
              f"검증 {len(yva):>6,}칸 · 원본예측 min {p.min():.3f} / "
              f"중앙값 {np.median(p):.2f}   ({time.time()-t1:.0f}s)", flush=True)
        del Xt, Xv, ds

    np.savez_compressed(OOF_NPZ, **{f"p{i}": d["p"].astype(np.float32) for i, d in enumerate(oof)},
                        **{f"y{i}": d["y"] for i, d in enumerate(oof)},
                        **{f"i{i}": d["iids"] for i, d in enumerate(oof)})

    def sc(d, p):
        return competition_score(d["y"], p, d["iids"], ctx.store_of_item,
                                 make_weights(1.0), ctx.n)

    base = {d["name"]: sc(d, np.maximum(d["p"], 1.0)) for d in oof}
    print(f"\n  현행(하한 1.0) : " +
          " · ".join(f"{k} {v:.4f}" for k, v in base.items()))
    print(f"                   가까움 {np.mean([base[d['name']] for d in oof if d['kind']=='가까움']):.4f}"
          f" · 멂 {np.mean([base[d['name']] for d in oof if d['kind']=='멂']):.4f}")

    # ============================================================ PART 2
    print("\n" + "=" * 98)
    print("① 하한 단독 스윕 — 폴드별 최적점이 일치하는가")
    print("=" * 98)
    print(f"  {'하한':>6s} " + "".join(f"{d['name']:>11s}" for d in oof)
          + f"{'평균':>10s}{'영향받는칸':>11s}")
    curve = {}
    for f in FLOORS:
        s = [sc(d, np.maximum(d["p"], f)) for d in oof]
        frac = np.mean([(d["p"] < f).mean() for d in oof])
        curve[float(f)] = s
        mark = "  ←현행" if abs(f - 1.0) < 1e-9 else ""
        if f <= 1.0 or abs(f * 20 - round(f * 20)) < 1e-9 and (round(f * 100) % 10 == 0 or f <= 1.5):
            print(f"  {f:>6.2f} " + "".join(f"{x:>11.4f}" for x in s)
                  + f"{np.mean(s):>10.4f}{100*frac:>10.1f}%{mark}")
    best_each = [FLOORS[int(np.argmin([curve[float(f)][i] for f in FLOORS]))]
                 for i in range(len(oof))]
    best_avg = FLOORS[int(np.argmin([np.mean(curve[float(f)]) for f in FLOORS]))]
    print(f"\n  폴드별 최적 하한: " +
          " · ".join(f"{d['name']} {b:.2f}" for d, b in zip(oof, best_each)))
    print(f"  평균 기준 최적: {best_avg:.2f}  "
          f"(현행 1.00 대비 {np.mean(curve[1.0]) - np.mean(curve[float(best_avg)]):+.4f})")
    print("  → 폴드마다 최적점이 흩어지면 노이즈다. 모여야 믿을 수 있다.")

    print("\n" + "=" * 98)
    print("② 전역 배율 스윕  p → max(c·p, 1.0)")
    print("=" * 98)
    mcurve = {}
    for c in MULTS:
        mcurve[float(c)] = [sc(d, np.maximum(c * d["p"], 1.0)) for d in oof]
    print(f"  {'배율':>6s} " + "".join(f"{d['name']:>11s}" for d in oof) + f"{'평균':>10s}")
    for c in MULTS:
        if abs(c * 100 - round(c * 100)) < 1e-9 and round(c * 100) % 2 == 0:
            s = mcurve[float(c)]
            mark = "  ←현행" if abs(c - 1.0) < 1e-9 else ""
            print(f"  {c:>6.2f} " + "".join(f"{x:>11.4f}" for x in s)
                  + f"{np.mean(s):>10.4f}{mark}")
    mb_each = [MULTS[int(np.argmin([mcurve[float(c)][i] for c in MULTS]))]
               for i in range(len(oof))]
    mb_avg = MULTS[int(np.argmin([np.mean(mcurve[float(c)]) for c in MULTS]))]
    print(f"\n  폴드별 최적 배율: " +
          " · ".join(f"{d['name']} {b:.2f}" for d, b in zip(oof, mb_each)))
    print(f"  평균 기준 최적: {mb_avg:.2f}  "
          f"(현행 1.00 대비 {np.mean(mcurve[1.0]) - np.mean(mcurve[float(mb_avg)]):+.4f})")

    print("\n" + "=" * 98)
    print("③ 2차원 격자  p → max(c·p, f)   (평균 기준 상위 8개)")
    print("=" * 98)
    grid = []
    for c in MULTS[::2]:
        for f in FLOORS[::2]:
            s = [sc(d, np.maximum(c * d["p"], f)) for d in oof]
            grid.append((float(c), float(f), s, float(np.mean(s))))
    grid.sort(key=lambda r: r[3])
    b0 = np.mean(curve[1.0])
    print(f"  {'배율':>6s}{'하한':>7s} " + "".join(f"{d['name']:>11s}" for d in oof)
          + f"{'평균':>10s}{'현행대비':>10s}{'폴드일관':>9s}")
    for c, f, s, m in grid[:8]:
        allb = all(s[i] < curve[1.0][i] for i in range(len(oof)))
        print(f"  {c:>6.2f}{f:>7.2f} " + "".join(f"{x:>11.4f}" for x in s)
              + f"{m:>10.4f}{b0-m:>+10.4f}{'○' if allb else '×':>8s}")

    print("\n" + "=" * 98)
    print("④ 구간별 진단 — 예측 구간마다 최적 배율이 다른가 (보정의 '모양')")
    print("=" * 98)
    P = np.concatenate([d["p"] for d in oof])
    Y = np.concatenate([d["y"] for d in oof])
    ok = Y != 0
    P, Y = P[ok], Y[ok]
    qs = np.quantile(P, np.linspace(0, 1, 11))
    print(f"  {'예측구간':>16s}{'칸수':>9s}{'실측중앙':>9s}{'예측중앙':>9s}"
          f"{'최적배율':>9s}{'그때이득':>9s}")
    shape = []
    for k in range(10):
        m = (P >= qs[k]) & (P < qs[k + 1] if k < 9 else P <= qs[10])
        if m.sum() < 50:
            continue
        p_, y_ = P[m], np.abs(Y[m])
        cs = np.arange(0.6, 1.61, 0.01)
        loss = [np.mean(2 * np.abs(y_ - np.maximum(c * p_, 1.0))
                        / (y_ + np.maximum(c * p_, 1.0))) for c in cs]
        bi = int(np.argmin(loss))
        shape.append((float(qs[k]), float(qs[k + 1]), int(m.sum()),
                      float(cs[bi]), float(loss[0 if False else bi]),
                      float(np.mean(2 * np.abs(y_ - np.maximum(p_, 1.0))
                                    / (y_ + np.maximum(p_, 1.0))))))
        print(f"  {qs[k]:>7.2f}~{qs[k+1]:<8.2f}{m.sum():>9,}{np.median(y_):>9.1f}"
              f"{np.median(p_):>9.2f}{cs[bi]:>9.2f}"
              f"{shape[-1][5]-loss[bi]:>+9.4f}")
    print("\n  → 최적 배율이 구간마다 **크게 다르면** 단순 배율/하한으로는 부족하고 '모양'이 필요하다는 뜻.")
    print("     비슷하면 전역 배율 하나로 충분하다.")

    # ============================================================ PART 3
    print("\n" + "=" * 98)
    print("PART 3 — 전체학습 → 테스트 예측을 **원본으로** 저장 (재학습 없이 제출본 생성용)")
    print("=" * 98)
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
    ds = lgb.Dataset(Xt, label=yt, feature_name=names,
                     categorical_feature=cats, free_raw_data=False)
    t1 = time.time()
    P_raw = np.mean([np.expm1(lgb.train(dict(PARAMS, seed=sd), ds,
                                        num_boost_round=ROUNDS).predict(Xte))
                     for sd in SUBMIT_SEEDS], 0)
    np.save(RAW_NPY, P_raw)
    print(f"  학습행 {Xt.shape[0]:,} · 시드 {len(SUBMIT_SEEDS)}개 {time.time()-t1:.0f}s")
    print(f"  저장: {os.path.basename(RAW_NPY)}  (하한 적용 전 원본, {P_raw.shape[0]:,}칸)")
    print(f"    min {P_raw.min():.3f} / 중앙값 {np.median(P_raw):.2f} / "
          f"평균 {P_raw.mean():.2f} / max {P_raw.max():.2f}")
    for f in (1.0, 1.2, 1.5, 2.0):
        print(f"    {f} 미만 비율 {100*(P_raw < f).mean():.1f}%")

    json.dump(dict(params=PARAMS, rounds=ROUNDS, seeds=list(SEEDS),
                   base={k: float(v) for k, v in base.items()},
                   floor_curve={str(k): [float(x) for x in v] for k, v in curve.items()},
                   mult_curve={str(k): [float(x) for x in v] for k, v in mcurve.items()},
                   grid_top=[[c, f, s, m] for c, f, s, m in grid[:20]],
                   bin_shape=shape,
                   best=dict(floor=float(best_avg), mult=float(mb_avg),
                             grid=[grid[0][0], grid[0][1]])),
              open(os.path.join(C.EXPERIMENTS, "phase6a_calibration.json"),
                   "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n저장: experiments/phase6a_calibration.json")
    print(f"총 {(time.time()-t0)/60:.0f}분")


if __name__ == "__main__":
    main()
