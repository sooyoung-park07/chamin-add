# -*- coding: utf-8 -*-
"""Phase 8-a — 보정 절차의 결함을 고치고, 더 나은 형태가 있는지 본다.

━━ 왜 ━━
Phase 6 보정은 실측 +0.0172로 크게 먹혔지만, **맞출 때 쓴 저울에 알려진 결함**이 있었다.

  우리 채점: 폴드 4개를 **각각** 채점 → 평균
             ↑ 각 폴드는 3개월. 그 기간에 안 팔린 계절품목은 T_i=0 → **통째로 제외**
  실제 채점: 1년 전체를 **한 번에** → 193품목이 전부 유효날짜를 갖고 각자 1표

이 불일치가 바로 OOF가 이득을 **정확히 2배 과소평가**한 원인으로 지목됐던 것이다.
그런데 원인을 알아놓고 **보정을 맞출 때는 그 저울을 계속 썼다.**

━━ 무엇을 하나 ━━
  1. OOF 재생성 — **시드별 예측**과 **horizon**을 함께 보관 (기존 OOF엔 없음)
  2. 채점 구성 진단 — 폴드별 채점 vs 합산 채점에서 **몇 품목이 실제로 채점되는가**
  3. 두 저울로 각각 3구간 보정을 재적합 → **파라미터가 바뀌는가**
  4. 형태 확장 후보 두 가지 (미시도 영역)
       (A) horizon별 보정  — 먼 날짜일수록 확신이 없어 더 압축돼 있을 것
       (B) 불확실성별 보정 — 시드 5개의 흩어짐이 곧 확신도. **압축의 원인을 직접 잰다**
     ※ 지금 보정은 "예측값이 작다"를 압축의 **대리 지표**로 쓰는 셈인데, (B)는 원인 자체를 쓴다.
  5. 전부 교차검증으로 판정 (합산 저울 기준)

전부 후처리라 모델 재학습은 PART 1 한 번뿐이다.
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

SEEDS = (42, 7, 2024, 913, 31)          # 불확실성 추정을 위해 5개
FLOOR = 1.0
SIGMA2 = 0.0032
NT = os.cpu_count()
GRID = np.round(np.arange(0.40, 1.36, 0.025), 3)
BND = [2.0, 10.0]
V9 = [0.55, 0.90, 1.02]

PARAMS = dict(objective="regression_l1", metric="l1", learning_rate=0.05,
              num_leaves=127, min_data_in_leaf=40, feature_fraction=0.65,
              bagging_fraction=0.85, bagging_freq=1, lambda_l2=1.0,
              verbosity=-1, num_threads=NT)
ROUNDS = 1000

# 폴드 순서: 0=F2겨울 1=F3봄 2=FAR봄 3=FAR겨울
FOLDS = [("F2 겨울", "2023-11-24", "2023-11-24", "2024-02-22"),
         ("F3 봄", "2024-02-23", "2024-02-23", "2024-06-08"),
         ("FAR-봄", "2023-11-24", "2024-02-23", "2024-06-08"),
         ("FAR-겨울", "2023-08-25", "2023-11-24", "2024-02-22")]
# 합산 단위: 겨울+봄을 이어붙여 6.5개월. 날짜가 안 겹치는 조합만 쓴다.
POOLS = [("가까움 합산 (F2+F3)", [0, 1]), ("멂 합산 (FAR겨울+FAR봄)", [3, 2])]
NPZ = os.path.join(C.EXPERIMENTS, "phase8a_oof.npz")


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

    print("=" * 98)
    print("Phase 8-a — 보정 저울 교정 + 형태 확장 탐색")
    print("=" * 98)

    # ============================================================ PART 1
    folds = []
    for fname, cut, v0, v1 in FOLDS:
        t1 = time.time()
        cut_col = int(np.searchsorted(np.array(dates), pd.Timestamp(cut)))
        ctx.set_proxy(F.pick_proxy_items(mat, dates, cut_col, ctx.store_codes))
        trn = [o for o in range(C.WINDOW - 1, nd - C.HORIZON)
               if dates[o] < pd.Timestamp(cut)]
        va = V.origins(dates, v0, v1, nd)
        Xtr, ytr, _ = F.build_samples(mat, dates, trn, ctx)
        Xva, yva, mva = F.build_samples(mat, dates, va, ctx)
        m = ytr != 0
        Xt = np.ascontiguousarray(Xtr[m][:, keep])
        yt = np.log1p(np.maximum(ytr[m], 1.0))
        Xv = np.ascontiguousarray(Xva[:, keep])
        del Xtr, Xva
        ds = lgb.Dataset(Xt, label=yt, feature_name=names,
                         categorical_feature=cats, free_raw_data=False)
        ps = np.stack([np.expm1(lgb.train(dict(PARAMS, seed=sd), ds,
                                          num_boost_round=ROUNDS).predict(Xv))
                       for sd in SEEDS])                     # (시드, 칸)
        folds.append(dict(name=fname, ps=ps, p=ps.mean(0), y=yva,
                          iids=mva[:, 2], h=mva[:, 1]))
        print(f"  [{fname:<8s}] 시드 {len(SEEDS)}개 완료 {time.time()-t1:.0f}s",
              flush=True)
        del Xt, Xv, ds
    np.savez_compressed(NPZ, **{f"ps{i}": d["ps"].astype(np.float32) for i, d in enumerate(folds)},
                        **{f"y{i}": d["y"] for i, d in enumerate(folds)},
                        **{f"i{i}": d["iids"] for i, d in enumerate(folds)},
                        **{f"h{i}": d["h"] for i, d in enumerate(folds)})

    # ============================================================ PART 2
    print("\n" + "=" * 98)
    print("① 채점 구성 진단 — 실제로 몇 품목이 채점되는가")
    print("=" * 98)
    print(f"  {'단위':<26s}{'채점 칸':>10s}{'채점된 품목':>12s}{'제외된 품목':>12s}")
    for i, d in enumerate(folds):
        n_it = len(np.unique(d["iids"][d["y"] != 0]))
        print(f"  {d['name']+' (단독)':<26s}{int((d['y']!=0).sum()):>10,}"
              f"{n_it:>12d}{ctx.n-n_it:>12d}")
    for lab, idx in POOLS:
        y = np.concatenate([folds[i]["y"] for i in idx])
        ii = np.concatenate([folds[i]["iids"] for i in idx])
        n_it = len(np.unique(ii[y != 0]))
        print(f"  {lab:<26s}{int((y!=0).sum()):>10,}{n_it:>12d}{ctx.n-n_it:>12d}")
    print(f"\n  전체 품목 {ctx.n}개 · 실제 테스트는 1년이라 사실상 전부 채점된다.")

    # ============================================================ 채점 도구
    def score(idx, get, pooled):
        """idx 폴드들을 채점. pooled=True 면 이어붙여 한 번에(품목이 1표씩)."""
        if pooled:
            y = np.concatenate([folds[i]["y"] for i in idx])
            p = np.concatenate([get(folds[i]) for i in idx])
            ii = np.concatenate([folds[i]["iids"] for i in idx])
            return competition_score(y, np.maximum(p, FLOOR), ii,
                                     ctx.store_of_item, make_weights(1.0), ctx.n)
        return float(np.mean([
            competition_score(folds[i]["y"], np.maximum(get(folds[i]), FLOOR),
                              folds[i]["iids"], ctx.store_of_item,
                              make_weights(1.0), ctx.n) for i in idx]))

    def seg(cs, bnd=BND, key=None):
        """구간 배율 변환기. key(d)->구간 기준값 (기본: 예측값 자신)."""
        def g(d):
            v = d["p"] if key is None else key(d)
            return d["p"] * np.asarray(cs)[np.digitize(v, bnd)]
        return g

    def fit(idx, pooled, bnd=BND, key=None, rounds=4):
        cs = [1.0] * (len(bnd) + 1)
        for _ in range(rounds):
            for k in range(len(cs)):
                best, bs = cs[k], 9.9
                for v in GRID:
                    cs[k] = v
                    s = score(idx, seg(cs, bnd, key), pooled)
                    if s < bs:
                        bs, best = s, v
                cs[k] = best
        return cs

    # ============================================================ PART 3
    print("\n" + "=" * 98)
    print("② 저울을 바꾸면 최적 배율이 달라지는가")
    print("=" * 98)
    ALL = [0, 1, 2, 3]
    raw = lambda d: d["p"]
    for pooled, lab in ((False, "폴드별 채점 (기존 저울)"), (True, "합산 채점 (교정된 저울)")):
        c = fit(ALL, pooled)
        b = score(ALL, raw, pooled)
        s9 = score(ALL, seg(V9), pooled)
        sn = score(ALL, seg(c), pooled)
        print(f"  [{lab}]")
        print(f"     무보정 {b:.4f} · v9(0.55,0.90,1.02) {s9:.4f} (+{b-s9:.4f})"
              f" · 재적합 ({c[0]:.2f},{c[1]:.2f},{c[2]:.2f}) {sn:.4f} (+{b-sn:.4f})")
    print("\n  ※ 합산 저울에서 최적 배율이 크게 달라지면, 기존 v9 값은 왜곡된 저울의 산물이다.")

    # ============================================================ PART 4/5
    print("\n" + "=" * 98)
    print("③ 형태 확장 후보 — horizon별 · 불확실성별")
    print("=" * 98)
    for d in folds:
        lg = np.log1p(np.maximum(d["ps"], 0))
        d["u"] = lg.std(0)                       # 시드 흩어짐 = 불확실성
    uq = np.quantile(np.concatenate([d["u"] for d in folds]), [1/3, 2/3])
    print(f"  불확실성(시드 로그공간 표준편차) 3등분 경계: "
          f"{uq[0]:.4f} / {uq[1]:.4f}")

    # 각 축에서 '압축 정도'가 실제로 다른지 먼저 진단
    print(f"\n  [진단] 축별로 최적 배율이 갈리는가 (합산 저울, 전체 적합)")
    for lab, key, bnd in (("horizon (1~2 / 3~5 / 6~7)", lambda d: d["h"], [2.5, 5.5]),
                          ("불확실성 (낮음/중간/높음)", lambda d: d["u"], list(uq))):
        c = fit(ALL, True, bnd, key)
        b = score(ALL, raw, True)
        s = score(ALL, seg(c, bnd, key), True)
        print(f"    {lab:<26s} 최적 ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})"
              f"   점수 {s:.4f}  (무보정 대비 +{b-s:.4f})")
    print("    ※ 세 배율이 서로 비슷하면 그 축은 압축과 무관하다는 뜻.")

    # ============================================================ PART 6
    print("\n" + "=" * 98)
    print("④ ★ 교차검증 — 합산 저울 기준 (한쪽에서 맞추고 다른 쪽에서 채점)")
    print("=" * 98)
    NEAR, FAR = POOLS[0][1], POOLS[1][1]
    CANDS = [
        ("현행 v9 (고정값)", None, BND, None),
        ("예측값 3구간 (재적합)", "fit", BND, None),
        ("horizon 3구간", "fit", [2.5, 5.5], lambda d: d["h"]),
        ("불확실성 3구간", "fit", list(uq), lambda d: d["u"]),
    ]
    print(f"  {'후보':<24s}{'가까움→멂':>12s}{'멂→가까움':>12s}{'평균 개선':>11s}{'일관':>6s}")
    out = []
    for lab, mode, bnd, key in CANDS:
        gains, picks = [], []
        for fi, ti in ((NEAR, FAR), (FAR, NEAR)):
            c = V9 if mode is None else fit(fi, True, bnd, key)
            picks.append([float(x) for x in c])
            gains.append(score(ti, raw, True) - score(ti, seg(c, bnd, key), True))
        ok = all(g > 0 for g in gains)
        out.append(dict(lab=lab, picks=picks, gains=gains,
                        mean=float(np.mean(gains)), ok=bool(ok)))
        print(f"  {lab:<24s}{gains[0]:>+12.4f}{gains[1]:>+12.4f}"
              f"{np.mean(gains):>+11.4f}{'○' if ok else '×':>5s}")
        if mode is not None:
            print(f"  {'':<24s}맞춘 값: {tuple(round(x,2) for x in picks[0])} / "
                  f"{tuple(round(x,2) for x in picks[1])}")

    print("\n" + "=" * 98)
    print("판정")
    print("=" * 98)
    base = [o for o in out if o["lab"].startswith("현행")][0]
    best = max(out, key=lambda o: o["mean"])
    print(f"  현행 v9 교차 개선 {base['mean']:+.4f}")
    print(f"  최고 후보  {best['lab']}  {best['mean']:+.4f}  일관 "
          f"{'○' if best['ok'] else '×'}")
    d = best["mean"] - base["mean"]
    if best["lab"].startswith("현행"):
        print("  → 현행 v9가 여전히 최선. 저울을 고쳐도 결론이 안 바뀐다.")
    elif d > SIGMA2 and best["ok"]:
        print(f"  → ★ {best['lab']} 가 v9보다 {d:+.4f} 낫다. 교체 검토.")
    else:
        print(f"  → v9 대비 {d:+.4f} 로 2σ 미달. 현행 유지.")

    json.dump(dict(pools=[[l, i] for l, i in POOLS], cands=out),
              open(os.path.join(C.EXPERIMENTS, "phase8a_recalib.json"),
                   "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n저장: experiments/phase8a_recalib.json · phase8a_oof.npz")
    print(f"총 {(time.time()-t0)/60:.0f}분")


if __name__ == "__main__":
    main()
