# -*- coding: utf-8 -*-
"""Phase 29 — 시즌 경계 램프 피처 이식. **앙상블 축이 아니라 피처 축이다.**

동기 (Phase 28). 봄 앙상블 이득의 90% 가 개장 주간 창 하나였고, 그 이득의 113% 가
화담숲 13품목에서 나왔다 — **램프 정보에 값어치가 있다는 것은 확인됐다.**
그런데 앙상블로는 못 먹는다: TEST 10개 중 개장 주간(3/29~31)을 예측하는 창이 **0개**다.
→ **B 를 섞을 게 아니라 그 정보를 A 에게 가르친다.** 이건 과녁이 있다 —
  T07(개장 13일 전) · T08(21일 후) · T04(소등 직후).

이식하는 5개 (`F.RAMP_KEYS`)
    d_to_hwadam_open / d_to_hwadam_close / d_to_ski_close   부호 있는 일수
    dayoff_run_len / dayoff_run_pos                          연휴 길이 / 며칠째
우리 `cal` 은 `hwadam_open`(0/1)·`month`·`doy_sin/cos` 뿐이라
**"개장 3일 전"과 "개장 30일 전"을 구분하지 못한다.** 그 공백을 메운다.
⚠️ 개장일을 **3/29** 로 고쳤다 (config 3/20 은 틀림 · 팀원 코드도 3/20 을 씀).

📌 사전등록 (결과 보기 전에 고정)
    후보는 **L1(5개 전부) 하나뿐**이다. L2·L3 은 분해 이해용이며 **채택 후보가 아니다**
    (규칙 ⑦ — N 개 중 최댓값 고르기는 실력 없이도 1.6σ 나온다).
    게이트: prod 평균 > +0.0015 **AND** prod 4/4 일관 **AND** raw 도 4/4 양수
    ※ raw 조건은 규칙 ⑲ 때문에 넣는다. 정수 스냅 뒤에서만 양수면 양자화 잡음이다.
    통과 → 전체학습 후 제출 1회 · 탈락 → 종료. **게이트 밖 재심 없음** (4연패 중).
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
from run_phase10c_thresholds import cell_weights

NT = os.cpu_count()
PARAMS = dict(objective="regression_l1", metric="l1", learning_rate=0.05,
              num_leaves=127, min_data_in_leaf=40, feature_fraction=0.65,
              bagging_fraction=0.85, bagging_freq=1, lambda_l2=1.0,
              verbosity=-1, num_threads=NT)
ROUNDS, SEEDS = 1000, (42, 7, 2024, 913, 31)
DROP3 = ["w_posmedian", "w_last14", "w_std"]

ARMS = [("L0 base(54)", []),
        ("L1 +ramp5(59)", F.RAMP_KEYS),                              # ← 유일한 후보
        ("L2 +hwadam2", ["d_to_hwadam_open", "d_to_hwadam_close"]),  # 분해용
        ("L3 +dayoff2", ["dayoff_run_len", "dayoff_run_pos"])]       # 분해용
CANDIDATE = "L1 +ramp5(59)"

FOLDS = [("F2 겨울", "2023-11-24", "2023-11-24", "2024-02-22"),
         ("F3 봄", "2024-02-23", "2024-02-23", "2024-06-08"),
         ("FAR-봄", "2023-11-24", "2024-02-23", "2024-06-08"),
         ("FAR-겨울", "2023-08-25", "2023-11-24", "2024-02-22")]
OUT = os.path.join(C.EXPERIMENTS, "phase29_ramp.json")


def post(raw, seg=True, snap=True):
    p = (np.where(raw < 1.8, 0.55 * raw, np.where(raw < 10.0, 0.90 * raw, 1.02 * raw))
         if seg else raw.copy())
    p = np.maximum(p, 1.0)
    if not snap:
        return p
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
    ctx = F.Context()
    tr = D.load_train()
    mat, dates = D.to_matrix(tr, ctx.items)
    nd = mat.shape[1]

    # 전체 피처 공간(ramp 포함)에서 v24 구성을 뺀 것이 base
    keep_all = F.active_columns(include=("ramp",))
    names_all = F.active_names(include=("ramp",))
    base_sub = [i for i, n in enumerate(names_all)
                if n not in DROP3 and n not in F.RAMP_KEYS]
    print(f"전체 활성 {len(names_all)}개 · base {len(base_sub)}개 "
          f"(= v24 54개인가: {len(base_sub) == 54})", flush=True)

    # 램프 피처 값 점검 (조용히 이상한 값이 들어가는 걸 막는다)
    smp = [pd.Timestamp("2024-03-26"), pd.Timestamp("2024-03-29"),
           pd.Timestamp("2024-04-05"), pd.Timestamp("2024-12-03"),
           pd.Timestamp("2024-09-16")]
    print("  램프 피처 점검:")
    for d in smp:
        v = F._ramp(d)
        print(f"    {d.date()}  개장까지 {v[0]:+6.0f}일 · 소등까지 {v[1]:+6.0f}일 · "
              f"스키종료까지 {v[2]:+6.0f}일 · 연휴 {v[3]:.0f}일중 {v[4]:.0f}일째", flush=True)

    res = {}
    for fname, cut, v0, v1 in FOLDS:
        cut_col = int(np.searchsorted(np.array(dates), pd.Timestamp(cut)))
        ctx.set_proxy(F.pick_proxy_items(mat, dates, cut_col, ctx.store_codes))
        trn = [o for o in range(C.WINDOW - 1, nd - C.HORIZON)
               if dates[o] < pd.Timestamp(cut)]
        va = V.origins(dates, v0, v1, nd)
        Xtr, ytr, _ = F.build_samples(mat, dates, trn, ctx)
        Xva, yva, mva = F.build_samples(mat, dates, va, ctx)
        m = ytr != 0
        Xt0 = np.ascontiguousarray(Xtr[m][:, keep_all])
        yt = np.log1p(np.maximum(ytr[m], 1.0))
        Xv0 = np.ascontiguousarray(Xva[:, keep_all])
        del Xtr, Xva
        W, _ = cell_weights(yva, mva[:, 2], ctx.store_of_item, ctx.n)

        for lab, add in ARMS:
            sub = [i for i in base_sub] + [i for i, n in enumerate(names_all)
                                           if n in add]
            sub = sorted(set(sub))
            nm = [names_all[i] for i in sub]
            cts = [c for c in F.CATEGORICAL if c in nm]
            Xt = np.ascontiguousarray(Xt0[:, sub])
            Xv = np.ascontiguousarray(Xv0[:, sub])
            ps = []
            for sd in SEEDS:
                ds = lgb.Dataset(Xt, label=yt, feature_name=nm,
                                 categorical_feature=cts, free_raw_data=False)
                ps.append(np.expm1(lgb.train(dict(PARAMS, seed=sd), ds,
                                             num_boost_round=ROUNDS).predict(Xv)))
                del ds
            p = np.mean(ps, 0)
            res[(fname, lab)] = {
                "n": len(nm),
                "prod": float((W * loss(yva, post(p, True, True))).sum()),
                "raw": float((W * loss(yva, post(p, False, False))).sum())}
            del Xt, Xv, ps
        del Xt0, Xv0
        b = res[(fname, "L0 base(54)")]
        print(f"  [{fname:<8s}] base prod {b['prod']:.5f} · raw {b['raw']:.5f}"
              + "".join(f" · {l.split()[0]} {b['prod']-res[(fname,l)]['prod']:+.5f}"
                        for l, _ in ARMS[1:])
              + f"  ({time.time()-t0:.0f}s)", flush=True)

    # ── 요약
    print("\n" + "=" * 96)
    for tag, nm in (("prod", "실제 파이프라인 (seg+스냅)"), ("raw", "raw (하한만) — 규칙 ⑲")):
        print(f"\n[{nm}]  base 대비 개선 (+가 좋음)")
        print(f"{'후보':<16s}{'피처':>5s}" + "".join(f"{f[0]:>11s}" for f in FOLDS)
              + f"{'평균':>11s}{'일관':>7s}")
        for lab, _ in ARMS[1:]:
            gs = [res[(f[0], "L0 base(54)")][tag] - res[(f[0], lab)][tag] for f in FOLDS]
            n = sum(g > 0 for g in gs)
            star = "  ← 후보" if lab == CANDIDATE else "  (분해용)"
            print(f"{lab:<16s}{res[(FOLDS[0][0], lab)]['n']:>5d}"
                  + "".join(f"{g:>+11.5f}" for g in gs)
                  + f"{np.mean(gs):>+11.5f}{n:>5d}/4{star}")

    gp = [res[(f[0], "L0 base(54)")]["prod"] - res[(f[0], CANDIDATE)]["prod"]
          for f in FOLDS]
    gr = [res[(f[0], "L0 base(54)")]["raw"] - res[(f[0], CANDIDATE)]["raw"]
          for f in FOLDS]
    ok = (np.mean(gp) > 0.0015) and (sum(g > 0 for g in gp) == 4) \
        and (sum(g > 0 for g in gr) == 4)
    print("\n" + "=" * 96)
    print(f"📌 사전등록 판정 — 후보 {CANDIDATE}")
    print(f"   prod 평균 {np.mean(gp):+.5f} (>+0.0015?) · prod 일관 "
          f"{sum(g>0 for g in gp)}/4 · **raw 일관 {sum(g>0 for g in gr)}/4**")
    print(f"   → {'✅ 통과. 전체학습 후 제출 1회' if ok else '❌ 탈락. 종료 (게이트 밖 재심 없음)'}")
    print("=" * 96)
    json.dump({f"{k[0]}|{k[1]}": v for k, v in res.items()},
              open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"저장: {os.path.basename(OUT)} · 총 {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
