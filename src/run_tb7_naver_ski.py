# -*- coding: utf-8 -*-
"""Tier B — TB7: "곤지암 스키장" 검색 모멘텀(신규 키워드), ski 업장군 게이팅.
run_tb6_ski_close.py와 동일한 저울(짝시드 5개·4폴드·raw/prod 동시 채점).

📌 사전등록 (experiments/tb7_naver_keyword_brief.md, 결과 보기 전에 기록 — 2026-08-15)

배경: TB4 계열("곤지암리조트"·"화담숲", 4가지 시점 변형)이 전부 게이트 미달로 종료된 뒤,
키워드 자체를 바꿔 재검토. 업장별 잔차 재진단(build_residual_baseline_store.py)에서
ski 업장군(포레스트릿·카페테리아)의 **12월**이 전체 표에서 가장 큰 단일 이탈이었다
(과소예측 ratio 2.80, 포레스트릿 단독 3.95). "곤지암 스키장" 원본 검색비율은 겨울
3개월(12~2월)에 뚜렷이 몰려 있고(1월 평균 78, 12월 59, 2월 42 vs 3월 1.9), ski 업장군
잔차와 상관 r=+0.51(td 당일, 순환이동 플라시보 96%ile — 우연 범위 밖)·+0.38(td 기준
lag7, 94%ile)로 저비용 스크리닝을 통과했다(상세: eda_naver2_screen.py 결과).

가설: origin(창이 끝나는 시점) 기준 최근 7일 검색량 평균이, ski 업장군에만 게이팅되어
주입되면 12월 개막 램프의 구조적 과소예측을 교정한다. TB2b의 교훈(리조트 전체에 뿌리면
나머지 7개 업장에 잡음)을 따라 처음부터 게이팅한다(features.py `ski_gate`).

⚠️ 알려진 리스크: TB4(당일, "곤지암리조트"+"화담숲")도 이미 4/4 통과하고 FAR-겨울에서
크게 이겼던 전례가 있다 — "겨울철 검색 관심도가 높다"는 같은 사실의 재확인일 가능성.
차이점은 이번엔 업장군 게이팅과 origin 기준 배포 가능 설계라는 것.

예상치: prod 평균 개선 0 ~ +0.010, 겨울 폴드(F2·FAR-겨울)에 집중될 것으로 예상. 스크린
단계의 r이 td 당일 기준이라 origin-lag7 평균으로 바꾸면서 다소 희석될 수 있음(TB2c가
TB2/TB2b 대비 겪은 것과 같은 패턴) — 낙관하지 않는다.

게이트: prod 평균 > +0.0015 AND prod 4/4 일관 AND raw 4/4 양수 (기존 TB 시리즈와 동일).
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

GROUP = "tb_naver_ski"
CAND_KEYS = ["naver_ski_last3_g"]
ARMS = [("L0 base(54)", []), (f"TB7 +naver_ski({54 + len(CAND_KEYS)})", CAND_KEYS)]
CANDIDATE = ARMS[1][0]
OUT = os.path.join(C.EXPERIMENTS, "phase_tb7_naver_ski.json")

FOLDS = [("F2 겨울", "2023-11-24", "2023-11-24", "2024-02-22"),
         ("F3 봄", "2024-02-23", "2024-02-23", "2024-06-08"),
         ("FAR-봄", "2023-11-24", "2024-02-23", "2024-06-08"),
         ("FAR-겨울", "2023-08-25", "2023-11-24", "2024-02-22")]


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

    keep_all = F.active_columns(include=(GROUP,))
    names_all = F.active_names(include=(GROUP,))
    base_sub = [i for i, n in enumerate(names_all)
                if n not in DROP3 and n not in CAND_KEYS]
    print(f"전체 활성 {len(names_all)}개 · base {len(base_sub)}개 "
          f"(= v24 54개인가: {len(base_sub) == 54})", flush=True)

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
        c = res[(fname, CANDIDATE)]
        print(f"  [{fname:<8s}] base prod {b['prod']:.5f} · raw {b['raw']:.5f}"
              f" · 후보 prod {b['prod']-c['prod']:+.5f} · raw {b['raw']-c['raw']:+.5f}"
              f"  ({time.time()-t0:.0f}s)", flush=True)

    print("\n" + "=" * 90)
    for tag, nm in (("prod", "실제 파이프라인 (seg+스냅)"), ("raw", "raw (하한만) — 규칙 ⑲")):
        gs = [res[(f[0], "L0 base(54)")][tag] - res[(f[0], CANDIDATE)][tag] for f in FOLDS]
        n = sum(g > 0 for g in gs)
        print(f"\n[{nm}]  base 대비 개선 (+가 좋음)")
        print("  " + " ".join(f"{f[0]}:{g:+.5f}" for f, g in zip(FOLDS, gs)))
        print(f"  평균 {np.mean(gs):+.5f} · 일관 {n}/4")

    gp = [res[(f[0], "L0 base(54)")]["prod"] - res[(f[0], CANDIDATE)]["prod"] for f in FOLDS]
    gr = [res[(f[0], "L0 base(54)")]["raw"] - res[(f[0], CANDIDATE)]["raw"] for f in FOLDS]
    ok = (np.mean(gp) > 0.0015) and (sum(g > 0 for g in gp) == 4) \
        and (sum(g > 0 for g in gr) == 4)
    print("\n" + "=" * 90)
    print(f"📌 사전등록 판정 — 후보 {CANDIDATE}")
    print(f"   prod 평균 {np.mean(gp):+.5f} (>+0.0015?) · prod 일관 "
          f"{sum(g>0 for g in gp)}/4 · raw 일관 {sum(g>0 for g in gr)}/4")
    print(f"   → {'✅ 통과' if ok else '❌ 탈락'}")
    print("=" * 90)
    json.dump({f"{k[0]}|{k[1]}": v for k, v in res.items()},
              open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"저장: {os.path.basename(OUT)} · 총 {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
