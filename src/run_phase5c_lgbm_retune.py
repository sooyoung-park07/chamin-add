# -*- coding: utf-8 -*-
"""Phase 5-c — LightGBM 좁은 재탐색.

XGBoost 40회 탐색(Phase 5-b)의 상위 10개가 한 방향을 가리켰다:
**"행은 다 쓰고(subsample 1.0) 잎은 작게 허용하고(mcw 10) 열만 줄인다(col 0.6~0.7)."**
그런데 현행 LightGBM은 정반대다 — bagging 0.85 / min_data_in_leaf 40 / feature_fraction 0.85.

그래서 여기서는 **무작위 탐색을 하지 않는다.** 이미 방향을 아는데 40번 뒤지는 건 낭비이고,
설정을 많이 볼수록 승자의 저주만 커진다. 대신 **한 번에 하나씩 바꾸는 사다리**로
"어느 축이 실제로 갈랐는가"까지 같이 알아낸다.

규약은 Phase 5-b와 동일:
  - F2+F3만 평가 (판정에 안 쓰는 F1은 제외)
  - **탐색 시드 (42,7) / 확인 시드 (2024,913,31) 분리**
  - 확인 개선폭이 2σ(0.003) 미만이면 기존 유지
  - ⚠️ Phase 5-b 교훈: CV 개선폭은 LB에서 **0.4배**로 줄었다. 폴드별 분해를 반드시 함께 볼 것.
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

SEARCH_SEEDS = (42, 7)
CONFIRM_SEEDS = (2024, 913, 31)
FLOOR = 1.0
SIGMA2 = 0.003
NT = os.cpu_count()

# 현행 v1 설정 (model_lgbm.PARAMS + ROUNDS) = 모든 비교의 원점
BASE = dict(objective="regression_l1", metric="l1", learning_rate=0.05,
            num_leaves=255, min_data_in_leaf=40, feature_fraction=0.85,
            bagging_fraction=0.85, bagging_freq=1, lambda_l2=1.0, lambda_l1=0.0,
            verbosity=-1, num_threads=NT)
BASE_ROUNDS = 1000

# XGB 방향을 하나씩 → 한꺼번에. (rounds 는 별도 키로 뺀다)
LADDER = [
    ("L0 현행(기준선)", {}, BASE_ROUNDS),
    ("L1 −bagging",     dict(bagging_fraction=1.0, bagging_freq=0), BASE_ROUNDS),
    ("L2 잎최소 40→10",  dict(min_data_in_leaf=10), BASE_ROUNDS),
    ("L3 열 0.85→0.65",  dict(feature_fraction=0.65), BASE_ROUNDS),
    ("L4 셋 다",         dict(bagging_fraction=1.0, bagging_freq=0,
                             min_data_in_leaf=10, feature_fraction=0.65), BASE_ROUNDS),
    ("L5 셋다+규제",      dict(bagging_fraction=1.0, bagging_freq=0,
                             min_data_in_leaf=10, feature_fraction=0.65,
                             lambda_l2=3.0, lambda_l1=2.0), BASE_ROUNDS),
    ("L6 셋다+1400r",    dict(bagging_fraction=1.0, bagging_freq=0,
                             min_data_in_leaf=10, feature_fraction=0.65), 1400),
    ("L7 L5+1400r",     dict(bagging_fraction=1.0, bagging_freq=0,
                             min_data_in_leaf=10, feature_fraction=0.65,
                             lambda_l2=3.0, lambda_l1=2.0), 1400),
    ("L8 L5+잎511",      dict(bagging_fraction=1.0, bagging_freq=0,
                             min_data_in_leaf=10, feature_fraction=0.65,
                             lambda_l2=3.0, lambda_l1=2.0, num_leaves=511),
     BASE_ROUNDS),
]
# ※ 원래 L7 은 `min_data_in_leaf=5` 였으나 스모크에서 L4(=10)와 **소수점까지 동일**하게 나왔다.
#   버그가 아니라 실제 현상 — 잎 255개에 학습행 18만~25만이면 잎당 평균 700행이라
#   임계값 10이든 5든 **한 번도 걸리지 않는다**(트리가 완전히 동일해짐).
#   → LightGBM 에서 `min_data_in_leaf` 는 이 규모에서 사실상 레버가 아니다. 더 쓸모 있는
#     축(규제 조합 · 부스팅 양 · 잎 개수)으로 교체.


def build_folds(fold_idx):
    """폴드별 **원시 배열**만 만들어 둔다.

    ⚠️ Dataset 을 폴드당 하나 만들어 설정마다 재사용하면 안 된다.
      LightGBM 은 Dataset 을 만들 때 `min_data_in_leaf` 로 **피처를 미리 걸러낸다**
      (`feature_pre_filter`). 그래서 40으로 만든 Dataset 에 10을 주면 거부당하고,
      스모크에서 실제로 L4(잎최소10)와 L7(잎최소5)가 **소수점까지 동일한 값**이 나왔다
      — 파라미터가 아예 안 먹은 것이다.
      → Dataset 은 **설정마다 새로 만든다.** (구성 비용은 폴드당 수 초)
    """
    ctx = F.Context()
    tr = D.load_train()
    mat, dates = D.to_matrix(tr, ctx.items)
    nd = mat.shape[1]
    fn_all = F.feature_names()
    keep = [i for i, k in enumerate(fn_all) if k not in set(F.PROF_KEYS)]
    names = [fn_all[i] for i in keep]
    cats = [c for c in F.CATEGORICAL if c in names]

    out = []
    for fi in fold_idx:
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
        del Xtr
        out.append(dict(name=fname, Xt=Xt, yt=yt, names=names, cats=cats,
                        Xv=np.ascontiguousarray(Xva[:, keep]),
                        y=yva, iids=mva[:, 2]))
        del Xva
        print(f"  [{fname}] 행렬 준비 완료  학습 {Xt.shape}", flush=True)
    return out, ctx


def evaluate(over, rounds, fds, seeds, ctx):
    p = dict(BASE, **over)
    scores = []
    for fd in fds:
        # 설정마다 새로 만든다 (위 build_folds 주석의 feature_pre_filter 이유)
        ds = lgb.Dataset(fd["Xt"], label=fd["yt"], feature_name=fd["names"],
                         categorical_feature=fd["cats"], free_raw_data=False)
        preds = []
        for sd in seeds:
            mdl = lgb.train(dict(p, seed=sd), ds, num_boost_round=rounds)
            preds.append(np.expm1(mdl.predict(fd["Xv"])))
        pr = np.maximum(np.mean(preds, 0), FLOOR)
        scores.append(competition_score(fd["y"], pr, fd["iids"],
                                        ctx.store_of_item, make_weights(1.0), ctx.n))
    return scores


def main():
    t0 = time.time()
    print("=" * 96)
    print("Phase 5-c — LightGBM 좁은 재탐색 (XGB 탐색이 가리킨 방향으로만)")
    print("=" * 96)
    fds, ctx = build_folds([1, 2])
    print()

    rows = []
    print(f"  {'설정':<18s} {'F2 겨울':>9s} {'F3 봄':>9s} {'F2+F3':>9s} "
          f"{'기준선 대비':>12s} {'시간':>7s}")
    base_s = None
    for label, over, rounds in LADDER:
        t1 = time.time()
        s = evaluate(over, rounds, fds, SEARCH_SEEDS, ctx)
        m = float(np.mean(s))
        if base_s is None:
            base_s = m
        d = base_s - m                       # 양수 = 좋아짐
        rows.append(dict(label=label, over=over, rounds=rounds,
                         f2=s[0], f3=s[1], f2f3=m, gain=d))
        mark = "  ★" if d > SIGMA2 else ("  ·" if d > 0 else "")
        print(f"  {label:<18s} {s[0]:>9.4f} {s[1]:>9.4f} {m:>9.4f} "
              f"{d:>+12.4f}{mark} {time.time()-t1:>6.0f}s", flush=True)

    print("\n  (부호 규약: 양수 = 기준선보다 좋아짐 · ★ = 2σ 초과)")

    # ------------------------------------------------------------ 확인 단계
    cand = sorted([r for r in rows if r["label"] != rows[0]["label"]],
                  key=lambda r: r["f2f3"])[:3]
    print("\n" + "=" * 96)
    print(f"확인 — 상위 3개 + 기준선을 **한 번도 안 쓴 시드** {CONFIRM_SEEDS} 로 재측정")
    print("=" * 96)
    conf = []
    for r in [rows[0]] + cand:
        s = evaluate(r["over"], r["rounds"], fds, CONFIRM_SEEDS, ctx)
        m = float(np.mean(s))
        conf.append(dict(label=r["label"], over=r["over"], rounds=r["rounds"],
                         f2=s[0], f3=s[1], f2f3=m, search=r["f2f3"]))
        print(f"  {r['label']:<18s} 탐색 {r['f2f3']:.4f} → 확인 {m:.4f} "
              f"({m-r['f2f3']:+.4f})   F2 {s[0]:.4f} · F3 {s[1]:.4f}", flush=True)

    cbase = conf[0]["f2f3"]
    cbest = min(conf, key=lambda r: r["f2f3"])
    gain = cbase - cbest["f2f3"]

    print("\n" + "=" * 96)
    print("판정")
    print("=" * 96)
    print(f"  확인 기준선(현행)  {cbase:.4f}")
    print(f"  확인 최고          {cbest['label']} {cbest['f2f3']:.4f}   개선 {gain:+.4f}")
    if gain > SIGMA2:
        print(f"  → ★ 채택: {cbest['over']} · rounds {cbest['rounds']}")
        # 폴드 편중 경고 — Phase 5-b 에서 CV 개선의 60%가 LB로 안 옮겨온 원인
        d2 = conf[0]["f2"] - cbest["f2"]
        d3 = conf[0]["f3"] - cbest["f3"]
        print(f"     폴드별 개선: F2 {d2:+.4f} · F3 {d3:+.4f}")
        if min(d2, d3) <= 0 or max(d2, d3) > 3 * max(min(d2, d3), 1e-9):
            print("     ⚠️ 개선이 한 폴드에 몰려 있다 — LB 전이율이 낮을 수 있다 (E05 사례)")
        else:
            print("     ✅ 두 폴드에 고르게 분산 — E05보다 LB 전이가 잘 될 가능성")
    else:
        print(f"  → 개선폭이 2σ({SIGMA2}) 미만. **현행 설정 유지가 정답.**")

    # XGBoost 튜닝본과의 비교 (Phase 5-b 확인시드 기준)
    print(f"\n  참고: 튜닝 XGBoost 확인시드 F2+F3 = 0.4680 (LB 0.480)")
    print(f"        현행 LightGBM v1 = 0.4725 (LB 0.4818)")

    json.dump(dict(base=BASE, base_rounds=BASE_ROUNDS,
                   search_seeds=list(SEARCH_SEEDS),
                   confirm_seeds=list(CONFIRM_SEEDS),
                   ladder=rows, confirm=conf,
                   verdict=dict(base=cbase, best=cbest["label"],
                                best_score=cbest["f2f3"], gain=gain, sigma2=SIGMA2)),
              open(os.path.join(C.EXPERIMENTS, "phase5c_lgbm_retune.json"),
                   "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n저장: experiments/phase5c_lgbm_retune.json")
    print(f"총 {(time.time()-t0)/60:.0f}분")


if __name__ == "__main__":
    main()
