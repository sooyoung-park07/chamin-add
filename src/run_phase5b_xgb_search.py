# -*- coding: utf-8 -*-
"""Phase 5-b — XGBoost 하이퍼파라미터 탐색.

XGBoost를 먼저 하는 이유 (Phase 5-a 정찰 결과):
  같은 조건 무튜닝에서 F2+F3 0.4739 로 LightGBM(0.4777)보다 앞섰고,
  v1 LightGBM 최종본(0.4725, leaves255·1000라운드·시드5)에 거의 근접했다.
  게다가 우세 폭이 **학습 데이터가 많을수록 커진다** (F1 −0.0058 → F2 +0.0038 → F3 +0.0037).
  실제 제출은 origin 497개로 F3(391)보다 많으니 유리한 방향이다.

━━━ 승자의 저주 방지 (이 스크립트의 핵심 설계) ━━━
σ=0.0014 인 저울로 설정 40개를 재서 최솟값을 고르면, **실력 차이가 전혀 없어도**
순수한 운으로 평균보다 2σ 이상 좋은 값이 나온다. 그대로 채택하면 LB에서 증발한다.
그래서:
  1. **탐색 시드 (42,7) 와 확인 시드 (2024,913,31) 를 분리한다.**
     확인 시드는 탐색 중 한 번도 쓰지 않는다. 시드에 대한 train/test 분리다.
  2. argmin 하나가 아니라 **상위 10개가 모여 있는 영역**을 함께 본다.
  3. 최종 후보는 확인 시드로 재측정하고, 기준선 대비 **2σ(0.003) 미만이면 기존 유지**.

판정은 3폴드 평균이 아니라 **F2+F3**(LB 대리지표). 탐색 중에는 F1을 아예 돌리지 않는다
— 판정에 안 쓰는데 시간의 21%를 먹기 때문. F1은 최종 후보에서만 스트레스 테스트로 본다.
"""
import os
import sys
import json
import time

import numpy as np
import pandas as pd
import xgboost as xgb

import config as C
import dataio as D
import features as F
import validate as V
from metrics import competition_score, make_weights

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
SEARCH_SEEDS = (42, 7)
CONFIRM_SEEDS = (2024, 913, 31)
FLOOR = 1.0
SIGMA2 = 0.003
NT = os.cpu_count()
RNG = np.random.default_rng(0)          # 탐색 재현성

# 정찰에서 쓴 무튜닝 기준선 — 모든 비교의 원점
BASE = dict(max_depth=8, min_child_weight=40, eta=0.05, subsample=0.85,
            colsample_bytree=0.85, reg_lambda=1.0, reg_alpha=0.0,
            grow_policy="depthwise", rounds=600)


def sample_config():
    """무작위 설정 하나. grow_policy 에 따라 깊이 파라미터가 달라진다.

    lossguide 는 LightGBM 의 leaf-wise 성장과 같은 방식이다 —
    XGBoost 로도 LightGBM 쪽 동작을 흉내낼 수 있어서 탐색 축으로 넣을 값어치가 있다.
    """
    gp = RNG.choice(["depthwise", "lossguide"])
    eta = float(RNG.choice([0.03, 0.05, 0.08]))
    k = float(RNG.choice([30.0, 45.0]))          # 부스팅 총량 (eta × rounds)
    cfg = dict(
        grow_policy=str(gp), eta=eta, rounds=int(round(k / eta)),
        min_child_weight=int(RNG.choice([5, 10, 20, 40, 80, 160])),
        subsample=float(RNG.choice([0.7, 0.8, 0.9, 1.0])),
        colsample_bytree=float(RNG.choice([0.6, 0.7, 0.85, 1.0])),
        reg_lambda=float(RNG.choice([0.5, 1.0, 3.0, 10.0, 30.0])),
        reg_alpha=float(RNG.choice([0.0, 0.5, 2.0])),
    )
    if gp == "depthwise":
        cfg["max_depth"] = int(RNG.choice([6, 7, 8, 9, 10]))
    else:
        cfg["max_depth"] = 0                      # 0 = 무제한 (leaf-wise)
        cfg["max_leaves"] = int(RNG.choice([63, 127, 255]))
    return cfg


def to_params(cfg, seed):
    p = dict(objective="reg:absoluteerror", tree_method="hist",
             max_cat_to_onehot=1,          # 193레벨 item_id 를 원-핫으로 터뜨리지 않기
             nthread=NT, seed=seed)
    for k, v in cfg.items():
        if k != "rounds":
            p[k] = v
    return p


def build_folds(fold_idx):
    """폴드별 DMatrix 를 **한 번만** 만들어 재사용한다 (설정마다 다시 만들면 낭비)."""
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
        Xt, Xv = Xtr[m][:, keep], Xva[:, keep]
        yt = np.log1p(np.maximum(ytr[m], 1.0))
        del Xtr, Xva

        dft = pd.DataFrame(Xt, columns=names)
        dfv = pd.DataFrame(Xv, columns=names)
        for c in cats:
            j = names.index(c)
            lv = np.unique(np.concatenate([Xt[:, j], Xv[:, j]])).astype(int)
            dft[c] = pd.Categorical(dft[c].astype(int), categories=lv)
            dfv[c] = pd.Categorical(dfv[c].astype(int), categories=lv)
        out.append(dict(name=fname,
                        dtr=xgb.DMatrix(dft, label=yt, enable_categorical=True),
                        dva=xgb.DMatrix(dfv, enable_categorical=True),
                        y=yva, iids=mva[:, 2]))
        del Xt, Xv, dft, dfv
        print(f"  [{fname}] DMatrix 준비 완료", flush=True)
    return out, ctx


def evaluate(cfg, fds, seeds, ctx):
    """설정 하나를 시드별로 학습 → 시드 평균 → 폴드별 점수 리스트."""
    scores = []
    for fd in fds:
        preds = []
        for sd in seeds:
            b = xgb.train(to_params(cfg, sd), fd["dtr"], num_boost_round=cfg["rounds"])
            preds.append(np.expm1(b.predict(fd["dva"])))
        p = np.maximum(np.mean(preds, 0), FLOOR)
        scores.append(competition_score(fd["y"], p, fd["iids"], ctx.store_of_item,
                                        make_weights(1.0), ctx.n))
    return scores


def fmt(cfg):
    d = (f"leaves{cfg['max_leaves']}" if cfg["grow_policy"] == "lossguide"
         else f"depth{cfg['max_depth']}")
    return (f"{cfg['grow_policy'][:4]}/{d} eta{cfg['eta']}×{cfg['rounds']} "
            f"mcw{cfg['min_child_weight']} sub{cfg['subsample']} "
            f"col{cfg['colsample_bytree']} L2:{cfg['reg_lambda']} L1:{cfg['reg_alpha']}")


def main():
    t0 = time.time()
    print("=" * 100)
    print(f"Phase 5-b — XGBoost 탐색 {N_TRIALS}개  "
          f"(탐색시드 {SEARCH_SEEDS} · 확인시드 {CONFIRM_SEEDS})")
    print("=" * 100)
    fds, ctx = build_folds([1, 2])          # F2 겨울 · F3 봄
    print()

    base = evaluate(BASE, fds, SEARCH_SEEDS, ctx)
    base_s = float(np.mean(base))
    print(f"기준선(무튜닝)  F2 {base[0]:.4f} · F3 {base[1]:.4f} → F2+F3 {base_s:.4f}\n")

    rows = []
    for t in range(N_TRIALS):
        cfg = sample_config()
        t1 = time.time()
        s = evaluate(cfg, fds, SEARCH_SEEDS, ctx)
        m = float(np.mean(s))
        rows.append(dict(cfg=cfg, f2=s[0], f3=s[1], f2f3=m))
        best = min(r["f2f3"] for r in rows)
        print(f"  [{t+1:2d}/{N_TRIALS}] {m:.4f} ({m-base_s:+.4f})  "
              f"{time.time()-t1:5.0f}s  best {best:.4f}  |  {fmt(cfg)}", flush=True)

    rows.sort(key=lambda r: r["f2f3"])

    print("\n" + "=" * 100)
    print("탐색 결과 상위 10  ← argmin 하나가 아니라 '영역'을 본다")
    print("=" * 100)
    for i, r in enumerate(rows[:10]):
        print(f"  {i+1:2d}. {r['f2f3']:.4f} ({r['f2f3']-base_s:+.4f})  {fmt(r['cfg'])}")

    print("\n[상위 10 vs 전체 — 어떤 파라미터가 실제로 갈렸나]")
    num_keys = ["eta", "rounds", "min_child_weight", "subsample",
                "colsample_bytree", "reg_lambda", "reg_alpha", "max_depth"]
    for k in num_keys:
        top = [r["cfg"].get(k) for r in rows[:10] if r["cfg"].get(k) is not None]
        allv = [r["cfg"].get(k) for r in rows if r["cfg"].get(k) is not None]
        if top:
            print(f"   {k:<20s} 상위10 중앙값 {np.median(top):>8.3g}   "
                  f"전체 중앙값 {np.median(allv):>8.3g}")
    for k in ["grow_policy"]:
        top = [r["cfg"][k] for r in rows[:10]]
        allv = [r["cfg"][k] for r in rows]
        print(f"   {k:<20s} 상위10 {dict((v, top.count(v)) for v in set(allv))}   "
              f"전체 {dict((v, allv.count(v)) for v in set(allv))}")

    # ------------------------------------------------------------ 확인 단계
    print("\n" + "=" * 100)
    print(f"확인 — 상위 5개 + 기준선을 **한 번도 안 쓴 시드** {CONFIRM_SEEDS} 로 재측정")
    print("=" * 100)
    print("  탐색 점수와 확인 점수의 차이가 곧 '노이즈에 얼마나 과적합했나' 이다.\n")
    cands = [("기준선", BASE)] + [(f"상위{i+1}", rows[i]["cfg"]) for i in range(min(5, len(rows)))]
    conf = []
    for label, cfg in cands:
        s = evaluate(cfg, fds, CONFIRM_SEEDS, ctx)
        m = float(np.mean(s))
        srch = base_s if label == "기준선" else rows[int(label[2:]) - 1]["f2f3"]
        conf.append(dict(label=label, cfg=cfg, f2=s[0], f3=s[1], f2f3=m, search=srch))
        print(f"  {label:<8s} 탐색 {srch:.4f} → 확인 {m:.4f}  ({m-srch:+.4f})   "
              f"F2 {s[0]:.4f} · F3 {s[1]:.4f}", flush=True)

    cb = min(conf, key=lambda r: r["f2f3"])
    base_conf = [r for r in conf if r["label"] == "기준선"][0]["f2f3"]
    gain = base_conf - cb["f2f3"]

    print("\n" + "=" * 100)
    print("판정")
    print("=" * 100)
    print(f"  확인 기준선   {base_conf:.4f}")
    print(f"  확인 최고     {cb['label']} {cb['f2f3']:.4f}   개선 {gain:+.4f}")
    if gain > SIGMA2:
        print(f"  → ★ 채택. {fmt(cb['cfg'])}")
    else:
        print(f"  → 개선폭이 2σ({SIGMA2}) 미만. **기존 설정 유지가 정답.**")
        print("     탐색 점수에서 보이던 개선은 대부분 노이즈였다는 뜻이다.")

    json.dump(dict(n_trials=N_TRIALS, search_seeds=list(SEARCH_SEEDS),
                   confirm_seeds=list(CONFIRM_SEEDS), base=BASE,
                   base_search=base_s, trials=rows, confirm=conf,
                   verdict=dict(base_confirm=base_conf, best=cb["label"],
                                best_score=cb["f2f3"], gain=gain, sigma2=SIGMA2)),
              open(os.path.join(C.EXPERIMENTS, "phase5b_xgb_search.json"),
                   "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n저장: experiments/phase5b_xgb_search.json")
    print(f"총 {(time.time()-t0)/60:.0f}분")


if __name__ == "__main__":
    main()
