# -*- coding: utf-8 -*-
"""Phase 9-b — 리조트 전체 피처 6개를 추가하면 성적이 오르는가.

━━ 왜 이 피처인가 ━━
코드를 다시 읽다 확인한 사실: **기존 57개 피처에 다른 영업장 정보가 하나도 없다.**
담하 불고기를 예측할 때 모델이 보는 것은 자기 품목과 담하 전체뿐이고,
**"오늘 리조트가 전체적으로 붐빈다"는 사실에 구조적으로 접근할 수 없다.**

이 프로젝트에서 효과가 났던 피처의 조건이 정확히 이것이었다:
  · 휴점 보정(+0.0036) — 창 통계가 **실제로 왜곡**되던 것을 바로잡음   → 채택
  · 요일 프로파일(−0.0014) — 모델이 `item_id`×`dow`로 **이미 알 수 있던** 정보 → 기각
  · 교차 품목(+0.0010) — 구조적으로 새 정보이나 `st_mean`과 일부 중복 → 유지

추가 6개 (`RESORT_KEYS`):
    r_last7 / r_prev7 / r_trend / r_dow  — 리조트 9곳 합계의 최근 수준·추세·요일
    **r_share / r_share_l7**             — 자기 영업장 ÷ 리조트 전체
`r_share` 가 핵심이다. 기존 `x_store_*`에는 "리조트가 붐빔"과 "우리 가게가 붐빔"이
섞여 있는데, 이 비율이 그 둘을 갈라준다.

━━ 채점 규약 (Phase 9-a에서 확정) ━━
**폴드별 채점 · 균등 가중**을 쓴다. 이 저울이 실제 순위를 4/4 재현했다(순위상관 1.00).
"이어붙여 채점"은 두 번 틀렸으므로 쓰지 않는다.
단, 균등 가중은 차이를 **절반쯤 축소**해서 보므로 담하·미라시아 3배 가중도 병기한다.
최종 판정은 **v9 보정을 얹은 상태**(= 실제 제출 파이프라인)에서 한다.
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
FLOOR = 1.0
SIGMA2 = 0.0032
NT = os.cpu_count()
V9_BND, V9_C = [2.0, 10.0], [0.55, 0.90, 1.02]

PARAMS = dict(objective="regression_l1", metric="l1", learning_rate=0.05,
              num_leaves=127, min_data_in_leaf=40, feature_fraction=0.65,
              bagging_fraction=0.85, bagging_freq=1, lambda_l2=1.0,
              verbosity=-1, num_threads=NT)
ROUNDS = 1000

FOLDS = [("F2 겨울", "2023-11-24", "2023-11-24", "2024-02-22"),
         ("F3 봄", "2024-02-23", "2024-02-23", "2024-06-08"),
         ("FAR-봄", "2023-11-24", "2024-02-23", "2024-06-08"),
         ("FAR-겨울", "2023-08-25", "2023-11-24", "2024-02-22")]


def main():
    t0 = time.time()
    ctx = F.Context()
    tr = D.load_train()
    mat, dates = D.to_matrix(tr, ctx.items)
    nd = mat.shape[1]
    fn_all = F.feature_names()

    drop_base = set(F.PROF_KEYS) | set(F.RESORT_KEYS)     # 기존 57개
    drop_new = set(F.PROF_KEYS)                           # 신규 63개
    SETS = [("기존 57개", [i for i, k in enumerate(fn_all) if k not in drop_base]),
            ("리조트 추가 63개", [i for i, k in enumerate(fn_all) if k not in drop_new])]

    print("=" * 96)
    print("Phase 9-b — 리조트 전체 피처 6개 추가")
    print("=" * 96)
    for lab, keep in SETS:
        print(f"  {lab:<18s} {len(keep)}개")
    print(f"  추가되는 것: {F.RESORT_KEYS}\n")

    folds = []
    for fname, cut, v0, v1 in FOLDS:
        cut_col = int(np.searchsorted(np.array(dates), pd.Timestamp(cut)))
        ctx.set_proxy(F.pick_proxy_items(mat, dates, cut_col, ctx.store_codes))
        trn = [o for o in range(C.WINDOW - 1, nd - C.HORIZON)
               if dates[o] < pd.Timestamp(cut)]
        va = V.origins(dates, v0, v1, nd)
        Xtr, ytr, _ = F.build_samples(mat, dates, trn, ctx)
        Xva, yva, mva = F.build_samples(mat, dates, va, ctx)
        m = ytr != 0
        folds.append(dict(name=fname, Xt=Xtr[m], yt=np.log1p(np.maximum(ytr[m], 1.0)),
                          Xv=Xva, y=yva, iids=mva[:, 2]))
        del Xtr, Xva
        print(f"  [{fname:<8s}] 학습 {folds[-1]['Xt'].shape[0]:>7,}행")
    print()

    def calib(p):
        return p * np.asarray(V9_C)[np.digitize(p, V9_BND)]

    def sc(d, p, high=1.0, use_calib=True):
        q = calib(p) if use_calib else p
        return competition_score(d["y"], np.maximum(q, FLOOR), d["iids"],
                                 ctx.store_of_item, make_weights(high), ctx.n)

    res, imps = {}, {}
    for lab, keep in SETS:
        t1 = time.time()
        names = [fn_all[i] for i in keep]
        cats = [c for c in F.CATEGORICAL if c in names]
        preds, gains = [], np.zeros(len(names))
        for d in folds:
            ds = lgb.Dataset(np.ascontiguousarray(d["Xt"][:, keep]), label=d["yt"],
                             feature_name=names, categorical_feature=cats,
                             free_raw_data=False)
            ms = [lgb.train(dict(PARAMS, seed=sd), ds, num_boost_round=ROUNDS)
                  for sd in SEEDS]
            Xv = np.ascontiguousarray(d["Xv"][:, keep])
            preds.append(np.mean([np.expm1(m.predict(Xv)) for m in ms], 0))
            g = np.mean([m.feature_importance("gain") for m in ms], 0)
            gains += g / g.sum()
        res[lab] = preds
        imps[lab] = dict(zip(names, gains / len(folds)))
        print(f"  {lab:<18s} 학습 완료 {time.time()-t1:.0f}s", flush=True)

    base, new = SETS[0][0], SETS[1][0]
    print("\n" + "=" * 96)
    print("① 성적 (v9 보정 적용 · 폴드별 채점)")
    print("=" * 96)
    for high in (1.0, 3.0):
        lbl = "균등 가중" if high == 1.0 else "담하·미라시아 3배"
        b = [sc(d, p, high) for d, p in zip(folds, res[base])]
        n_ = [sc(d, p, high) for d, p in zip(folds, res[new])]
        ok = all(n_[i] < b[i] for i in range(4))
        print(f"\n  [{lbl}]")
        print(f"    {'':<18s}" + "".join(f"{d['name']:>11s}" for d in folds)
              + f"{'평균':>10s}")
        print(f"    {base:<18s}" + "".join(f"{x:>11.4f}" for x in b)
              + f"{np.mean(b):>10.4f}")
        print(f"    {new:<18s}" + "".join(f"{x:>11.4f}" for x in n_)
              + f"{np.mean(n_):>10.4f}")
        print(f"    {'개선':<18s}" + "".join(f"{b[i]-n_[i]:>+11.4f}" for i in range(4))
              + f"{np.mean(b)-np.mean(n_):>+10.4f}   4폴드 일관 {'○' if ok else '×'}")

    print("\n  [참고] 보정 없이")
    b0 = [sc(d, p, 1.0, False) for d, p in zip(folds, res[base])]
    n0 = [sc(d, p, 1.0, False) for d, p in zip(folds, res[new])]
    print(f"    {base} {np.mean(b0):.4f} · {new} {np.mean(n0):.4f}"
          f"  개선 {np.mean(b0)-np.mean(n0):+.4f}")

    print("\n" + "=" * 96)
    print("② 새 피처를 모델이 실제로 쓰는가 (gain 비중, 폴드 평균)")
    print("=" * 96)
    iv = imps[new]
    for k in F.RESORT_KEYS:
        rank = sorted(iv, key=lambda x: -iv[x]).index(k) + 1
        print(f"  {k:<14s} {100*iv[k]:>6.2f}%   전체 {len(iv)}개 중 {rank}위")
    print(f"\n  참고 상위 5개: " + " · ".join(
        f"{k}({100*iv[k]:.1f}%)" for k in sorted(iv, key=lambda x: -iv[x])[:5]))

    print("\n" + "=" * 96)
    print("③ 영업장별 — 어디서 좋아지나 (v9 보정 · 균등 가중)")
    print("=" * 96)
    y = np.concatenate([d["y"] for d in folds])
    ii = np.concatenate([d["iids"] for d in folds])
    ps = {}
    for lab in (base, new):
        p = np.concatenate([np.maximum(calib(x), FLOOR) for x in res[lab]])
        _, per, _, _ = competition_score(y, p, ii, ctx.store_of_item, None,
                                         ctx.n, return_parts=True)
        ps[lab] = per
    print(f"  {'영업장':<14s}{'품목':>6s}{'기존':>10s}{'추가':>10s}{'개선':>10s}")
    for s in sorted(ps[base], key=lambda k: ps[base][k] - ps[new][k], reverse=True):
        print(f"  {s:<14s}{int((np.array(ctx.store_of_item)==s).sum()):>6d}"
              f"{ps[base][s]:>10.4f}{ps[new][s]:>10.4f}"
              f"{ps[base][s]-ps[new][s]:>+10.4f}")

    print("\n" + "=" * 96)
    print("판정")
    print("=" * 96)
    b = [sc(d, p) for d, p in zip(folds, res[base])]
    n_ = [sc(d, p) for d, p in zip(folds, res[new])]
    g = float(np.mean(b) - np.mean(n_))
    ok = all(n_[i] < b[i] for i in range(4))
    print(f"  균등 가중 개선 {g:+.4f} · 4폴드 일관 {'○' if ok else '×'} · 문턱 2σ={SIGMA2}")
    print(f"  ※ 균등 가중은 실제 차이를 절반쯤 축소해 본다(Phase 9-a) → 실제로는 약 {2*g:+.4f} 기대")
    if g > SIGMA2 and ok:
        print("  → ★ 채택 권고. 제출로 확인.")
    elif g > 0 and ok:
        print("  → ⚠️ 2σ 미달이나 4폴드 일관. 축소 배율을 감안하면 제출해볼 값어치 있음.")
    else:
        print("  → 기각 또는 판정 불가.")

    json.dump(dict(base=b, new=n_, gain=g, consistent=bool(ok),
                   resort_importance={k: float(iv[k]) for k in F.RESORT_KEYS},
                   per_store={k: {s: float(v) for s, v in d.items()}
                              for k, d in ps.items()}),
              open(os.path.join(C.EXPERIMENTS, "phase9b_resort.json"),
                   "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n저장: experiments/phase9b_resort.json")
    print(f"총 {(time.time()-t0)/60:.0f}분")


if __name__ == "__main__":
    main()
