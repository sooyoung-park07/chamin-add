# -*- coding: utf-8 -*-
"""Phase 9-e — 이름 그룹의 '거친 쪽 끝'을 찾는다.

━━ 9-c / 9-d 에서 나온 추세 ━━
| 설정 | 레벨 수 | gain 순위 | 개선 |
|---|---|---|---|
| KMeans k=12 | 12 | 9위 | **+0.0026** ← 최고 |
| KMeans k=25 | 25 | 5위 | +0.0001 |
| KMeans k=45 | 45 | 4위 | +0.0006 |
| 임계값 t=0.25 | 156 | 2위 | +0.0003 |
| 임계값 t=0.35 | 151 | 2위 | −0.0013 |

**레벨이 적을수록 좋다. 그런데 12가 우리가 시험한 최소값이다** — 사다리 끝에서 이겼다.
(잎 개수 실험에서 `leaves 15` 가 끝에서 이겼던 것과 같은 상황.)

━━ 왜 거친 게 좋은가 (9-d에서 규명) ━━
레벨이 많아지면 `name_cluster` 가 **`item_id` 의 복사본**이 된다. 그러면
품목별 암기 통로가 두 배가 되고, Phase 5-e에서 확인한 "세밀한 기억은 시간이 지나면 낡는다"에 걸린다.
gain 은 오르는데(2위) 성적은 안 오르는 게 그 신호였다.

━━ 이번에 시험할 것 ━━
  ① KMeans k = 6 / 8 / 10   — 거친 쪽 끝 찾기
  ② **임계값 + 단독 병합**   — 새 설계. 깨끗한 그룹만 남기고 나머지는 '기타' 하나로.
     9-c k=12 의 "실질 그룹 11개 + 나머지 한 덩어리" 구조를, **억지 병합 없이** 재현한다.
     t=0.35 → 23그룹 + 기타 = 24레벨 · t=0.25 → 19그룹 + 기타 = 20레벨

채점: 폴드별 · 균등 가중 · v9 보정 (Phase 9-a 검증 저울).
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

# (이름, 설정 함수)
VARIANTS = [
    ("KMeans k=6", lambda c: c.set_name_kmeans(6)),
    ("KMeans k=8", lambda c: c.set_name_kmeans(8)),
    ("KMeans k=10", lambda c: c.set_name_kmeans(10)),
    ("임계0.25+기타", lambda c: c.set_name_threshold(0.25, True)),
    ("임계0.35+기타", lambda c: c.set_name_threshold(0.35, True)),
]

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
    BASE_KEEP = F.active_columns()
    NAME_COL = fn_all.index("name_cluster")
    WITH_KEEP = BASE_KEEP + [NAME_COL]

    print("=" * 96)
    print("Phase 9-e — 이름 그룹의 거친 쪽 끝 찾기")
    print("=" * 96)
    from collections import Counter
    for lab, setter in VARIANTS:
        setter(ctx)
        cnt = Counter(ctx.name_cluster.tolist())
        multi = {c: n for c, n in cnt.items() if n > 1}
        print(f"  {lab:<14s} 레벨 {ctx.n_name_levels:>3d}개 · 최대 그룹 "
              f"{max(cnt.values()):>3d}품목 · 2품목이상 {len(multi)}개")
    print()

    folds = []
    for fname, cut, v0, v1 in FOLDS:
        cut_col = int(np.searchsorted(np.array(dates), pd.Timestamp(cut)))
        ctx.set_proxy(F.pick_proxy_items(mat, dates, cut_col, ctx.store_codes))
        trn = [o for o in range(C.WINDOW - 1, nd - C.HORIZON)
               if dates[o] < pd.Timestamp(cut)]
        va = V.origins(dates, v0, v1, nd)
        Xtr, ytr, mtr = F.build_samples(mat, dates, trn, ctx)
        Xva, yva, mva = F.build_samples(mat, dates, va, ctx)
        m = ytr != 0
        folds.append(dict(name=fname, Xt=Xtr[m], yt=np.log1p(np.maximum(ytr[m], 1.0)),
                          it=mtr[m, 2], Xv=Xva, iv=mva[:, 2],
                          y=yva, iids=mva[:, 2]))
        del Xtr, Xva
        print(f"  [{fname:<8s}] 학습 {folds[-1]['Xt'].shape[0]:>7,}행")
    print()

    def calib(p):
        return p * np.asarray(V9_C)[np.digitize(p, V9_BND)]

    def sc(d, p):
        return competition_score(d["y"], np.maximum(calib(p), FLOOR), d["iids"],
                                 ctx.store_of_item, make_weights(1.0), ctx.n)

    def run(keep):
        names = [fn_all[i] for i in keep]
        cats = [c for c in F.CATEGORICAL if c in names]
        preds, imp = [], np.zeros(len(names))
        for d in folds:
            ds = lgb.Dataset(np.ascontiguousarray(d["Xt"][:, keep]), label=d["yt"],
                             feature_name=names, categorical_feature=cats,
                             free_raw_data=False)
            ms = [lgb.train(dict(PARAMS, seed=sd), ds, num_boost_round=ROUNDS)
                  for sd in SEEDS]
            Xv = np.ascontiguousarray(d["Xv"][:, keep])
            preds.append(np.mean([np.expm1(m.predict(Xv)) for m in ms], 0))
            g = np.mean([m.feature_importance("gain") for m in ms], 0)
            imp += g / g.sum()
        return preds, dict(zip(names, imp / len(folds)))

    results, levels = {}, {}
    t1 = time.time()
    results["기존 57개"] = run(BASE_KEEP)
    print(f"  기존 57개 학습 완료 {time.time()-t1:.0f}s", flush=True)
    for lab, setter in VARIANTS:
        setter(ctx)
        levels[lab] = ctx.n_name_levels
        for d in folds:
            d["Xt"][:, NAME_COL] = ctx.name_cluster[d["it"]]
            d["Xv"][:, NAME_COL] = ctx.name_cluster[d["iv"]]
        t1 = time.time()
        results[lab] = run(WITH_KEEP)
        print(f"  {lab} 학습 완료 {time.time()-t1:.0f}s", flush=True)

    base_lab = "기존 57개"
    b = [sc(d, p) for d, p in zip(folds, results[base_lab][0])]

    print("\n" + "=" * 96)
    print("① 성적 (v9 보정 · 폴드별 채점 · 균등 가중)")
    print("=" * 96)
    print(f"  {'':<14s}{'레벨':>5s}" + "".join(f"{d['name']:>11s}" for d in folds)
          + f"{'평균':>10s}{'개선':>10s}{'일관':>6s}{'gain순위':>9s}")
    rows = []
    for lab, (preds, imp) in results.items():
        s = [sc(d, p) for d, p in zip(folds, preds)]
        g = float(np.mean(b) - np.mean(s))
        ok = all(s[i] < b[i] for i in range(4))
        if "name_cluster" in imp:
            rk = sorted(imp, key=lambda x: -imp[x]).index("name_cluster") + 1
            gtxt = f"{rk}위({100*imp['name_cluster']:.1f}%)"
        else:
            gtxt = "—"
        rows.append(dict(lab=lab, levels=levels.get(lab, 0), s=s,
                         mean=float(np.mean(s)), gain=g, ok=bool(ok), gtxt=gtxt))
        print(f"  {lab:<14s}{levels.get(lab, 0):>5d}"
              + "".join(f"{x:>11.4f}" for x in s)
              + f"{np.mean(s):>10.4f}{g:>+10.4f}{'○' if ok else '×':>5s}{gtxt:>10s}")

    print("\n  [지난 실험 참고]  KMeans k=12 +0.0026(9위) · k=25 +0.0001(5위) · "
          "k=45 +0.0006(4위) · 임계0.25 +0.0003(2위) · 임계0.35 −0.0013(2위)")

    print("\n" + "=" * 96)
    print("② 영업장별 (최고 후보)")
    print("=" * 96)
    best = min([r for r in rows if r["lab"] != base_lab], key=lambda r: r["mean"])
    y = np.concatenate([d["y"] for d in folds])
    ii = np.concatenate([d["iids"] for d in folds])
    ps = {}
    for lab in (base_lab, best["lab"]):
        p = np.concatenate([np.maximum(calib(x), FLOOR) for x in results[lab][0]])
        _, per, _, _ = competition_score(y, p, ii, ctx.store_of_item, None,
                                         ctx.n, return_parts=True)
        ps[lab] = per
    st = np.array(ctx.store_of_item)
    print(f"  {'영업장':<14s}{'품목':>6s}{'기존':>10s}{'최고후보':>10s}{'개선':>10s}")
    for s in sorted(ps[base_lab], key=lambda k: int((st == k).sum())):
        print(f"  {s:<14s}{int((st==s).sum()):>6d}{ps[base_lab][s]:>10.4f}"
              f"{ps[best['lab']][s]:>10.4f}{ps[base_lab][s]-ps[best['lab']][s]:>+10.4f}")

    print("\n" + "=" * 96)
    print("판정")
    print("=" * 96)
    print(f"  최고 {best['lab']} (레벨 {best['levels']})  개선 {best['gain']:+.4f} · "
          f"4폴드 일관 {'○' if best['ok'] else '×'} · 문턱 2σ={SIGMA2}")
    if best["gain"] > SIGMA2 and best["ok"]:
        print("  → ★ 채택 권고. 제출로 확인.")
    elif best["gain"] > 0 and best["ok"]:
        print("  → ⚠️ 2σ 미달이나 4폴드 일관. 제출 1회로 확인할 값어치 있음.")
    else:
        print("  → 기각. **이름 그룹 축 종료.**")

    json.dump(dict(rows=rows, per_store={k: {s: float(v) for s, v in d.items()}
                                         for k, d in ps.items()}),
              open(os.path.join(C.EXPERIMENTS, "phase9e_coarse.json"),
                   "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n저장: experiments/phase9e_coarse.json")
    print(f"총 {(time.time()-t0)/60:.0f}분")


if __name__ == "__main__":
    main()
