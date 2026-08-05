# -*- coding: utf-8 -*-
"""Phase 9-c — 메뉴 '이름' 군집 피처 1개를 추가하면 성적이 오르는가.

━━ 동기 ━━
**같은 메뉴가 여러 영업장에 흩어져 있는데 지금은 완전히 남남이다.**
    스프라이트 6곳 · 아메리카노 7곳 · 공깃밥 5곳 · 코카콜라 3곳 · 카페라떼 5곳
`item_id` 는 이들을 전부 별개 품목으로 본다. 그래서 **화담숲카페(품목 5개뿐)의 아메리카노**는
포레스트릿 아메리카노에서 아무것도 못 빌려온다.
공식 지표는 **드문 품목에도 1표**를 주므로, 이 지점이 실제 점수에 걸린다.

━━ 설계 (Phase 9-b 실패에서 배운 것 반영) ━━
리조트 피처는 6개를 통째로 넣었다가 기각됐다. 실패 경로가 둘이었다:
    ① 품목마다 관계 방향이 반대 (화담숲은 리조트가 붐빌 때 휴점)
    ② 모든 품목에 같은 값 = **'날짜 도장'** → 암기 통로
이름 군집은 **둘 다 해당 없다** — 품목 고정 속성이라 시간과 무관하고, 품목별로 값이 다르다.
그리고 **벡터 8차원 대신 범주형 1개**로 압축해 피처 예산 경쟁을 최소화했다.

군집 수 k 를 3가지 시험한다. 작으면 기존 `category`(6분류)와 비슷해지고,
크면 `item_id` 와 비슷해져 빌려오는 효과가 사라진다. 그 사이 어딘가가 최적일 것이다.

━━ 채점 규약 (Phase 9-a 확정) ━━
**폴드별 채점 · 균등 가중 · v9 보정 적용**. 이 저울이 실제 순위를 4/4 재현했다.
균등 가중은 차이를 절반쯤 축소하므로 **"내부 차이 × 2 ≈ 실제 차이"** 로 환산한다.
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
KS = [12, 25, 45]
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
    BASE_KEEP = F.active_columns()                       # 확정 57개 (name 제외)
    NAME_COL = fn_all.index("name_cluster")
    WITH_KEEP = BASE_KEEP + [NAME_COL]                   # 58개

    print("=" * 96)
    print("Phase 9-c — 메뉴 이름 군집 피처")
    print("=" * 96)
    print(f"  기존 {len(BASE_KEEP)}개 · 추가 시 {len(WITH_KEEP)}개 (범주형 1개만 늘어남)")
    print(f"  군집 수 시험: {KS}\n")

    # 군집이 무엇을 묶는지 (k=25 기준 표본)
    ctx.set_name_k(25)
    from collections import Counter
    cnt = Counter(ctx.name_cluster.tolist())
    multi = [c for c, n in cnt.items() if n >= 3]
    print("  [k=25일 때 여러 영업장에 걸친 군집 표본]")
    shown = 0
    for c in sorted(multi, key=lambda c: -cnt[c]):
        idx = np.where(ctx.name_cluster == c)[0]
        sts = set(np.array(ctx.store_of_item)[idx])
        if len(sts) >= 3 and shown < 6:
            ms = [ctx.menus[i] for i in idx][:4]
            print(f"    군집 {c:>2d} ({cnt[c]:>2d}품목 · 영업장 {len(sts)}곳): {' · '.join(ms)}")
            shown += 1
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
        # 품목 인덱스를 보관해두면 k를 바꿀 때 행렬을 다시 안 만들어도 된다
        folds.append(dict(name=fname, Xt=Xtr[m], yt=np.log1p(np.maximum(ytr[m], 1.0)),
                          it=mtr[m, 2], Xv=Xva, iv=mva[:, 2],
                          y=yva, iids=mva[:, 2]))
        del Xtr, Xva
        print(f"  [{fname:<8s}] 학습 {folds[-1]['Xt'].shape[0]:>7,}행")
    print()

    def calib(p):
        return p * np.asarray(V9_C)[np.digitize(p, V9_BND)]

    def sc(d, p, high=1.0):
        return competition_score(d["y"], np.maximum(calib(p), FLOOR), d["iids"],
                                 ctx.store_of_item, make_weights(high), ctx.n)

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

    results = {}
    t1 = time.time()
    results["기존 57개"] = run(BASE_KEEP)
    print(f"  기존 57개 학습 완료 {time.time()-t1:.0f}s", flush=True)

    for k in KS:
        # 이름 군집은 **품목 이름만으로** 정해지므로, 보관해둔 품목 인덱스로
        # 해당 열만 갈아끼우면 된다 (행렬 재생성 불필요 · 폴드 무관).
        ctx.set_name_k(k)
        for d in folds:
            d["Xt"][:, NAME_COL] = ctx.name_cluster[d["it"]]
            d["Xv"][:, NAME_COL] = ctx.name_cluster[d["iv"]]
        t1 = time.time()
        results[f"이름군집 k={k}"] = run(WITH_KEEP)
        print(f"  이름군집 k={k} 학습 완료 {time.time()-t1:.0f}s", flush=True)

    base_lab = "기존 57개"
    b = [sc(d, p) for d, p in zip(folds, results[base_lab][0])]

    print("\n" + "=" * 96)
    print("① 성적 (v9 보정 · 폴드별 채점 · 균등 가중)")
    print("=" * 96)
    print(f"  {'':<16s}" + "".join(f"{d['name']:>11s}" for d in folds)
          + f"{'평균':>10s}{'개선':>10s}{'일관':>6s}{'실제환산':>10s}")
    rows = []
    for lab, (preds, _) in results.items():
        s = [sc(d, p) for d, p in zip(folds, preds)]
        g = float(np.mean(b) - np.mean(s))
        ok = all(s[i] < b[i] for i in range(4))
        rows.append(dict(lab=lab, s=s, mean=float(np.mean(s)), gain=g, ok=bool(ok)))
        mark = "  ←현행" if lab == base_lab else ""
        print(f"  {lab:<16s}" + "".join(f"{x:>11.4f}" for x in s)
              + f"{np.mean(s):>10.4f}{g:>+10.4f}{'○' if ok else '×':>5s}"
              + f"{2*g:>+10.4f}{mark}")

    print("\n" + "=" * 96)
    print("② 모델이 이름 군집을 쓰는가 (gain 비중)")
    print("=" * 96)
    for lab, (_, imp) in results.items():
        if "name_cluster" in imp:
            rank = sorted(imp, key=lambda x: -imp[x]).index("name_cluster") + 1
            print(f"  {lab:<16s} {100*imp['name_cluster']:>6.2f}%   "
                  f"{len(imp)}개 중 {rank}위")

    print("\n" + "=" * 96)
    print("③ 영업장별 — 품목 수가 적은 곳에서 이득이 나는가 (최고 후보 기준)")
    print("=" * 96)
    best = min([r for r in rows if r["lab"] != base_lab], key=lambda r: r["mean"])
    y = np.concatenate([d["y"] for d in folds])
    ii = np.concatenate([d["iids"] for d in folds])
    ps = {}
    for lab in (base_lab, best["lab"]):
        p = np.concatenate([np.maximum(calib(x), FLOOR)
                            for x in results[lab][0]])
        _, per, _, _ = competition_score(y, p, ii, ctx.store_of_item, None,
                                         ctx.n, return_parts=True)
        ps[lab] = per
    print(f"  {'영업장':<14s}{'품목':>6s}{'기존':>10s}{best['lab']:>14s}{'개선':>10s}")
    for s in sorted(ps[base_lab], key=lambda k: int((np.array(ctx.store_of_item) == k).sum())):
        n = int((np.array(ctx.store_of_item) == s).sum())
        print(f"  {s:<14s}{n:>6d}{ps[base_lab][s]:>10.4f}"
              f"{ps[best['lab']][s]:>14.4f}"
              f"{ps[base_lab][s]-ps[best['lab']][s]:>+10.4f}")
    print("\n  ※ 품목 적은 곳(화담숲카페 5 · 화담숲주막 8 · 포레스트릿 12)에서 이득이 크면 가설대로다.")

    print("\n" + "=" * 96)
    print("판정")
    print("=" * 96)
    print(f"  최고 후보 {best['lab']}  개선 {best['gain']:+.4f} · 4폴드 일관 "
          f"{'○' if best['ok'] else '×'} · 실제 환산 약 {2*best['gain']:+.4f}")
    if best["gain"] > SIGMA2 and best["ok"]:
        print("  → ★ 채택 권고. 제출로 확인.")
    elif best["gain"] > 0 and best["ok"]:
        print("  → ⚠️ 2σ 미달이나 4폴드 일관. 환산치가 2σ를 넘으면 제출 검토.")
    else:
        print("  → 기각.")

    json.dump(dict(ks=KS, rows=rows,
                   importance={lab: {k: float(v) for k, v in imp.items()
                                     if k == "name_cluster"}
                               for lab, (_, imp) in results.items()},
                   per_store={k: {s: float(v) for s, v in d.items()}
                              for k, d in ps.items()}),
              open(os.path.join(C.EXPERIMENTS, "phase9c_namecluster.json"),
                   "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n저장: experiments/phase9c_namecluster.json")
    print(f"총 {(time.time()-t0)/60:.0f}분")


if __name__ == "__main__":
    main()
