# -*- coding: utf-8 -*-
"""Phase 9-d — 이름 그룹 재설계본 평가. (9-c의 KMeans 방식을 대체)

━━ 9-c에서 배운 것 ━━
KMeans(k 고정)로 193품목을 억지로 나눴더니 **k=12에서 129품목(67%)이 한 덩어리**가 됐다.
그런데 **그 k=12가 제일 좋았다**(+0.0026). 뒤집어 보면:
    k=12 = 의미 있는 그룹 11개 + 나머지 전부 한 덩어리
즉 **확실한 것만 묶고 나머지는 안 건드릴 때 가장 잘 됐다.** k를 늘려 애매한 것까지
억지로 쪼개면(25, 45) 오히려 나빠졌다.

━━ 재설계 ━━
거리 임계값 기반 병합. 가까운 것만 붙고 먼 것은 **혼자 남는다.**
혼자 남은 품목의 그룹 ID 는 사실상 `item_id` 와 같아 **정보 손실도 가짜 그룹도 없다.**
9-c의 "129품목 한 덩어리"는 '이것들은 다 같은 종류'라는 **거짓 신호**였고,
그게 F2 폴드에서 나빠진 원인일 수 있다.

임계값 0.35 기준: 2품목 이상 그룹 23개(65품목) + 단독 128개, 최대 그룹 7품목, **쓰레기 그룹 없음.**

━━ 9-c에서 확인된 것(재설계본에서도 봐야 할 것) ━━
품목 수가 적은 영업장일수록 이득이 컸다(화담숲카페 5품목 +0.0083 · 화담숲주막 8품목 +0.0062
· 담하 42품목 −0.0017). **가설대로 '드문 품목이 정보를 빌려온다'가 작동한다.**
재설계본에서 그 패턴이 더 선명해지는지 본다.

채점: 폴드별 · 균등 가중 · v9 보정 적용 (Phase 9-a에서 검증된 저울).
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
THRS = [0.25, 0.35, 0.45]
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
    BASE_KEEP = F.active_columns()
    NAME_COL = fn_all.index("name_cluster")
    WITH_KEEP = BASE_KEEP + [NAME_COL]

    print("=" * 96)
    print("Phase 9-d — 이름 그룹 (거리 임계값 방식)")
    print("=" * 96)
    from collections import Counter
    for thr in THRS:
        ctx.set_name_threshold(thr)
        cnt = Counter(ctx.name_cluster.tolist())
        multi = {c: n for c, n in cnt.items() if n > 1}
        print(f"  임계값 {thr}  →  그룹 {len(cnt)}개 · 2품목이상 {len(multi)}개"
              f"(품목 {sum(multi.values())}개) · 단독 {len(cnt)-len(multi)}개"
              f" · 최대 {max(cnt.values())}품목")
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

    results = {}
    t1 = time.time()
    results["기존 57개"] = run(BASE_KEEP)
    print(f"  기존 57개 학습 완료 {time.time()-t1:.0f}s", flush=True)
    for thr in THRS:
        ctx.set_name_threshold(thr)
        for d in folds:
            d["Xt"][:, NAME_COL] = ctx.name_cluster[d["it"]]
            d["Xv"][:, NAME_COL] = ctx.name_cluster[d["iv"]]
        t1 = time.time()
        results[f"이름그룹 t={thr}"] = run(WITH_KEEP)
        print(f"  이름그룹 t={thr} 학습 완료 {time.time()-t1:.0f}s", flush=True)

    base_lab = "기존 57개"
    b = [sc(d, p) for d, p in zip(folds, results[base_lab][0])]

    print("\n" + "=" * 96)
    print("① 성적 (v9 보정 · 폴드별 채점 · 균등 가중)")
    print("=" * 96)
    print(f"  {'':<16s}" + "".join(f"{d['name']:>11s}" for d in folds)
          + f"{'평균':>10s}{'개선':>10s}{'일관':>6s}")
    rows = []
    for lab, (preds, _) in results.items():
        s = [sc(d, p) for d, p in zip(folds, preds)]
        g = float(np.mean(b) - np.mean(s))
        ok = all(s[i] < b[i] for i in range(4))
        rows.append(dict(lab=lab, s=s, mean=float(np.mean(s)), gain=g, ok=bool(ok)))
        mark = "  ←현행" if lab == base_lab else ""
        print(f"  {lab:<16s}" + "".join(f"{x:>11.4f}" for x in s)
              + f"{np.mean(s):>10.4f}{g:>+10.4f}{'○' if ok else '×':>5s}{mark}")
    print(f"\n  [참고] 9-c KMeans 최고(k=12) 는 +0.0026 · 일관 × 였다.")

    print("\n" + "=" * 96)
    print("② gain 비중")
    print("=" * 96)
    for lab, (_, imp) in results.items():
        if "name_cluster" in imp:
            rank = sorted(imp, key=lambda x: -imp[x]).index("name_cluster") + 1
            print(f"  {lab:<16s} {100*imp['name_cluster']:>6.2f}%   {len(imp)}개 중 {rank}위")

    print("\n" + "=" * 96)
    print("③ 영업장별 — 품목 적은 곳에서 이득이 나는가 (최고 후보)")
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
    print(f"  {'영업장':<14s}{'품목':>6s}{'기존':>10s}{'재설계':>10s}{'개선':>10s}")
    for s in sorted(ps[base_lab], key=lambda k: int((st == k).sum())):
        print(f"  {s:<14s}{int((st==s).sum()):>6d}{ps[base_lab][s]:>10.4f}"
              f"{ps[best['lab']][s]:>10.4f}{ps[base_lab][s]-ps[best['lab']][s]:>+10.4f}")

    print("\n" + "=" * 96)
    print("판정")
    print("=" * 96)
    print(f"  최고 {best['lab']}  개선 {best['gain']:+.4f} · 4폴드 일관 "
          f"{'○' if best['ok'] else '×'} · 문턱 2σ={SIGMA2}")
    if best["gain"] > SIGMA2 and best["ok"]:
        print("  → ★ 채택 권고. 제출로 확인.")
    elif best["gain"] > 0 and best["ok"]:
        print("  → ⚠️ 2σ 미달이나 4폴드 일관. 제출 1회로 확인할 값어치 있음.")
    else:
        print("  → 기각 또는 판정 불가.")

    json.dump(dict(thrs=THRS, rows=rows,
                   per_store={k: {s: float(v) for s, v in d.items()}
                              for k, d in ps.items()}),
              open(os.path.join(C.EXPERIMENTS, "phase9d_namegroup.json"),
                   "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n저장: experiments/phase9d_namegroup.json")
    print(f"총 {(time.time()-t0)/60:.0f}분")


if __name__ == "__main__":
    main()
