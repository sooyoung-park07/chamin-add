# -*- coding: utf-8 -*-
"""Phase 7-a — 모델 구조 분리에 **여유가 있는가**. 만들기 전에 헤드룸부터 잰다.

질문: **전문가 모델이 글로벌 모델을 이기는 구간이 실제로 있는가?**

분리에는 대가가 있다 — 모델당 학습량이 줄어든다.
    세그먼트: 계절 72품목 / 상시 98 / B2B 23  → B2B 는 3.8만 행 (전체의 12%)
    horizon : 7등분                            → 각 4.6만 행 (전체의 14%)
이 프로젝트에서 **데이터가 줄면 답이 뒤집히는** 걸 세 번 봤다(item_id · F1 · FAR 폴드).
게다가 트리는 이미 `store`·`cluster` 로 뿌리에서 갈라친다 — 명시적 분리의 순이득이 없을 수도 있다.

━━ 설계 ━━
같은 폴드에서 글로벌 1개와 전문가 N개를 학습해, **각 구간에서 누가 이기는지** 직접 비교한다.
그리고 전문가들의 예측을 **조립해서 전체 점수**도 낸다 — 이게 실제로 궁금한 숫자다.

교란 차단: 전문가가 지면 "분리가 나쁜 것"인지 "leaves127이 적은 데이터에 과한 것"인지 모른다.
→ 전문가는 **같은 설정(leaves127)과 작은 설정(leaves31) 둘 다** 돌려서 분리한다.

※ 이 비교는 같은 데이터·같은 시점이라 지금까지 우리를 속인 거리·용량 문제와 얽히지 않는다.
   따라서 **폴드 판정을 믿어도 된다**(LB 없이 결정 가능).
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

SEEDS = (2024, 913)
FLOOR = 1.0
SIGMA2 = 0.0032
NT = os.cpu_count()

BASE = dict(objective="regression_l1", metric="l1", learning_rate=0.05,
            num_leaves=127, min_data_in_leaf=40, feature_fraction=0.65,
            bagging_fraction=0.85, bagging_freq=1, lambda_l2=1.0,
            verbosity=-1, num_threads=NT)
SMALL = dict(BASE, num_leaves=31)
ROUNDS = 1000

FOLDS = [("F2 겨울", "2023-11-24", "2023-11-24", "2024-02-22"),
         ("F3 봄", "2024-02-23", "2024-02-23", "2024-06-08"),
         ("FAR-봄", "2023-11-24", "2024-02-23", "2024-06-08"),
         ("FAR-겨울", "2023-08-25", "2023-11-24", "2024-02-22")]
GNAME = ["계절매장", "상시", "B2B"]
GRP3 = {"hwadam": 0, "ski": 0, "green": 0, "always": 1, "b2b": 2}


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
    item_grp = np.array([GRP3[C.STORE_CLUSTER[s]] for s in ctx.store_of_item])

    print("=" * 98)
    print("Phase 7-a — 구조 분리 헤드룸 측정")
    print("=" * 98)
    for gi, gn in enumerate(GNAME):
        print(f"  {gn:<8s} 품목 {int((item_grp == gi).sum()):>3d}개  "
              f"({', '.join(sorted(set(np.array(ctx.store_of_item)[item_grp == gi])))})")
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
        folds.append(dict(
            name=fname,
            Xt=np.ascontiguousarray(Xtr[m][:, keep]),
            yt=np.log1p(np.maximum(ytr[m], 1.0)),
            gt=item_grp[mtr[m, 2]], ht=mtr[m, 1],
            Xv=np.ascontiguousarray(Xva[:, keep]),
            y=yva, iids=mva[:, 2], hv=mva[:, 1], gv=item_grp[mva[:, 2]]))
        del Xtr, Xva
        print(f"  [{fname:<8s}] 학습 {folds[-1]['Xt'].shape[0]:>7,}행 · "
              f"검증 {len(yva):>6,}칸")
    print()

    def train_pred(Xt, yt, Xv, params):
        ds = lgb.Dataset(Xt, label=yt, feature_name=names,
                         categorical_feature=cats, free_raw_data=False)
        return np.mean([np.expm1(lgb.train(dict(params, seed=sd), ds,
                                           num_boost_round=ROUNDS).predict(Xv))
                        for sd in SEEDS], 0)

    def sc(d, p, mask=None):
        """공식 지표. mask 로 부분집합만 채점 가능(빠진 품목은 자동 제외)."""
        if mask is None:
            mask = np.ones(len(p), bool)
        return competition_score(d["y"][mask], np.maximum(p[mask], FLOOR),
                                 d["iids"][mask], ctx.store_of_item,
                                 make_weights(1.0), ctx.n)

    # ---------------------------------------------------------- 글로벌 기준선
    print("=" * 98)
    print("학습 중 — 글로벌 / 세그먼트 전문가 / horizon 전문가")
    print("=" * 98)
    for d in folds:
        t1 = time.time()
        d["p_glob"] = train_pred(d["Xt"], d["yt"], d["Xv"], BASE)
        # 세그먼트 전문가 (같은 설정 / 작은 설정)
        for tag, prm in (("seg", BASE), ("segS", SMALL)):
            p = np.zeros_like(d["p_glob"])
            for gi in range(3):
                mt, mv = d["gt"] == gi, d["gv"] == gi
                p[mv] = train_pred(d["Xt"][mt], d["yt"][mt], d["Xv"][mv], prm)
            d[f"p_{tag}"] = p
        # horizon 전문가
        for tag, prm in (("hor", BASE), ("horS", SMALL)):
            p = np.zeros_like(d["p_glob"])
            for h in range(1, C.HORIZON + 1):
                mt, mv = d["ht"] == h, d["hv"] == h
                p[mv] = train_pred(d["Xt"][mt], d["yt"][mt], d["Xv"][mv], prm)
            d[f"p_{tag}"] = p
        print(f"  [{d['name']}] 완료 {time.time()-t1:.0f}s", flush=True)

    # ---------------------------------------------------------- 세그먼트별
    print("\n" + "=" * 98)
    print("① 세그먼트별 — 전문가가 글로벌을 이기는가 (해당 구간만 채점)")
    print("=" * 98)
    seg_rows = []
    print(f"  {'세그먼트':<10s}{'':<8s}" + "".join(f"{d['name']:>11s}" for d in folds)
          + f"{'평균':>10s}{'글로벌대비':>11s}{'일관':>6s}")
    for gi, gn in enumerate(GNAME):
        g = [sc(d, d["p_glob"], d["gv"] == gi) for d in folds]
        print(f"  {gn:<10s}{'글로벌':<8s}" + "".join(f"{x:>11.4f}" for x in g)
              + f"{np.mean(g):>10.4f}")
        for tag, lab in (("seg", "전문가"), ("segS", "전문가(작은)")):
            s = [sc(d, d[f"p_{tag}"], d["gv"] == gi) for d in folds]
            ok = all(s[i] < g[i] for i in range(4))
            seg_rows.append(dict(grp=gn, kind=lab, s=s, mean=float(np.mean(s)),
                                 gain=float(np.mean(g) - np.mean(s)), ok=bool(ok)))
            print(f"  {'':<10s}{lab:<8s}" + "".join(f"{x:>11.4f}" for x in s)
                  + f"{np.mean(s):>10.4f}{np.mean(g)-np.mean(s):>+11.4f}"
                  + f"{'○' if ok else '×':>5s}")

    # ---------------------------------------------------------- horizon별
    print("\n" + "=" * 98)
    print("② horizon별 — 전문가가 글로벌을 이기는가")
    print("=" * 98)
    print(f"  {'h':>3s}{'글로벌':>10s}{'전문가':>10s}{'차이':>9s}"
          f"{'전문가(작은)':>13s}{'차이':>9s}")
    hor_rows = []
    for h in range(1, C.HORIZON + 1):
        g = float(np.mean([sc(d, d["p_glob"], d["hv"] == h) for d in folds]))
        a = float(np.mean([sc(d, d["p_hor"], d["hv"] == h) for d in folds]))
        b = float(np.mean([sc(d, d["p_horS"], d["hv"] == h) for d in folds]))
        hor_rows.append(dict(h=h, glob=g, hor=a, horS=b))
        print(f"  {h:>3d}{g:>10.4f}{a:>10.4f}{g-a:>+9.4f}{b:>13.4f}{g-b:>+9.4f}")

    # ---------------------------------------------------------- 조립 전체
    print("\n" + "=" * 98)
    print("③ ★ 조립해서 전체 점수 — 실제로 궁금한 숫자")
    print("=" * 98)
    tot = {}
    for tag, lab in (("glob", "글로벌 (현행)"), ("seg", "세그먼트 분리"),
                     ("segS", "세그먼트 분리(작은)"), ("hor", "horizon 분리"),
                     ("horS", "horizon 분리(작은)")):
        s = [sc(d, d[f"p_{tag}"]) for d in folds]
        tot[tag] = dict(lab=lab, s=s, mean=float(np.mean(s)))
    gm = tot["glob"]["mean"]
    print(f"  {'':<22s}" + "".join(f"{d['name']:>11s}" for d in folds)
          + f"{'평균':>10s}{'현행대비':>10s}{'일관':>6s}")
    for tag in ("glob", "seg", "segS", "hor", "horS"):
        t = tot[tag]
        ok = all(t["s"][i] < tot["glob"]["s"][i] for i in range(4))
        mark = "  ←현행" if tag == "glob" else ""
        print(f"  {t['lab']:<22s}" + "".join(f"{x:>11.4f}" for x in t["s"])
              + f"{t['mean']:>10.4f}{gm-t['mean']:>+10.4f}"
              + f"{'○' if ok else '×':>5s}{mark}")

    # ---------------------------------------------------------- 판정
    print("\n" + "=" * 98)
    print("판정")
    print("=" * 98)
    best = min((t for k, t in tot.items() if k != "glob"), key=lambda t: t["mean"])
    bg = gm - best["mean"]
    okb = all(best["s"][i] < tot["glob"]["s"][i] for i in range(4))
    print(f"  최고 분리안: {best['lab']}  {best['mean']:.4f}  "
          f"현행 대비 {bg:+.4f}  4폴드 일관 {'○' if okb else '×'}  (2σ={SIGMA2})")
    if bg > SIGMA2 and okb:
        print("  → ★ 여유 있음. 본격 구축 진행.")
    elif bg > 0:
        print("  → ⚠️ 방향은 있으나 2σ 미달. 구축 비용 대비 회수 불투명.")
    else:
        print("  → ❌ 여유 없음. **트리가 이미 세그먼트/horizon 을 스스로 처리하고 있다.**")
        print("     분리로 잃는 학습량이 얻는 특화보다 크다. 구조 축 종료 → B(앙상블+보정)로.")

    json.dump(dict(seg=seg_rows, hor=hor_rows,
                   total={k: dict(lab=v["lab"], s=v["s"], mean=v["mean"])
                          for k, v in tot.items()},
                   best=best["lab"], best_gain=bg, consistent=bool(okb)),
              open(os.path.join(C.EXPERIMENTS, "phase7a_headroom.json"),
                   "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n저장: experiments/phase7a_headroom.json")
    print(f"총 {(time.time()-t0)/60:.0f}분")


if __name__ == "__main__":
    main()
