# -*- coding: utf-8 -*-
"""Phase 5-f — 작은 쪽으로 사다리 연장 + 제출본 생성. **판정은 FAR 기준.**

Phase 5-e에서 확정된 것: 예측 거리가 멀어지면 용량 순위가 뒤집힌다(상관 +0.982).
실제 채점은 거리 0~11개월이므로 **작은 모델이 유리**하다. `leaves63`가 사다리 끝에서 1위였으니
더 작은 쪽에 최적이 있을 수 있다.

━━━ 같은 실수를 반복하지 않기 위한 장치 두 개 ━━━
Phase 5-e의 교훈은 "검증 구간이 고정된 채 설정을 많이 보면 그 구간에 과적합한다"였다.
이제 FAR 구간 하나에 설정 10개를 태우면 **똑같은 함정에 다시 빠진다.** 그래서:

  ① **독립적인 FAR 폴드 2개**를 쓴다 (계절이 다름). 둘 다에서 이겨야 채택.
       FAR-봄   : 학습 ~2023-11-23 → 검증 2024-02-23~06-08  (거리 ~3개월)
       FAR-겨울 : 학습 ~2023-08-25 → 검증 2023-11-24~2024-02-22 (거리 ~3개월)
  ② 탐색 시드 (42,7) / **확인 시드 (2024,913,31) 분리** — 난수 과적합 방지 (기존 규약)

①이 새로 추가된 장치다. 시드 분리는 난수만 막고 구간 과적합은 못 막는다는 걸 비싸게 배웠다.

마지막에 이긴 설정으로 전체 학습 → `submission_v6_small.csv` 생성.
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
SUBMIT_SEEDS = (42, 7, 2024, 913, 31)
FLOOR = 1.0
SIGMA2 = 0.003
NT = os.cpu_count()

# 거리 ~3개월짜리 폴드 2개. (학습 종료일, 검증 시작, 검증 끝)
FAR_FOLDS = [
    ("FAR-봄",   "2023-11-24", "2024-02-23", "2024-06-08"),
    ("FAR-겨울", "2023-08-25", "2023-11-24", "2024-02-22"),
]

COMMON = dict(objective="regression_l1", metric="l1", learning_rate=0.05,
              min_data_in_leaf=40, feature_fraction=0.85,
              bagging_fraction=0.85, bagging_freq=1, lambda_l2=1.0,
              verbosity=-1, num_threads=NT)

LADDER = [
    ("S0 leaves255 (=v1)", dict(num_leaves=255), 1000),
    ("S1 leaves127",       dict(num_leaves=127), 1000),
    ("S2 leaves63",        dict(num_leaves=63), 1000),
    ("S3 leaves31",        dict(num_leaves=31), 1000),
    ("S4 leaves15",        dict(num_leaves=15), 1000),
    # 작은 트리는 1000라운드로 덜 학습됐을 수 있다 → 부스팅을 더 준다
    ("S5 leaves63 ×2000r", dict(num_leaves=63), 2000),
    ("S6 leaves31 ×2000r", dict(num_leaves=31), 2000),
    # 용량을 줄이는 다른 방법 — 잎 개수 대신 규제로
    ("S7 leaves63 강규제",  dict(num_leaves=63, min_data_in_leaf=100,
                              lambda_l2=10.0), 1000),
    ("S8 leaves127 강규제", dict(num_leaves=127, min_data_in_leaf=100,
                              lambda_l2=10.0), 1000),
    ("S9 leaves63 ff0.65", dict(num_leaves=63, feature_fraction=0.65), 1000),
]


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
    print("Phase 5-f — 작은 쪽 사다리 · 판정은 FAR(거리 3개월) 폴드 2개")
    print("=" * 98)

    folds = []
    for fname, cut, v0, v1 in FAR_FOLDS:
        cut_col = int(np.searchsorted(np.array(dates), pd.Timestamp(cut)))
        ctx.set_proxy(F.pick_proxy_items(mat, dates, cut_col, ctx.store_codes))
        trn = [o for o in range(C.WINDOW - 1, nd - C.HORIZON)
               if dates[o] < pd.Timestamp(cut)]
        va = V.origins(dates, v0, v1, nd)
        Xtr, ytr, _ = F.build_samples(mat, dates, trn, ctx)
        Xva, yva, mva = F.build_samples(mat, dates, va, ctx)
        m = ytr != 0
        folds.append(dict(
            name=fname, Xt=np.ascontiguousarray(Xtr[m][:, keep]),
            yt=np.log1p(np.maximum(ytr[m], 1.0)),
            Xv=np.ascontiguousarray(Xva[:, keep]), y=yva, iids=mva[:, 2]))
        del Xtr, Xva
        print(f"  [{fname}] 학습 {dates[trn[0]].date()}~{dates[trn[-1]].date()} "
              f"({len(trn)} origin · {folds[-1]['Xt'].shape[0]:,}행) → "
              f"검증 {v0}~{v1} ({len(va)} origin)")
    print()

    def evaluate(over, rounds, seeds):
        p = dict(COMMON, **over)
        out = []
        for fd in folds:
            ds = lgb.Dataset(fd["Xt"], label=fd["yt"], feature_name=names,
                             categorical_feature=cats, free_raw_data=False)
            pr = np.mean([np.expm1(lgb.train(dict(p, seed=sd), ds,
                                             num_boost_round=rounds).predict(fd["Xv"]))
                          for sd in seeds], 0)
            out.append(competition_score(fd["y"], np.maximum(pr, FLOOR), fd["iids"],
                                         ctx.store_of_item, make_weights(1.0), ctx.n))
        return out

    rows, base = [], None
    print(f"  {'설정':<20s} {'FAR-봄':>9s} {'FAR-겨울':>9s} {'평균':>9s} "
          f"{'기준선 대비':>12s} {'일관':>5s} {'시간':>7s}")
    for label, over, rounds in LADDER:
        t1 = time.time()
        s = evaluate(over, rounds, SEARCH_SEEDS)
        m = float(np.mean(s))
        if base is None:
            base = s[:]
        d = float(np.mean(base)) - m
        both = (s[0] < base[0]) and (s[1] < base[1])       # 두 폴드 모두 개선?
        rows.append(dict(label=label, over=over, rounds=rounds,
                         spring=s[0], winter=s[1], mean=m, gain=d, both=both))
        mark = "  ★" if (d > SIGMA2 and both) else ("  ·" if d > 0 else "")
        print(f"  {label:<20s} {s[0]:>9.4f} {s[1]:>9.4f} {m:>9.4f} {d:>+12.4f}"
              f"{mark:<3s} {'○' if both else '×':>4s} {time.time()-t1:>6.0f}s",
              flush=True)

    print("\n  (★ = 2σ 초과 **그리고** 두 폴드 모두 개선 · '일관 ×'는 한쪽만 좋아진 것 = 신뢰 못 함)")

    # ------------------------------------------------------------ 확인
    cand = sorted([r for r in rows[1:] if r["both"]], key=lambda r: r["mean"])[:3]
    if not cand:
        cand = sorted(rows[1:], key=lambda r: r["mean"])[:3]
        print("\n  ⚠️ 두 폴드 모두 개선한 설정이 없다 — 평균 상위 3개로 확인 진행")

    print("\n" + "=" * 98)
    print(f"확인 — 기준선 + 후보 {len(cand)}개를 **안 쓴 시드** {CONFIRM_SEEDS} 로 재측정")
    print("=" * 98)
    conf = []
    for r in [rows[0]] + cand:
        s = evaluate(r["over"], r["rounds"], CONFIRM_SEEDS)
        conf.append(dict(label=r["label"], over=r["over"], rounds=r["rounds"],
                         spring=s[0], winter=s[1], mean=float(np.mean(s)),
                         search=r["mean"]))
        print(f"  {r['label']:<20s} 탐색 {r['mean']:.4f} → 확인 {np.mean(s):.4f} "
              f"({np.mean(s)-r['mean']:+.4f})   봄 {s[0]:.4f} · 겨울 {s[1]:.4f}",
              flush=True)

    cbase = conf[0]
    cbest = min(conf, key=lambda r: r["mean"])
    gain = cbase["mean"] - cbest["mean"]
    ok = (cbest["spring"] < cbase["spring"]) and (cbest["winter"] < cbase["winter"])

    print("\n" + "=" * 98)
    print("판정")
    print("=" * 98)
    print(f"  확인 기준선(v1 leaves255)  {cbase['mean']:.4f}")
    print(f"  확인 최고  {cbest['label']}  {cbest['mean']:.4f}   개선 {gain:+.4f}   "
          f"두 폴드 일관 {'○' if ok else '×'}")
    adopt = gain > SIGMA2 and ok
    print("  → ★ 채택" if adopt else "  → 기준선 유지 (2σ 미달이거나 폴드 불일치)")

    win = cbest if adopt else cbase
    json.dump(dict(far_folds=FAR_FOLDS, ladder=rows, confirm=conf,
                   adopt=adopt, winner=dict(label=win["label"], over=win["over"],
                                            rounds=win["rounds"], cv=win["mean"])),
              open(os.path.join(C.EXPERIMENTS, "phase5f_small.json"),
                   "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # ------------------------------------------------------------ 제출본
    print("\n" + "=" * 98)
    print(f"전체 데이터 학습 → 제출본  (설정: {win['label']})")
    print("=" * 98)
    for f in folds:
        f["Xt"] = f["Xv"] = None
    ctx.set_proxy(F.pick_proxy_items(mat, dates, nd, ctx.store_codes))
    Xa, ya, _ = F.build_samples(mat, dates,
                                list(range(C.WINDOW - 1, nd - C.HORIZON)), ctx)
    m = ya != 0
    Xt = np.ascontiguousarray(Xa[m][:, keep])
    yt = np.log1p(np.maximum(ya[m], 1.0))
    del Xa
    p = dict(COMMON, **win["over"])
    ds = lgb.Dataset(Xt, label=yt, feature_name=names,
                     categorical_feature=cats, free_raw_data=False)
    t1 = time.time()
    models = [lgb.train(dict(p, seed=sd), ds, num_boost_round=win["rounds"])
              for sd in SUBMIT_SEEDS]
    print(f"  학습행 {Xt.shape[0]:,} · 시드 {len(SUBMIT_SEEDS)}개 {time.time()-t1:.0f}s")

    tpl = D.load_submission_template()
    rk = tpl.columns[0]
    out = tpl.copy()
    out[ctx.items] = out[ctx.items].astype(float)
    for t in range(C.N_TEST):
        te = D.load_test(t)
        tmat, tdates = D.to_matrix(te, ctx.items)
        X, _, _ = F.build_samples(tmat, tdates, [C.WINDOW - 1], ctx,
                                  with_target=False)
        pr = np.mean([np.expm1(mdl.predict(X[:, keep])) for mdl in models], 0)
        pr = np.maximum(pr, FLOOR).reshape(C.HORIZON, ctx.n)   # 평균 뒤 하한 1회
        for h in range(1, C.HORIZON + 1):
            out.loc[out.index[out[rk] == f"TEST_{t:02d}+{h}일"][0], ctx.items] = pr[h - 1]
    out[ctx.items] = out[ctx.items].round(2)
    path = os.path.join(C.SUBMISSIONS, "submission_v6_small.csv")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    v = out[ctx.items].values.astype(float)
    assert not np.isnan(v).any() and (v >= FLOOR - 1e-9).all()
    print(f"\n  저장: submission_v6_small.csv  {out.shape}")
    print(f"    min {v.min():.2f} / 중앙값 {np.median(v):.2f} / "
          f"평균 {v.mean():.2f} / max {v.max():.2f}")
    for tag in ("v1", "v3_xgb"):
        q = os.path.join(C.SUBMISSIONS, f"submission_{tag}.csv")
        if os.path.exists(q):
            o = pd.read_csv(q)[ctx.items].values.astype(float)
            print(f"    {tag} 대비 로그공간 상관 "
                  f"{np.corrcoef(np.log1p(o).ravel(), np.log1p(v).ravel())[0,1]:.4f} · "
                  f"평균절대차 {np.abs(o-v).mean():.2f}")
    print(f"\n총 {(time.time()-t0)/60:.0f}분")


if __name__ == "__main__":
    main()
