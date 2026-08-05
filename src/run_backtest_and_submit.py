# -*- coding: utf-8 -*-
"""공식 지표로 백테스트 → 블렌딩 비율 확정 → 전체 학습 → 제출파일 생성."""
import os
import json
import numpy as np
import pandas as pd

import config as C
import dataio as D
import features as F
import model_lgbm as M
import validate as V
from metrics import competition_score, flat_smape, make_weights

STAMP = "v2"
OUT_CSV = os.path.join(C.SUBMISSIONS, f"submission_{STAMP}.csv")


def main():
    ctx = F.Context()
    tr = D.load_train()
    mat, dates = D.to_matrix(tr, ctx.items)
    nd = mat.shape[1]
    # 확정 피처 57개. build_samples 는 기각된 그룹(prof·resort)까지 만들므로 열을 잘라 쓴다.
    # ※ 직접 거르지 말고 F.active_columns() 를 쓸 것 — 새 그룹이 추가돼도 안 낡는다.
    KEEP = F.active_columns()
    fn = F.active_names()
    ev = V.Evaluator(ctx.items)

    def score(y, p, iids, high=1.0):
        return competition_score(y, p, iids, ctx.store_of_item,
                                 make_weights(high), ctx.n)

    # ---------- 제출 형식 스모크 테스트 (긴 학습 전에 먼저) ----------
    tpl = D.load_submission_template()
    rk = tpl.columns[0]
    assert list(tpl.columns[1:]) == ctx.items
    chk = tpl.copy()
    chk[ctx.items] = chk[ctx.items].astype(float)
    for t in range(C.N_TEST):
        for h in range(1, C.HORIZON + 1):
            ridx = chk.index[chk[rk] == f"TEST_{t:02d}+{h}일"]
            assert len(ridx) == 1
            chk.loc[ridx[0], ctx.items] = 1.23
    assert (chk[ctx.items].values == 1.23).all()
    print(f"제출 형식 스모크 테스트 통과 {chk.shape}\n")

    # ---------- 백테스트 ----------
    print("=" * 76)
    print("공식 지표 백테스트 (균등 가중 기준)")
    print("=" * 76)
    folds = []
    for name, d0, d1 in V.FOLDS:
        # 인원수 프록시는 **그 폴드의 학습구간까지만**으로 고른다 (Phase 4-f, 누수 차단).
        # 이걸 빠뜨리면 Context() 기본값(각 영업장 첫 품목)이 그대로 쓰여 cross 피처가 무의미해진다.
        cut = int(np.searchsorted(np.array(dates), pd.Timestamp(d0)))
        ctx.set_proxy(F.pick_proxy_items(mat, dates, cut, ctx.store_codes))
        va = V.origins(dates, d0, d1, nd)
        trn = V.train_origins(dates, d0, nd)
        Xtr, ytr, _ = F.build_samples(mat, dates, trn, ctx)
        Xva, yva, mva = F.build_samples(mat, dates, va, ctx)
        iids = mva[:, 2]
        models = M.fit(Xtr[:, KEEP], ytr, fn)
        p = M.predict(models, Xva[:, KEEP])
        rule = np.maximum(V.rule_baseline(mat, dates, va), M.FLOOR)
        folds.append(dict(name=name, y=yva, p=p, rule=rule, iids=iids))
        print(f"  [{name}]  규칙 {score(yva, rule, iids):.4f} | "
              f"LGBM {score(yva, p, iids):.4f}   (학습 origin {len(trn)})")

    def agg(fn_pred, high=1.0):
        return np.mean([score(f["y"], fn_pred(f), f["iids"], high) for f in folds])

    print(f"\n  규칙 베이스라인 폴드평균 : {agg(lambda f: f['rule']):.4f}")
    print(f"  LGBM          폴드평균 : {agg(lambda f: f['p']):.4f}")

    print("\n[블렌딩 a·LGBM + (1-a)·규칙]")
    best_a, best_s = 1.0, 9.0
    for a in [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]:
        s = agg(lambda f, a=a: np.maximum(a * f["p"] + (1 - a) * f["rule"], M.FLOOR))
        tag = ""
        if s < best_s:
            best_s, best_a, tag = s, a, "  ←최적"
        print(f"   a={a:.2f}   {s:.4f}{tag}")

    def final_pred(f):
        return np.maximum(best_a * f["p"] + (1 - best_a) * f["rule"], M.FLOOR)

    print(f"\n[가중치 민감도]  (최종 a={best_a})")
    for h in (1.0,) + C.WEIGHT_CANDIDATES:
        lbl = "균등" if h == 1.0 else f"담하·미라시아 {h:g}x"
        print(f"   {lbl:<20s} {agg(final_pred, h):.4f}")
    print(f"   {'flat(참고)':<20s} "
          f"{np.mean([flat_smape(f['y'], final_pred(f)) for f in folds]):.4f}")

    print("\n[영업장별]")
    y_all = np.concatenate([f["y"] for f in folds])
    p_all = np.concatenate([final_pred(f) for f in folds])
    i_all = np.concatenate([f["iids"] for f in folds])
    _, per_store, item_sc, item_cnt = competition_score(
        y_all, p_all, i_all, ctx.store_of_item, None, ctx.n, return_parts=True)
    for s, v in sorted(per_store.items(), key=lambda x: -x[1]):
        mark = " ★가중높음" if s in C.HIGH_WEIGHT_STORES else ""
        print(f"   {s:<14s} {v:.4f}{mark}")

    print("\n[오차 최악 품목 10개 — 다음 개선 타깃]")
    order = np.argsort(-np.nan_to_num(item_sc, nan=-1))
    for i in order[:10]:
        print(f"   {ctx.items[i]:<40s} {item_sc[i]:.3f}  유효날짜 {item_cnt[i]:>4d}")

    # ---------- 전체 학습 + 제출 ----------
    print("\n" + "=" * 76)
    print("전체 데이터 학습 → TEST_00~09 예측")
    print("=" * 76)
    # 최종 모델의 프록시는 학습 데이터 전체로 고른다 (검증 구간이 따로 없으므로 누수 아님).
    ctx.set_proxy(F.pick_proxy_items(mat, dates, nd, ctx.store_codes))
    all_origins = list(range(C.WINDOW - 1, nd - C.HORIZON))
    Xa, ya, _ = F.build_samples(mat, dates, all_origins, ctx)
    print(f"  학습 샘플 {int((ya != 0).sum()):,} (전체 {len(ya):,} 중 유효)")
    models = M.fit(Xa[:, KEEP], ya, fn)
    print(f"  시드 {len(M.SEEDS)}개 학습 완료")

    out = tpl.copy()
    out[ctx.items] = out[ctx.items].astype(float)
    for t in range(C.N_TEST):
        te = D.load_test(t)
        tmat, tdates = D.to_matrix(te, ctx.items)
        assert tmat.shape == (ctx.n, C.WINDOW)
        X, _, _ = F.build_samples(tmat, tdates, [C.WINDOW - 1], ctx, with_target=False)
        p = M.predict(models, X[:, KEEP])
        if best_a < 1.0:
            rule = np.maximum(
                V.rule_baseline(tmat, tdates, [C.WINDOW - 1]), M.FLOOR)
            p = np.maximum(best_a * p + (1 - best_a) * rule, M.FLOOR)
        p = p.reshape(C.HORIZON, ctx.n)
        for h in range(1, C.HORIZON + 1):
            ridx = out.index[out[rk] == f"TEST_{t:02d}+{h}일"]
            out.loc[ridx[0], ctx.items] = p[h - 1]
        print(f"  TEST_{t:02d}  {tdates[0].date()}~{tdates[-1].date()}  →  "
              f"{(tdates[-1] + pd.Timedelta(days=1)).date()}~"
              f"{(tdates[-1] + pd.Timedelta(days=7)).date()}")

    out[ctx.items] = out[ctx.items].round(2)
    os.makedirs(C.SUBMISSIONS, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    v = out[ctx.items].values.astype(float)
    print(f"\n저장: {OUT_CSV}  {out.shape}")
    print(f"  예측값 min {v.min():.2f} / 중앙값 {np.median(v):.2f} / "
          f"평균 {v.mean():.2f} / max {v.max():.2f}")
    print(f"  1 미만 비율 {100 * (v < 1).mean():.1f}%  (하한 {M.FLOOR} 적용)")

    json.dump({"blend_a": best_a, "cv_official_uniform": best_s,
               "seeds": list(M.SEEDS), "rounds": M.ROUNDS,
               "leaves": M.PARAMS["num_leaves"], "floor": M.FLOOR},
              open(os.path.join(C.EXPERIMENTS, f"config_{STAMP}.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"\n공식지표 CV(균등) = {best_s:.4f}   [통과선 {C.PASS_PUBLIC}]")


if __name__ == "__main__":
    main()
