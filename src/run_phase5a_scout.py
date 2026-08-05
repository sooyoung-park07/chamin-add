# -*- coding: utf-8 -*-
"""Phase 5-a 정찰 — LightGBM / XGBoost / CatBoost 를 같은 조건에서 돌려
   '알고리즘 분업 + 앙상블'에 CPU 시간을 투자할 값어치가 있는지 **먼저** 판정한다.

판정할 것 두 가지:
  1. 세 모델이 **다르게 틀리는가** — 잔차 상관 ρ
     ρ > 0.98 이면 사실상 같은 모델 3개다. 분업의 전제가 무너진다.
  2. 기본설정 단순평균의 이득이 **2σ(0.003)** 를 넘는가

튜닝 전 기본설정으로 재는 이유: 튜닝은 CPU 시간이 크게 든다(설정 1개당
LGBM 3분 · XGB 5분 · Cat 10분). 그 투자를 할지 말지를 먼저 정해야 한다.
여기서 부정이 나오면 팀을 구조(horizon별 7모델 · 세그먼트 분리) 쪽으로
돌리는 게 기대값이 크다.

규약 (log.md 확정 사항 준수):
  - 피처는 확정본 57개 = 전체 62개에서 `prof` 5개 제외 (4-e 기각)
  - 프록시는 **폴드마다 학습구간까지만**으로 계산해 주입 (누수 차단)
  - 학습은 y != 0 행만, 음수는 1로. 타깃 log1p, 목적함수 L1
  - **공유·앙상블용 예측은 하한 적용 전(pre-floor)** 로 저장한다.
    max(·,1)은 비선형이라 '각자 하한 → 평균' != '평균 → 하한' 이기 때문.
  - 판정은 3폴드 평균이 아니라 **F2+F3**(LB 대리지표)로 한다.

산출: experiments/phase5a_scout.json  (요약)
      experiments/phase5a_oof.npz     (폴드별·알고리즘별·시드별 예측 — 재학습 없이
                                       나중에 가중치 탐색에 그대로 쓴다)
"""
import os
import json
import time

import numpy as np
import pandas as pd

import config as C
import dataio as D
import features as F
import validate as V
from metrics import competition_score, make_weights

# ---------------------------------------------------------------- 공통 설정
# 스크리닝 급(leaves127 / 600라운드). 세 라이브러리를 '대략 같은 용량'으로 맞춘다.
#   LightGBM  num_leaves=127  (leaf-wise)
#   XGBoost   max_depth=8     (level-wise)
#   CatBoost  depth=8         (oblivious, 대칭트리라 2^8=256 leaf 고정)
# ※ 공정한 대결이 아니라 **다양성 측정**이 목적이다. 순위는 튜닝 후에 정한다.
ROUNDS = 600
LR = 0.05
SEEDS = (42, 7)
FLOOR = 1.0
NT = os.cpu_count()
SIGMA2 = 0.003          # Phase 4-b에서 측정한 2σ (판정 문턱)


# ---------------------------------------------------------------- 알고리즘별 어댑터
def run_lgbm(Xt, yt, Xv, names, cats, seeds):
    import lightgbm as lgb
    p = dict(objective="regression_l1", metric="l1", learning_rate=LR,
             num_leaves=127, min_data_in_leaf=40, feature_fraction=0.85,
             bagging_fraction=0.85, bagging_freq=1, lambda_l2=1.0,
             verbosity=-1, num_threads=NT)
    out = []
    for sd in seeds:
        ds = lgb.Dataset(Xt, label=yt, feature_name=names,
                         categorical_feature=cats, free_raw_data=False)
        m = lgb.train(dict(p, seed=sd), ds, num_boost_round=ROUNDS)
        out.append(np.expm1(m.predict(Xv)))
    return np.array(out)


def _cat_frames(Xt, Xv, names, cats):
    """범주형 열을 category dtype 으로. 카테고리는 train+val 합집합으로 고정한다.
    (한쪽에만 있는 레벨이 NaN 으로 떨어지는 사고 방지 — 예: F1 학습구간에 없는 달)"""
    dft = pd.DataFrame(Xt, columns=names)
    dfv = pd.DataFrame(Xv, columns=names)
    for c in cats:
        j = names.index(c)
        lv = np.unique(np.concatenate([Xt[:, j], Xv[:, j]])).astype(int)
        dft[c] = pd.Categorical(dft[c].astype(int), categories=lv)
        dfv[c] = pd.Categorical(dfv[c].astype(int), categories=lv)
    return dft, dfv


def run_xgb(Xt, yt, Xv, names, cats, seeds):
    import xgboost as xgb
    dft, dfv = _cat_frames(Xt, Xv, names, cats)
    dtr = xgb.DMatrix(dft, label=yt, enable_categorical=True)
    dva = xgb.DMatrix(dfv, enable_categorical=True)
    del dft, dfv
    # max_cat_to_onehot=1 → 193레벨 item_id 를 원-핫으로 터뜨리지 않고
    # 분할(partition) 기반으로 처리. 이걸 안 주면 item_id 신호가 뭉개진다.
    p = dict(objective="reg:absoluteerror", eta=LR, max_depth=8,
             min_child_weight=40, subsample=0.85, colsample_bytree=0.85,
             reg_lambda=1.0, tree_method="hist", max_cat_to_onehot=1,
             nthread=NT)
    out = []
    for sd in seeds:
        b = xgb.train(dict(p, seed=sd), dtr, num_boost_round=ROUNDS)
        out.append(np.expm1(b.predict(dva)))
    return np.array(out)


def run_cat(Xt, yt, Xv, names, cats, seeds):
    from catboost import CatBoostRegressor, Pool
    conv = {c: int for c in cats}
    ptr = Pool(pd.DataFrame(Xt, columns=names).astype(conv),
               label=yt, cat_features=cats)
    pva = Pool(pd.DataFrame(Xv, columns=names).astype(conv), cat_features=cats)
    out = []
    for sd in seeds:
        m = CatBoostRegressor(loss_function="MAE", learning_rate=LR, depth=8,
                              iterations=ROUNDS, l2_leaf_reg=1.0,
                              random_seed=sd, verbose=0, thread_count=NT)
        m.fit(ptr)
        out.append(np.expm1(m.predict(pva)))
    return np.array(out)


ALGOS = [("LightGBM", run_lgbm), ("XGBoost", run_xgb), ("CatBoost", run_cat)]


# ---------------------------------------------------------------- 본체
def main():
    t_all = time.time()
    ctx = F.Context()
    tr = D.load_train()
    mat, dates = D.to_matrix(tr, ctx.items)
    nd = mat.shape[1]

    fn_all = F.feature_names()
    keep = [i for i, k in enumerate(fn_all) if k not in set(F.PROF_KEYS)]
    names = [fn_all[i] for i in keep]
    cats = [c for c in F.CATEGORICAL if c in names]

    print("=" * 88)
    print("Phase 5-a 정찰 — 알고리즘 분업에 값어치가 있는가")
    print("=" * 88)
    print(f"피처 {len(names)}개 (prof 제외) · 범주형 {len(cats)}개 · "
          f"시드 {SEEDS} · {ROUNDS}라운드 · 스레드 {NT}\n")

    folds = []
    for fi, (fname, d0, d1) in enumerate(V.FOLDS):
        t0 = time.time()
        cut = int(np.searchsorted(np.array(dates), pd.Timestamp(d0)))
        ctx.set_proxy(F.pick_proxy_items(mat, dates, cut, ctx.store_codes))
        va = V.origins(dates, d0, d1, nd)
        trn = V.train_origins(dates, d0, nd)
        Xtr, ytr, _ = F.build_samples(mat, dates, trn, ctx)
        Xva, yva, mva = F.build_samples(mat, dates, va, ctx)

        m = ytr != 0
        Xt = np.ascontiguousarray(Xtr[m][:, keep])
        yt = np.log1p(np.maximum(ytr[m], 1.0))
        Xv = np.ascontiguousarray(Xva[:, keep])
        del Xtr, Xva

        print(f"[{fname}]  학습 origin {len(trn)} · 학습행 {Xt.shape[0]:,} · "
              f"검증행 {Xv.shape[0]:,}   (행렬 {time.time()-t0:.0f}s)")

        preds = {}
        for aname, runner in ALGOS:
            t1 = time.time()
            preds[aname] = runner(Xt, yt, Xv, names, cats, SEEDS)
            p = np.maximum(preds[aname].mean(0), FLOOR)
            s = competition_score(yva, p, mva[:, 2], ctx.store_of_item,
                                  make_weights(1.0), ctx.n)
            print(f"    {aname:<10s} {time.time()-t1:6.0f}s   {s:.4f}")

        folds.append(dict(name=fname, y=yva, iids=mva[:, 2], preds=preds))
        del Xt, Xv
        print()

    # ------------------------------------------------------------ 채점 도구
    def sc(fd, p):
        return competition_score(fd["y"], np.maximum(p, FLOOR), fd["iids"],
                                 ctx.store_of_item, make_weights(1.0), ctx.n)

    def summarize(label, getter, table):
        """getter(fold) -> pre-floor 예측. 3폴드/ F2+F3 요약 후 표에 기록."""
        v = [sc(fd, getter(fd)) for fd in folds]
        table[label] = dict(folds=v, cv3=float(np.mean(v)),
                            f2f3=float(np.mean(v[1:])))
        return table[label]

    single = {}
    for aname, _ in ALGOS:
        summarize(aname, lambda fd, a=aname: fd["preds"][a].mean(0), single)

    print("=" * 88)
    print("1) 단일 모델 (튜닝 전 기본설정)")
    print("=" * 88)
    print(f"  {'':<12s} {'F1 가을':>9s} {'F2 겨울':>9s} {'F3 봄':>9s} "
          f"{'3폴드':>9s} {'F2+F3':>9s}")
    for k, v in single.items():
        print(f"  {k:<12s} " + " ".join(f"{x:>9.4f}" for x in v["folds"])
              + f" {v['cv3']:>9.4f} {v['f2f3']:>9.4f}")
    best_solo = min(single.values(), key=lambda v: v["f2f3"])["f2f3"]
    best_name = min(single, key=lambda k: single[k]["f2f3"])

    # ------------------------------------------------------------ 잔차 상관
    print("\n" + "=" * 88)
    print("2) 세 모델이 '다르게 틀리는가' — 잔차 상관 ρ  (F2+F3, 유효행만)")
    print("=" * 88)
    print("   잔차 = log1p(예측) − log1p(실측). SMAPE가 비율오차라 로그 공간이 맞다.")
    res = {}
    for aname, _ in ALGOS:
        parts = []
        for fd in folds[1:]:
            m = fd["y"] > 0          # 음수(환불) 14행은 log1p 가 정의 안 됨 → 먼저 걸러낸다
            parts.append(np.log1p(np.maximum(fd["preds"][aname].mean(0)[m], 0))
                         - np.log1p(fd["y"][m]))
        res[aname] = np.concatenate(parts)

    keys = [a for a, _ in ALGOS]
    print(f"\n  {'':<12s}" + "".join(f"{k:>12s}" for k in keys))
    rho = {}
    for a in keys:
        row = []
        for b in keys:
            c = 1.0 if a == b else float(np.corrcoef(res[a], res[b])[0, 1])
            rho[f"{a}|{b}"] = c
            row.append(c)
        print(f"  {a:<12s}" + "".join(f"{x:>12.4f}" for x in row))
    pair_rho = [(a, b, rho[f"{a}|{b}"]) for i, a in enumerate(keys)
                for b in keys[i + 1:]]
    worst = max(pair_rho, key=lambda t: t[2])
    print(f"\n  최고 상관 쌍: {worst[0]} ↔ {worst[1]}  ρ={worst[2]:.4f}")
    print("  판독:  ρ>0.98 사실상 같은 모델 / 0.92~0.98 전형적 GBDT 3형제 / "
          "<0.90 진짜 다름")

    # ------------------------------------------------------------ 앙상블
    print("\n" + "=" * 88)
    print("3) 합쳐서 이득이 있는가")
    print("=" * 88)

    def arith(fd, ks):
        return np.mean([fd["preds"][k].mean(0) for k in ks], axis=0)

    def geo(fd, ks):
        """로그 공간 평균 = 기하평균. 비율오차 지표에는 이쪽이 자연스럽다."""
        return np.expm1(np.mean(
            [np.log1p(np.maximum(fd["preds"][k].mean(0), 0)) for k in ks], axis=0))

    ens = {}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            summarize(f"{a[:3]}+{b[:3]}", lambda fd, ks=(a, b): arith(fd, ks), ens)
    summarize("셋 산술평균", lambda fd: arith(fd, keys), ens)
    summarize("셋 기하평균", lambda fd: geo(fd, keys), ens)

    # 부호 규약: **양수 = 최고 단일보다 좋아짐** (아래 '판정'과 동일)
    print(f"  {'':<14s} {'3폴드':>9s} {'F2+F3':>9s} {'최고단일 대비 개선':>16s}")
    for k, v in ens.items():
        g = best_solo - v["f2f3"]
        mark = "  ★" if g > SIGMA2 else ("  ·" if g > 0 else "")
        print(f"  {k:<14s} {v['cv3']:>9.4f} {v['f2f3']:>9.4f} {g:>+16.4f}{mark}")
    print(f"\n  (최고 단일 = {best_name} {best_solo:.4f} · 문턱 2σ = {SIGMA2})")

    # ------------------------------------------------------------ 덤: 시드 평균 방식
    print("\n" + "=" * 88)
    print("4) 덤 — 시드 앙상블을 로그 공간에서 평균하면? (지금은 원공간 산술평균)")
    print("=" * 88)
    seedcmp = {}
    for aname, _ in ALGOS:
        a = np.mean([sc(fd, fd["preds"][aname].mean(0)) for fd in folds[1:]])
        g = np.mean([sc(fd, np.expm1(np.log1p(np.maximum(fd["preds"][aname], 0)).mean(0)))
                     for fd in folds[1:]])
        seedcmp[aname] = dict(arith=float(a), geo=float(g), diff=float(g - a))
        print(f"  {aname:<12s} 산술 {a:.4f} → 기하 {g:.4f}   ({g-a:+.4f})")
    print(f"  ※ 시드 2개뿐이라 차이가 작다. 방향만 참고하고 시드5에서 재확인할 것.")

    # ------------------------------------------------------------ 판정
    best_ens = min(ens.values(), key=lambda v: v["f2f3"])
    best_ens_k = min(ens, key=lambda k: ens[k]["f2f3"])
    gain = best_solo - best_ens["f2f3"]
    print("\n" + "=" * 88)
    print("판정")
    print("=" * 88)
    print(f"  최고 상관 ρ = {worst[2]:.4f}")
    print(f"  최고 앙상블 = {best_ens_k} {best_ens['f2f3']:.4f} "
          f"(최고 단일 {best_solo:.4f} 대비 {gain:+.4f} 개선)")
    if gain > SIGMA2 and worst[2] < 0.98:
        print("  → ★ 분업 진행. 튜닝으로 개별 성능이 오르면 이득은 더 커진다.")
    elif gain > SIGMA2:
        print("  → ⚠️ 이득은 있으나 상관이 높다. 튜닝보다 **구조 다양화**"
              "(horizon별·세그먼트)가 기대값이 클 수 있다.")
    else:
        print("  → ❌ 기본설정 이득이 측정 한계 안. 알고리즘 분업 재고 필요.")
        print("     대안: 두 명을 구조(horizon별 7모델 · 세그먼트 분리)로 돌린다.")

    # ------------------------------------------------------------ 저장
    os.makedirs(C.EXPERIMENTS, exist_ok=True)
    json.dump(dict(config=dict(rounds=ROUNDS, lr=LR, seeds=list(SEEDS),
                               n_features=len(names), floor=FLOOR),
                   single=single, rho=rho, ensemble=ens, seed_avg=seedcmp,
                   verdict=dict(max_rho=worst[2], best_solo=best_solo,
                                best_ens=best_ens_k, gain=gain, sigma2=SIGMA2)),
              open(os.path.join(C.EXPERIMENTS, "phase5a_scout.json"),
                   "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    npz = {}
    for fi, fd in enumerate(folds):
        npz[f"y{fi}"] = fd["y"]
        npz[f"i{fi}"] = fd["iids"]
        for aname, _ in ALGOS:
            npz[f"p{fi}_{aname}"] = fd["preds"][aname].astype(np.float32)
    np.savez_compressed(os.path.join(C.EXPERIMENTS, "phase5a_oof.npz"), **npz)

    print(f"\n저장: experiments/phase5a_scout.json · phase5a_oof.npz "
          f"(pre-floor 예측 — 재학습 없이 가중치 탐색에 재사용)")
    print(f"총 {(time.time()-t_all)/60:.1f}분")


if __name__ == "__main__":
    main()
