# -*- coding: utf-8 -*-
"""Phase 24 — KNORA-E (동적 앙상블 선택)를 직접 구현해서 실패를 눈으로 본다.

**방법 (Ko, Sabourin & Britto 2008 / 회귀판 = OLA·DES-Reg)**
    각 예측점 x 마다:
      1. 라벨을 아는 풀(DSEL)에서 **피처공간 최근접 이웃 k개**를 찾는다
      2. 그 이웃들에서 A 와 B 의 **국소 오차**를 각각 잰다
      3. 국소 오차가 작은 쪽 모델의 예측을 **그 칸에 채택**한다
    전역 가중치가 아니라 칸마다 모델을 고른다 — 정확히 "구분해서 합치기"다.

**두 판을 나란히 돌린다 — 이게 이 실험의 전부다**
    · 낙관판: DSEL = 채점하는 폴드 자신 (같은 origin 은 제외 — 창 21일이 겹쳐 사실상 복제)
              → 문헌 벤치마크가 보고하는 조건. 검증셋과 테스트셋이 같은 분포.
    · 정직판: DSEL = **겨울(F2)**, 채점 = **봄(F3)**
              → 우리 실제 조건. 학습 이후 먼 미래를 맞히는 것과 같은 구조.

**후처리 2종에서 반복한다 (규칙 ⑲)**
    `하한만`(raw 진실) 과 `seg+스냅`(실제 파이프라인). 두 값이 어긋나면 raw 를 믿는다.

**거리공간**: 활성 54개 중 **범주형 7개를 뺀 47개 수치 피처**를 DSEL 기준으로 표준화.
    (범주 코드를 유클리드 좌표로 쓰면 '5번 매장과 6번 매장이 가깝다'는 헛소리가 된다.)
**국소 오차**: 이웃들의 SMAPE 항 평균. DSEL 은 **채점 대상 칸(y≠0)만** 넣는다 —
    y=0 인 칸은 애초에 유능함을 판정할 근거가 없다.
"""
import os
import time

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

import config as C
import dataio as D
import features as F
import validate as V
from run_phase10c_thresholds import cell_weights

NPZ = os.path.join(C.EXPERIMENTS, "phase21c_oof.npz")
DROP3 = ["w_posmedian", "w_last14", "w_std"]
KGRID = [1, 5, 15, 50, 150, 500]
WGRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
POSTS = [("하한만 (raw)", dict(seg=False, snap=False)),
         ("seg+스냅 (실제)", dict(seg=True, snap=True))]

FITF = ("F2 겨울", "2023-11-24", "2023-11-24", "2024-02-22")
SCOF = ("F3 봄", "2024-02-23", "2024-02-23", "2024-06-08")


def post(raw, seg=True, snap=True):
    p = (np.where(raw < 1.8, 0.55 * raw, np.where(raw < 10.0, 0.90 * raw, 1.02 * raw))
         if seg else raw.copy())
    p = np.maximum(p, 1.0)
    if not snap:
        return p
    k = np.maximum(np.floor(p), 1.0)
    return np.maximum(np.where(p >= np.sqrt(k * (k + 1.0)), k + 1.0, k), 1.0)


def loss(a, p):
    a, p = np.abs(a), np.abs(p)
    den = a + p
    out = np.zeros(len(a))
    m = den > 0
    out[m] = 2.0 * np.abs(a[m] - p[m]) / den[m]
    return out


def build_X(ctx, mat, dates, nd, cut, v0, v1, keep):
    """폴드의 검증 행 피처. 행 순서는 phase21c_oof.npz 와 동일(origin, horizon, item)."""
    cut_col = int(np.searchsorted(np.array(dates), pd.Timestamp(cut)))
    ctx.set_proxy(F.pick_proxy_items(mat, dates, cut_col, ctx.store_codes))
    va = V.origins(dates, v0, v1, nd)
    X, y, meta = F.build_samples(mat, dates, va, ctx)
    return X[:, keep], y, meta, len(va)


def main():
    t0 = time.time()
    z = np.load(NPZ)
    ctx = F.Context()
    tr = D.load_train()
    mat, dates = D.to_matrix(tr, ctx.items)
    nd = mat.shape[1]
    keep0, names0 = F.active_columns(), F.active_names()
    idx = [i for i, n in enumerate(names0) if n not in DROP3]
    keep = [keep0[i] for i in idx]
    names = [names0[i] for i in idx]
    num = [i for i, n in enumerate(names) if n not in F.CATEGORICAL]
    print(f"거리공간: 54개 중 범주형 {len(names)-len(num)}개 제외 → **{len(num)}개 수치 피처**")
    print(f"  제외된 것: {[n for n in names if n in F.CATEGORICAL]}\n", flush=True)

    fold = {}
    for tag, (fn, cut, v0, v1) in (("fit", FITF), ("sco", SCOF)):
        X, y, meta, n_org = build_X(ctx, mat, dates, nd, cut, v0, v1, keep)
        assert np.allclose(y, z[f"y|{fn}"]), f"{fn} 타깃 불일치 — 행 순서 붕괴"
        W, _ = cell_weights(y, meta[:, 2], ctx.store_of_item, ctx.n)
        fold[tag] = dict(name=fn, X=X[:, num].astype(np.float64), y=y, W=W,
                         iid=meta[:, 2].astype(int), n_org=n_org,
                         org=np.repeat(np.arange(n_org), C.HORIZON * ctx.n),
                         A=z[f"ours|{fn}"], B=z[f"tm_ridge|{fn}"])
        print(f"  [{fn}] 행 {len(y):,} · origin {n_org}개 · 채점칸 {int((y!=0).sum()):,}"
              f"  ({time.time()-t0:.0f}s)", flush=True)

    S, Fi = fold["sco"], fold["fit"]

    for pname, kw in POSTS:
        print("\n" + "=" * 98)
        print(f"[{pname}]")
        # 기준선
        base = float((S["W"] * loss(S["y"], post(S["A"], **kw))).sum())
        bsolo = float((S["W"] * loss(S["y"], post(S["B"], **kw))).sum())
        gw = min(WGRID, key=lambda w: float(
            (Fi["W"] * loss(Fi["y"], post((1 - w) * Fi["A"] + w * Fi["B"], **kw))).sum()))
        gsc = float((S["W"] * loss(S["y"], post(
            (1 - gw) * S["A"] + gw * S["B"], **kw))).sum())
        print(f"  기준선 — A 단독 {base:.5f} · B 단독 {bsolo:.5f} · "
              f"전역 w={gw} (겨울에서 고름) {gsc:.5f} ({base-gsc:+.5f})")

        # 칸별 손실 (DSEL 유능도 계산용)
        for d in (S, Fi):
            d["lA"] = loss(d["y"], post(d["A"], **kw))
            d["lB"] = loss(d["y"], post(d["B"], **kw))

        for arm in ("낙관판 (DSEL=봄 자신, 같은 origin 제외)", "정직판 (DSEL=겨울)"):
            insample = arm.startswith("낙관")
            src = S if insample else Fi
            pool = src["y"] != 0                       # 채점 칸만 DSEL 에 넣는다
            mu, sd = src["X"][pool].mean(0), src["X"][pool].std(0) + 1e-9
            Xd = (src["X"][pool] - mu) / sd
            Xq = (S["X"] - mu) / sd
            lA, lB = src["lA"][pool], src["lB"][pool]
            porg = src["org"][pool]

            kmax = max(KGRID) + (S["n_org"] and 0)
            kfetch = min(len(Xd), kmax * (6 if insample else 1))
            nn = NearestNeighbors(n_neighbors=kfetch, algorithm="brute",
                                  metric="euclidean", n_jobs=-1).fit(Xd)
            _, ind = nn.kneighbors(Xq)
            print(f"\n  ── {arm}  · DSEL {int(pool.sum()):,}행 "
                  f"({time.time()-t0:.0f}s)", flush=True)
            if insample:
                # 같은 origin 이웃은 창 21/28 이 겹쳐 사실상 복제 → 마스킹
                same = porg[ind] == S["org"][:, None]
                ind = np.where(same, -1, ind)

            print(f"  {'k':>5s}{'점수':>11s}{'A단독대비':>11s}{'전역w대비':>11s}"
                  f"{'B선택률':>9s}{'선택일치':>9s}")
            for k in KGRID:
                if insample:
                    sel = np.full((len(Xq), k), -1, dtype=np.int64)
                    for r in range(len(Xq)):
                        v = ind[r][ind[r] >= 0][:k]
                        sel[r, :len(v)] = v
                    ok = sel >= 0
                    ea = np.where(ok, lA[np.maximum(sel, 0)], 0).sum(1) / np.maximum(ok.sum(1), 1)
                    eb = np.where(ok, lB[np.maximum(sel, 0)], 0).sum(1) / np.maximum(ok.sum(1), 1)
                else:
                    sel = ind[:, :k]
                    ea, eb = lA[sel].mean(1), lB[sel].mean(1)
                useB = eb < ea
                p = np.where(useB, post(S["B"], **kw), post(S["A"], **kw))
                sc = float((S["W"] * loss(S["y"], p)).sum())
                # 사후 최적(반칙)과 얼마나 같은 선택을 했나
                truth = S["lB"] < S["lA"]
                agree = float((useB == truth)[S["y"] != 0].mean())
                print(f"  {k:>5d}{sc:>11.5f}{base-sc:>+11.5f}{gsc-sc:>+11.5f}"
                      f"{100*useB.mean():>8.1f}%{100*agree:>8.1f}%")

            if not insample:
                # 유능도 자체가 전이되는가 — 겨울 국소판정 vs 봄 국소판정
                sel = ind[:, :50]
                wint = (lB[sel].mean(1) < lA[sel].mean(1))
                print(f"    ※ 무작위로 찍어도 선택일치는 "
                      f"{100*max((S['lB']<S['lA'])[S['y']!=0].mean(), 1-(S['lB']<S['lA'])[S['y']!=0].mean()):.1f}% "
                      f"나온다 (다수 클래스 비율).")

    print("\n" + "=" * 98)
    print("""읽는 법
  · '선택일치' = KNORA 가 고른 모델이 실제로 그 칸에서 나았던 비율.
    다수 클래스 비율(항상 A 를 고를 때의 정답률)을 못 넘으면 **국소 유능도에 정보가 없다**는 뜻이다.
  · 낙관판이 좋고 정직판이 나쁘면, 이 방법론이 논문에서 되고 여기서 안 되는 이유가
    **알고리즘이 아니라 DSEL 과 타깃의 분포 차이**임이 증명된다.
  · '전역w대비' 가 음수면, 칸마다 고르는 것이 스칼라 하나보다 나쁘다는 뜻이다.""")
    print("=" * 98)
    print(f"총 {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
