# -*- coding: utf-8 -*-
"""Phase 23 — 방법론 조사가 내놓은 주장 3개를 우리 데이터로 직접 검증한다.

조사(24 에이전트, 15개 방법 전부 탈락)에서 나온 **검증 가능한 주장**:

  주장 ① "이론적 천장이 +0.0002~0.0006 뿐이다. 그런데 너희가 관측한 ±0.001~0.003 은
          그보다 5~15배 크다. 그 차이는 블렌드가 아니라 **정수 스냅 격자를 넘나드는
          이산 점프**에서 왔다."
      → 검증: 스냅을 뺀 채 같은 w 스윕을 돌린다. 이득이 작고 매끄러워지면 주장이 맞다.
        우리 메모리에 이미 방증이 있다 — "양자화는 시드 노이즈를 흡수하지 않고 키운다".

  주장 ② "k(=σ_B/σ_A)는 점수비가 아니라 로그잔차비여야 하고, SMAPE 포화 때문에
          점수비는 σ비를 **과소평가**한다."
      → 우리는 이미 21-f 에서 로그잔차로 쟀다. 방향이 맞는지 대조한다.
        (에이전트들에겐 점수비만 줬으므로 그들은 이 측정을 몰랐다.)

  주장 ③ "라우팅의 성패는 이질성 크기가 아니라 **레짐 간 지속성 p** 이고 손익분기는
          p ≈ 0.5~0.6 이다. 이 데이터의 p 는 0.3 근처라 **오라클조차 손해**다."
      → 검증: 품목별 우위를 겨울 폴드와 봄 폴드에서 각각 재고 상관을 본다 (n=193).
        그리고 실제로 겨울에서 고른 라우팅을 봄에 적용해 본다.
        ⭐ Phase 22 가 창 10개(검정력 없음)로 검증한 것을 **품목 193개**로 다시 한다.
"""
import os

import numpy as np

import config as C
import features as F
from run_phase10c_thresholds import cell_weights

NPZ = os.path.join(C.EXPERIMENTS, "phase21c_oof.npz")
FOLDS = ["F2 겨울", "F3 봄", "FAR-봄", "FAR-겨울"]
WGRID = np.round(np.arange(0.0, 0.55, 0.05), 3)


def post(raw, snap=True, seg=True):
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


def wstat(w, x):
    m = np.sum(w * x) / np.sum(w)
    return m, np.sqrt(np.sum(w * (x - m) ** 2) / np.sum(w))


def main():
    z = np.load(NPZ)
    ctx = F.Context()
    D_ = {}
    for f in FOLDS:
        y, iid = z[f"y|{f}"], z[f"iid|{f}"].astype(int)
        W, _ = cell_weights(y, iid, ctx.store_of_item, ctx.n)
        D_[f] = dict(y=y, iid=iid, W=W, A=z[f"ours|{f}"], B=z[f"tm_ridge|{f}"])

    # ───────────────────────────── 주장 ① 스냅이 증폭기인가
    print("=" * 96)
    print("① 정수 스냅을 빼면 이득이 작고 매끄러워지는가 (4폴드 평균 · w=0 대비)")
    print(f"{'w':>6s}" + "".join(f"{n:>16s}" for n in
                                 ("seg+스냅(실제)", "seg만(스냅X)", "하한만")))
    curves = {}
    for tag, kw in (("snap", dict(snap=True, seg=True)),
                    ("nosnap", dict(snap=False, seg=True)),
                    ("floor", dict(snap=False, seg=False))):
        g = []
        for w in WGRID:
            per = [float((D_[f]["W"] * loss(D_[f]["y"], post(
                       (1 - w) * D_[f]["A"] + w * D_[f]["B"], **kw))).sum())
                   for f in FOLDS]
            g.append(np.array(per))
        curves[tag] = np.array(g)                      # (len(WGRID), 4)
    for i, w in enumerate(WGRID):
        row = [(curves[t][0] - curves[t][i]).mean() for t in ("snap", "nosnap", "floor")]
        print(f"{w:>6.2f}" + "".join(f"{v:>+16.5f}" for v in row))

    print("\n  폴드별 '창별 요동'의 크기 — 곡선이 얼마나 울퉁불퉁한가")
    for t, nm in (("snap", "seg+스냅"), ("nosnap", "seg만"), ("floor", "하한만")):
        d2 = np.diff(curves[t], n=2, axis=0)           # 2차 차분 = 매끄러움의 반대
        print(f"    {nm:<10s} 2차차분 RMS {np.sqrt((d2**2).mean()):.6f}"
              f" · w=0.10 이득 {(curves[t][0]-curves[t][2]).mean():+.5f}")
    print("  → 스냅판만 크고 거칠면, 관측된 이득의 정체는 **양자화 잡음**이다.")

    # ───────────────────────────── 주장 ② k 는 점수비보다 큰가 작은가
    print("\n" + "=" * 96)
    print("② 점수비 vs 실측 로그잔차비 — 조사는 '점수비가 σ비를 과소평가'한다고 했다")
    print(f"{'폴드':<10s}{'점수비':>9s}{'σ비(실측)':>11s}{'차이':>9s}   판정")
    for f in FOLDS:
        d = D_[f]
        sA = float((d["W"] * loss(d["y"], post(d["A"]))).sum())
        sB = float((d["W"] * loss(d["y"], post(d["B"]))).sum())
        sc = d["y"] != 0
        lg = lambda v: np.log1p(np.maximum(v, 0.0))
        a = lg(np.maximum(np.abs(d["y"][sc]), 1.0))
        _, s1 = wstat(d["W"][sc], lg(np.maximum(d["A"][sc], 1.0)) - a)
        _, s2 = wstat(d["W"][sc], lg(d["B"][sc]) - a)
        sr, kr = sB / sA, s2 / s1
        print(f"{f:<10s}{sr:>9.4f}{kr:>11.4f}{kr-sr:>+9.4f}   "
              f"{'과소평가(조사 주장대로)' if kr > sr else '과대평가(조사와 반대)'}")

    # ───────────────────────────── 주장 ③ 품목별 우위의 계절 간 지속성
    print("\n" + "=" * 96)
    print("③ 품목별 우위가 계절을 넘어 지속되는가 (n=193 · 겨울 F2 ↔ 봄 F3)")

    def per_item(f):
        d = D_[f]
        pa = loss(d["y"], post(d["A"])) * d["W"]
        pb = loss(d["y"], post(d["B"])) * d["W"]
        wt = np.bincount(d["iid"], weights=d["W"], minlength=ctx.n)
        sa = np.bincount(d["iid"], weights=pa, minlength=ctx.n)
        sb = np.bincount(d["iid"], weights=pb, minlength=ctx.n)
        ok = wt > 0
        adv = np.full(ctx.n, np.nan)
        adv[ok] = (sa[ok] - sb[ok]) / wt[ok]           # + 이면 그 품목은 B 가 낫다
        return adv, wt

    aw, ww = per_item("F2 겨울")
    as_, ws = per_item("F3 봄")
    both = ~np.isnan(aw) & ~np.isnan(as_)
    p_hat = float(np.corrcoef(aw[both], as_[both])[0, 1])
    print(f"  겨울·봄 모두 채점된 품목 {both.sum()}개 · **지속성 p̂ = {p_hat:+.3f}**")
    print(f"  B 가 나은 품목: 겨울 {int((aw[both]>0).sum())}개 · 봄 {int((as_[both]>0).sum())}개"
          f" · 양쪽 다 {int(((aw[both]>0)&(as_[both]>0)).sum())}개")
    print(f"  조사가 제시한 손익분기: p ≈ 0.5~0.6 (그 아래면 오라클조차 손해)")

    # 실제 라우팅: 겨울에서 품목별 최적 w 를 고르고 봄에 적용
    print("\n  실제 교차 라우팅 (겨울에서 고른 품목별 w → 봄에 적용)")
    def item_scores(f, ws_grid):
        d = D_[f]
        out = {}
        for w in ws_grid:
            l = loss(d["y"], post((1 - w) * d["A"] + w * d["B"])) * d["W"]
            out[w] = np.bincount(d["iid"], weights=l, minlength=ctx.n)
        return out
    grid = [0.0, 0.1, 0.2, 0.3]
    win, spr = item_scores("F2 겨울", grid), item_scores("F3 봄", grid)
    base_spr = spr[0.0].sum()
    pick_w = np.array([min(grid, key=lambda w: win[w][i]) for i in range(ctx.n)])
    cross = sum(spr[pick_w[i]][i] for i in range(ctx.n))
    oracle = sum(min(spr[w][i] for w in grid) for i in range(ctx.n))
    glob = min((spr[w].sum() for w in grid))
    print(f"    봄 · A 단독            {base_spr:.5f}")
    print(f"    봄 · 전역 최적 w       {glob:.5f}   ({base_spr-glob:+.5f})")
    print(f"    봄 · 겨울에서 고른 w   {cross:.5f}   ({base_spr-cross:+.5f})  ← 정직한 라우팅")
    print(f"    봄 · 오라클(봄에서 고름) {oracle:.5f}   ({base_spr-oracle:+.5f})  ← 반칙, 천장")
    print(f"    선택된 w 분포: " +
          " ".join(f"w={w}:{int((pick_w==w).sum())}개" for w in grid))
    print("\n" + "=" * 96)
    if base_spr - cross > 0:
        print("  → 교차 라우팅이 이득이다. 지속성이 실재한다는 뜻 — 추가 조사 가치 있음.")
    else:
        print("  → **교차 라우팅이 손해다.** 겨울에서 배운 품목별 우위가 봄에 안 통한다.")
        print("     Phase 22 를 창 10개가 아니라 품목 193개에서 다시 한 셈이고 답이 같다.")
        print("     **조건부 결합 축, 정량적 근거와 함께 영구 종료.**")
    print("=" * 96)


if __name__ == "__main__":
    main()
