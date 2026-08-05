# -*- coding: utf-8 -*-
"""Phase 6-d — 보정을 더 정교하게 할 여지가 있는가 (세분화 · 영업장군집별).

배경: 3구간 보정이 실측 **+0.0172** (OOF 예측 +0.0086의 2배). v9 = 대회 3등.
      그런데 T1 까지 열어 6분할 교차검증을 해도 **최적이 v9 설정 그대로**였다.
      → 같은 형태(3구간·전역) 안에서는 더 짤 게 없다.

그래서 **형태를 바꿔** 두 가지를 본다:
  A. 구간을 5개로 세분화 — 보정 곡선의 해상도를 올린다
  B. **영업장 군집별로 따로 보정** — 계절매장(오차 0.70)과 B2B(0.34)는 사실상 다른 문제다.
     공식 지표가 영업장별로 평균내므로 군집별 보정이 구조적으로 맞을 수 있다.

자유도가 늘어나므로 **6분할 교차검증이 필수**다(2폴드 맞춤 → 나머지 2폴드 채점).
낙관 수치와 정직 수치의 차이가 곧 과적합분이고, 그게 크면 기각한다.

최적화는 전격자 대신 **좌표하강**(한 구간씩 번갈아 최적화)을 쓴다 — 5구간 전격자는 너무 크다.
"""
import os
import json

import numpy as np

import config as C
import features as F
from metrics import competition_score, make_weights

NAMES = ["F2 겨울", "F3 봄", "FAR-봄", "FAR-겨울"]
FLOOR = 1.0
SIGMA2 = 0.0032
GRID = np.round(np.arange(0.40, 1.41, 0.025), 3)
SPLITS = [((0, 1), (2, 3)), ((2, 3), (0, 1)), ((0, 3), (1, 2)),
          ((1, 2), (0, 3)), ((0, 2), (1, 3)), ((1, 3), (0, 2))]

V9 = ([2.0, 10.0], [0.55, 0.90, 1.02])          # 현재 채택본


def main():
    ctx = F.Context()
    z = np.load(os.path.join(C.EXPERIMENTS, "phase6a_oof.npz"))
    folds = [dict(name=NAMES[i], p=z[f"p{i}"].astype(np.float64),
                  y=z[f"y{i}"], iids=z[f"i{i}"]) for i in range(4)]
    # 품목 → 군집 3그룹 (계절 / 상시 / B2B)
    grp3 = {"hwadam": 0, "ski": 0, "green": 0, "always": 1, "b2b": 2}
    item_grp = np.array([grp3[C.STORE_CLUSTER[s]] for s in ctx.store_of_item])
    GNAME = ["계절매장", "상시", "B2B"]
    for d in folds:
        d["g"] = item_grp[d["iids"]]

    def sc(d, p):
        return competition_score(d["y"], np.maximum(p, FLOOR), d["iids"],
                                 ctx.store_of_item, make_weights(1.0), ctx.n)

    def seg_apply(p, bnd, cs, mask=None):
        """구간별 배율. mask 가 주어지면 그 행에만 적용."""
        idx = np.digitize(p, bnd)
        q = p * np.asarray(cs)[idx]
        return np.where(mask, q, p) if mask is not None else q

    base = [sc(d, d["p"]) for d in folds]
    B = float(np.mean(base))
    ref = [sc(d, seg_apply(d["p"], *V9)) for d in folds]
    R = float(np.mean(ref))
    print("=" * 96)
    print("Phase 6-d — 보정 형태를 바꿔서 더 짤 게 있는가")
    print("=" * 96)
    print(f"  무보정 {B:.4f} · v9(3구간) {R:.4f} · OOF 개선 {B-R:+.4f}")
    print(f"  ※ 실측: 무보정 0.46883 → v9 0.45161 = +0.0172 (OOF의 2배)\n")

    def fit(bnd, fold_idx, mask_g=None, rounds=4):
        """좌표하강으로 구간 배율을 맞춘다. mask_g 가 있으면 그 군집만."""
        cs = [1.0] * (len(bnd) + 1)
        for _ in range(rounds):
            for k in range(len(cs)):
                best, bs = cs[k], 9.9
                for v in GRID:
                    cs[k] = v
                    s = np.mean([sc(folds[i],
                                    seg_apply(folds[i]["p"], bnd, cs,
                                              None if mask_g is None
                                              else folds[i]["g"] == mask_g))
                                 for i in fold_idx])
                    if s < bs:
                        bs, best = s, v
                cs[k] = best
        return cs

    # ------------------------------------------------------------ A. 5구간
    print("=" * 96)
    print("A. 5구간 세분화  (경계 1.5 / 2.5 / 5 / 15)")
    print("=" * 96)
    BND5 = [1.5, 2.5, 5.0, 15.0]
    c5_all = fit(BND5, range(4))
    s5 = [sc(d, seg_apply(d["p"], BND5, c5_all)) for d in folds]
    print(f"  4폴드 전체 최적(낙관): {[f'{x:.2f}' for x in c5_all]}")
    print(f"    " + " · ".join(f"{d['name']} {x:.4f}" for d, x in zip(folds, s5))
          + f"  평균 {np.mean(s5):.4f}  v9대비 {R-np.mean(s5):+.4f}")
    gains5 = []
    for fi, ti in SPLITS:
        c = fit(BND5, fi)
        new = float(np.mean([sc(folds[i], seg_apply(folds[i]["p"], BND5, c)) for i in ti]))
        old = float(np.mean([ref[i] for i in ti]))
        gains5.append(old - new)
    g5 = float(np.mean(gains5))
    print(f"  ★ 교차검증 v9 대비 개선 {g5:+.4f}  "
          f"({sum(1 for g in gains5 if g > 0)}/6 개선)  "
          f"{'→ 채택' if g5 > SIGMA2 else '→ 기각 (2σ 미달)'}")

    # ------------------------------------------------------------ B. 군집별
    print("\n" + "=" * 96)
    print("B. 영업장 군집별 3구간 보정  (계절매장 / 상시 / B2B)")
    print("=" * 96)
    for gi, gn in enumerate(GNAME):
        n = int((item_grp == gi).sum())
        print(f"  {gn:<8s} 품목 {n:>3d}개")
    cg_all = {}
    for gi in range(3):
        cg_all[gi] = fit(V9[0], range(4), mask_g=gi)

    def apply_grp(d, cg):
        q = d["p"].copy()
        for gi in range(3):
            m = d["g"] == gi
            q[m] = seg_apply(d["p"], V9[0], cg[gi])[m]
        return q
    sg = [sc(d, apply_grp(d, cg_all)) for d in folds]
    print(f"\n  4폴드 전체 최적(낙관):")
    for gi, gn in enumerate(GNAME):
        print(f"    {gn:<8s} {[f'{x:.2f}' for x in cg_all[gi]]}")
    print(f"    " + " · ".join(f"{d['name']} {x:.4f}" for d, x in zip(folds, sg))
          + f"  평균 {np.mean(sg):.4f}  v9대비 {R-np.mean(sg):+.4f}")
    gainsg = []
    for fi, ti in SPLITS:
        cg = {gi: fit(V9[0], fi, mask_g=gi) for gi in range(3)}
        new = float(np.mean([sc(folds[i], apply_grp(folds[i], cg)) for i in ti]))
        old = float(np.mean([ref[i] for i in ti]))
        gainsg.append(old - new)
    gg = float(np.mean(gainsg))
    print(f"  ★ 교차검증 v9 대비 개선 {gg:+.4f}  "
          f"({sum(1 for g in gainsg if g > 0)}/6 개선)  "
          f"{'→ 채택' if gg > SIGMA2 else '→ 기각 (2σ 미달)'}")

    print("\n" + "=" * 96)
    print("판정")
    print("=" * 96)
    print(f"  A 5구간   교차검증 {g5:+.4f}  (낙관 {R-np.mean(s5):+.4f}, "
          f"과적합분 {abs((R-np.mean(s5))-g5):.4f})")
    print(f"  B 군집별  교차검증 {gg:+.4f}  (낙관 {R-np.mean(sg):+.4f}, "
          f"과적합분 {abs((R-np.mean(sg))-gg):.4f})")
    win = "A 5구간" if g5 > gg else "B 군집별"
    bg = max(g5, gg)
    if bg > SIGMA2:
        print(f"  → ★ {win} 채택 권고 (v9 대비 {bg:+.4f})")
    else:
        print(f"  → 둘 다 2σ 미달. **v9 유지가 정답.** 보정 축은 여기서 종료.")

    json.dump(dict(base=base, v9=ref, seg5=dict(bnd=BND5, c=c5_all,
                                                naive=float(R - np.mean(s5)), cv=g5),
                   grp=dict(c={GNAME[k]: cg_all[k] for k in cg_all},
                            naive=float(R - np.mean(sg)), cv=gg)),
              open(os.path.join(C.EXPERIMENTS, "phase6d_finer.json"),
                   "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n저장: experiments/phase6d_finer.json")


if __name__ == "__main__":
    main()
