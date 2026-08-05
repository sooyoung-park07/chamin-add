# -*- coding: utf-8 -*-
"""Phase 5-a 후속 — 저장된 OOF로 **가중치를 제대로 줬을 때** 앙상블 이득을 다시 본다.

동기: 정찰(run_phase5a_scout)의 등가중 평균은 CatBoost가 단독으로 0.011 나쁘기 때문에
      끌려 내려간다. "앙상블이 값어치가 있나"를 등가중 하나로 판정하면 과소평가한다.

핵심은 **정직성 검사**다. 가중치를 F2에서 맞춰 F3에서 채점하고, 반대로도 해본다.
같은 폴드에서 맞추고 같은 폴드에서 채점한 숫자는 언제나 좋아 보이기 때문이다
(자유도 2개를 origin 30개에 맞추는 셈).

재학습 없음 — experiments/phase5a_oof.npz 만 읽는다. 수 초면 끝난다.
"""
import os
import json
import itertools

import numpy as np

import config as C
import features as F
from metrics import competition_score, make_weights

ALGOS = ["LightGBM", "XGBoost", "CatBoost"]
FOLD_NAMES = ["F1 가을", "F2 겨울", "F3 봄"]
FLOOR = 1.0
STEP = 0.05          # 심플렉스 격자 간격


def main():
    ctx = F.Context()
    z = np.load(os.path.join(C.EXPERIMENTS, "phase5a_oof.npz"))

    folds = []
    for fi in range(3):
        folds.append(dict(
            name=FOLD_NAMES[fi], y=z[f"y{fi}"], iids=z[f"i{fi}"],
            # 시드 평균까지만 미리 해둔다 (각자의 '최종 예측'에 해당). pre-floor.
            P=np.stack([z[f"p{fi}_{a}"].mean(0) for a in ALGOS]),
        ))

    def sc(fd, w):
        p = np.maximum(np.tensordot(w, fd["P"], axes=1), FLOOR)
        return competition_score(fd["y"], p, fd["iids"], ctx.store_of_item,
                                 make_weights(1.0), ctx.n)

    def avg(w, idx):
        return float(np.mean([sc(folds[i], w) for i in idx]))

    grid = [np.array(c) for c in itertools.product(
        np.arange(0, 1 + 1e-9, STEP), repeat=3) if abs(sum(c) - 1) < 1e-9]
    eq = np.ones(3) / 3

    print("=" * 88)
    print("Phase 5-a 후속 — 가중치를 최적화하면 앙상블 이득이 달라지는가")
    print("=" * 88)
    solo = {a: avg(np.eye(3)[k], [1, 2]) for k, a in enumerate(ALGOS)}
    best_solo_k = min(solo, key=solo.get)
    best_solo = solo[best_solo_k]
    print("  단일(F2+F3):  " + " · ".join(f"{a} {v:.4f}" for a, v in solo.items()))
    print(f"  최고 단일 = {best_solo_k} {best_solo:.4f}")
    print(f"  등가중 평균  {avg(eq, [1, 2]):.4f}   "
          f"(개선 {best_solo - avg(eq, [1, 2]):+.4f})")

    # ---------------------------------------------------------- 같은 폴드에서 맞추고 채점 (낙관적)
    w_in = min(grid, key=lambda w: avg(w, [1, 2]))
    s_in = avg(w_in, [1, 2])
    print("\n" + "-" * 88)
    print("A) F2+F3에서 맞추고 F2+F3에서 채점  ← 낙관 편향. 이 숫자를 믿으면 안 된다")
    print("-" * 88)
    print(f"  최적 가중 {dict(zip(ALGOS, np.round(w_in, 2)))}  →  {s_in:.4f}   "
          f"(개선 {best_solo - s_in:+.4f})")

    # ---------------------------------------------------------- 폴드 교차 (정직)
    print("\n" + "-" * 88)
    print("B) 한 폴드에서 맞추고 **다른 폴드에서 채점**  ← 이게 진짜 이득")
    print("-" * 88)
    honest = []
    for fit_i, test_i in [(1, 2), (2, 1)]:
        w = min(grid, key=lambda w: avg(w, [fit_i]))
        s_w = avg(w, [test_i])
        s_eq = avg(eq, [test_i])
        s_best = avg(np.eye(3)[ALGOS.index(best_solo_k)], [test_i])
        # 축소(shrinkage): 등가중 쪽으로 절반 당긴다. 자유도 대비 표본이 적을 때의 정석.
        w_sh = 0.5 * w + 0.5 * eq
        s_sh = avg(w_sh, [test_i])
        honest.append(dict(fit=FOLD_NAMES[fit_i], test=FOLD_NAMES[test_i],
                           w=w.tolist(), weighted=s_w, shrunk=s_sh,
                           equal=s_eq, best_solo=s_best))
        print(f"\n  [{FOLD_NAMES[fit_i]}에서 맞춤 → {FOLD_NAMES[test_i]}에서 채점]"
              f"  가중치 {dict(zip(ALGOS, np.round(w, 2)))}")
        print(f"    최고 단일({best_solo_k}) {s_best:.4f}")
        print(f"    등가중              {s_eq:.4f}   ({s_best - s_eq:+.4f})")
        print(f"    최적가중            {s_w:.4f}   ({s_best - s_w:+.4f})")
        print(f"    축소가중(50%)       {s_sh:.4f}   ({s_best - s_sh:+.4f})")

    g_eq = float(np.mean([h["best_solo"] - h["equal"] for h in honest]))
    g_w = float(np.mean([h["best_solo"] - h["weighted"] for h in honest]))
    g_sh = float(np.mean([h["best_solo"] - h["shrunk"] for h in honest]))

    print("\n" + "=" * 88)
    print("정리 — 폴드 교차 평균 개선폭 (양수 = 좋아짐)")
    print("=" * 88)
    print(f"  등가중      {g_eq:+.4f}")
    print(f"  최적가중    {g_w:+.4f}")
    print(f"  축소가중    {g_sh:+.4f}   ← 실전에서 쓸 값")
    print(f"\n  참고: 낙관 편향 버전은 {best_solo - s_in:+.4f} 였다. "
          f"차이 {abs((best_solo - s_in) - g_w):.4f} 만큼이 가중치 과적합.")

    json.dump(dict(solo=solo, best_solo=best_solo_k, equal_f2f3=avg(eq, [1, 2]),
                   optimistic=dict(w=w_in.tolist(), score=s_in),
                   honest=honest, gains=dict(equal=g_eq, weighted=g_w, shrunk=g_sh)),
              open(os.path.join(C.EXPERIMENTS, "phase5a_weights.json"),
                   "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n저장: experiments/phase5a_weights.json")


if __name__ == "__main__":
    main()
