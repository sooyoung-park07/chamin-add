# -*- coding: utf-8 -*-
"""Phase 9-a — 내부 채점이 틀린 이유가 **영업장 가중치** 때문인가.

━━ 가설 ━━
대회 규칙: 영업장 가중 SMAPE이고 **담하·미라시아 가중치가 더 높다(값 비공개)**.
우리 내부 채점: **균등 가중**을 쓴다(값을 모르니 가장 보수적인 선택이라고 판단했었음).

초기에 민감도를 재보고 "균등 0.6185 / 2배 0.6131 로 폭이 0.009뿐이니 우열은 안 뒤집힌다"고
결론냈는데, 그건 **규칙 베이스라인 하나로** 잰 것이었다. **모델 간 순위가 뒤집히는지는 안 봤다.**

━━ 지금 검증할 수 있는 이유 ━━
보정 변형 4개(v8/v9/v10/v11)는 **완전히 같은 원본 예측**에서 나왔고, **실제 채점 점수를 전부 안다.**
→ 내부 채점의 가중치만 바꿔가며 "어느 가중치에서 실제 순위를 재현하는가"를 직접 확인할 수 있다.

실제 합산 점수 (Public+Private)/2:
    v9  0.45161  <  v10 0.45544  <  v11 0.46146  <  v8 0.46883      (낮을수록 좋음)

균등 가중 내부 채점은 v9와 v11을 **거의 동률**로 봤다(0.4983 vs 0.4982).
실제로는 v9가 0.0099 낫다. **가중치를 올리면 이 차이가 드러나는가?**
"""
import os
import json

import numpy as np

import config as C
import features as F
from metrics import competition_score, make_weights

OOF = os.path.join(C.EXPERIMENTS, "phase8a_oof.npz")
NAMES = ["F2 겨울", "F3 봄", "FAR-봄", "FAR-겨울"]
FLOOR = 1.0
BND = [2.0, 10.0]

# (이름, 배율, 실제 합산 점수)
CANDS = [
    ("v8  무보정", [1.00, 1.00, 1.00], 0.46883),
    ("v9  채택본", [0.55, 0.90, 1.02], 0.45161),
    ("v10 낮은구간만", [0.55, 1.00, 1.00], 0.45544),
    ("v11 저울교정", [0.60, 0.93, 1.18], 0.46146),
]
HIGHS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0]


def spearman(a, b):
    ra = np.argsort(np.argsort(a)) + 1
    rb = np.argsort(np.argsort(b)) + 1
    n = len(a)
    return 1 - 6 * float(((ra - rb) ** 2).sum()) / (n * (n * n - 1))


def main():
    ctx = F.Context()
    z = np.load(OOF)
    folds = [dict(name=NAMES[i], p=z[f"ps{i}"].astype(np.float64).mean(0),
                  y=z[f"y{i}"], iids=z[f"i{i}"]) for i in range(4)]
    stores = np.array(ctx.store_of_item)

    print("=" * 96)
    print("Phase 9-a — 내부 채점의 영업장 가중치가 문제인가")
    print("=" * 96)
    n_hi = int(np.isin(stores, C.HIGH_WEIGHT_STORES).sum())
    print(f"  가중치 높은 영업장: {C.HIGH_WEIGHT_STORES}  (품목 {n_hi}개 / 전체 {ctx.n}개)")
    print(f"  실제 순위: " + " < ".join(
        f"{n.split()[0]}({t:.4f})" for n, _, t in sorted(CANDS, key=lambda r: r[2])))
    print()

    def apply(p, cs):
        return p * np.asarray(cs)[np.digitize(p, BND)]

    def sc(cs, high, pooled=False):
        w = make_weights(high)
        if pooled:
            y = np.concatenate([d["y"] for d in folds])
            p = np.concatenate([apply(d["p"], cs) for d in folds])
            ii = np.concatenate([d["iids"] for d in folds])
            return competition_score(y, np.maximum(p, FLOOR), ii, stores, w, ctx.n)
        return float(np.mean([competition_score(
            d["y"], np.maximum(apply(d["p"], cs), FLOOR), d["iids"],
            stores, w, ctx.n) for d in folds]))

    truth = np.array([t for _, _, t in CANDS])

    print("=" * 96)
    print("① 가중치를 올려가며 내부 점수 — 실제 순위를 재현하는가")
    print("=" * 96)
    print(f"  {'담하·미라시아 가중':>18s}" + "".join(f"{n:>16s}" for n, _, _ in CANDS)
          + f"{'실제와 순위상관':>16s}{'v9<v11?':>10s}")
    rows = []
    for h in HIGHS:
        s = np.array([sc(cs, h) for _, cs, _ in CANDS])
        rho = spearman(s, truth)
        ok = s[1] < s[3]                      # v9 가 v11 보다 좋게 나오는가
        rows.append(dict(high=h, scores=s.tolist(), rho=rho, v9_beats_v11=bool(ok)))
        print(f"  {h:>18.1f}" + "".join(f"{x:>16.4f}" for x in s)
              + f"{rho:>16.2f}{'○' if ok else '×':>9s}")

    print("\n  ※ '실제와 순위상관' 1.00 = 내부 채점이 실제 순위를 완전히 재현.")
    print("     'v9<v11?' 은 실제로 0.0099 차이나는 두 후보를 내부 채점이 올바로 가르는가.")

    print("\n" + "=" * 96)
    print("② 실제 점수 차이를 얼마나 잘 재현하는가 (v9 기준 상대 차이)")
    print("=" * 96)
    d_true = truth - truth[1]
    print(f"  {'가중':>6s}" + "".join(f"{n.split()[0]:>12s}" for n, _, _ in CANDS))
    print(f"  {'실제':>6s}" + "".join(f"{x:>+12.4f}" for x in d_true))
    print("  " + "-" * 60)
    best_h, best_err = None, 9.9
    for r in rows:
        s = np.array(r["scores"])
        d = s - s[1]
        err = float(np.abs(d - d_true).mean())
        if err < best_err:
            best_err, best_h = err, r["high"]
        print(f"  {r['high']:>6.1f}" + "".join(f"{x:>+12.4f}" for x in d)
              + f"   평균오차 {err:.4f}")
    print(f"\n  → 실제 차이를 가장 잘 재현하는 가중치: **{best_h}** (평균오차 {best_err:.4f})")

    print("\n" + "=" * 96)
    print("③ 영업장별로 뜯어보기 — v9 vs v11 이 어디서 갈리나")
    print("=" * 96)
    y = np.concatenate([d["y"] for d in folds])
    ii = np.concatenate([d["iids"] for d in folds])
    per = {}
    for lab, cs in (("v9", CANDS[1][1]), ("v11", CANDS[3][1])):
        p = np.concatenate([apply(d["p"], cs) for d in folds])
        _, ps_, _, _ = competition_score(y, np.maximum(p, FLOOR), ii, stores,
                                         None, ctx.n, return_parts=True)
        per[lab] = ps_
    print(f"  {'영업장':<14s}{'품목수':>7s}{'v9':>10s}{'v11':>10s}{'차이':>10s}{'가중':>7s}")
    tot = 0.0
    for s in sorted(per["v9"], key=lambda k: per["v11"][k] - per["v9"][k]):
        n = int((stores == s).sum())
        d = per["v11"][s] - per["v9"][s]
        tot += d
        hi = "★높음" if s in C.HIGH_WEIGHT_STORES else ""
        print(f"  {s:<14s}{n:>7d}{per['v9'][s]:>10.4f}{per['v11'][s]:>10.4f}"
              f"{d:>+10.4f}{hi:>9s}")
    print(f"  {'(균등 평균)':<14s}{'':>7s}{'':>10s}{'':>10s}{tot/len(per['v9']):>+10.4f}")
    print("\n  → v11이 나빠진 곳이 **가중치 높은 영업장에 몰려 있으면**, 균등 가중 채점이")
    print("     그 손해를 축소해서 봤다는 뜻이다.")

    json.dump(dict(rows=rows, best_high=best_h, best_err=best_err,
                   per_store={k: {s: float(v) for s, v in d.items()}
                              for k, d in per.items()}),
              open(os.path.join(C.EXPERIMENTS, "phase9a_weight_check.json"),
                   "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n저장: experiments/phase9a_weight_check.json")


if __name__ == "__main__":
    main()
