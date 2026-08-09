# -*- coding: utf-8 -*-
"""Phase 21-d — 21-c 결과 정독. 헤드라인 판정이 놓친 것을 본다.

21-c 의 사전등록 게이트는 **blend(0.68·ridge + 0.32·같은요일평균)** 변형에 걸려 있었다.
그런데 격자표를 보면 **ridge 단독**이 재료로 더 낫다 (+0.00184 vs +0.00085).
왜 그런지, 그리고 그게 게이트를 통과할 자격이 있는지 폴드별로 뜯는다.

핵심 질문 3개
  ① ridge 단독이 왜 blend 보다 나은가 — 그들 baseline(같은요일 양수평균)은
     우리 `d_posmean` 피처와 같은 재료다. 이미 아는 걸 다시 섞는 셈.
  ② 어느 폴드가 반대로 가는가 — 겨울인가?
  ③ 이 비교의 문턱은 정말 2σ=0.003 인가 — **같은 두 벡터를 다른 w 로 섞는 것**은
     '후처리만 다름'과 같은 부류라 시드 무작위성이 상쇄된다. 문턱이 달라야 한다.
"""
import json
import os

import numpy as np

import config as C

_P = os.path.join(C.EXPERIMENTS, "phase21c_blend.json")
# ⚠️ 21-c 가 encoding 지정 없이 저장해 Windows 기본(cp949)으로 쓰였다. 둘 다 시도한다.
for _enc in ("utf-8", "cp949"):
    try:
        J = json.load(open(_P, encoding=_enc))
        break
    except UnicodeDecodeError:
        continue
FOLDS = ["F2 겨울", "F3 봄", "FAR-봄", "FAR-겨울"]
WS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def w_star(rho, k):
    """오차분산 최소 가중치. k = σ_theirs / σ_ours."""
    return (1 - rho * k) / (1 + k * k - 2 * rho * k)


def main():
    print("=" * 88)
    print("① ρ(잔차) — blend 는 그들 baseline 때문에 우리와 더 닮는다")
    print(f"{'폴드':<10s} {'ρ blend':>9s} {'ρ ridge':>9s} {'차이':>8s}   해석")
    for f in FOLDS:
        s = J[f]
        d = s["rho_blend"] - s["rho_ridge"]
        note = "baseline 이 상관을 크게 올림" if d > 0.03 else "차이 작음"
        print(f"{f:<10s} {s['rho_blend']:>9.4f} {s['rho_ridge']:>9.4f} {d:>+8.4f}   {note}")
    rb = np.mean([J[f]["rho_blend"] for f in FOLDS])
    rr = np.mean([J[f]["rho_ridge"] for f in FOLDS])
    print(f"{'평균':<10s} {rb:>9.4f} {rr:>9.4f} {rb-rr:>+8.4f}")
    print("  ※ 참고: 트리끼리는 LGBM↔XGB 0.9648 · LGBM↔Cat 0.9440 (Phase 5-a)")
    print(f"  → ridge 단독 {rr:.3f} 는 CatBoost보다도 이질적이다. 축은 실제로 열렸다.")

    print("\n" + "=" * 88)
    print("② 폴드별 w 격자 — ridge 단독 (w=0 대비 개선, +가 좋음)")
    print(f"{'w':>6s} " + "".join(f"{f:>12s}" for f in FOLDS) + f"{'평균':>11s}{'일관':>7s}")
    for w in WS:
        g = [J[f]["sweep_ridge"]["0.0"] - J[f]["sweep_ridge"][str(w)] for f in FOLDS]
        n = sum(x > 0 for x in g)
        print(f"{w:>6.2f} " + "".join(f"{x:>+12.5f}" for x in g)
              + f"{np.mean(g):>+11.5f}{n:>5d}/4")

    print("\n" + "=" * 88)
    print("③ 폴드별 최적 w — 부호가 뒤집히는가 (Phase 5-a 병리 재현 여부)")
    print(f"{'폴드':<10s} {'우리':>9s} {'그들ridge':>10s} {'k':>7s} {'ρ':>7s} "
          f"{'이론 w*':>9s} {'실측 최적w':>10s} {'그때 이득':>10s}")
    for f in FOLDS:
        s = J[f]
        sw = {float(k): v for k, v in s["sweep_ridge"].items()}
        base = sw[0.0]
        bw = min(sw, key=lambda w: sw[w])
        # k 는 점수비 대용이 아니라 잔차 표준편차비가 옳지만, 여기선 solo 점수비를 쓴다
        k = s["solo_tm"] / s["solo_ours"]
        ws = w_star(s["rho_ridge"], k)
        print(f"{f:<10s} {base:>9.5f} {'':>10s} {k:>7.3f} {s['rho_ridge']:>7.4f} "
              f"{ws:>+9.2f} {bw:>10.2f} {base-sw[bw]:>+10.5f}")
    print("  ※ k 는 blend 기준 점수비(ridge 단독 solo 점수는 21-c 가 안 남겼다) — 근사치다.")

    print("\n" + "=" * 88)
    print("④ 문턱 재검토 — 이 비교에 2σ=0.003 이 맞는가")
    print("""
  우리 분해능 표(메모리):
    · 후처리만 다름 (같은 .npy 에 다른 함수)  → 무작위성 출처 없음 → 문턱 ≈ 0
    · 재학습 동반 (피처·모델·시드)            → 2σ̂ ≈ 0.002~0.003

  w 격자 비교는 **어느 쪽인가?**
    한 폴드 안에서 w=0 과 w=0.10 은 *완전히 같은* LGBM 5시드 벡터와
    *완전히 같은* Ridge 벡터를 쓴다. 시드 난수가 양쪽에 동일하게 들어가므로
    **w 간 비교에는 시드 노이즈가 안 들어온다** → '후처리만 다름'에 가깝다.

  → 즉 **+0.0018 을 '2σ 미달이라 잡음'이라고 부르는 건 틀렸다.**
     이 숫자는 재현되는 값이다. 진짜 위험은 다른 데 있다:
     **폴드 4개가 TEST 의 10개 고정 창을 대표하는가** (규칙 ⑮ — 램프 부스트가
     4/4 통과하고도 실전에서 죽은 이유). 그건 σ 로 못 막는다.
""")


if __name__ == "__main__":
    main()
