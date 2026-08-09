# -*- coding: utf-8 -*-
"""Phase 21-f — 21-e 의 판정을 다시 읽는다. **내 사전등록 규칙이 틀린 전제 위에 있었다.**

21-e 규칙: gain(B) − gain(C) 를 '거리 효과'로 읽었다.
           B 와 C 는 둘 다 origin 209개니까 학습량이 같다고 본 것이다.
**그 전제가 거짓이다.**
    B (cut 11-24, 뒤 209개) → 학습행 143,500   ← 봄·여름·가을 구간
    C (cut 08-25, 209개)    → 학습행 118,996   ← 겨울(화담숲 0) 포함
    같은 origin 수인데 **학습행이 21% 다르다.** 계절마다 양수 비율이 다르기 때문이다.
Ridge 는 3,201열짜리 원핫이라 굶주림의 척도는 origin 이 아니라 **행**이다.

그래서 x축을 행으로 바꿔 다시 본다. 거리 0 인 세 점(A·B·B2)으로 추세를 만들고,
그 추세가 C 의 행수에서 무엇을 예측하는지 본다. **예측과 실측의 차이 = 순수 거리 효과.**

⚠️ 이건 결과를 본 뒤에 저울을 손보는 것이다(규칙 ②가 경고한 동작).
   그래서 아래 수치는 '판정'이 아니라 '해석'으로만 쓴다. 원래 게이트(+0.003·4/4)는
   여전히 탈락이며, 이 문서가 그걸 뒤집지 않는다.

같이 하는 것: σ 를 점수비 대용이 아니라 **가중 로그잔차 표준편차**로 제대로 재서
             문턱 ρ < σ₁/σ₂ 를 엄밀하게 확인한다.
"""
import json
import os

import numpy as np

import config as C
import dataio as D
import features as F
from run_phase10c_thresholds import cell_weights

FOLDS = ["F2 겨울", "F3 봄", "FAR-봄", "FAR-겨울"]


def load_json(path):
    for enc in ("utf-8", "cp949"):
        try:
            return json.load(open(path, encoding=enc))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(path)


def wstat(w, x):
    m = np.sum(w * x) / np.sum(w)
    v = np.sum(w * (x - m) ** 2) / np.sum(w)
    return m, np.sqrt(v)


def wcorr(w, x, y):
    mx, sx = wstat(w, x)
    my, sy = wstat(w, y)
    return float(np.sum(w * (x - mx) * (y - my)) / np.sum(w) / (sx * sy))


def w_star(rho, k):
    return (1 - rho * k) / (1 + k * k - 2 * rho * k)


def main():
    # ── ① σ 를 제대로 잰다 (점수비 대용 폐기)
    z = np.load(os.path.join(C.EXPERIMENTS, "phase21c_oof.npz"))
    ctx = F.Context()
    print("=" * 94)
    print("① σ 를 가중 로그잔차 표준편차로 다시 잰다 (지금까지 쓴 '점수비'는 대용품이었다)")
    print(f"{'폴드':<10s}{'σ 우리':>9s}{'σ 그들':>9s}{'k=σ₂/σ₁':>10s}"
          f"{'ρ(가중)':>9s}{'문턱 1/k':>9s}{'통과':>6s}{'이론 w*':>9s}{'실측 최적w':>10s}")
    e = load_json(os.path.join(C.EXPERIMENTS, "phase21c_blend.json"))
    for f in FOLDS:
        y = z[f"y|{f}"]
        ours = z[f"ours|{f}"]
        tm = z[f"tm_ridge|{f}"]
        iid = z[f"iid|{f}"].astype(int)
        W, _ = cell_weights(y, iid, ctx.store_of_item, ctx.n)
        sc = y != 0
        lg = lambda v: np.log1p(np.maximum(v, 0.0))
        a = lg(np.maximum(np.abs(y[sc]), 1.0))
        r1 = lg(np.maximum(ours[sc], 1.0)) - a
        r2 = lg(tm[sc]) - a
        w = W[sc]
        _, s1 = wstat(w, r1)
        _, s2 = wstat(w, r2)
        rho = wcorr(w, r1, r2)
        k = s2 / s1
        ws = w_star(rho, k)
        sw = {float(kk): v for kk, v in e[f]["sweep_ridge"].items()}
        bw = min(sw, key=lambda t: sw[t])
        print(f"{f:<10s}{s1:>9.4f}{s2:>9.4f}{k:>10.3f}{rho:>9.4f}{1/k:>9.4f}"
              f"{'✅' if rho < 1/k else '❌':>6s}{ws:>+9.2f}{bw:>10.2f}")
    print("  → 이론 w* 와 실측 최적 w 가 같은 자리를 가리키면 산수가 현상을 설명한 것이다.")

    # ── ② x축을 origin 이 아니라 '학습행'으로
    d = load_json(os.path.join(C.EXPERIMENTS, "phase21e_datavolume.json"))
    pts = [(v["rows"], v["sweep"]["0.0"] - v["sweep"]["0.1"], k) for k, v in d.items()]
    pts.sort()
    print("\n" + "=" * 94)
    print("② x축을 '학습행'으로 바꾸면 네 점이 한 곡선에 앉는가")
    print(f"{'학습행':>10s}{'거리':>7s}{'w=0.10 이득':>13s}   실험")
    for rows, g, lab in pts:
        dist = "3개월" if lab.startswith("C") else "0"
        print(f"{rows:>10,d}{dist:>7s}{g:>+13.5f}   {lab}")

    far = [p for p in pts if p[2].startswith("C")][0]
    near = sorted([p for p in pts if not p[2].startswith("C")])
    xs = np.array([p[0] for p in near], float)
    ys = np.array([p[1] for p in near], float)
    lo = [p for p in near if p[0] < far[0]][-1]
    hi = [p for p in near if p[0] > far[0]][0]
    pred = lo[1] + (far[0] - lo[0]) / (hi[0] - lo[0]) * (hi[1] - lo[1])
    print(f"\n  거리 0 인 세 점 중 C 를 감싸는 두 점: {lo[0]:,}행({lo[1]:+.5f}) ~ "
          f"{hi[0]:,}행({hi[1]:+.5f})")
    print(f"  그 추세가 C 의 {far[0]:,}행에서 예측하는 값 : {pred:+.5f}")
    print(f"  C 실측                                    : {far[1]:+.5f}")
    print(f"  → **순수 거리 효과 = {far[1]-pred:+.5f}**")
    span = ys.max() - ys.min()
    print(f"  → 학습량 효과(107k→185k 구간) = {span:+.5f}  ·  "
          f"거리 효과의 약 {abs(span/(far[1]-pred)):.0f}배")

    # ── ③ 실제 제출 학습량에서는?
    print("\n" + "=" * 94)
    print("③ 실제 제출(320,012행)에서는 — ⚠️ 외삽이다. 믿지 말고 방향만 본다")
    sl = (hi[1] - lo[1]) / (hi[0] - lo[0])
    top = max(near)
    sl_top = (top[1] - hi[1]) / (top[0] - hi[0]) if top[0] != hi[0] else sl
    for nm, s, base in (("아래 구간 기울기", sl, hi), ("위 구간 기울기", sl_top, top)):
        ext = base[1] + (320_012 - base[0]) * s
        print(f"  {nm} {s*1e5:+.5f}/10만행 → 320,012행 외삽 {ext:+.5f}")
    print("  ※ 수확체감이 거의 확실하므로 위 값은 상한이다. 하한은 A 의 +0.00361 에서")
    print("    거리 페널티(3개월당 약 0.001, TEST 평균 6개월이면 약 0.002)를 뺀 +0.0016 근처.")

    print("\n" + "=" * 94)
    print("""④ 결론 — 무엇이 확정이고 무엇이 아닌가

  확정 (측정)
    · ρ(잔차) ≈ 0.87 — Ridge 는 CatBoost(0.944)보다 이질적이다. 축은 열려 있다.
    · 팀원 blend 보다 **ridge 단독**이 재료로 낫다 (baseline 이 우리 d_posmean 과 겹침).
    · 등가중(w=0.5)은 −0.011. 작은 w(0.05~0.15)에서만 이득.
    · 이득이 **학습량과 강하게 붙어 있다**. 107k행 −0.006 → 185k행 +0.004.

  확정 아님 (해석)
    · '거리 효과가 작다'는 건 점 네 개에서 뽑은 것이다. 규칙 ⑧대로 다른 설계에서
      재현돼야 믿을 수 있는데, 데이터가 1.5년뿐이라 재현할 설계가 없다.
    · 실제 제출 학습량에서의 이득은 외삽이다.

  ⚠️ 그리고 원래 게이트는 여전히 탈락이다 (+0.0018 < +0.003 · 3/4 < 4/4).
     위 해석이 그걸 뒤집지 않는다. 이건 '기각'과 '한 번 제출해 볼 값어치' 사이의
     회색지대이고, 이 프로젝트의 규칙은 그럴 때 **제출로 판정하라**고 말한다
     ("내부 검증은 명백한 악화 거르기에만 쓰고 실제 우열은 제출로 판정한다").""")
    print("=" * 94)


if __name__ == "__main__":
    main()
