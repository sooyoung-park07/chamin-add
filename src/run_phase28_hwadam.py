# -*- coding: utf-8 -*-
"""Phase 28 — 개장 램프를 겨냥할 수 있는가. **내부 데이터만으로 세 가지를 확인한다.**

사용자 지적이 옳다: 2024-03-29 발견(봄 이득의 90%)은 `phase21c_oof.npz` 에서 나왔고
**테스트 점수를 하나도 쓰지 않았다.** 오염된 것은 "T07·T08 이 좋았다"는 사실이지
개장 발견이 아니다. 그러니 개장을 겨냥하는 것 자체는 규칙 위반이 아니다.

확인할 것 세 개:
  ① **기전이 진짜인가** — 2024-03-29 창의 이득이 정말 **화담숲 품목**에 몰려 있는가?
     몰려 있으면 `days_to_hwadam_open` 이 원인이라는 이야기가 선다.
     전 업장에 퍼져 있으면 그냥 운 좋은 창 하나일 뿐이다.
  ② **품목 축으로 옮기면 표본이 늘어나는가** — 영업장 9개 × 폴드 4개로 보면
     origin 1개(n=1)가 아니라 4폴드 재현성을 따질 수 있다.
     **화담숲이 4/4 폴드에서 Ridge 우위면 자유도 1짜리 규칙이 선다.**
  ③ **애초에 겨냥할 과녁이 TEST 에 있는가** — 화담숲 개장은 실측 3/29~31 인데
     TEST 10개 창이 예측하는 날짜에 그게 들어 있는가?

⚠️ ③ 은 테스트 **입력**(창 날짜)만 본다. 점수가 아니다. 합법이다.
"""
import os

import numpy as np
import pandas as pd

import config as C
import dataio as D
import features as F
import validate as V
from run_phase10c_thresholds import cell_weights

NPZ = os.path.join(C.EXPERIMENTS, "phase21c_oof.npz")
W = 0.20
FOLDS = [("F2 겨울", "2023-11-24", "2024-02-22"),
         ("F3 봄", "2024-02-23", "2024-06-08"),
         ("FAR-봄", "2024-02-23", "2024-06-08"),
         ("FAR-겨울", "2023-11-24", "2024-02-22")]


def post(raw, seg=False, snap=False):
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


def main():
    z = np.load(NPZ)
    ctx = F.Context()
    tr = D.load_train()
    mat, dates = D.to_matrix(tr, ctx.items)
    nd = mat.shape[1]
    items = ctx.items
    store_of = np.array([str(k).split("_", 1)[0] for k in items])
    stores = sorted(set(store_of))
    blk = C.HORIZON * ctx.n

    # ───────────────────── ① 2024-03-29 창의 이득이 화담숲에 몰려 있는가
    print("=" * 92)
    print("① 봄 이득의 90% 를 낸 창(2024-03-29 · 예측 3/30~4/5)을 영업장별로 뜯는다")
    va = V.origins(dates, "2024-02-23", "2024-06-08", nd)
    tgt = [i for i, o in enumerate(va)
           if pd.Timestamp(dates[o]) == pd.Timestamp("2024-03-29")]
    assert len(tgt) == 1, "2024-03-29 origin 을 못 찾음"
    i0 = tgt[0]
    y, iid = z["y|F3 봄"], z["iid|F3 봄"].astype(int)
    A, B = z["ours|F3 봄"], z["tm_ridge|F3 봄"]
    Wt, _ = cell_weights(y, iid, ctx.store_of_item, ctx.n)
    s = slice(i0 * blk, (i0 + 1) * blk)
    ga = Wt[s] * loss(y[s], post(A[s]))
    gb = Wt[s] * loss(y[s], post((1 - W) * A[s] + W * B[s]))
    st_row = store_of[iid[s]]
    tot = (ga - gb).sum()
    print(f"  창 전체 이득 {tot:+.5f}  (봄 폴드 16창 합계 +0.00330 의 {100*tot/0.00330:.0f}%)")
    print(f"  {'영업장':<16s}{'품목':>5s}{'이득':>11s}{'비중':>8s}")
    parts = []
    for st in stores:
        m = st_row == st
        v = float((ga[m] - gb[m]).sum())
        parts.append((v, st, int((store_of == st).sum())))
    for v, st, n in sorted(parts, reverse=True):
        print(f"  {st:<16s}{n:>5d}{v:>+11.5f}{100*v/tot:>7.0f}%")
    hw = sum(v for v, st, _ in parts if st.startswith("화담숲"))
    print(f"\n  → 화담숲 2개 업장(13품목) 이 {hw:+.5f} = 전체의 **{100*hw/tot:.0f}%**")
    verdict = ("✅ 기전 확인 — days_to_hwadam_open 이 원인이라는 이야기가 선다"
               if hw / tot > 0.5 else
               "❌ 화담숲에 몰려 있지 않다 — 그냥 운 좋은 창 하나였다")
    print(f"     {verdict}")

    # ───────────────────── ② 영업장 축 4폴드 재현성
    print("\n" + "=" * 92)
    print("② 영업장별 혼합 이득 — 4폴드에서 부호가 재현되는가 (raw · w=0.20)")
    print(f"  {'영업장':<16s}" + "".join(f"{f[0]:>11s}" for f in FOLDS)
          + f"{'평균':>11s}{'일관':>7s}")
    tab = {}
    for fn, v0, v1 in FOLDS:
        yy, ii = z[f"y|{fn}"], z[f"iid|{fn}"].astype(int)
        AA, BB = z[f"ours|{fn}"], z[f"tm_ridge|{fn}"]
        WW, _ = cell_weights(yy, ii, ctx.store_of_item, ctx.n)
        la = WW * loss(yy, post(AA))
        lb = WW * loss(yy, post((1 - W) * AA + W * BB))
        sr = store_of[ii]
        for st in stores:
            m = sr == st
            tab[(fn, st)] = float((la[m] - lb[m]).sum())
    hits = []
    for st in stores:
        vs = [tab[(f[0], st)] for f in FOLDS]
        mn = float(np.mean(vs))
        cons = sum(v > 0 for v in vs) if mn > 0 else sum(v < 0 for v in vs)
        mark = "✅" if cons == 4 else ""
        print(f"  {st:<16s}" + "".join(f"{v:>+11.5f}" for v in vs)
              + f"{mn:>+11.5f}{cons:>5d}/4 {mark}")
        if cons == 4 and mn > 0:
            hits.append((st, mn))
    print(f"\n  4/4 양수 영업장: {[h[0] for h in hits] if hits else '없음'}")
    if hits:
        print(f"  합계 기대이득 {sum(h[1] for h in hits):+.5f} / 폴드")

    # ───────────────────── ③ TEST 에 과녁이 있는가 (입력만 사용)
    print("\n" + "=" * 92)
    print("③ 화담숲 개장(실측 3/29~31)이 TEST 예측 날짜에 들어 있는가 — 창 날짜만 본다")
    print(f"  {'파일':<9s}{'예측 구간':>26s}{'개장(3/29~31) 포함':>20s}")
    hit_any = False
    for t in range(C.N_TEST):
        te = D.load_test(t)
        _, td = D.to_matrix(te, items)
        end = pd.Timestamp(td[-1])
        f0, f1 = end + pd.Timedelta(days=1), end + pd.Timedelta(days=7)
        inc = any((d.month == 3 and 29 <= d.day <= 31)
                  for d in pd.date_range(f0, f1))
        hit_any |= inc
        print(f"  TEST_{t:02d}  {str(f0.date())+' ~ '+str(f1.date()):>26s}"
              f"{('🎯 포함' if inc else '·'):>20s}")
    print(f"\n  → {'과녁 있음' if hit_any else '**과녁이 없다. TEST 10개 중 개장 주간을 예측하는 창이 0개다.**'}")

    # 참고: 달력 대응 내부 origin vs 실측
    print("\n" + "=" * 92)
    print("④ 참고 — 달력이 같은 내부 origin 이 TEST 창을 예측했는가")
    print("  (T07·T08 실측 파일당 약 −0.0017 · T09 약 +0.0055 — 이건 제출로 안 값이다)")
    for lab, d in (("T07 ↔ 2024-03-15", "2024-03-15"),
                   ("T08 ↔ 2024-04-19", "2024-04-19"),
                   ("T09 ↔ 2024-05-24", "2024-05-24")):
        j = [i for i, o in enumerate(va) if pd.Timestamp(dates[o]) == pd.Timestamp(d)]
        if not j:
            continue
        sl = slice(j[0] * blk, (j[0] + 1) * blk)
        g = float((Wt[sl] * loss(y[sl], post(A[sl]))).sum()
                  - (Wt[sl] * loss(y[sl], post((1 - W) * A[sl] + W * B[sl]))).sum())
        print(f"  {lab:<20s} 내부 이득 {g:+.5f}")
    print("""
  세 내부값이 +0.0002~+0.0004 로 서로 거의 같은데 실측은 −0.0017 / −0.0017 / +0.0055 로
  퍼졌다. **내부 창 단위 측정은 대응하는 TEST 창을 예측하지 못한다** (규칙 ⑮).""")
    print("=" * 92)


if __name__ == "__main__":
    main()
