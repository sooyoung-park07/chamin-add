# -*- coding: utf-8 -*-
"""Phase 12 — Q6 · Q7 진단. **학습 0회 · 제출 0회.** (브리핑 3편 5-3절)

Q6  봄에 남은 여지 0.03~0.05 가 **어디에(어느 주에)** 몰려 있는가?
    방법: 주별로 [우리 기여] 와 [품목×주 사후 최적 정수상수의 기여] 를 비교.
    차이 = "주간 수준을 완벽히 알면 회수되는 여지"의 상한. 몰려 있으면 국소 처방,
    퍼져 있으면 구조적 한계. 겨울 폴드를 대조군으로 같이 잰다.
    재현성: F3(근거리 학습)과 FAR-봄(원거리 학습)은 검증 타깃이 같고 학습만 다르다
    → 두 설계에서 주별 패턴이 같아야 믿는다 (규칙 8).

Q7  창이 급변할 때 한쪽으로 체계적으로 틀리는가?
    방법: 창 추세 r = (last7+1)/(prev7+1) 로 칸을 5구간으로 나누고
    구간별 [부호 편향 ln(P/A)] 와 [사후 최적 배율] 을 잰다.
    판정: 배율이 추세에 단조 + 독립 2구간(겨울/봄) 모두 같은 방향일 때만 신호.
    결정타: 겨울에서 맞춘 배율을 봄에 적용(전이 시험) — 이득이 나야 실전 가치가 있다.

사전 기대 (실행 전 기록):
    Q6: 방향 미정. 몰린다면 3/16~4/7 (스키 폐장 + 화담숲 재개장) 예상.
    Q7: 트리가 평균으로 당기므로 급상승 창에서 배율 > 1 (과소예측) 예상.
"""
import os
import numpy as np
import pandas as pd

import config as C
import dataio as D
import features as F
import validate as V
from run_phase10c_thresholds import cell_weights

FOLDS = [
    ("F2 겨울",  "2023-11-24", "2023-11-24", "2024-02-22"),
    ("F3 봄",    "2024-02-23", "2024-02-23", "2024-06-08"),
    ("FAR-봄",   "2023-11-24", "2024-02-23", "2024-06-08"),
    ("FAR-겨울", "2023-08-25", "2023-11-24", "2024-02-22"),
]

EVENTS = {  # 주 시작일 기준 표시용
    "2024-03-11": "스키 폐장 무렵",
    "2024-03-25": "화담숲 재개장(3/29)",
    "2023-11-27": "화담숲 폐장(11/30)",
}


def seg_base(raw):
    p = np.where(raw < 1.8, 0.55 * raw, np.where(raw < 10.0, 0.90 * raw, 1.02 * raw))
    return np.maximum(p, 1.0)


def gsnap(p):
    k = np.maximum(np.floor(p), 1.0)
    return np.maximum(np.where(p >= np.sqrt(k * (k + 1.0)), k + 1.0, k), 1.0)


def loss(a, p):
    a = np.abs(a); p = np.abs(p)
    den = a + p
    out = np.zeros(len(a))
    m = den > 0
    out[m] = 2.0 * np.abs(a[m] - p[m]) / den[m]
    return out


def main():
    ctx = F.Context()
    tr = D.load_train()
    mat, dates = D.to_matrix(tr, ctx.items)
    dser = pd.Series(pd.DatetimeIndex(dates))
    nd = mat.shape[1]
    ctx.set_proxy(F.pick_proxy_items(mat, dates, nd, ctx.store_codes))
    d20 = np.load(os.path.join(C.EXPERIMENTS, "phase10e_oof20.npz"))

    fo = []
    for fi, (name, cut, v0, v1) in enumerate(FOLDS):
        va = V.origins(dates, v0, v1, nd)
        _, yva, mva = F.build_samples(mat, dates, va, ctx)
        assert np.array_equal(yva, d20[f"y{fi}"]), f"{name}: y 순서 불일치"
        raw = d20[f"ps{fi}"].mean(0).astype(np.float64)
        W, valid = cell_weights(yva, mva[:, 2], ctx.store_of_item, ctx.n)
        tdi = mva[:, 0] + mva[:, 1]                      # 대상일 인덱스
        tdate = pd.DatetimeIndex(dser.values[tdi])
        base = seg_base(raw)
        ours = gsnap(base)
        fo.append(dict(name=name, y=yva, W=W, valid=valid, iid=mva[:, 2],
                       origin=mva[:, 0], tdate=tdate, base=base, ours=ours))
        sc = float((W * loss(yva, ours)).sum())
        print(f"  [{name:<8s}] 칸 {len(yva):>6,} · v17 점수 {sc:.5f}")

    # ================================================================ Q6
    print("\n" + "=" * 100)
    print("Q6 — 여지가 어느 주에 몰려 있는가  (여지 = 우리 기여 − 품목×주 사후 최적 정수상수 기여)")
    print("=" * 100)

    def weekly(f):
        wk = (f["tdate"] - pd.to_timedelta(f["tdate"].weekday, unit="D")).normalize()
        ours_c = f["W"] * loss(f["y"], f["ours"])
        rows = {}
        for (it, w), idx in pd.Series(range(len(wk))).groupby(
                [pd.Series(f["iid"]), pd.Series(wk)]).groups.items():
            idx = np.asarray(idx)
            a = np.abs(f["y"][idx]); Wg = f["W"][idx]
            v = f["valid"][idx]
            if not v.any():
                continue
            a_v, W_v = a[v], Wg[v]
            cmax = int(a_v.max()) + 1
            cand = np.arange(1, cmax + 1)
            pen = np.array([(W_v * (2 * np.abs(a_v - c) / (a_v + c))).sum() for c in cand])
            best = float(pen.min())
            o = float(ours_c[idx].sum())
            r = rows.setdefault(w, dict(ours=0.0, orac=0.0, by_store={}))
            r["ours"] += o
            r["orac"] += best
            s = ctx.store_of_item[it]
            r["by_store"][s] = r["by_store"].get(s, 0.0) + (o - best)
        return rows

    weekly_res = {}
    for f in fo:
        weekly_res[f["name"]] = weekly(f)

    for pair, ctrl in (("F3 봄", "FAR-봄"), ("F2 겨울", "FAR-겨울")):
        wA, wB = weekly_res[pair], weekly_res[ctrl]
        weeks = sorted(wA.keys())
        gapsA = np.array([wA[w]["ours"] - wA[w]["orac"] for w in weeks])
        gapsB = np.array([wB[w]["ours"] - wB[w]["orac"] for w in weeks if w in wB])
        tot = gapsA.sum()
        print(f"\n### {pair}  (재현 대조 = {ctrl})   여지 합계 {tot:+.5f}")
        print(f"  {'주 시작':<12s}{'우리 기여':>10s}{'주간상수':>10s}{'여지':>10s}{'비중':>7s}"
              f"{'재현폴드':>10s}  {'이벤트 / 여지 상위 영업장'}")
        for i, w in enumerate(weeks):
            g = gapsA[i]
            gB = wB[w]["ours"] - wB[w]["orac"] if w in wB else float("nan")
            ev = EVENTS.get(str(w.date()), "")
            top = sorted(wA[w]["by_store"].items(), key=lambda x: -x[1])[:2]
            tops = " · ".join(f"{s}{v:+.4f}" for s, v in top if abs(v) > 5e-4)
            mark = " ★" if tot > 0 and g > 1.5 * tot / len(weeks) else ""
            print(f"  {str(w.date()):<12s}{wA[w]['ours']:>10.5f}{wA[w]['orac']:>10.5f}"
                  f"{g:>+10.5f}{100 * g / tot if tot else 0:>6.1f}%{gB:>+10.5f}  {ev} {tops}{mark}")
        srt = np.sort(gapsA)[::-1]
        print(f"  → 상위 3주 비중 {100 * srt[:3].sum() / tot:.0f}%"
              f" (균등이면 {100 * 3 / len(weeks):.0f}%)"
              f" · 두 설계의 주별 여지 상관 "
              f"{np.corrcoef(gapsA[:len(gapsB)], gapsB)[0, 1]:.3f}")

    # ================================================================ Q7
    print("\n" + "=" * 100)
    print("Q7 — 창 추세 축.  r = (last7+1)/(prev7+1),  배율은 사후 최적(스냅 유지)")
    print("=" * 100)
    BINS = [0.0, 0.5, 0.8, 1.25, 2.0, np.inf]
    LAB = ["급하강<0.5", "하강0.5-0.8", "평탄0.8-1.25", "상승1.25-2", "급상승>2"]
    MS = np.round(np.arange(0.60, 1.61, 0.01), 2)

    def trend_of(f):
        r = np.empty(len(f["y"]))
        for o in np.unique(f["origin"]):
            m = f["origin"] == o
            it = f["iid"][m]
            l7 = mat[:, o - 6:o + 1].mean(1)
            p7 = mat[:, o - 13:o - 6].mean(1)
            r[m] = (l7[it] + 1.0) / (p7[it] + 1.0)
        return r

    def bin_table(f, r):
        out = []
        bi = np.digitize(r, BINS) - 1
        for b in range(5):
            m = (bi == b) & f["valid"]
            if m.sum() < 30:
                out.append(None)
                continue
            a = np.abs(f["y"][m]); W = f["W"][m]; base = f["base"][m]
            bias = float(np.average(np.log(np.maximum(f["ours"][m], 1.0) / np.maximum(a, 1.0)),
                                    weights=W))
            cur = float((W * (2 * np.abs(a - gsnap(base)) / (a + gsnap(base)))).sum())
            pen = [float((W * (2 * np.abs(a - gsnap(mm * base)) / (a + gsnap(mm * base)))).sum())
                   for mm in MS]
            j = int(np.argmin(pen))
            out.append(dict(n=int(m.sum()), wsh=float(W.sum()), bias=bias,
                            mult=float(MS[j]), gain=cur - pen[j], cur=cur))
        return out

    trends, tables = {}, {}
    for f in fo:
        trends[f["name"]] = trend_of(f)
        tables[f["name"]] = bin_table(f, trends[f["name"]])

    for name in ("F2 겨울", "F3 봄"):
        t = tables[name]
        print(f"\n### {name}")
        print(f"  {'추세 구간':<14s}{'칸수':>8s}{'가중비중':>9s}{'부호편향':>10s}{'최적배율':>9s}{'개선':>10s}")
        for lab, row in zip(LAB, t):
            if row is None:
                print(f"  {lab:<14s}{'(표본<30)':>8s}")
                continue
            print(f"  {lab:<14s}{row['n']:>8,}{row['wsh']:>8.1%}{row['bias']:>+10.4f}"
                  f"{row['mult']:>9.2f}{row['gain']:>+10.5f}")

    # 전이 시험 — 겨울에서 맞춘 배율을 봄에, 그 반대도
    print("\n### 전이 시험 (규칙 3 — 한 구간에서 맞추고 다른 구간에서 채점)")
    for src, dst in (("F2 겨울", "F3 봄"), ("F3 봄", "F2 겨울")):
        ms = [row["mult"] if row else 1.0 for row in tables[src]]
        f = [x for x in fo if x["name"] == dst][0]
        bi = np.digitize(trends[dst], BINS) - 1
        mvec = np.array([ms[b] for b in bi])
        a = np.abs(f["y"]); W = f["W"]
        cur = float((W * loss(f["y"], f["ours"])).sum())
        new = float((W * loss(f["y"], gsnap(mvec * f["base"]))).sum())
        print(f"  {src} 배율 {[f'{m:.2f}' for m in ms]} → {dst} 적용: {cur - new:+.5f}")


if __name__ == "__main__":
    main()
