# -*- coding: utf-8 -*-
"""경쟁 분석 — 구성비 인공 상관을 제거한 엄밀한 재검증.

1차 분석에서 담하~미라시아 잔차 상관이 -0.857로 나왔으나, 통제변수로
'세 영업장의 합계'를 썼기 때문에 **구성비 제약**이 걸린다. 라그로타 점유율이
5~10%로 작아서 담하+미라시아 ≈ 합계가 되고, 합계를 통제하면 둘은
**자동으로 -1에 가까운 음의 상관**을 갖게 된다. 즉 그 수치는 증거가 아니다.

바로잡는 법: 두 영업장을 **제외한** 나머지 영업장의 합계를 공통 수요 대리변수로 쓴다.
이러면 구성비 제약이 사라진다. 추가로 **플라시보 쌍**(경쟁할 이유가 없는 조합)을
같이 계산해 기준선을 만든다.
"""
import itertools

import numpy as np
import pandas as pd

import dataio as D

pd.set_option("display.width", 250)

tr = D.load_train()
day = tr.groupby(["store", "date"])["qty"].sum().unstack(0).fillna(0)
day.index = pd.to_datetime(day.index)
STORES = list(day.columns)
SEASONAL = ["화담숲주막", "화담숲카페"]          # 겨울 완전 휴점 → 제외
CORE = [s for s in STORES if s not in SEASONAL]


def pair_resid_corr(a, b, df):
    """a,b를 '제외한' 나머지 합계 + 요일 + 월을 통제한 뒤의 잔차 상관."""
    others = [c for c in CORE if c not in (a, b)]
    d = df[(df[a] > 0) & (df[b] > 0) & (df[others].sum(axis=1) > 0)]
    if len(d) < 100:
        return np.nan, len(d)
    idx = d.index
    dow = pd.get_dummies(idx.dayofweek, prefix="d").values.astype(float)
    mon = pd.get_dummies(idx.month, prefix="m").values.astype(float)
    ctrl = np.log1p(d[others].sum(axis=1).values).reshape(-1, 1)
    X = np.hstack([np.ones((len(d), 1)), dow[:, 1:], mon[:, 1:], ctrl])
    r = {}
    for c in (a, b):
        y = np.log1p(d[c].values)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r[c] = y - X @ beta
    return float(np.corrcoef(r[a], r[b])[0, 1]), len(d)


print("=" * 96)
print("① 모든 영업장 쌍의 잔차 상관 — 통제변수에서 두 당사자를 제외 (인공 상관 없음)")
print("=" * 96)
print("   음수 = 한쪽이 잘될 때 다른 쪽이 안 됨 (경쟁 신호)")
print("   양수 = 같이 오르내림 (공통 손님/보완재)")
print()
rows = []
for a, b in itertools.combinations(CORE, 2):
    c, n = pair_resid_corr(a, b, day)
    rows.append((a, b, c, n))
rows.sort(key=lambda x: (x[2] if not np.isnan(x[2]) else 0))
print(f"  {'영업장 A':<14s} {'영업장 B':<14s} {'상관':>7s} {'일수':>6s}   비고")
for a, b, c, n in rows:
    note = ""
    if {a, b} == {"담하", "미라시아"}:
        note = "← 같은 건물, 1차에서 -0.857 나왔던 쌍"
    elif {a, b} == {"카페테리아", "포레스트릿"}:
        note = "← 둘 다 슬로프"
    elif {a, b} <= {"담하", "라그로타", "미라시아"}:
        note = "← 같은 건물(빌리지센터)"
    print(f"  {a:<14s} {b:<14s} {c:>7.3f} {n:>6d}   {note}")

vals = [r[2] for r in rows if not np.isnan(r[2])]
print(f"\n  전체 쌍 평균 {np.mean(vals):+.3f} · 최소 {min(vals):+.3f} · 최대 {max(vals):+.3f}")
print("  → 대부분이 0 근처면 '경쟁의 증거 없음'. 특정 쌍만 뚜렷이 음수여야 경쟁이다.")

print()
print("=" * 96)
print("② 담하 ↔ 미라시아 집중 검증")
print("=" * 96)
c_ex, n_ex = pair_resid_corr("담하", "미라시아", day)
print(f"  통제=나머지 영업장 합계 (올바른 방법) : {c_ex:+.3f}  (일수 {n_ex})")

# 잘못된 방법 재현 — 비교용
ALWAYS = ["담하", "라그로타", "미라시아"]
s = day[ALWAYS][(day[ALWAYS] > 0).all(axis=1)]
idx = s.index
dow = pd.get_dummies(idx.dayofweek, prefix="d").values.astype(float)
mon = pd.get_dummies(idx.month, prefix="m").values.astype(float)
tot = np.log1p(s.sum(axis=1).values).reshape(-1, 1)
X = np.hstack([np.ones((len(s), 1)), dow[:, 1:], mon[:, 1:], tot])
rr = {}
for c in ("담하", "미라시아"):
    y = np.log1p(s[c].values)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    rr[c] = y - X @ beta
print(f"  통제=세 영업장 합계 (구성비 함정)     : {np.corrcoef(rr['담하'], rr['미라시아'])[0,1]:+.3f}")
print(f"  라그로타 평균 점유율 {s['라그로타'].sum()/s.sum().sum():.3f} "
      f"→ 담하+미라시아가 합계의 대부분이라 자동으로 -1에 가까워진다")

print()
print("=" * 96)
print("③ 담하의 점유율 하락은 진짜인가 — 절대 수준으로 확인")
print("=" * 96)
ym = day.index.strftime("%Y-%m")
mm = day.groupby(ym).mean().round(0)
print("  월별 일평균 (상시 3사):")
print(mm[ALWAYS].T.to_string())
print("\n  2023 vs 2024 같은 달 비교:")
print(f"  {'':<10s} {'1월':>14s} {'2월':>14s} {'4월':>14s} {'5월':>14s}")
for st in ALWAYS:
    cells = []
    for mth in ["01", "02", "04", "05"]:
        a = mm.loc[f"2023-{mth}", st]
        b = mm.loc[f"2024-{mth}", st]
        cells.append(f"{a:.0f}→{b:.0f} ({100*(b/a-1):+.0f}%)")
    print(f"  {st:<10s} " + " ".join(f"{c:>14s}" for c in cells))
