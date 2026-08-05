# -*- coding: utf-8 -*-
"""영업장 간 경쟁(카니발리제이션)이 존재하는가?

어려운 점: 계절/요일 때문에 모든 영업장이 같이 오르내린다. 그 '공통 성분'을
걷어낸 **잔차**에서 서로 반대로 움직이는지를 봐야 경쟁이 보인다.

주의해야 할 함정 — **구성비 인공 상관(compositional artifact)**:
전체 수요를 통제하면 각 영업장은 '점유율'만 남는데, 점유율의 합은 1이라
아무 관계가 없어도 평균적으로 음의 상관이 생긴다(9개면 약 -1/8 = -0.125).
따라서 "음수면 경쟁"이 아니라 **"-0.125보다 확실히 더 음수여야 경쟁"**이다.

그래서 더 믿을 만한 증거로 **자연 실험**을 함께 본다:
  · 라그로타 2023-12 신메뉴 5종 동시 출시
  · 담하 한우불고기'정식' 2023-06-02 출시
이런 일회성 사건 전후로 다른 영업장이 반대로 움직였는지.
"""
import numpy as np
import pandas as pd

import config as C
import dataio as D

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)

tr = D.load_train()
day = tr.groupby(["store", "date"])["qty"].sum().unstack(0).fillna(0)
day.index = pd.to_datetime(day.index)
ALWAYS = ["담하", "라그로타", "미라시아"]          # 빌리지센터 상시 3사 = 직접 경쟁 후보
SKI = ["카페테리아", "포레스트릿"]                  # 슬로프 2사

print("=" * 100)
print("① 상시 3사(담하·라그로타·미라시아)의 월별 '점유율'")
print("   — 같은 건물(빌리지센터)에 있어 직접 경쟁 관계일 가능성이 가장 높다")
print("=" * 100)
sub = day[ALWAYS]
open_all = (sub > 0).all(axis=1)                  # 셋 다 영업한 날만
s = sub[open_all]
share = s.div(s.sum(axis=1), axis=0)
m = share.groupby(share.index.strftime("%Y-%m")).mean().round(3)
print(m.T.to_string())
print("\n  전반기(2023-01~09) vs 후반기(2023-12~2024-06) 평균 점유율:")
h1 = share[share.index < "2023-10"].mean()
h2 = share[share.index >= "2023-12"].mean()
for st in ALWAYS:
    print(f"    {st:<8s} {h1[st]:.3f} → {h2[st]:.3f}   ({h2[st]-h1[st]:+.3f})")

print()
print("=" * 100)
print("② 공통 성분(요일·월·전체수요)을 걷어낸 잔차 상관")
print("=" * 100)


def residuals(df, cols):
    """log 판매량에서 요일·월·전체수요를 회귀로 제거한 잔차."""
    idx = df.index
    dow = pd.get_dummies(idx.dayofweek, prefix="d").values.astype(float)
    mon = pd.get_dummies(idx.month, prefix="m").values.astype(float)
    tot = np.log1p(df[cols].sum(axis=1).values).reshape(-1, 1)
    X = np.hstack([np.ones((len(idx), 1)), dow[:, 1:], mon[:, 1:], tot])
    out = {}
    for c in cols:
        y = np.log1p(df[c].values)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        out[c] = y - X @ beta
    return pd.DataFrame(out, index=idx)


res = residuals(s, ALWAYS)
corr = res.corr().round(3)
print("  [상시 3사 잔차 상관]  (구성비 인공 상관 기준선 ≈ -0.50, 3개일 때)")
print(corr.to_string())
n = len(ALWAYS)
print(f"\n  ※ {n}개 영업장이면 인공 상관 기준선 = -1/{n-1} = {-1/(n-1):.3f}")
print("     이보다 '더 음수'여야 실제 경쟁이라 볼 수 있다.")

skid = day[SKI]
so = skid[(skid > 0).all(axis=1)]
res2 = residuals(so, SKI)
print(f"\n  [슬로프 2사 잔차 상관] 카페테리아 ~ 포레스트릿 = "
      f"{res2.corr().iloc[0,1]:.3f}  (인공 기준선 -1.000, 2개일 때는 판별 불가)")

print()
print("=" * 100)
print("③ 자연 실험 A — 라그로타 2023-12-08 신메뉴 5종 동시 출시")
print("   경쟁이 있다면: 라그로타가 오르고 담하·미라시아가 내려가야 한다")
print("=" * 100)
EV = pd.Timestamp("2023-12-08")
win = 56                                            # 전후 8주
pre = day[(day.index >= EV - pd.Timedelta(days=win)) & (day.index < EV)]
post = day[(day.index >= EV) & (day.index < EV + pd.Timedelta(days=win))]
print(f"  전 {pre.index.min().date()}~{pre.index.max().date()} / "
      f"후 {post.index.min().date()}~{post.index.max().date()}")
print(f"\n  {'영업장':<14s} {'전 일평균':>9s} {'후 일평균':>9s} {'변화율':>8s}")
rows = {}
for st in day.columns:
    a, b = pre[st].mean(), post[st].mean()
    rows[st] = (a, b, (b / a - 1) if a > 0 else np.nan)
    print(f"  {st:<14s} {a:>9.1f} {b:>9.1f} {100*rows[st][2]:>7.1f}%"
          if a > 0 else f"  {st:<14s} {a:>9.1f} {b:>9.1f}      휴점")
print("\n  → 겨울로 넘어가며 스키 영업장이 통째로 오르므로, **상시 3사 안에서의 점유율**로 봐야 한다")
p1 = pre[ALWAYS].sum() / pre[ALWAYS].sum().sum()
p2 = post[ALWAYS].sum() / post[ALWAYS].sum().sum()
print(f"\n  {'':<10s} {'전 점유율':>9s} {'후 점유율':>9s} {'변화':>8s}")
for st in ALWAYS:
    print(f"  {st:<10s} {p1[st]:>9.3f} {p2[st]:>9.3f} {p2[st]-p1[st]:>+8.3f}")

print()
print("=" * 100)
print("④ 자연 실험 B — 담하 '한우불고기 정식' 2023-06-02 출시")
print("=" * 100)
EV2 = pd.Timestamp("2023-06-02")
pre2 = day[(day.index >= EV2 - pd.Timedelta(days=56)) & (day.index < EV2)]
post2 = day[(day.index >= EV2) & (day.index < EV2 + pd.Timedelta(days=56))]
q1 = pre2[ALWAYS].sum() / pre2[ALWAYS].sum().sum()
q2 = post2[ALWAYS].sum() / post2[ALWAYS].sum().sum()
print(f"  {'':<10s} {'전 점유율':>9s} {'후 점유율':>9s} {'변화':>8s}")
for st in ALWAYS:
    print(f"  {st:<10s} {q1[st]:>9.3f} {q2[st]:>9.3f} {q2[st]-q1[st]:>+8.3f}")

print()
print("=" * 100)
print("⑤ 그럼 리조트 '전체 수요'는 얼마나 안정적인가?")
print("   (경쟁이 제로섬이려면 전체는 일정해야 한다)")
print("=" * 100)
tot = day.sum(axis=1)
mm = tot.groupby(tot.index.strftime("%Y-%m")).mean().round(0)
print("  월별 일평균 전체 판매량:")
print(mm.to_string())
al = day[ALWAYS].sum(axis=1)
mm2 = al.groupby(al.index.strftime("%Y-%m")).mean().round(0)
print("\n  상시 3사만:")
print(mm2.to_string())
print(f"\n  전체 변동계수(CV) {tot.std()/tot.mean():.2f} · "
      f"상시3사 CV {al.std()/al.mean():.2f}")
print("  → CV가 크면 '파이 자체가 계속 변한다'는 뜻이라 제로섬 경쟁 가정이 약해진다")
