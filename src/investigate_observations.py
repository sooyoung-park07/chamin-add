# -*- coding: utf-8 -*-
"""엑셀을 보며 나온 관찰/가설들을 데이터로 검증."""
import numpy as np
import pandas as pd

import config as C
import dataio as D

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)

tr = D.load_train()
tr["ym"] = tr["date"].dt.strftime("%Y-%m")
tr["m"] = tr["date"].dt.month
tr["y"] = tr["date"].dt.year
tr["dow"] = tr["date"].dt.dayofweek
DOW = ["월", "화", "수", "목", "금", "토", "일"]
MONTHS = sorted(tr["ym"].unique())


def mrow(sub, label):
    p = sub.groupby("ym")["qty"].sum().reindex(MONTHS, fill_value=0)
    return pd.Series(p.values, index=MONTHS, name=label)


def show(df, title):
    print(f"\n[{title}]")
    print(df.to_string())


print("=" * 110)
print("Q1. 느티나무 — 대여료가 단체메뉴 패턴을 안 따라가나? 2024년 급감?")
print("=" * 110)
nt = tr[tr["store"] == "느티나무 셀프BBQ"]
grp = {
    "대여료류": nt["menu"].str.contains("대여료|의자 추가"),
    "단체주류음료": nt["menu"].str.contains(r"\(단체\)|병\(단체\)|BBQ55"),
    "그외(소모품·식품)": ~(nt["menu"].str.contains("대여료|의자 추가") |
                    nt["menu"].str.contains(r"\(단체\)|병\(단체\)|BBQ55")),
}
tbl = pd.DataFrame({k: mrow(nt[v], k) for k, v in grp.items()}).T
show(tbl, "느티나무 그룹별 월별 판매량")
print("\n  그룹 간 월별 상관계수:")
print(tbl.T.corr().round(3).to_string())
print("\n  2023 vs 2024 같은 달 비교 (4·5·6월):")
for g in tbl.index:
    a = tbl.loc[g, ["2023-04", "2023-05", "2023-06"]].sum()
    b = tbl.loc[g, ["2024-04", "2024-05", "2024-06"]].sum()
    print(f"    {g:<16s} 2023: {a:>6.0f}  2024: {b:>6.0f}   ({100*b/max(a,1):.0f}%)")
print("  ※ 2024-06은 6/15까지만이라 불리함에 유의")

print()
print("=" * 110)
print("Q2. 담하 — 2023-01 폭발? 2023-10 강세? 한우불고기 vs 정식 = 메뉴 교체?")
print("=" * 110)
dh = tr[tr["store"] == "담하"]
show(pd.DataFrame([mrow(dh, "담하 전체")]), "담하 월별 총합")
bul = dh[dh["menu"] == "담하 한우 불고기"]
bulj = dh[dh["menu"] == "담하 한우 불고기 정식"]
comb = pd.DataFrame([mrow(bul, "한우불고기"), mrow(bulj, "한우불고기 정식")])
comb.loc["합계"] = comb.sum()
show(comb, "한우 불고기 / 정식 / 합계")
print(f"\n  정식 첫 판매일: {bulj[bulj['qty']>0]['date'].min().date()}")
print("  → 정식 출시 전후로 '합계'가 안정적이면 수요 분산(카니발리제이션) 가설이 맞다")
print("\n  2023-01 무슨 일? 담하 상위 품목의 2023-01 판매량:")
jan = dh[dh["ym"] == "2023-01"].groupby("menu")["qty"].sum().nlargest(6)
avg = dh[dh["ym"] != "2023-01"].groupby(["menu", "ym"])["qty"].sum().groupby("menu").mean()
for m, v in jan.items():
    print(f"    {m:<30s} 2023-01 {v:>6.0f}  (다른달 평균 {avg.get(m,0):>6.0f})")

print()
print("=" * 110)
print("Q3. 콜라 개수 = 손님 수 프록시인가? 콜라와 사이다는 같이 가나?")
print("=" * 110)
day = tr.groupby(["store", "date"])["qty"].sum().rename("store_total").reset_index()
print(f"  {'영업장':<14s} {'콜라류~총합':>11s} {'콜라~사이다':>11s} {'공깃밥~총합':>11s}")
for s in sorted(tr["store"].unique()):
    sub = tr[tr["store"] == s]
    st = day[day["store"] == s].set_index("date")["store_total"]
    cola = sub[sub["menu"].str.contains("콜라|코카콜라")].groupby("date")["qty"].sum()
    spr = sub[sub["menu"].str.contains("스프라이트")].groupby("date")["qty"].sum()
    rice = sub[sub["menu"].str.contains("공깃밥")].groupby("date")["qty"].sum()
    idx = st[st > 0].index

    def cc(a, b):
        if len(a) == 0 or len(b) == 0:
            return np.nan
        j = a.reindex(idx).fillna(0), b.reindex(idx).fillna(0)
        return np.corrcoef(j[0], j[1])[0, 1]
    print(f"  {s:<14s} {cc(cola, st):>11.3f} {cc(cola, spr):>11.3f} {cc(rice, st):>11.3f}")

print()
print("=" * 110)
print("Q4. 라그로타 — 2023 후반 신메뉴가 기존 메뉴 수요를 분산시켰나?")
print("=" * 110)
lg = tr[tr["store"] == "라그로타"]
first = lg[lg["qty"] > 0].groupby("menu")["date"].min().sort_values()
print("  품목별 첫 판매일 (늦은 순 8개):")
for m, d in first.tail(8).items():
    print(f"    {d.date()}  {m}")
new = set(first[first > "2023-07-01"].index)
print(f"\n  2023-07 이후 신설: {len(new)}개")
old_ = lg[~lg["menu"].isin(new)]
new_ = lg[lg["menu"].isin(new)]
show(pd.DataFrame([mrow(old_, "기존메뉴"), mrow(new_, "신설메뉴"),
                   mrow(lg, "전체")]), "라그로타 기존 vs 신설")

print()
print("=" * 110)
print("Q5. 미라시아 — 2024가 2023보다 많나? 9·10·11월 강세의 정체?")
print("=" * 110)
mr = tr[tr["store"] == "미라시아"]
show(pd.DataFrame([mrow(mr, "미라시아 전체")]), "미라시아 월별")
print("\n  9·10·11월에 특히 늘어난 품목 (그 3개월 합 vs 나머지달 평균×3):")
aut = mr[mr["m"].isin([9, 10, 11])].groupby("menu")["qty"].sum()
oth = mr[~mr["m"].isin([9, 10, 11])].groupby("menu")["qty"].sum() / 15 * 3
rat = (aut / oth.replace(0, np.nan)).dropna().sort_values(ascending=False)
for m in rat.head(8).index:
    print(f"    {m:<34s} 가을 {aut[m]:>6.0f} vs 기대 {oth[m]:>6.0f}  ({rat[m]:.2f}배)")

print()
print("=" * 110)
print("Q6. 연회장 — 음식 수요가 '대관 건수'와 정비례하는가?")
print("=" * 110)
yh = tr[tr["store"] == "연회장"]
room_kw = "Conference|Convention Hall|Grand Ballroom|OPUS"
rooms = yh[yh["menu"].str.contains(room_kw)]
food = yh[~yh["menu"].str.contains(room_kw)]
rd = rooms.groupby("date")["qty"].sum()
fd = food.groupby("date")["qty"].sum()
both = pd.concat([rd.rename("대관"), fd.rename("음식")], axis=1).fillna(0)
active = both[(both.sum(axis=1) > 0)]
print(f"  일자 {len(active)}일 기준")
print(f"  대관 ~ 음식 상관계수 : {active['대관'].corr(active['음식']):.3f}")
print(f"  대관 있는 날 {int((active['대관']>0).sum())}일 · 음식 있는 날 {int((active['음식']>0).sum())}일")
print(f"  대관 있는 날의 음식 평균 : {active[active['대관']>0]['음식'].mean():.1f}")
print(f"  대관 없는 날의 음식 평균 : {active[active['대관']==0]['음식'].mean():.1f}")

print()
print("=" * 110)
print("Q7. 카페테리아 — 여름엔 메뉴를 줄여서 운영하나? (월별 '판매된 품목 수')")
print("=" * 110)
for s in ["카페테리아", "포레스트릿"]:
    sub = tr[tr["store"] == s]
    cnt = sub[sub["qty"] > 0].groupby("ym")["menu"].nunique().reindex(MONTHS, fill_value=0)
    tot = sub.groupby("ym")["qty"].sum().reindex(MONTHS, fill_value=0)
    show(pd.DataFrame([cnt.rename("판매품목수"), tot.rename("총판매량")]), s)

print()
print("=" * 110)
print("Q8. 화담숲주막 — 왜 4월·10월이 7·8월보다 강한가?")
print("=" * 110)
hd = tr[tr["store"].isin(["화담숲주막", "화담숲카페"])]
show(pd.DataFrame([mrow(hd[hd["store"] == "화담숲주막"], "화담숲주막"),
                   mrow(hd[hd["store"] == "화담숲카페"], "화담숲카페")]), "화담숲 월별")
opend = hd.groupby(["store", "date"])["qty"].sum().reset_index()
opend["ym"] = opend["date"].dt.strftime("%Y-%m")
od = opend[opend["qty"] > 0].groupby(["store", "ym"]).size().unstack(fill_value=0)
show(od.reindex(columns=MONTHS, fill_value=0), "화담숲 월별 영업일수")
print("\n  → 영업일수로 나눈 '하루 평균'으로 봐야 계절 강도를 제대로 비교할 수 있다")
per = (pd.DataFrame([mrow(hd[hd["store"] == "화담숲주막"], "화담숲주막")]).iloc[0]
       / od.loc["화담숲주막"].replace(0, np.nan))
show(pd.DataFrame([per.round(0).rename("주막 하루평균")]), "휴점일 보정 후")

print()
print("=" * 110)
print("Q9. 요일 — '단체' 붙은 메뉴만 평일에 몰리나?")
print("=" * 110)
tr["is_group"] = tr["menu"].str.contains(r"단체|BBQ55|Open Food|패키지")
op = tr.groupby(["store", "date"])["qty"].sum().reset_index()
op = op[op["qty"] > 0][["store", "date"]]
trs = tr.merge(op, on=["store", "date"])
for label, flag in [("단체성 메뉴", True), ("일반 메뉴", False)]:
    sub = trs[trs["is_group"] == flag]
    prof = sub.groupby("dow")["qty"].mean().reindex(range(7))
    prof = (prof / prof.mean()).round(2)
    print(f"  {label:<10s} " + "  ".join(f"{DOW[i]} {prof[i]:.2f}" for i in range(7)))
print("  (1.00 = 평균. 값이 크면 그 요일에 몰림)")
print("\n  개별 확인 — 사용자가 지목한 메뉴들:")
for key in ["느티나무 셀프BBQ_BBQ55(단체)", "라그로타_Open Food",
            "라그로타_미션 서드 카베르네 쉬라", "미라시아_(단체)브런치주중 36,000",
            "담하_(단체) 황태해장국 3/27까지"]:
    sub = trs[trs["key"] == key]
    if len(sub) == 0:
        continue
    prof = sub.groupby("dow")["qty"].mean().reindex(range(7)).fillna(0)
    prof = (prof / max(prof.mean(), 1e-9)).round(2)
    print(f"    {key:<38s} " + " ".join(f"{DOW[i]}{prof[i]:.1f}" for i in range(7)))
