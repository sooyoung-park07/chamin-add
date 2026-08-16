# -*- coding: utf-8 -*-
"""Tier B — TB8: "화담숲 단풍" 검색량, td(예측 대상일) 기준 lag 7~18일 구간 평균.

EDA(eda_naver2_placebo.py) 근거: hwadam 업장군 잔차와의 상관이 td 기준 lag 1~35일
**전 구간에서 한 번도 음수·0 근처로 안 떨어지는 넓은 고원**(level 변형, r=+0.28~+0.46,
피크는 lag 8~12). TB4d 방식 정식 플라시보(무작위 순열×5변형×lag1~35 스캔)에서 관측치가
백분위 100% — 우연 범위 밖.

lag 하한을 7로 잡은 이유: process_naver_slope.py(TB4d)와 동일한 이유 — HORIZON=7이라
h=1~7 중 어떤 예측이든 td-7 ≤ origin이 항상 성립해야 실전 배포 가능(미래 정보 누출
방지). 상한 18은 위 고원이 아직 튼튼한 지점(r≈+0.38)까지만 잡아 lag=9 단일 최댓값에
과적합하지 않는다(TB4c의 다중비교 교훈). 조회는 origin이 아니라 **td 기준**으로 한다
(process_naver_slope.py와 동일 관례 — shift(7)이 이미 h=7까지 안전 마진을 갖고 있어
td로 바로 조회해도 미래 누출이 없다).

⚠️ 2026-08-15 수정: SRC를 naver_trend2.csv(12개월 청크+키워드5개 공유 척도)에서
naver_trend3_solo.csv(전체기간 단일 호출·키워드 단독)로 교체 — 이유는
process_naver_ski.py 상단 주석 참고. 연도를 가로지르는 폴드(FAR-봄·FAR-겨울)에서
척도 불일치가 진짜 신호와 섞여 보일 위험을 없애기 위함.

입력: data/tierb/naver_trend3_solo.csv (date, hwadam_foliage, ski_field)
출력: data/tierb/naver_hwadam_foliage_lag.csv (date, hwadam_foliage_lag7_18)
"""
import os

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "data", "tierb", "naver_trend3_solo.csv")
OUT = os.path.join(BASE, "data", "tierb", "naver_hwadam_foliage_lag.csv")
LAG_LO, LAG_HI = 7, 18  # td 기준 lag. LAG_LO=HORIZON(7)이 안전 최소치.


def main():
    df = pd.read_csv(SRC, parse_dates=["date"]).sort_values("date")
    s = df.set_index("date")["hwadam_foliage"].asfreq("D").fillna(0.0)
    # date=td 기준. shift(LAG_LO) 후 (LAG_HI-LAG_LO+1)일 rolling
    # = [td-LAG_HI, td-LAG_LO] 구간 평균 — td-7 <= origin 이므로 h=1~7 전부 안전.
    lag_avg = s.shift(LAG_LO).rolling(LAG_HI - LAG_LO + 1, min_periods=1).mean().fillna(0.0)
    out = lag_avg.reset_index()
    out.columns = ["date", "hwadam_foliage_lag7_18"]
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"저장: {OUT} · {len(out)}행")
    print(out.tail())


if __name__ == "__main__":
    main()
