# -*- coding: utf-8 -*-
"""Tier B — TB7: "곤지암 스키장" 검색량, origin(창이 끝나는 시점) 기준 최근 3일 평균.

⚠️ 2026-08-15 재설계(창 7일→3일): `eda_naver_ski_placebo.py`로 뒤늦게 lag1~35
전체를 스캔해보니, ski_field는 화담숲 단풍(고원형)과 달리 **lag=1일 r=+0.5157에서
시작해 lag이 늘수록 매끄럽게 단조 감소**하는 곡선이었다(lag=7: +0.417 · lag=13:
+0.36 부근). origin 기준 **7일** 평균은 td 기준으로 보면 h=1은 lag 1~7(강한 구간)을
쓰지만 h=7은 lag 7~13(이미 꺾인 구간)까지 끌어다 써서 신호를 불필요하게 희석시켰다.
origin은 어떤 h에도 정확히 td-h(=배포 가능한 최소 lag)에 해당하므로, **origin 자체의
값에 최대한 가깝게** 창을 좁혀야 한다 — 최근 3일 평균(origin-2~origin)으로 줄이면
h=1은 lag 1~3(0.52~0.48), h=7도 lag 7~9(0.42~0.39)로 감쇠곡선의 훨씬 강한 구간에
머문다. 단일 origin 값(창 없음, 노이즈 그대로)까지 좁히지 않은 이유는 ski_field의
일별 변동성(std≈28, 0~100 척도)이 커서 최소한의 평활은 필요하다고 판단했기 때문.

이전 버전(창 7일, `naver_ski_last7.csv`)은 TB2c의 설계 패턴만 빌려 정했을 뿐 ski_field
자체의 lag 곡선을 스캔하지 않고 고른 것이었다 — 사용자 지적으로 뒤늦게 근거를 확인.

⚠️ 2026-08-15 (별도) 척도 버그 수정: 원래 SRC는 naver_trend2.csv(12개월씩 청크 호출 +
키워드 5개를 한 호출에 같이 묶음)였는데, 네이버 데이터랩은 호출 1건 안에서만 0~100
정규화를 하고 그 안의 여러 키워드끼리도 척도를 공유한다 — 그 결과 연도마다 "100"이
가리키는 절대 검색량이 달라졌다(실측: ski_field 연도별 최댓값이 100·91.8·100으로
들쭉날쭉). naver_trend3_solo.csv는 같은 키워드를 전체 기간 한 번의 호출로, 키워드
단독으로 다시 받아 기간 전체에서 일관된 척도로 만든 것.

입력: data/tierb/naver_trend3_solo.csv (date, ski_field, hwadam_foliage)
출력: data/tierb/naver_ski_last3.csv (date, ski_last3)
"""
import os

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "data", "tierb", "naver_trend3_solo.csv")
OUT = os.path.join(BASE, "data", "tierb", "naver_ski_last3.csv")
WINDOW = 3


def main():
    df = pd.read_csv(SRC, parse_dates=["date"]).sort_values("date")
    s = df.set_index("date")["ski_field"].asfreq("D").fillna(0.0)
    # date=origin 기준, [origin-2, origin] 3일 평균 (origin 당일 포함 — 이미 알고 있는 값)
    last3 = s.rolling(WINDOW, min_periods=1).mean()
    out = last3.reset_index()
    out.columns = ["date", "ski_last3"]
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"저장: {OUT} · {len(out)}행")
    print(out.tail())


if __name__ == "__main__":
    main()
