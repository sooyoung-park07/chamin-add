# -*- coding: utf-8 -*-
"""피처 생성 — 28일 창 + 달력 + 도메인 + 품목 정적속성.

핵심 규약: **테스트에서 쓸 수 있는 정보만 쓴다.**
  창 통계는 주어진 28일에서만, 달력은 예측 대상일에서 (미래여도 확정된 사실이라 누수 아님).
  작년 동기(YoY)는 학습기간이 1.5년뿐이라 효과 없음이 확인되어 제외.

피처 그룹 (Phase 4 ablation 단위):
  win  : 창 전체 통계 14개
  dow  : 예측 대상 요일과 같은 요일의 창 내 통계 5개
  cal  : 달력·도메인 플래그 13개
  ctx  : 지난주 같은요일 값 + 영업장 집계 + horizon 4개
  item : 품목 정적속성 7개 (범주형 5 + 가격 + 가중영업장)
"""
import os
import re
import warnings

import numpy as np
import pandas as pd

import config as C
import dataio as D

WIN_KEYS = ["w_mean", "w_std", "w_max", "w_median", "w_nzratio", "w_posmean",
            "w_posmedian", "w_last", "w_last7", "w_last14", "w_prev7",
            "w_trend", "w_days_since", "w_last7_nz"]
DOW_KEYS = ["d_mean", "d_max", "d_posmean", "d_nzratio", "d_last"]
# 휴점일(영업장 합계=0) 보정 — 창 안에 휴점이 섞이면 위의 통계가 희석된다.
# 프로파일링 결과: 라그로타 13% · 포레스트릿 28% · 화담숲 46%가 휴점일.
CLOSED_KEYS = ["st_closed_ratio", "st_closed_last7", "w_mean_open",
               "w_nzratio_open", "d_mean_open", "st_days_since_open"]
# 품목별 '요일 프로파일' — 학습기간 전체에서 계산한 정적 피처.
# 근거(관찰): '단체' 성격 메뉴는 목·금에 몰리고(평균의 1.7~1.8배, 일요일 0.17배),
#   일반 메뉴는 토·일에 몰린다(1.5배). 즉 **요일의 의미가 품목마다 정반대**다.
#   창(28일) 안의 같은 요일 표본은 4개뿐이라 노이즈가 커서 이 패턴을 잡기 어렵다.
#   → 학습기간 전체로 안정적인 요일 프로파일을 만들어 정적 피처로 준다.
# ※ 누수 방지: 프로파일은 반드시 **해당 폴드의 학습 구간까지만**으로 계산한다.
PROF_KEYS = ["p_dow_idx", "p_weekend_ratio", "p_dow_peak", "p_dow_spread",
             "p_is_group"]
# 교차 품목(cross-item) 피처 — **모델이 스스로 만들 수 없는 정보**.
# 지금 학습 행 하나에는 '자기 품목'의 통계만 들어 있어서, 같은 영업장 다른 품목이
# 어떻게 팔렸는지는 구조적으로 접근이 불가능하다.
# 근거(관찰): 어떤 품목은 '선택'이 아니라 '인원수'를 따라간다.
#   화담숲주막 콜라~영업장합계 0.960 · 포레스트릿 0.916 · 느티나무 0.905
#   담하는 콜라가 0.362로 약하고 공깃밥이 0.788 (한식당이라 밥이 인원수를 따라감)
# → 영업장마다 '인원수 온도계' 품목을 데이터로 골라, 그 최근 수준을 모든 품목에 알려준다.
# 침투율(share) = 이 품목 / 영업장 합계. 절대 수준보다 안정적이라 곱셈 구조를 흉내낸다.
CROSS_KEYS = ["x_store_last7", "x_store_prev7", "x_store_trend", "x_store_dow",
              "x_proxy_last7", "x_proxy_dow", "x_share_win", "x_share_dow"]
# 리조트 전체(9개 영업장 합계) 피처 — **지금까지 완전히 빠져 있던 축**.
# 확인 결과 기존 57개 피처에는 **다른 영업장 정보가 하나도 없다.** 자기 품목과 자기 영업장뿐이다.
# 즉 "오늘 리조트 전체가 붐빈다"는 사실에 모델이 **구조적으로 접근할 수 없었다.**
# 자기 영업장 합계(x_store_*)가 그걸 간접적으로 담지만, 거기엔
#   "리조트가 붐빔" + "우리 가게만 붐빔" 이 섞여 있어 분리가 안 된다.
# → r_share(자기 영업장 ÷ 리조트 전체)가 그 둘을 갈라주는 핵심 피처다.
# 대회 규칙 안전: 주어진 28일 창 안의 데이터만 쓴다(창 밖 조회 없음).
RESORT_KEYS = ["r_last7", "r_prev7", "r_trend", "r_dow", "r_share", "r_share_l7"]
CTX_KEYS = ["sd_lastweek", "st_mean", "st_nz", "horizon"]
CAL_KEYS = ["dow", "month", "day", "is_weekend", "is_holiday", "is_holiday_eve",
            "is_dayoff", "doy_sin", "doy_cos", "hwadam_open", "ski_season",
            "ski_peak", "foliage"]
# 시즌 경계까지의 **부호 있는 거리** + 연휴 구조 (Phase 29, 팀원 저장소에서 이식).
# 동기: 위 CAL 은 `hwadam_open` 0/1 과 `month` 뿐이라 **"개장 3일 전"과 "개장 30일 전"을
#   구분하지 못한다.** Phase 28 에서 봄 앙상블 이득의 90% 가 개장 주간 창 하나였고
#   그 이득의 113% 가 화담숲 13품목에서 나왔다 — 즉 램프 정보에 값어치가 있다.
#   앙상블로는 못 먹는다(TEST 에 개장 주간 창이 0개). **모델에 가르치는 쪽이 옳다.**
# ⚠️ 개장일은 **3/29** 다. config.HWADAM_OPEN 의 3/20 은 틀렸고(Phase 5-b 실측 3/29~31)
#   팀원 코드도 3/20 을 쓴다. 이식하면서 고친다.
RAMP_KEYS = ["d_to_hwadam_open", "d_to_hwadam_close", "d_to_ski_close",
             "dayoff_run_len", "dayoff_run_pos"]
HWADAM_OPEN_MD, HWADAM_CLOSE_MD, SKI_CLOSE_MD = (3, 29), (11, 30), (3, 5)

# ⚠️ Tier B (대회 규정 밖 — 외부 뉴스 조사). RAMP_KEYS 는 개장일을 고정 3/29 상수로
#   쓰지만 실측은 매년 다르다. 뉴스/보도자료 교차확인(2026-08-10):
#     2023-03-31 (LG 보도자료 lg.co.kr/media/release/26016, 2023-03-14 작성)
#     2024-03-29 (한국경제 202403295890K, 화담채 개관 동시 보도, 2024-03-29 작성)
#     2025-03-28 (헤럴드경제 article/10436390, 2025-03-07 작성 "오는 28일 개원")
#   자료 없는 연도는 HWADAM_OPEN_MD 고정값으로 대체(안전한 하위호환).
HWADAM_OPEN_ACTUAL = {2023: (3, 31), 2024: (3, 29), 2025: (3, 28)}
TB_RAMP_KEYS = ["d_to_hwadam_open_actual"]

# ⚠️ Tier B — 기상청 API허브 ASOS(이천 관측소 203) 실측 일자료. 예측 대상일(td)의 **실제**
#   기온/강수/적설을 쓴다 — 실전 배포라면 미래 날씨는 예보치라 이건 "완벽한 예보가 있다면"의
#   상한선 실험이다(재현 CSV: data/tierb/weather_icheon.csv, fetch_weather.py로 수집).
TB_WEATHER_KEYS = ["wthr_ta_avg", "wthr_rn_day", "wthr_sd_max"]
# ⚠️ Tier B — TB2b(TB2 재시도). log_tierb.md TB2 판정: 겨울 2폴드 손해·봄 2폴드 이득으로
#   4/4 게이트 탈락. 원인: 위 TB_WEATHER_KEYS가 `np.tile`로 9개 업장 193개 전 품목에
#   동일한 날씨값을 그대로 복사한다 — README가 경고한 Phase 9-b 실수(리조트 공통 피처는
#   업장마다 부호가 반대라 기각)와 같은 패턴. 실제 날씨 노출 업장은 `STORE_CLUSTER` 기준
#   hwadam(화담숲주막·화담숲카페, 12~2월 완전 휴점)·green(느티나무 셀프BBQ, 연중 영업)
#   둘뿐이다. 이번엔 그 두 업장군에만 값을 주고, 화담숲은 개장기간(`HWADAM_OPEN`,
#   `_calendar()`의 hwadam_open과 동일 정의)에만 켠다. 경계를 TB2 폴드 결과(2/23)로
#   맞추지 않기 위해 **결과와 무관하게 이미 존재하던 도메인 상수**를 그대로 재사용한다.
TB_WEATHER_GATED_KEYS = ["wthr_ta_avg_g", "wthr_rn_day_g", "wthr_sd_max_g"]
# ⚠️ Tier B — TB2c. TB2/TB2b는 둘 다 예측 대상일(td)의 실측 날씨라 "완벽예보가 있다면"의
#   상한선 실험이었다. TB2c는 **origin(창이 끝나는 날, 이미 다 지난 시점) 기준 최근 7일
#   실측 날씨 추세**로 바꾼 배포 가능 버전이다 — td가 아니라 origin까지의 정보만 쓰므로
#   실전에서도 그대로 계산 가능하다. ta_avg는 7일 평균, rn_day는 7일 합, sd_max는 7일
#   최댓값. 게이팅(화담숲=개장기간만·느티나무=연중)은 TB2b와 동일하게 유지 — 개장기간
#   판정은 예측 대상일 td 기준(그 날 화담숲이 열려 있는지가 중요하지, origin이 아니다).
TB_WEATHER_TREND_KEYS = ["wthr_ta_avg_last7_g", "wthr_rn_day_last7sum_g", "wthr_sd_max_last7_g"]
# ⚠️ Tier B — TB2d. TB2·TB2b·TB2c 세 번 전부 FAR-겨울(학습 데이터 최소 폴드)에서 손해가
#   났는데, 공통점은 겨울엔 화담숲 게이트가 꺼지고 **느티나무(green) 단독 신호만 남는다**는
#   것이었다 — 그 얇아진 신호가 매번 잡음원으로 의심됨. TB2d는 TB2c와 원본 데이터·추세
#   설계(origin 기준 최근 7일)는 동일하되, **느티나무를 게이트에서 완전히 빼고 화담숲
#   (개장기간만, 계절성이 훨씬 뚜렷함)만 남긴다** — 겨울엔 항상 0이 되므로 FAR-겨울에서
#   더 이상 잡음을 만들 수 없다는 게 핵심 가설.
TB_WEATHER_HWADAM_KEYS = ["wthr_ta_avg_last7_hw", "wthr_rn_day_last7sum_hw", "wthr_sd_max_last7_hw"]
# ⚠️ Tier B — TB2e. 진단(2026-08-14): FAR-겨울·F2 검증 구간에서 화담숲 게이트가 켜지는
#   날은 6.59%(11/25~11/30, 개장기간 끝자락)뿐인데도 TB2d(화담숲 단독)가 FAR-겨울에서
#   -0.00517 손해를 냈다 — 값의 93% 이상이 상수 0인 열치고는 손해가 너무 커서, "느티나무와
#   섞여서" 문제가 아니라 열 추가 자체가 feature_fraction=0.65 표본추출을 흔든 부작용
#   (규칙 ⑯)일 가능성이 크다는 뜻. 그래도 TB2d가 느티나무를 통째로 지우며 F2의 실신호까지
#   버린 건 별개 문제라, **화담숲과 느티나무를 하나의 게이트 값으로 묶지 않고 완전히
#   독립된 열로 분리**해서 그 손실만이라도 되살릴 수 있는지 시험한다. 화담숲 쪽은
#   TB_WEATHER_HWADAM_KEYS를 그대로 재사용(값·게이트 동일), 느티나무는 아래 별도 열.
TB_WEATHER_GREEN_KEYS = ["wthr_ta_avg_last7_gr", "wthr_rn_day_last7sum_gr", "wthr_sd_max_last7_gr"]
# ⚠️ Tier B — 한국관광공사 지역별(경기 광주시) 방문자수. 마찬가지로 예측 대상일 실측치
#   (data/tierb/visitors_gwangju.csv, fetch_visitors.py). local/outside/foreign 3종.
TB_VISIT_KEYS = ["visit_local", "visit_outside", "visit_foreign"]
# ⚠️ Tier B — 네이버 검색어트렌드(NAVER API HUB). "곤지암리조트"/"화담숲" 검색량의
#   7일 이동평균 대비 모멘텀(당일 포함, 실측치 상한선 실험).
#   (data/tierb/naver_trend_mom.csv, fetch_naver_trend.py + process_naver_trend.py)
TB_NAVER_KEYS = ["naver_gonjiam_mom", "naver_hwadam_mom"]
# ⚠️ Tier B — OpenAI 메뉴명 임베딩(text-embedding-3-small). 철자(TF-IDF) 대신 **의미**로
#   가장 가까운 다른 품목 1개를 골라, 기존 x_proxy_*(CROSS_KEYS)와 같은 방식으로
#   그 품목의 최근 판매량을 피처화한다. (data/tierb/menu_embeddings.npy,
#   fetch_menu_embeddings.py — D.item_order() 순서와 정렬됨)
# ⚠️ Tier B — 네이버 검색어트렌드, 선행지표 버전. 위 TB_NAVER_KEYS와 원본 데이터는 같지만
#   예측 대상일(td)이 아니라 창이 끝나는 시점(origin)의 검색 모멘텀을 쓴다 — 진짜 "그때까지
#   알 수 있는 정보만으로 미래를 맞히는가"를 테스트. TB_NAVER_KEYS와 절대 같은 실험에서
#   동시에 켜지 않는다(규칙: 한 번에 후보 하나).
TB_NAVER_LEAD_KEYS = ["naver_gonjiam_lead", "naver_hwadam_lead"]
# ⚠️ Tier B — 네이버 검색 모멘텀, lag 18~33일 전 구간 평균(곤지암만). EDA(eda_naver_lag.py)
#   근거: 이 구간에서 상관이 고립 봉우리 없이 완만하게 이어짐(r 0.05~0.19). 화담숲은
#   전 lag에서 상관 없어 제외. (data/tierb/naver_trend_lag.csv, process_naver_lag.py)
TB_NAVER_LAG_KEYS = ["naver_gonjiam_lag18_33"]
# ⚠️ Tier B — 네이버 검색량 7일 기울기(slope7)의 lag 7~29일 구간 평균(곤지암만).
#   EDA(eda_naver_residual_curve.py) 근거: 베이스 모델 잔차와의 상관이 lag 1~29 구간에서
#   넓게 양수(r 0.14~0.28), 고립 봉우리 아님. lag 하한 7 = HORIZON(7)과 맞춰 h=1~7 전부
#   미래 정보 누출 없이 안전하게 설계.
#   (data/tierb/naver_slope_lag.csv, process_naver_slope.py)
TB_NAVER_SLOPE_KEYS = ["naver_gonjiam_slope_lag7_29"]
# ⚠️ Tier B — 화담숲 검색량 7일 변동성(std7)의 lag 20~35일 구간 평균. EDA 근거: 잔차와의
#   상관이 lag 20~35에서 고원(r 0.15~0.22). 원본 수요 기준 EDA에선 무신호였으나 잔차 기준+
#   변동성 조합에서만 나타남 — 모델이 기존에 갖지 않은 종류의 정보일 가능성.
#   (data/tierb/naver_std_lag.csv, process_naver_std.py)
TB_NAVER_STD_KEYS = ["hwadam_std_lag20_35"]
# ⚠️ Tier B — RAMP_KEYS는 스키장 폐장일을 고정 3/5 상수로 쓰지만 실측은 매년 다르다.
#   뉴스 교차확인(2026-08-11):
#     2024-03-01 (전국 스키장 폐장일 정리 기사, "곤지암 스키장 폐장일 2024년 3월 1일")
#     2025-03-03 (곤지암리조트 자체 공지 "슬로프 및 눈썰매장 이용 안내" ver.2/18, "3/3(월) 17:00 폐장")
#   2023년은 신뢰할 만한 출처를 못 찾음 — SKI_CLOSE_MD 고정값(3/5)으로 대체(안전한 하위호환,
#   HWADAM_OPEN_ACTUAL과 동일 관례).
SKI_CLOSE_ACTUAL = {2024: (3, 1), 2025: (3, 3)}
TB_RAMP_KEYS = ["d_to_hwadam_open_actual", "d_to_ski_close_actual"]

TB_EMBED_KEYS = ["emb_sim_last7", "emb_sim_dow"]
ITEM_KEYS = ["store", "cluster", "category", "season_hint", "item_id",
             "price", "is_high_weight"]
# 메뉴 '이름'으로 만든 군집 — 품목 하나짜리 범주형 피처.
# 동기: **같은 메뉴가 여러 영업장에 흩어져 있는데 지금은 완전히 남남이다.**
#   스프라이트 6곳 · 아메리카노 7곳 · 공깃밥 5곳 · 코카콜라 3곳 …
#   `item_id` 는 이들을 전부 별개로 보므로, 화담숲카페(품목 5개뿐)의 아메리카노는
#   포레스트릿 아메리카노에서 아무것도 못 빌려온다.
#   공식 지표는 **드문 품목에도 1표**를 주므로 이 지점이 실제로 점수에 걸린다.
# 설계 주의(Phase 9-b 교훈): 벡터 8차원을 통째로 넣으면 피처 예산 경쟁이 커진다.
#   → **범주형 1개**로 압축한다. 그리고 정적 속성이라 '날짜 도장'이 될 수 없다.
NAME_KEYS = ["name_cluster"]
NAME_THRESHOLD = 0.35           # set_name_threshold() 로 바꾼다

FEATURE_GROUPS = {"win": WIN_KEYS, "dow": DOW_KEYS, "closed": CLOSED_KEYS,
                  "prof": PROF_KEYS, "cross": CROSS_KEYS, "resort": RESORT_KEYS,
                  "ctx": CTX_KEYS, "cal": CAL_KEYS, "ramp": RAMP_KEYS,
                  "item": ITEM_KEYS, "name": NAME_KEYS,
                  "tb_ramp": TB_RAMP_KEYS, "tb_weather": TB_WEATHER_KEYS,
                  "tb_weather_v2": TB_WEATHER_GATED_KEYS,
                  "tb_visit": TB_VISIT_KEYS, "tb_embed": TB_EMBED_KEYS,
                  "tb_naver": TB_NAVER_KEYS, "tb_naver_lead": TB_NAVER_LEAD_KEYS, "tb_naver_lag": TB_NAVER_LAG_KEYS,
                  "tb_naver_slope": TB_NAVER_SLOPE_KEYS, "tb_naver_std": TB_NAVER_STD_KEYS,
                  "tb_ski_close": ["d_to_ski_close_actual"],
                  "tb_weather_trend": TB_WEATHER_TREND_KEYS,
                  "tb_weather_trend_hw": TB_WEATHER_HWADAM_KEYS,
                  "tb_weather_split": TB_WEATHER_HWADAM_KEYS + TB_WEATHER_GREEN_KEYS}
CATEGORICAL = ["store", "cluster", "category", "season_hint", "item_id",
               "dow", "month", "name_cluster"]
# price는 5등급으로 묶되 **숫자(순서형)로 둔다**. categorical로 바꿨더니
# 0.5159 → 0.5186 으로 나빠졌다 — 가격 등급은 순서가 있는데 범주로 만들면 그 순서를 버린다.

_HOLIDAYS = set(pd.to_datetime(C.HOLIDAYS))


def feature_names():
    # ※ 이 순서는 build_samples 의 열 연결 순서와 **반드시** 일치해야 한다.
    return (WIN_KEYS + DOW_KEYS + CLOSED_KEYS + PROF_KEYS + CROSS_KEYS
            + RESORT_KEYS + CTX_KEYS + CAL_KEYS + RAMP_KEYS + ITEM_KEYS
            + NAME_KEYS + TB_RAMP_KEYS + TB_WEATHER_KEYS + TB_VISIT_KEYS
            + TB_EMBED_KEYS + TB_NAVER_KEYS + TB_NAVER_LEAD_KEYS + TB_NAVER_LAG_KEYS
            + TB_NAVER_SLOPE_KEYS + TB_NAVER_STD_KEYS + TB_WEATHER_GATED_KEYS
            + TB_WEATHER_TREND_KEYS + TB_WEATHER_HWADAM_KEYS + TB_WEATHER_GREEN_KEYS)


# ⚠️ 학습에서 제외하는 그룹. build_samples 는 여전히 이 열들을 만들지만 **쓰지 않는다.**
#   prof   : Phase 4-e 기각 (모델이 item_id×dow 로 이미 알 수 있던 정보)
#   resort : Phase 9-b 기각 (품목마다 방향이 반대 + 날짜 도장 역할)
#   name   : **검증 전.** 통과하면 여기서 뺀다.
#   ramp   : **검증 전 (Phase 29).** 기본 OFF 로 둔다 — 켜면 기존 스크립트가 전부 조용히
#            v24 와 다른 모델을 만든다. 실험 스크립트가 명시적으로 켠다:
#              keep = F.active_columns(include=("ramp",))
# 남겨두는 이유는 재현성과 재시도 방지 기록이다.
#   tb_ramp/tb_weather/tb_visit : Tier B(대회 규정 밖). tb_ramp는 게이트 탈락(log_tierb.md TB1).
#   tb_weather_v2 : TB2b, 게이트 탈락(log_tierb.md TB2b, 3/4).
#   tb_weather_trend : TB2c, 게이트 탈락(log_tierb.md TB2c, 3/4).
#   tb_weather_trend_hw : TB2d, 게이트 탈락(log_tierb.md TB2d, 2/4).
#   tb_weather_split : TB2e, **검증 전.** 화담숲·느티나무를 독립 열로 분리한 버전.
DROPPED = (set(PROF_KEYS) | set(RESORT_KEYS) | set(NAME_KEYS) | set(RAMP_KEYS)
           | set(TB_RAMP_KEYS) | set(TB_WEATHER_KEYS) | set(TB_VISIT_KEYS)
           | set(TB_EMBED_KEYS) | set(TB_WEATHER_GATED_KEYS) | set(TB_NAVER_KEYS)
           | set(TB_NAVER_LEAD_KEYS) | set(TB_NAVER_LAG_KEYS)
           | set(TB_NAVER_SLOPE_KEYS) | set(TB_NAVER_STD_KEYS)
           | set(TB_WEATHER_TREND_KEYS) | set(TB_WEATHER_HWADAM_KEYS)
           | set(TB_WEATHER_GREEN_KEYS))


def _norm_menu(m):
    """온·냉 표기와 용량·가격 군더더기를 걷어낸다.
    **의미가 다른 표시('단체' 등)는 남긴다** — 단체 메뉴는 수요 패턴이 실제로 다르다."""
    s = m.lower()
    s = re.sub(r"\bhot\b|\bice\b|아이스(?!크림|티)", "", s)   # 아이스크림·아이스티는 보존
    s = re.sub(r"[\s\(\)\[\],\.·]", "", s)
    s = re.sub(r"\d+", "", s)
    return re.sub(r"원|ea|ps|인석|g\b", "", s)


def _name_kmeans(menus, k, seed=0):
    """KMeans(k 고정) 방식. 193품목을 억지로 k개로 나눈다.

    ⚠️ 이게 **거친 쪽에서는 오히려 잘 됐다**(Phase 9-c). k=12 에서 129품목이 한 덩어리로
    뭉쳤는데, 그 덩어리가 사실상 '기타' 역할을 해서 **레벨이 12개뿐 = 암기 불가**가 됐다.
    반대로 임계값 방식(9-d)은 단독 128개가 생겨 레벨 151개 → `item_id` 복사본이 되어
    gain 2위를 먹으면서도 성적은 안 올랐다. **거친 게 미덕이다.**
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.cluster import KMeans
    clean = [re.sub(r"\s+", " ", m).strip() for m in menus]
    Xn = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3),
                         min_df=2).fit_transform(clean)
    Z = TruncatedSVD(n_components=min(30, Xn.shape[1] - 1),
                     random_state=seed).fit_transform(Xn)
    return KMeans(n_clusters=k, n_init=10,
                  random_state=seed).fit_predict(Z).astype(np.int32)


def _name_groups(menus, thr=NAME_THRESHOLD, merge_singletons=False):
    """이름이 **충분히 비슷한 것만** 묶고 나머지는 각자 단독으로 둔다.

    거리 임계값 기반 병합. 가까운 것만 붙고 먼 것은 혼자 남는다.
    임계값 0.35 기준: 2품목 이상 그룹 23개(65품목) + 단독 128개, 최대 그룹 7품목.
      아메리카노 7품목/4곳 · 스프라이트 6/6곳 · 카페라떼 5/3곳 · 공깃밥 3/3곳 · 콜라 3/3곳
      (정식)된장찌개↔(후식)된장찌개 · 브런치 2인↔4인 · 대여료 3만↔6만↔9만 …
    KMeans 와 달리 **관계없는 품목을 억지로 붙이지 않는다.**

    ⚠️ **`merge_singletons=True` 를 쓸 것.** 그냥 두면 단독 품목이 128개나 생겨
      레벨이 151개가 되고 **`item_id` 의 복사본**이 되어 버린다 — 이것이 9-d 실패의 원인이다
      (gain 2위를 먹으면서 성적은 −0.0013. 품목별 암기 통로가 두 배가 됨).
      단독끼리 하나의 '기타' 그룹으로 합치면 **깨끗한 그룹 + 기타 하나** 구조가 되어,
      9-c에서 유일하게 효과가 있었던 k=12 의 구조를 **억지 병합 없이** 재현한다.

    판매 데이터를 전혀 안 쓰므로 누수가 구조적으로 불가능하고,
    제출 템플릿의 컬럼명만 사용하므로 대회 규칙상 안전하다.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import AgglomerativeClustering
    Xn = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3),
                         min_df=1).fit_transform([_norm_menu(m) for m in menus])
    lb = AgglomerativeClustering(
        n_clusters=None, distance_threshold=thr, metric="cosine",
        linkage="average").fit_predict(Xn.toarray())
    if not merge_singletons:
        return lb.astype(np.int32)
    keep = np.bincount(lb) > 1                      # 2품목 이상인 그룹만 유지
    remap = {c: i for i, c in enumerate(np.where(keep)[0])}
    other = len(remap)                              # 나머지는 전부 '기타'
    return np.array([remap.get(c, other) for c in lb], dtype=np.int32)


def _dropped(include=()):
    """DROPPED 에서 include 로 지정한 그룹만 되살린다."""
    d = set(DROPPED)
    for g in include:
        d -= set(FEATURE_GROUPS[g])
    return d


def active_columns(include=()):
    """**확정 피처 57개**의 열 인덱스. 모든 학습 코드는 이걸 써야 한다.

    직접 `k not in PROF_KEYS` 같은 식으로 거르면 새 그룹이 추가될 때마다
    스크립트가 조용히 낡는다(실제로 Phase 5-a에서 그 사고가 났다).

    include : 검증 중인 그룹을 실험 스크립트가 **명시적으로** 켤 때만 쓴다.
              예) active_columns(include=("ramp",))
    """
    dr = _dropped(include)
    return [i for i, k in enumerate(feature_names()) if k not in dr]


def active_names(include=()):
    dr = _dropped(include)
    return [k for k in feature_names() if k not in dr]


def pick_proxy_items(mat, dates, upto_col, store_codes, min_nz=0.4):
    """영업장별 '인원수 온도계' 품목을 데이터로 고른다.

    ※ 중요 — 자기 자신을 뺀 나머지 합계와 상관을 잰다.
      그냥 영업장 총합과 재면 **대표 메뉴가 총합의 큰 부분을 차지해 자기 자신과
      상관을 재는 꼴**이 된다(순환 논리). 실제로 그렇게 했더니 화담숲주막=해물파전,
      포레스트릿=꼬치어묵처럼 그 가게 1위 품목이 뽑혔고, 결과적으로 영업장 합계 피처와
      중복이라 새 정보가 없었다.
    upto_col 미만만 사용 → 검증 구간 누수 차단.
    """
    sub = mat[:, :upto_col]
    proxy = np.zeros(len(np.unique(store_codes)), dtype=np.int64)
    for sc in np.unique(store_codes):
        m = np.where(store_codes == sc)[0]
        tot = sub[m].sum(0)
        openday = tot > 0
        if openday.sum() < 30:
            proxy[sc] = m[0]
            continue
        best, best_c = m[0], -2.0
        for i in m:
            v = sub[i][openday]
            if (v > 0).mean() < min_nz or v.std() == 0:
                continue
            rest = tot[openday] - v            # ← 자기 자신 제외
            if rest.std() == 0:
                continue
            c = np.corrcoef(v, rest)[0, 1]
            if np.isfinite(c) and c > best_c:
                best, best_c = i, c
        proxy[sc] = best
    return proxy


# 이름만으로 '단체성'을 판별 — 관찰로 확인된 패턴들
GROUP_PAT = r"단체|BBQ\d+|Open Food|패키지|Platter|대여료|그늘집|Conference|Ballroom|Convention|OPUS"


def item_dow_profile(mat, dates, upto_col, store_codes):
    """품목별 요일 프로파일 (n_items, 7). 평균 1로 정규화.

    upto_col : 이 열 인덱스 **미만**만 사용 → 검증 구간 정보가 새지 않게 한다.
    영업일(그 영업장 합계>0)만 세어 휴점 때문에 요일이 왜곡되는 것을 막는다.
    """
    sub = mat[:, :upto_col]
    dws = np.array([d.dayofweek for d in dates[:upto_col]])
    # 영업장이 문 연 날만
    open_day = np.zeros(sub.shape, dtype=bool)
    for sc in np.unique(store_codes):
        m = store_codes == sc
        open_day[m] = (sub[m].sum(0) > 0)[None, :]
    prof = np.ones((sub.shape[0], 7), dtype=np.float32)
    for w in range(7):
        col = (dws == w)
        if not col.any():
            continue
        vals = np.where(open_day[:, col], sub[:, col], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            prof[:, w] = np.nan_to_num(np.nanmean(vals, axis=1))
    mean = prof.mean(1, keepdims=True)
    prof = np.divide(prof, mean, out=np.ones_like(prof), where=mean > 0)
    return prof.astype(np.float32)


class Context:
    """품목 정적 속성 묶음. 품목 순서는 제출 열 순서로 고정."""

    def __init__(self):
        self.items = D.item_order()
        self.n = len(self.items)
        stores = [k.split("_")[0] for k in self.items]
        self.store_of_item = np.array(stores)
        self.store_codes = pd.Categorical(stores).codes.astype(np.int32)
        self.store_names = list(pd.Categorical(stores).categories)
        self.cluster_codes = pd.Categorical(
            [C.STORE_CLUSTER[s] for s in stores]).codes.astype(np.int32)
        # TB2b — 날씨 게이팅용 업장군 마스크(화담숲=개장기간만, 느티나무=연중, 나머지=0).
        self._hwadam_mask = np.array(
            [1.0 if C.STORE_CLUSTER[s] == "hwadam" else 0.0 for s in stores], np.float32)
        self._green_mask = np.array(
            [1.0 if C.STORE_CLUSTER[s] == "green" else 0.0 for s in stores], np.float32)
        lab = D.load_labels()
        raw_price = np.array(
            [D.label_of(lab, k, "price_krw", 0) or 0 for k in self.items], np.float32)
        self.raw_price = raw_price
        # Phase 2 결론: 원본 숫자보다 5등급 범주가 낫다 (0.5171 → 0.5159).
        # 트리는 단조변환에 불변이라 log는 무의미했고, 값을 '묶는' 범주화만 효과가 있었다.
        self.price = np.digitize(raw_price, C.PRICE_BANDS).astype(np.float32)
        self.cat_codes = pd.Categorical(
            [D.label_of(lab, k, "category", "?") for k in self.items]).codes.astype(np.int32)
        self.season_codes = pd.Categorical(
            [D.label_of(lab, k, "season", "무관") for k in self.items]).codes.astype(np.int32)
        self.is_high_w = np.array(
            [1 if s in C.HIGH_WEIGHT_STORES else 0 for s in stores], np.int32)
        menus = [k.split("_", 1)[1] for k in self.items]
        self.menus = menus
        self.is_group = np.array(
            [1 if re.search(GROUP_PAT, m) else 0 for m in menus], np.int32)
        self.set_name_threshold(NAME_THRESHOLD)
        self.dow_profile = None      # set_profile()로 폴드마다 주입
        self.proxy_idx = None        # set_proxy()로 폴드마다 주입
        self.set_profile(np.ones((self.n, 7), np.float32))   # 기본값
        self.set_proxy(np.array([np.where(self.store_codes == sc)[0][0]
                                 for sc in np.unique(self.store_codes)]))
        self.embed_nn_of_item = _embed_nn(self.items)   # Tier B — tb_embed 그룹용
        self._item_static = np.column_stack([
            self.store_codes, self.cluster_codes, self.cat_codes,
            self.season_codes, np.arange(self.n), self.price, self.is_high_w
        ]).astype(np.float32)

    @property
    def item_static(self):
        return self._item_static

    def set_name_threshold(self, thr, merge_singletons=False):
        """이름 그룹 임계값을 바꾼다. 이름만 쓰므로 폴드마다 다시 계산할 필요가 없다.
        작을수록 엄격(거의 같은 이름만 묶임), 클수록 느슨."""
        self._set_name(_name_groups(self.menus, thr, merge_singletons))

    def set_name_kmeans(self, k):
        """KMeans 방식으로 k개 군집. 거친 쪽(k 작음)에서만 값어치가 있다."""
        self._set_name(_name_kmeans(self.menus, k))

    def _set_name(self, lb):
        self.name_cluster = lb
        self.n_name_levels = int(lb.max()) + 1
        self._name_static = lb.reshape(-1, 1).astype(np.float32)

    @property
    def name_static(self):
        return self._name_static

    def set_proxy(self, proxy_idx):
        """영업장별 인원수 프록시 품목 인덱스 주입. 품목별로 펼쳐 둔다."""
        self.proxy_idx = proxy_idx
        self.proxy_of_item = proxy_idx[self.store_codes]

    def set_profile(self, prof):
        """폴드 학습구간에서 계산한 요일 프로파일 주입 (n_items, 7)."""
        self.dow_profile = prof
        wk = prof[:, :4].mean(1)
        we = prof[:, 5:].mean(1)
        self._prof_static = np.column_stack([
            np.divide(we, wk, out=np.ones_like(we), where=wk > 0),   # 주말/평일 비
            prof.argmax(1).astype(np.float32),                       # 최강 요일
            prof.std(1),                                             # 요일 편차 크기
            self.is_group.astype(np.float32),
        ]).astype(np.float32)


_CAL_CACHE = {}


def _calendar(d):
    if d in _CAL_CACHE:
        return _CAL_CACHE[d]
    doy = d.dayofyear
    md = (d.month, d.day)
    hw = C.HWADAM_OPEN[0] <= md <= C.HWADAM_OPEN[1]
    ski = d.month in C.SKI_SEASON_MONTHS or (d.month == 3 and d.day <= 5)
    peak = md >= C.SKI_PEAK[0] or md <= C.SKI_PEAK[1]
    fol = C.FOLIAGE[0] <= md <= C.FOLIAGE[1]
    v = np.array([
        d.dayofweek, d.month, d.day,
        1 if d.dayofweek >= 5 else 0,
        1 if d in _HOLIDAYS else 0,
        1 if (d + pd.Timedelta(days=1)) in _HOLIDAYS else 0,
        1 if (d.dayofweek >= 5 or d in _HOLIDAYS) else 0,
        np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
        int(hw), int(ski), int(peak and ski), int(fol),
    ], dtype=np.float32)
    _CAL_CACHE[d] = v
    return v


_RAMP_CACHE = {}


def _signed_days_to(d, month, day):
    """가장 가까운 (month,day) 까지의 **부호 있는** 일수. +면 아직 안 왔고 −면 지났다.

    연도를 셋(작년·올해·내년) 다 놓고 절댓값 최소를 고른다 — 그래야 12/31↔1/1 이 붙는다.
    `month`(1~12)나 `doy_sin` 으로는 표현할 수 없는 정보다.
    """
    cands = (pd.Timestamp(year=y, month=month, day=day)
             for y in (d.year - 1, d.year, d.year + 1))
    return float(min(((c - d).days for c in cands), key=abs))


def _dayoff_run(d):
    """연휴 덩어리의 (길이, 이 날이 몇째 날인가). 평일이면 (0,0).

    '토요일'과 '5일 연휴의 3일째'는 수요가 다른데 `is_dayoff` 0/1 로는 같아 보인다.
    """
    off = lambda x: x.dayofweek >= 5 or x in _HOLIDAYS
    if not off(d):
        return 0.0, 0.0
    s = e = d
    for _ in range(10):
        if off(s - pd.Timedelta(days=1)):
            s -= pd.Timedelta(days=1)
        else:
            break
    for _ in range(10):
        if off(e + pd.Timedelta(days=1)):
            e += pd.Timedelta(days=1)
        else:
            break
    return float((e - s).days + 1), float((d - s).days + 1)


def _ramp(d):
    """RAMP_KEYS 5종. 전부 타깃일 하나로 정해지므로 품목 무관(창 밖 조회 없음 = 규칙 안전)."""
    if d in _RAMP_CACHE:
        return _RAMP_CACHE[d]
    rl, rp = _dayoff_run(d)
    v = np.array([_signed_days_to(d, *HWADAM_OPEN_MD),
                  _signed_days_to(d, *HWADAM_CLOSE_MD),
                  _signed_days_to(d, *SKI_CLOSE_MD), rl, rp], dtype=np.float32)
    _RAMP_CACHE[d] = v
    return v


_TB_RAMP_CACHE = {}


def _signed_days_to_actual(d):
    """Tier B — 연도별 실측 개장일까지 부호 있는 일수(HWADAM_OPEN_ACTUAL).
    자료 없는 연도는 HWADAM_OPEN_MD 고정 상수로 대체(_signed_days_to와 동일 로직)."""
    cands = []
    for y in (d.year - 1, d.year, d.year + 1):
        month, day = HWADAM_OPEN_ACTUAL.get(y, HWADAM_OPEN_MD)
        cands.append(pd.Timestamp(year=y, month=month, day=day))
    return float(min(((c - d).days for c in cands), key=abs))

def _signed_days_to_ski_close_actual(d):
    """Tier B — 연도별 실측 스키장 폐장일까지 부호 있는 일수(SKI_CLOSE_ACTUAL).
    자료 없는 연도는 SKI_CLOSE_MD 고정 상수로 대체."""
    cands = []
    for y in (d.year - 1, d.year, d.year + 1):
        month, day = SKI_CLOSE_ACTUAL.get(y, SKI_CLOSE_MD)
        cands.append(pd.Timestamp(year=y, month=month, day=day))
    return float(min(((c - d).days for c in cands), key=abs))

def _ramp_actual(d):
    """TB_RAMP_KEYS. _ramp()의 d_to_hwadam_open·d_to_ski_close를 실측치로 교체."""
    if d in _TB_RAMP_CACHE:
        return _TB_RAMP_CACHE[d]
    v = np.array([_signed_days_to_actual(d),
                  _signed_days_to_ski_close_actual(d)], dtype=np.float32)
    _TB_RAMP_CACHE[d] = v
    return v


def _load_tierb_csv(name, cols):
    """data/tierb/<name>.csv → {date: np.array(cols)} 딕셔너리. 없으면 빈 딕셔너리(전부 0으로 대체)."""
    path = os.path.join(C.DATA, "tierb", f"{name}.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    out = {}
    for _, row in df.iterrows():
        out[row["date"]] = np.array([row[c] for c in cols], dtype=np.float32)
    return out


_WEATHER = None
_VISIT = None
_NAVER = None


def _weather(d):
    """TB_WEATHER_KEYS(3). 기상청 실측(예측 대상일 td 기준 — 완벽예보 상한선 실험, 캐시 lazy-load)."""
    global _WEATHER
    if _WEATHER is None:
        _WEATHER = _load_tierb_csv("weather_icheon", ["ta_avg", "rn_day", "sd_max"])
    return _WEATHER.get(d, np.zeros(len(TB_WEATHER_KEYS), dtype=np.float32))


def _weather_gated(td, ctx):
    """TB_WEATHER_GATED_KEYS(3), TB2b. 화담숲=개장기간만(HWADAM_OPEN, `_calendar()`의
    hwadam_open과 동일 정의) · 느티나무=연중 · 나머지 7개 업장=0. TB2와 동일하게 예측
    대상일(td)의 실측값이라 완벽예보 상한선 전제도 동일하다. 게이트 경계는 TB2 폴드
    결과가 아니라 기존 도메인 상수(HWADAM_OPEN)로 고정했다."""
    w = _weather(td)
    md = (td.month, td.day)
    hw_open = C.HWADAM_OPEN[0] <= md <= C.HWADAM_OPEN[1]
    gate = ctx._hwadam_mask * (1.0 if hw_open else 0.0) + ctx._green_mask
    return gate[:, None] * w[None, :]


_WEATHER_LAST7_CACHE = {}


def _weather_last7(origin_date):
    """TB_WEATHER_TREND_KEYS 원본(게이팅 전), TB2c. origin_date를 포함한 최근 7일
    실측의 ta_avg 평균 · rn_day 합 · sd_max 최댓값. td가 아니라 **origin**(이미 지난
    시점) 기준이라 실전 배포 가능 — TB2/TB2b의 "완벽예보 상한선" 전제가 없다."""
    if origin_date in _WEATHER_LAST7_CACHE:
        return _WEATHER_LAST7_CACHE[origin_date]
    global _WEATHER
    if _WEATHER is None:
        _WEATHER = _load_tierb_csv("weather_icheon", ["ta_avg", "rn_day", "sd_max"])
    days = [origin_date - pd.Timedelta(days=i) for i in range(7)]
    vals = [_WEATHER[d] for d in days if d in _WEATHER]
    if vals:
        arr = np.stack(vals)
        out = np.array([arr[:, 0].mean(), arr[:, 1].sum(), arr[:, 2].max()],
                        dtype=np.float32)
    else:
        out = np.zeros(3, dtype=np.float32)
    _WEATHER_LAST7_CACHE[origin_date] = out
    return out


def _weather_trend_gated(w7, td, ctx):
    """TB_WEATHER_TREND_KEYS(3), TB2c. `_weather_last7(origin)` 결과를 TB2b와 동일한
    업장군 게이트(화담숲=개장기간만·느티나무=연중)로 좁힌다. 개장기간 판정은 예측
    대상일(td) 기준 — 그날 화담숲이 실제로 열려 있는지가 중요하지 origin이 아니다."""
    md = (td.month, td.day)
    hw_open = C.HWADAM_OPEN[0] <= md <= C.HWADAM_OPEN[1]
    gate = ctx._hwadam_mask * (1.0 if hw_open else 0.0) + ctx._green_mask
    return gate[:, None] * w7[None, :]


def _weather_trend_hwadam(w7, td, ctx):
    """TB_WEATHER_HWADAM_KEYS(3), TB2d. `_weather_trend_gated`와 원본 값·개장기간
    판정(td 기준)은 동일하지만, **느티나무(green) 항을 완전히 뺀다.** TB2/TB2b/TB2c
    셋 다 겨울엔 화담숲이 꺼지고 느티나무 단독 신호만 남으면서 FAR-겨울(학습 데이터
    최소 폴드)에서 손해를 봤다 — 화담숲만 남기면 겨울엔 항상 0이 되어 그 잡음원이
    구조적으로 사라진다는 게 이 후보의 핵심 가설."""
    md = (td.month, td.day)
    hw_open = C.HWADAM_OPEN[0] <= md <= C.HWADAM_OPEN[1]
    gate = ctx._hwadam_mask * (1.0 if hw_open else 0.0)
    return gate[:, None] * w7[None, :]


def _weather_trend_green(w7, ctx):
    """TB_WEATHER_GREEN_KEYS(3), TB2e. 느티나무(green) 전용 열 — 연중 영업이라 계절
    게이트 없이 항상 켜진다. `_weather_trend_hwadam`의 화담숲 열과 **완전히 분리된
    별도 열**이라, 트리가 두 업장의 날씨 반응을 서로 다른 계수로 학습할 수 있다
    (TB2d처럼 하나의 게이트 값으로 묶여 서로의 신호를 지우는 문제를 피한다)."""
    return ctx._green_mask[:, None] * w7[None, :]


def _visit(d):
    """TB_VISIT_KEYS(3). 관광공사 실측 방문자수(경기 광주시, 예측 대상일 td 기준)."""
    global _VISIT
    if _VISIT is None:
        _VISIT = _load_tierb_csv("visitors_gwangju", ["local", "outside", "foreign"])
    return _VISIT.get(d, np.zeros(len(TB_VISIT_KEYS), dtype=np.float32))

def _naver(d):
    """TB_NAVER_KEYS(2). 네이버 검색어트렌드 7일 모멘텀(예측 대상일 td 당일 포함)."""
    global _NAVER
    if _NAVER is None:
        _NAVER = _load_tierb_csv("naver_trend_mom", ["gonjiam_mom", "hwadam_mom"])
    return _NAVER.get(d, np.zeros(len(TB_NAVER_KEYS), dtype=np.float32))

def _naver_lead(d):
    """TB_NAVER_LEAD_KEYS(2). 네이버 검색 모멘텀 — 예측 대상일이 아니라 창이 끝나는 시점(origin)
    기준. 선행지표로서 실제 작동하는지 테스트하는 버전(미래 정보 미포함)."""
    global _NAVER
    if _NAVER is None:
        _NAVER = _load_tierb_csv("naver_trend_mom", ["gonjiam_mom", "hwadam_mom"])
    return _NAVER.get(d, np.zeros(len(TB_NAVER_LEAD_KEYS), dtype=np.float32))

_NAVER_LAG = None

def _naver_lag(d):
    """TB_NAVER_LAG_KEYS(1). 곤지암 검색 모멘텀 lag 18~33일 구간 평균(origin 기준)."""
    global _NAVER_LAG
    if _NAVER_LAG is None:
        _NAVER_LAG = _load_tierb_csv("naver_trend_lag", ["gonjiam_lag18_33"])
    return _NAVER_LAG.get(d, np.zeros(len(TB_NAVER_LAG_KEYS), dtype=np.float32))
_NAVER_SLOPE = None


def _naver_slope(td):
    """TB_NAVER_SLOPE_KEYS(1). 곤지암 검색량 7일 기울기의 lag 7~29일 구간 평균(td 기준)."""
    global _NAVER_SLOPE
    if _NAVER_SLOPE is None:
        _NAVER_SLOPE = _load_tierb_csv("naver_slope_lag", ["gonjiam_slope_lag7_29"])
    return _NAVER_SLOPE.get(td, np.zeros(len(TB_NAVER_SLOPE_KEYS), dtype=np.float32)) 

_NAVER_STD = None


def _naver_std(td):
    """TB_NAVER_STD_KEYS(1). 화담숲 검색량 7일 변동성의 lag 20~35일 구간 평균(td 기준)."""
    global _NAVER_STD
    if _NAVER_STD is None:
        _NAVER_STD = _load_tierb_csv("naver_std_lag", ["hwadam_std_lag20_35"])
    return _NAVER_STD.get(td, np.zeros(len(TB_NAVER_STD_KEYS), dtype=np.float32))

def _embed_nn(items):
    """품목별 '의미상 가장 가까운 다른 품목' 인덱스(자기 자신 제외).

    data/tierb/menu_embeddings.npy가 없으면(아직 임베딩 안 받음) 자기 자신 인덱스를
    반환해 x_proxy_* 와 값이 겹치지 않게 하되 조용히 죽지는 않게 한다(호출부에서 무해).
    """
    path = os.path.join(C.DATA, "tierb", "menu_embeddings.npy")
    n = len(items)
    if not os.path.exists(path):
        return np.arange(n)
    emb = np.load(path)
    if emb.shape[0] != n:
        raise ValueError(f"menu_embeddings.npy 행 수({emb.shape[0]}) != 품목 수({n}) "
                          "— D.item_order() 변경 후 재생성 필요")
    norm = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
    sim = norm @ norm.T
    np.fill_diagonal(sim, -np.inf)
    return sim.argmax(1)


def _window_stats(win):
    pos = win > 0
    cnt = pos.sum(1)
    posmean = np.where(cnt > 0, (win * pos).sum(1) / np.maximum(cnt, 1), 0.0)
    rev = pos[:, ::-1]
    days_since = np.where(rev.any(1), rev.argmax(1), C.WINDOW).astype(np.float32)
    last7, prev7 = win[:, -7:].mean(1), win[:, -14:-7].mean(1)
    masked = np.where(pos, win, np.nan)          # 0을 NaN으로 → 양수만의 중앙값
    with warnings.catch_warnings():              # 전부 0인 품목 = All-NaN, 0으로 처리
        warnings.simplefilter("ignore", RuntimeWarning)
        posmed = np.nan_to_num(np.nanmedian(masked, axis=1)).astype(np.float32)
    return np.column_stack([
        win.mean(1), win.std(1), win.max(1), np.median(win, axis=1),
        cnt / C.WINDOW, posmean, posmed, win[:, -1], last7,
        win[:, -14:].mean(1), prev7, last7 / np.maximum(prev7, 0.5),
        days_since, (win[:, -7:] > 0).sum(1) / 7.0,
    ]).astype(np.float32)


def _dow_stats(win, wdows, target_dow):
    sel = win[:, wdows == target_dow]
    if sel.shape[1] == 0:
        return np.zeros((win.shape[0], len(DOW_KEYS)), np.float32)
    pos = sel > 0
    cnt = pos.sum(1)
    pmean = np.where(cnt > 0, (sel * pos).sum(1) / np.maximum(cnt, 1), 0.0)
    return np.column_stack([
        sel.mean(1), sel.max(1), pmean, cnt / sel.shape[1], sel[:, -1],
    ]).astype(np.float32)


def _open_mask(win, store_codes):
    """(n_items, 28) 불리언 — 그 품목의 영업장이 그날 문을 열었는가.
    영업장 합계 > 0 을 '영업'으로 본다 (프로파일링으로 검증한 정의)."""
    om = np.zeros(win.shape, dtype=bool)
    for sc in np.unique(store_codes):
        m = store_codes == sc
        om[m] = (win[m].sum(0) > 0)[None, :]
    return om


def _closed_stats(win, om, wdows, target_dow):
    """휴점일을 걷어낸 통계. 창이 전부 휴점이면 0으로 떨어뜨린다."""
    n_open = om.sum(1)
    open_win = np.where(om, win, 0.0)
    w_mean_open = np.where(n_open > 0, open_win.sum(1) / np.maximum(n_open, 1), 0.0)
    nz_open = ((win > 0) & om).sum(1)
    w_nz_open = np.where(n_open > 0, nz_open / np.maximum(n_open, 1), 0.0)
    sel = wdows == target_dow
    n_od = (om[:, sel]).sum(1)
    d_mean_open = np.where(n_od > 0,
                           np.where(om[:, sel], win[:, sel], 0.0).sum(1) / np.maximum(n_od, 1),
                           0.0)
    rev = om[:, ::-1]
    since_open = np.where(rev.any(1), rev.argmax(1), C.WINDOW).astype(np.float32)
    return np.column_stack([
        1.0 - om.mean(1), 1.0 - om[:, -7:].mean(1),
        w_mean_open, w_nz_open, d_mean_open, since_open,
    ]).astype(np.float32)


def build_samples(mat, dates, origin_list, ctx, with_target=True, target_mat=None):
    """(X, y, meta) 생성. meta 열 = [origin, horizon, item_idx].

    origin      : 창의 마지막 날 인덱스. 타깃은 origin+h (h=1..7).
    target_mat  : 타깃만 다른 행렬에서 가져올 때 사용(스파이크 캡 A/B용).
                  None이면 mat과 동일.
    """
    n = ctx.n
    tmat = mat if target_mat is None else target_mat
    Xs, ys, ms = [], [], []
    idx_arange = np.arange(n, dtype=np.float32)
    for o in origin_list:
        lo = o - C.WINDOW + 1
        win = mat[:, lo:o + 1]
        wdows = np.array([d.dayofweek for d in dates[lo:o + 1]])
        wstat = _window_stats(win)
        w7 = _weather_last7(dates[o])   # TB2c — origin 기준 최근 7일 날씨(호라이즌 h와 무관)
        om = _open_mask(win, ctx.store_codes)
        st_mean = np.zeros(n, np.float32)
        st_nz = np.zeros(n, np.float32)
        st_daily = np.zeros((n, C.WINDOW), np.float32)   # 그 품목이 속한 영업장의 일별 합계
        for sc in np.unique(ctx.store_codes):
            m = ctx.store_codes == sc
            st_mean[m] = win[m].mean()
            st_nz[m] = (win[m] > 0).mean()
            st_daily[m] = win[m].sum(0)[None, :]
        proxy_win = win[ctx.proxy_of_item]               # (n, 28) 프록시 품목 시계열
        embed_win = win[ctx.embed_nn_of_item]             # (n, 28) 임베딩 최근접 품목 시계열
        st_sum = st_daily.sum(1)
        share_win = np.divide(win.sum(1), st_sum, out=np.zeros(n, np.float32),
                              where=st_sum > 0)
        # ---- 리조트 전체(9곳 합계) ----
        rs_daily = win.sum(0)                            # (28,) 리조트 일별 총합
        rs_l7 = float(rs_daily[-7:].mean())
        rs_p7 = float(rs_daily[-14:-7].mean())
        rs_sum = float(rs_daily.sum())
        rs_sum_l7 = float(rs_daily[-7:].sum())
        # 자기 영업장이 리조트에서 차지하는 몫 — "리조트가 붐빔"과 "우리 가게가 붐빔"을 가른다
        r_share = (st_sum / rs_sum if rs_sum > 0 else np.zeros(n, np.float32))
        st_l7_sum = st_daily[:, -7:].sum(1)
        r_share_l7 = (st_l7_sum / rs_sum_l7 if rs_sum_l7 > 0
                      else np.zeros(n, np.float32))
        for h in range(1, C.HORIZON + 1):
            td = dates[o] + pd.Timedelta(days=h)
            dstat = _dow_stats(win, wdows, td.dayofweek)
            cstat = _closed_stats(win, om, wdows, td.dayofweek)
            ctxm = np.column_stack([
                mat[:, o + h - 7], st_mean, st_nz, np.full(n, h, np.float32)
            ]).astype(np.float32)
            # 요일 프로파일: 예측 대상 요일에 해당하는 값 + 정적 요약 4개
            pstat = np.column_stack([
                ctx.dow_profile[:, td.dayofweek], ctx._prof_static
            ]).astype(np.float32)
            # 교차 품목: 영업장 수준 최근/요일 통계 + 인원수 프록시 + 침투율
            sel = wdows == td.dayofweek
            st_l7 = st_daily[:, -7:].mean(1)
            st_p7 = st_daily[:, -14:-7].mean(1)
            st_dw = st_daily[:, sel].mean(1)
            it_dw = win[:, sel].mean(1)
            xstat = np.column_stack([
                st_l7, st_p7, st_l7 / np.maximum(st_p7, 0.5), st_dw,
                proxy_win[:, -7:].mean(1), proxy_win[:, sel].mean(1),
                share_win,
                np.divide(it_dw, st_dw, out=np.zeros(n, np.float32), where=st_dw > 0),
            ]).astype(np.float32)
            # 리조트 전체: 앞의 셋은 모든 품목에 같은 값(그날 리조트가 얼마나 붐비나),
            # 뒤의 둘은 영업장마다 다른 값(그 안에서 우리 몫이 얼마나 되나)
            rstat = np.column_stack([
                np.full(n, rs_l7, np.float32), np.full(n, rs_p7, np.float32),
                np.full(n, rs_l7 / max(rs_p7, 0.5), np.float32),
                np.full(n, float(rs_daily[sel].mean()) if sel.any() else 0.0,
                        np.float32),
                r_share, r_share_l7,
            ]).astype(np.float32)
            Xs.append(np.concatenate([
                wstat, dstat, cstat, pstat, xstat, rstat, ctxm,
                np.tile(_calendar(td), (n, 1)), np.tile(_ramp(td), (n, 1)),
                ctx.item_static, ctx.name_static,
                np.tile(_ramp_actual(td), (n, 1)),
                np.tile(_weather(td), (n, 1)), np.tile(_visit(td), (n, 1)),
                np.column_stack([embed_win[:, -7:].mean(1),
                                 embed_win[:, sel].mean(1)]).astype(np.float32),
                np.tile(_naver(td), (n, 1)),
                np.tile(_naver_lead(dates[o]), (n, 1)),
                np.tile(_naver_lag(dates[o]), (n, 1)),
                np.tile(_naver_slope(td), (n, 1)),
                np.tile(_naver_std(td), (n, 1)),
                _weather_gated(td, ctx), _weather_trend_gated(w7, td, ctx),
                _weather_trend_hwadam(w7, td, ctx), _weather_trend_green(w7, ctx)],
                axis=1))
            if with_target:
                ys.append(tmat[:, o + h])
            ms.append(np.column_stack([np.full(n, o), np.full(n, h), idx_arange]))
    X = np.concatenate(Xs, 0)
    meta = np.concatenate(ms, 0).astype(np.int32)
    return X, (np.concatenate(ys) if with_target else None), meta


def group_columns(group):
    """피처 그룹 이름 → 열 인덱스 목록 (ablation용)."""
    fn = feature_names()
    keys = set(FEATURE_GROUPS[group])
    return [i for i, k in enumerate(fn) if k in keys]
