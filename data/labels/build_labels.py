# -*- coding: utf-8 -*-
"""menu_labels.json 자체 제작 스크립트 (원본 라벨 파일이 없어서 규칙 기반으로 재구성).

주의: price_krw는 메뉴명에 숫자가 박혀있는 항목(대여료 등)만 정확하고, 나머지는
카테고리별 통상 가격대로 추정한 값이다. 팀원의 실제 라벨과 다를 수 있음.

실행: python build_labels.py  (이 폴더에서)
출력: menu_labels.json
"""
import json
import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import pandas as pd

SUB_PATH = os.path.join(os.path.dirname(__file__), '..', 'sample_submission.csv')

ALCOHOL = ['참이슬', '처음처럼', '카스', '테라', '하이네켄', '버드와이저', '스텔라',
           '막걸리', '소주', 'Sileni', '서드', '와인', 'Wine', '칵테일', '하이볼',
           '토닉', 'G-Charge', 'Cass Beer', '명인안동소주']
COFFEE_TEA = ['아메리카노', '카페라떼', '얼그레이']
SOFT_DRINK = ['콜라', '스프라이트', '사이다', '생수', '코카콜라', 'Coke']
ADE_JUICE = ['에이드', '아이스티', '식혜', '미숫가루']
GOODS_FEE = ['대여료', '이용료', '그늘집', '수저세트', '접시', '컵', '샷 추가',
             'Conference', 'Ballroom', 'Convention', 'OPUS']
SIDE = ['추가', '샐러드', 'Platter', '골뱅이', '떡볶이', '핫도그', '꼬치어묵',
        '소시지', '쌈장', '쌈야채', '라면사리', '메밀면 사리', '허브솔트',
        '주먹밥', '쿠키', 'Cookie']
MEAL = ['비빔밥', '국밥', '찌개', '전골', '탕', '냉면', '파스타', '피자', '스테이크',
        '리조또', '돈까스', '우동', '짜장', '짬뽕', '불고기', 'BBQ', '삼겹', '갈비',
        '닭발', '돼지고기', '갑오징어', '꼬막', '새우', '랍스타', '브런치', '한우',
        '스튜', '스파게티', '샹궈', '설렁탕', '순대', '해물파전', '해장국', '미역국',
        '지짐', '갱시기', '뻥스크림']

SUMMER = ['냉면', '아이스', 'ICE', '자몽리치', '애플망고', '핑크레몬', '복숭아 아이스티']
WINTER = ['어묵', '전골', '국밥', '갈비탕', '설렁탕', '찌개', '해장국', '미역국',
          'HOT', '호빵', '닭발']

PRICE_PATTERNS = re.compile(r'([\d]{1,3}(?:,\d{3})+|\d{4,6})\s*원?')


def guess_category(menu):
    for kw in ALCOHOL:
        if kw in menu:
            return '주류'
    for kw in COFFEE_TEA + SOFT_DRINK + ADE_JUICE:
        if kw in menu:
            return '음료'
    for kw in GOODS_FEE:
        if kw in menu:
            return '기타·용품'
    for kw in MEAL:
        if kw in menu:
            return '식사·메인'
    for kw in SIDE:
        if kw in menu:
            return '사이드·안주'
    return '기타·용품'


def guess_season(menu):
    for kw in WINTER:
        if kw in menu:
            return '겨울'
    for kw in SUMMER:
        if kw in menu:
            return '여름'
    return '무관'


CATEGORY_PRICE_DEFAULT = {
    '주류': 6000,
    '음료': 4500,
    '식사·메인': 13000,
    '사이드·안주': 6000,
    '기타·용품': 2000,
}


def guess_price(menu, category):
    m = PRICE_PATTERNS.search(menu)
    if m:
        val = int(m.group(1).replace(',', ''))
        if 500 <= val <= 200000:
            return val
    # 카테고리 내에서 조금 더 세분화
    if category == '식사·메인':
        if any(k in menu for k in ['한우', '랍스타', '스테이크', '파스타', '리조또', 'BBQ', '양갈비']):
            return 26000
        if '단체' in menu or '패키지' in menu:
            return 32000
        return CATEGORY_PRICE_DEFAULT['식사·메인']
    if category == '주류':
        if any(k in menu for k in ['와인', 'Wine', 'Sileni', '서드']):
            return 14000
        if '막걸리' in menu:
            return 8500
        return CATEGORY_PRICE_DEFAULT['주류']
    return CATEGORY_PRICE_DEFAULT[category]


def main():
    sub = pd.read_csv(SUB_PATH, encoding='utf-8-sig')
    items = list(sub.columns[1:])
    assert len(items) == 193, f'품목 수 불일치: {len(items)}'

    records = []
    for key in items:
        store, menu = key.split('_', 1)
        category = guess_category(menu)
        season = guess_season(menu)
        price = guess_price(menu, category)
        records.append({
            'store': store,
            'menu': menu,
            'price_krw': price,
            'category': category,
            'season': season,
        })

    out_path = os.path.join(os.path.dirname(__file__), 'menu_labels.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'items': records}, f, ensure_ascii=False, indent=1)

    print(f'저장: {out_path} ({len(records)}개 품목)')
    cat_counts = pd.Series([r['category'] for r in records]).value_counts()
    season_counts = pd.Series([r['season'] for r in records]).value_counts()
    print('\ncategory 분포:')
    print(cat_counts)
    print('\nseason 분포:')
    print(season_counts)


if __name__ == '__main__':
    main()
