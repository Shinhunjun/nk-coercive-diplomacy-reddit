# Hanoi Summit 3-Period DID Analysis Report

**분석일**: 2025-12-12  
**연구 질문**: 싱가포르 회담과 하노이 회담 결렬이 북한에 대한 Reddit 여론 framing에 미친 영향

---

## 분석 기간 정의

| Period | 기간 | 설명 | NK Posts |
|--------|------|------|----------|
| **1. Pre-Singapore** | 2017.01 ~ 2018.05 | 싱가포르 회담 전 (긴장기) | 5,822 |
| **2. Singapore-Hanoi** | 2018.06 ~ 2019.02 | 싱가포르 ~ 하노이 회담 | 1,699 |
| **3. Post-Hanoi** | 2019.03 ~ 2019.06 | 하노이 결렬 후 | 1,486 |

**Control Group**: China  
**Treatment Group**: North Korea  
**종속변수**: Framing Score (GPT-4o-mini 분류, -2 ~ +2 척도)

---

## 주요 결과

### 1. Difference-in-Differences 분석

| 비교 | DID 추정치 | P-value | Cohen's d | 유의성 |
|------|-----------|---------|-----------|--------|
| **싱가포르 효과** (P1→P2) | **+0.959** | **0.002** | **1.645** | ✅ **유의미** |
| 하노이 붕괴 효과 (P2→P3) | -0.388 | 0.320 | -0.697 | ❌ 비유의미 |
| 전체 변화 (P1→P3) | +0.571 | 0.122 | 1.021 | ❌ 비유의미 |

### 2. NK Framing 변화 (기간별 평균)

```
Period 1 (Pre-Singapore):    -0.69 (부정적)
Period 2 (Singapore-Hanoi):  +0.04 (중립)     → +0.73 개선
Period 3 (Post-Hanoi):       +0.05 (중립)     → 거의 변화 없음
```

### 3. T-test 결과 (NK 단독)

| 비교 | T-statistic | P-value | 결과 |
|------|-------------|---------|------|
| Period 1 vs 2 | -2.592 | **0.016** | ✅ 유의미한 개선 |
| Period 2 vs 3 | -0.036 | 0.972 | ❌ 차이 없음 |
| Period 1 vs 3 | -1.935 | 0.068 | ⚠️ 경계선 |

---

## 해석

### ✅ 싱가포르 회담 효과 (2018년 6월)

- **매우 강한 긍정적 효과** (Cohen's d = 1.645, Large Effect)
- NK framing이 -0.69 → +0.04로 크게 개선
- China 대비 훨씬 큰 개선폭 (+0.96점 DID)
- 통계적으로 매우 유의미 (p = 0.002)

### ❓ 하노이 회담 결렬 효과 (2019년 2월)

- 현재 데이터에서는 **급격한 하락 미발생**
- NK framing 유지: +0.04 → +0.05
- **주의**: Post-Hanoi 기간이 4개월에 불과하여 통계적 검정력 부족 가능

### 📋 추가 분석 필요

- 2019년 7-12월 데이터 수집으로 하노이 결렬의 **장기 효과** 확인 필요
- 긍정적 framing이 유지되는지, 아니면 지연된 붕괴(delayed crash)가 있는지 검증

---

## 파일 구조

```
scripts/
├── hanoi_3period_did_analysis.py    # 3-period DID 분석 스크립트
└── collect_nk_hanoi_extended.py     # 2019년 7-12월 데이터 수집 스크립트

data/results/
└── hanoi_3period_did_results.json   # 분석 결과 (JSON)
```

---

## 다음 단계

1. [ ] 2019년 7-12월 NK 데이터 수집
2. [ ] 확장된 데이터로 4-period 분석 또는 ITS 분석
3. [ ] 시각화 (월별 framing 변화 그래프)
