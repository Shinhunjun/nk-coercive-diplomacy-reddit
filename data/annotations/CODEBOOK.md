# Framing Annotation Codebook

**Version:** 1.1 (Theory-Grounded)  
**Last Updated:** 2024-12-28  
**Status:** In Progress (Batch 1 완료)

---

## 이론적 기반 (Theoretical Foundation)

본 분류 체계는 다음 세 가지 국제정치학 이론에 기반합니다:

### 1. Coercive Diplomacy Theory (George, 1991)
>
> "The distinction between **brute force** and **coercive diplomacy** lies in whether physical military action has been taken."

| 개념 | 정의 | 우리 분류 |
|------|------|----------|
| **Brute Force** | 실제 군사적 행동 (미사일 발사, 핵실험 등) | → **THREAT** |
| **Coercive Diplomacy** | 군사력 *위협*을 통한 외교적 압박 | → **DIPLOMACY** |

### 2. Escalation Ladder (Kahn, 1965)
>
> 갈등은 단계적으로 고조되며, 각 단계는 질적으로 다른 행위를 포함한다.

| 단계 | 행위 유형 | 우리 분류 |
|------|----------|----------|
| 낮은 단계 | 외교적 항의, 비난, 유감 표명 | → **DIPLOMACY** |
| 중간 단계 | 경제 제재, 무역 조치 | → **ECONOMIC** |
| 높은 단계 | 군사적 시위, 무력 사용 | → **THREAT** |

### 3. Securitization Theory (Buzan et al., 1998)
>
> 언어적 행위(speech act)가 이슈를 "안보 위협"으로 규정하는지 여부

- **Securitizing Move** (위협으로 규정하는 담론) → 맥락에 따라 THREAT 또는 DIPLOMACY
- **핵심:** 실제 군사적 *조치*가 있는지가 최종 판단 기준

---

## 분류 결정 흐름도 (Decision Flowchart)

```
[포스트 읽기]
     ↓
[1단계] 군사적 *조치*가 언급되었는가?
     ├── YES: 미사일 발사, 핵실험, 무기 배치, 군사훈련 → THREAT
     └── NO: 아래로 ↓

[2단계] 외교적 *조치*가 언급되었는가?
     ├── YES: 회담, 협상, 합의, 비난, 유감 표명 → DIPLOMACY
     └── NO: 아래로 ↓

[3단계] 경제적 *조치*가 언급되었는가?
     ├── YES: 제재, 무역, 경제 조치 → ECONOMIC
     └── NO: 아래로 ↓

[4단계] 인권/인도주의 이슈인가?
     ├── YES: 인권, 난민, 인도 지원 → HUMANITARIAN
     └── NO: → NEUTRAL
```

---

## 핵심 원칙 (Core Principles)

### 원칙 1: 국가 중립적 관점

> **"특정 국가의 입장에서 생각하지 않는다."**

- ❌ 서방 국가 입장에서 "이 행동이 위협이 되는가?"를 판단하지 않음
- ✅ **행위의 성격** 자체를 판단: 군사적 조치인가, 외교적 조치인가?

### 원칙 2: 조치(Action) 기반 분류

> **"레토릭이 아닌 실제 조치의 성격으로 판단한다."**

- 강한 비난 + 군사적 조치 없음 → **DIPLOMACY**
- 약한 표현 + 미사일 발사 → **THREAT**

**George (1991)의 핵심 구분:** *"물리적 군사 행동이 발생했는가?"*

---

## 프레임별 정의 및 예시

### THREAT (위협)

**이론적 근거:** George의 "Brute Force", Kahn의 높은 에스컬레이션 단계

**정의:** 실제 군사적 행동 또는 구체적 군사적 조치가 명시된 경우

**Keywords:**

- 미사일 발사/시험
- 핵실험, 핵무장
- 군사훈련, 군 배치
- 무력 시위

**예시:**

- "North Korea launches ballistic missile over Japan" → THREAT
- "US deploys carrier strike group to Korean peninsula" → THREAT

---

### DIPLOMACY (외교)

**이론적 근거:** George의 "Coercive Diplomacy", Kahn의 낮은 에스컬레이션 단계

**정의:** 대화, 협상, 또는 군사적 조치가 없는 외교적 압박/비난

**Keywords:**

- 회담, 협상, 합의, 대화
- 비난, 유감 표명, 성명
- 관계 중단/재개 선언

**예시:**

- "North Korea rejects dialogue pledge, says will never have talks" → DIPLOMACY (입장 표명)
- "Trump calls Kim 'rocket man' at UN speech" → DIPLOMACY (비난, 군사조치 없음)
- "Singapore Summit produces joint declaration" → DIPLOMACY

---

### NEUTRAL (중립)

**정의:** 단순 보도, 사실 전달, 분석 기사. 특정 프레임으로 분류 어려움.

**예시:**

- "Kim Jong-un's health reportedly deteriorating" → NEUTRAL
- "Analysis: What does Kim's new year speech mean?" → NEUTRAL

---

### ECONOMIC (경제)

**정의:** 경제 제재, 무역, 금융 조치 관련

**예시:**

- "UN imposes new sanctions on North Korea" → ECONOMIC
- "China restricts coal imports from North Korea" → ECONOMIC

---

### HUMANITARIAN (인도주의)

**정의:** 인권, 난민, 인도주의적 지원 관련

**예시:**

- "Report documents human rights abuses in North Korea" → HUMANITARIAN
- "Food aid arrives in North Korea" → HUMANITARIAN

---

## IRR 결과 및 코드북 발전 과정

| Batch | 샘플 | 일치율 | Cohen's κ | 비고 |
|-------|------|--------|-----------|------|
| Pilot | 100 | 76.0% | 0.688 | 초기 기준, 불명확한 케이스 다수 |
| Batch 1 | 197 | 88.3% | 0.842 | 이론 기반 기준 적용 후 |

**개선 요인:** George (1991)의 "군사적 조치 여부" 기준 명확화

---

## TODO

- [ ] Batch 2~6 어노테이션 완료
- [ ] 최종 코드북 확정
- [ ] Method 섹션 반영
