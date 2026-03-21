# Week 2 — 실습 결과 총정리

> **작성일**: 2026-03-21
> **구성**: Python 스크립트 4개 + FastAPI 웹앱 4개

---

## 목차

1. [Python 스크립트 실행 결과](#1-python-스크립트-실행-결과)
   - [01 — 선형 회귀 (용수철 법칙)](#01--선형-회귀-용수철-법칙)
   - [02 — 비지도 학습 (K-Means 군집화)](#02--비지도-학습-k-means-군집화)
   - [03 — 데이터 전처리 (정규화)](#03--데이터-전처리-정규화)
   - [04 — 경사 하강법](#04--경사-하강법)
2. [FastAPI 웹앱 제작 결과](#2-fastapi-웹앱-제작-결과)
   - [LinRegSpr — 훅의 법칙 TF 신경망](#linregspr--훅의-법칙-tf-신경망-시각화)
   - [KMeansCluster — K-Means 군집화](#kmeansclusterm--k-means-군집화-시각화)
   - [DataPrep — 데이터 전처리](#dataprep--데이터-전처리-시각화)
   - [GradDesc — 경사 하강법](#graddesc--경사-하강법-시각화)
3. [파일 구조](#3-파일-구조)
4. [핵심 개념 요약](#4-핵심-개념-요약)

---

## 1. Python 스크립트 실행 결과

### 01 — 선형 회귀 (용수철 법칙)

**파일**: `01_linear_regression_spring.py`
**라이브러리**: TensorFlow 2.21.0, NumPy, Matplotlib
**주제**: 훅의 법칙(F = kx) — AI가 공식 없이 데이터만 보고 스스로 회귀식 학습

#### 실험 설정
| 항목 | 값 |
|------|-----|
| 초기 길이 | 10 cm |
| 1 kg당 늘어남 | 2 cm |
| **실제 공식** | `길이 = 2.00 × 무게 + 10.00` |
| 노이즈 표준편차 | 1.5 cm |

#### 실행 결과
```
TensorFlow Version: 2.21.0

[데이터 확인]
무게(kg): [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10.]
측정된 길이(cm): [10.75 11.79 14.97 18.28 17.65 19.65 24.37 25.15 25.30 28.81 29.30]

[학습 완료]
예측된 식: 길이 = 2.02 × 무게 + 10.27
실제 식  : 길이 = 2.00 × 무게 + 10.00

[예측 테스트]
15 kg 추를 매달았을 때 예측 길이: 40.58 cm
이론상 실제 길이              : 40.00 cm

출력 파일: outputs/spring_fitting.png
```

#### 학습 결과 분석
| | 기울기 (w) | 절편 (b) |
|--|:----------:|:--------:|
| **AI가 학습한 값** | **2.02** | **10.27** |
| 실제 값 | 2.00 | 10.00 |
| 오차 | +1.0% | +0.27 cm |

> **결론**: AI가 노이즈가 섞인 데이터 11개만 보고 실제 공식에 99% 수준으로 근접하게 학습 성공

#### 출력 이미지

![선형 회귀 — 용수철 피팅](./outputs/spring_fitting.png)

---

### 02 — 비지도 학습 (K-Means 군집화)

**파일**: `02_unsupervised_clustering.py`
**라이브러리**: NumPy, Matplotlib
**주제**: 정답 없이 고객 데이터를 3개 그룹으로 자동 분류

#### 데이터 설정
| 항목 | 값 |
|------|-----|
| 전체 데이터 수 | 90개 |
| 그룹 수 | 3개 |
| 그룹 1 중심 | (2, 2) |
| 그룹 2 중심 | (8, 3) |
| 그룹 3 중심 | (5, 8) |

#### 실행 결과
```
=== 비지도 학습 (Unsupervised Learning) 예제: 군집화 (Clustering) ===
데이터 개수: 90개

[초기 중심점]
[[5.411  8.948]
 [4.513  8.394]
 [5.701  7.299]]

[학습 완료된 중심점]
[[5.198  8.632]
 [4.829  7.835]
 [4.937  2.483]]

출력 파일: outputs/02_clustering.png
```

#### 중심점 수렴 결과
| 클러스터 | 초기 중심 | 최종 중심 | 이동 거리 |
|:--------:|:---------:|:---------:|:---------:|
| C1 | (5.41, 8.95) | (5.20, 8.63) | ~0.38 |
| C2 | (4.51, 8.39) | (4.83, 7.84) | ~0.64 |
| C3 | (5.70, 7.30) | (4.94, 2.48) | **~4.83** |

> **결론**: C3 중심점이 4.83 이동하며 가장 큰 변화 — 초기 중심이 그룹 3 영역과 멀었으나 수렴 성공

#### 출력 이미지

![K-Means 군집화](./outputs/02_clustering.png)

---

### 03 — 데이터 전처리 (정규화)

**파일**: `03_data_preprocessing.py`
**라이브러리**: NumPy, Matplotlib
**주제**: 단위가 극단적으로 다른 연봉(원)과 나이(세)를 Min-Max Scaling으로 0~1 변환

#### 데이터 설정
| 특성 | 원본 범위 | 변환 후 범위 |
|------|:--------:|:-----------:|
| 연봉 | 31,440,915 ~ 97,893,690 원 | 0.0 ~ 1.0 |
| 나이 | 20 ~ 59 세 | 0.0 ~ 1.0 |

#### 실행 결과
```
=== 데이터 전처리 (Data Preprocessing) 예제: 정규화 (Normalization) ===

[데이터 비교]
연봉(원본): 최소 31,440,915, 최대 97,893,690
연봉(변환): 최소 0.0, 최대 1.0
------------------------------
나이(원본): 최소 20, 최대 59
나이(변환): 최소 0.0, 최대 1.0

출력 파일: outputs/03_preprocessing.png
```

#### Min-Max 공식
```
x' = (x - x_min) / (x_max - x_min)
```

> **결론**: 6,600만 원 차이 vs. 39세 차이 → 스케일 통일 후 머신러닝 모델 공정하게 학습 가능

#### 출력 이미지

![데이터 전처리](./outputs/03_preprocessing.png)

---

### 04 — 경사 하강법

**파일**: `04_gradient_descent_vis.py`
**라이브러리**: NumPy, Matplotlib
**주제**: f(x) = x² 손실 함수에서 최솟값(x=0)을 경사 하강법으로 탐색

#### 설정
| 항목 | 값 |
|------|-----|
| 손실 함수 | f(x) = x² |
| 미분 (기울기) | f'(x) = 2x |
| 시작 위치 | x = -4.0 |
| 학습률 (α) | 0.1 |
| 업데이트 규칙 | x := x - α × 2x |

#### 실행 결과 (전체 20스텝)
```
=== 최적화 예제: 경사 하강법 (Gradient Descent) ===
시작 위치: x = -4.0

Step  1: x = -3.2000, Loss = 16.0000
Step  2: x = -2.5600, Loss = 10.2400
Step  3: x = -2.0480, Loss =  6.5536
Step  4: x = -1.6384, Loss =  4.1943
Step  5: x = -1.3107, Loss =  2.6844
Step  6: x = -1.0486, Loss =  1.7180
Step  7: x = -0.8389, Loss =  1.0995
Step  8: x = -0.6711, Loss =  0.7037
Step  9: x = -0.5369, Loss =  0.4504
Step 10: x = -0.4295, Loss =  0.2882
Step 11: x = -0.3436, Loss =  0.1845
Step 12: x = -0.2749, Loss =  0.1181
Step 13: x = -0.2199, Loss =  0.0756
Step 14: x = -0.1759, Loss =  0.0484
Step 15: x = -0.1407, Loss =  0.0309
Step 16: x = -0.1126, Loss =  0.0198
Step 17: x = -0.0901, Loss =  0.0127
Step 18: x = -0.0721, Loss =  0.0081
Step 19: x = -0.0576, Loss =  0.0052
Step 20: x = -0.0461, Loss =  0.0033

최종 위치: x = -0.0461 (목표값 0.0에 매우 가까움)
출력 파일: outputs/04_gradient_descent.png
```

#### 수렴 분석
| 단계 | x 값 | Loss | 누적 감소율 |
|:----:|:-----:|:----:|:-----------:|
| 시작 | -4.000 | 16.00 | — |
| Step 5 | -1.311 | 2.68 | -83.2% |
| Step 10 | -0.430 | 0.29 | -98.2% |
| Step 20 | -0.046 | 0.003 | **-99.98%** |

> **결론**: 20스텝만에 Loss를 16.0 → 0.003 으로 **99.98% 감소**, 매 스텝 0.64배씩 지수적 감소

#### 출력 이미지

![경사 하강법](./outputs/04_gradient_descent.png)

---

## 2. FastAPI 웹앱 제작 결과

> **공통 기술 스택**: Tailwind CSS + FastAPI + matplotlib (Agg) + pure NumPy
> **설계 방법론**: bkit PDCA (Plan → Design → Do → Check → Act)
> **공통 UI 테마**: 다크 (#0f0f1a / #1a1a2e), 3-column 12-col grid

---

### LinRegSpr — 훅의 법칙 TF 신경망 시각화

| 항목 | 내용 |
|------|------|
| 경로 | `week2/LinRegSpr/` |
| 포트 | 8769 |
| 알고리즘 | TensorFlow Dense(1) 선형 회귀 |
| 물리 법칙 | F = k·x → x = (m × g) / k |
| 설정 | k = 50 N/m, g = 9.81 m/s², n=200 |

#### API 엔드포인트
| Method | URL | 기능 |
|--------|-----|------|
| `POST` | `/train` | TF 모델 학습 + 3개 PNG 생성 |
| `POST` | `/predict` | 질량 입력 → 늘어난 길이 예측 |
| `GET` | `/health` | 서버 상태 / 모델 학습 여부 |

#### 학습 결과 (epochs=500, lr=0.01, n=200)
```
epochs_actual        : 152  (EarlyStopping 조기 종료)
final_loss (MSE)     : 1.85e-05
R² Score             : 0.9997
MAE                  : 0.362 cm
learned_k            : 49.885 N/m  (실제: 50.0 N/m)
k 상대 오차           : 0.23 %
weight_w             : 0.19665  (이론: g/k = 0.1962)
bias_b               : -0.00162  (이론: 0)
```

#### 예측 테스트 결과 (질량 3.0 kg)
```
predicted_extension  : 58.834 cm
true_extension       : 58.860 cm
error_percent        : 0.045 %
```

#### 생성 시각화 (PNG 4개)
| 파일 | 내용 |
|------|------|
| `loss_curve.png` | Train/Val Loss 곡선 (선형 + 로그 스케일 2패널) |
| `predictions.png` | 4패널 분석 (메인 피팅, 잔차, 예측 vs 실제, 잔차 히스토그램) |
| `spring_diagram.png` | 용수철 물리 다이어그램 + 모델 성능 지표 테이블 |
| `single_prediction.png` | 단일 예측 시각화 + 모델 vs 실제 막대 비교 |

#### UI 주요 기능
- Epochs / Learning Rate / 샘플 수 슬라이더 조절 후 Train 버튼
- 학습 결과: R², MSE, MAE, 학습된 k값, w, b 실시간 표시
- 질량(kg) 입력 → 늘어난 길이(cm) 예측 + 오차율
- 4개 탭: Loss Curve / Model Analysis / Spring Diagram / Prediction

#### 출력 이미지

**Loss Curve (Train / Validation)**
![Loss Curve](./LinRegSpr/output/loss_curve.png)

**4-Panel Analysis (피팅 + 잔차 + 산점도 + 히스토그램)**
![Predictions Analysis](./LinRegSpr/output/predictions.png)

**Spring Diagram + Model Report**
![Spring Diagram](./LinRegSpr/output/spring_diagram.png)

**Single Prediction (3.0 kg)**
![Single Prediction](./LinRegSpr/output/single_prediction.png)

---

### KMeansCluster — K-Means 군집화 시각화

| 항목 | 내용 |
|------|------|
| 경로 | `week2/KMeansCluster/` |
| 포트 | 8766 |
| 설계 일치율 | **99%** (2회 PDCA 반복) |
| 알고리즘 | Pure NumPy K-Means (Corner-spread 초기화) |

#### API 엔드포인트
| Method | URL | 기능 |
|--------|-----|------|
| `POST` | `/api/cluster` | K-Means 실행 + 4개 PNG 생성 |
| `POST` | `/api/predict` | 신규 고객 클러스터 예측 |
| `GET` | `/api/wcss` | WCSS 수렴 히스토리 반환 |
| `GET` | `/api/health` | 서버 상태 확인 |

#### UI 주요 기능
- 고객 수 슬라이더 (30~300명), K값 슬라이더 (2~6)
- 클러스터 자동 명칭 배정 (Regular / Premium / VIP)
- 클러스터 크기 비율 바 차트 + WCSS 스파크라인
- 신규 고객 구매금액·방문횟수 입력 → 클러스터 예측

#### 출력 이미지

**Cluster Plot (산점도 + 중심점)**
![Cluster Plot](./KMeansCluster/output/cluster_plot.png)

**WCSS 수렴 곡선 (반복별)**
![WCSS Plot](./KMeansCluster/output/wcss_plot.png)

**K값 선택 Elbow 그래프**
![K Selection](./KMeansCluster/output/k_selection_plot.png)

**신규 고객 예측 결과**
![Prediction Plot](./KMeansCluster/output/prediction_plot.png)

---

### DataPrep — 데이터 전처리 시각화

| 항목 | 내용 |
|------|------|
| 경로 | `week2/DataPrep/` |
| 포트 | 8767 |
| 설계 일치율 | **99%** (2회 PDCA 반복) |
| 알고리즘 | Pure NumPy Min-Max Scaling + Z-Score |

#### API 엔드포인트
| Method | URL | 기능 |
|--------|-----|------|
| `POST` | `/api/normalize` | 정규화 실행 + 5개 PNG 생성 |
| `POST` | `/api/predict` | 연봉·나이 입력 → 정규화값 예측 |
| `GET` | `/api/health` | 서버 상태 확인 |

#### UI 주요 기능
- 정규화 방법 선택 (Min-Max / Z-Score / Both)
- 직급 빠른 입력 프리셋 (신입 / 주니어 / 시니어 / 임원)
- 퍼센타일 바, 공식 문자열 실시간 표시

#### 출력 이미지

**원본 데이터 분포 (히스토그램 + 산점도)**
![Raw Distribution](./DataPrep/output/raw_distribution.png)

**Min-Max vs Z-Score 정규화 비교**
![Normalized Distribution](./DataPrep/output/normalized_distribution.png)

**정규화 전·후 나란히 비교**
![Comparison](./DataPrep/output/comparison_plot.png)

**분산 비교 (Raw / Min-Max / Z-Score)**
![Variance Comparison](./DataPrep/output/variance_comparison.png)

**입력값 예측 위치 시각화**
![Prediction](./DataPrep/output/prediction_plot.png)

---

### GradDesc — 경사 하강법 시각화

| 항목 | 내용 |
|------|------|
| 경로 | `week2/GradDesc/` |
| 포트 | 8768 |
| 설계 일치율 | **98%** |
| 알고리즘 | Pure NumPy Gradient Descent on f(x) = x² |

#### API 엔드포인트
| Method | URL | 기능 |
|--------|-----|------|
| `POST` | `/api/simulate` | GD 실행 + 3개 PNG 생성 + step 데이터 반환 |
| `GET` | `/api/presets` | 4개 사전 설정 반환 |
| `GET` | `/api/health` | 서버 상태 확인 |

#### 파라미터 검증 (Pydantic)
| 파라미터 | 기본값 | 범위 |
|----------|:------:|:----:|
| x_start | 8.0 | -20 ~ +20 |
| learning_rate (α) | 0.1 | 0 초과 ~ 2.0 |
| max_steps | 100 | 5 ~ 500 |
| tolerance | 1e-8 | — |

#### 시뮬레이션 결과 (α=0.1, x₀=8.0)
```
n_steps       : 86
converged     : True
final_x       : 3.71e-08  (≈ 0)
final_loss    : ~0.0
convergence%  : 100.0%
```

#### UI 주요 기능
- x₀ / α 슬라이더 (α 색상 코딩: teal→amber→red), α > 1.0 발산 경고 배너
- 4개 프리셋: Too Slow (0.01) / Optimal (0.1) / Fast (0.45) / Diverge (1.05)
- 실시간 공식 카드 (x₀ → x₁ 계산), Step 상세 테이블 (최대 50행)

#### 출력 이미지

**경사 하강법 경로 (포물선 위 이동 시각화)**
![GD Path](./GradDesc/output/gd_path.png)

**Loss 수렴 곡선 + Gradient 크기**
![Loss Curve](./GradDesc/output/loss_curve.png)

**학습률 4개 비교 (Too Slow / Optimal / Fast / Diverge)**
![Learning Rate Comparison](./GradDesc/output/comparison.png)

---

## 3. 파일 구조

```
week2/
├── 01_linear_regression_spring.py     # TF 선형회귀 (훅의 법칙) 스크립트
├── 02_unsupervised_clustering.py      # NumPy K-Means 군집화 스크립트
├── 03_data_preprocessing.py           # NumPy Min-Max 정규화 스크립트
├── 04_gradient_descent_vis.py         # NumPy 경사 하강법 스크립트
│
├── outputs/                           # 스크립트 출력 PNG
│   ├── spring_fitting.png             # 01 결과
│   ├── 02_clustering.png              # 02 결과
│   ├── 03_preprocessing.png           # 03 결과
│   └── 04_gradient_descent.png        # 04 결과
│
├── LinRegSpr/                         # 웹앱 01 — TF 훅의 법칙 시각화
│   ├── model.py                       # HookesLawModel (TF Dense + 4개 plot)
│   ├── main.py                        # FastAPI: /train /predict /health
│   ├── static/index.html              # Tailwind CSS 대시보드
│   └── output/                        # loss_curve / predictions / spring_diagram / single_prediction
│
├── KMeansCluster/                     # 웹앱 02 — K-Means 군집화 시각화
│   ├── kmeans.py                      # Pure NumPy K-Means 알고리즘
│   ├── visualizer.py                  # matplotlib 4개 플롯
│   ├── main.py                        # FastAPI: /api/cluster /predict /wcss
│   ├── static/index.html              # Tailwind CSS 대시보드
│   ├── output/                        # cluster / wcss / k_selection / prediction
│   └── docs/                          # PDCA 문서 (plan/design/analysis)
│
├── DataPrep/                          # 웹앱 03 — 데이터 전처리 시각화
│   ├── preprocessing.py               # MinMaxScaler / StandardScaler (pure numpy)
│   ├── visualizer.py                  # matplotlib 5개 플롯
│   ├── main.py                        # FastAPI: /api/normalize /predict
│   ├── static/index.html              # Tailwind CSS 대시보드
│   ├── output/                        # raw / normalized / comparison / variance / prediction
│   └── docs/                          # PDCA 문서
│
├── GradDesc/                          # 웹앱 04 — 경사 하강법 시각화
│   ├── gradient_descent.py            # GDStep / GDResult / run_gradient_descent()
│   ├── visualizer.py                  # matplotlib 3개 플롯
│   ├── main.py                        # FastAPI: /api/simulate /presets
│   ├── static/index.html              # Tailwind CSS 대시보드
│   ├── output/                        # gd_path / loss_curve / comparison
│   └── docs/                          # PDCA 문서 (plan/design/analysis)
│
├── README000.md                       # Week2 Lab4 선형 회귀 상세 설명
└── README001.md                       # 이 파일 — 전체 결과 정리
```

---

## 4. 핵심 개념 요약

### 스크립트 vs 웹앱 비교

| # | 주제 | 스크립트 | 웹앱 | 알고리즘 |
|---|------|----------|------|----------|
| 01 | 선형 회귀 (훅의 법칙) | `01_linear_regression_spring.py` | `LinRegSpr/` | TensorFlow Dense(1) |
| 02 | K-Means 군집화 | `02_unsupervised_clustering.py` | `KMeansCluster/` | Pure NumPy K-Means |
| 03 | 데이터 전처리 | `03_data_preprocessing.py` | `DataPrep/` | Pure NumPy Min-Max / Z-Score |
| 04 | 경사 하강법 | `04_gradient_descent_vis.py` | `GradDesc/` | Pure NumPy GD |

### 알고리즘별 핵심 공식

| 알고리즘 | 핵심 공식 | 주요 결과 |
|----------|-----------|-----------|
| TF 선형 회귀 | `ŷ = w·x + b` | w=0.19665, b≈0, k_learned=49.885 N/m (오차 0.23%) |
| K-Means | `argmin Σ‖xᵢ - μₖ‖²` | 3클러스터 완전 분리, WCSS 수렴 확인 |
| Min-Max | `x' = (x - xₘᵢₙ) / (xₘₐₓ - xₘᵢₙ)` | 연봉 6.6천만 범위 → [0, 1] |
| Z-Score | `x' = (x - μ) / σ` | 평균 0, 표준편차 1로 변환 |
| 경사 하강법 | `x := x - α · 2x` | 86스텝, loss 100% 감소 (64 → ~0) |

### 웹앱 PDCA 설계 일치율

| 웹앱 | PDCA 반복 | 일치율 |
|------|:---------:|:------:|
| LinRegSpr | — | — |
| KMeansCluster | 2회 | **99%** ✅ |
| DataPrep | 2회 | **99%** ✅ |
| GradDesc | 1회 | **98%** ✅ |

### 서버 실행 방법

```bash
# 웹앱 01 — LinRegSpr (훅의 법칙 TF)
cd week2/LinRegSpr
..\..\..\.venv\Scripts\uvicorn main:app --host 127.0.0.1 --port 8769

# 웹앱 02 — KMeansCluster
cd week2/KMeansCluster
..\..\..\.venv\Scripts\uvicorn main:app --host 127.0.0.1 --port 8766

# 웹앱 03 — DataPrep
cd week2/DataPrep
..\..\..\.venv\Scripts\uvicorn main:app --host 127.0.0.1 --port 8767

# 웹앱 04 — GradDesc
cd week2/GradDesc
..\..\..\.venv\Scripts\uvicorn main:app --host 127.0.0.1 --port 8768
```

| 웹앱 | 접속 주소 |
|------|-----------|
| LinRegSpr | http://127.0.0.1:8769 |
| KMeansCluster | http://127.0.0.1:8766 |
| DataPrep | http://127.0.0.1:8767 |
| GradDesc | http://127.0.0.1:8768 |
