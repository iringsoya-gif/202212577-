# Week 4 — PySide6 인터랙티브 Neural Network 튜너 설계 및 구현 계획

> **작성일**: 2026-04-03
> **학번/이름**: 202212577 박성안
> **프로젝트**: Physics Neural Network Tuner — PySide6 인터랙티브 앱
> **경로**: `week4/`

---

## 1. 제작 배경 및 전체 흐름

### 이번 과제의 목표

> Week 4의 Python 스크립트 4개를 **PySide6 인터랙티브 앱**으로 설계하고 구현 계획을 수립

```
Week 4 Python 스크립트 4개
(01~04_*.py — TensorFlow/Keras로 물리 데이터 학습)
        ↓  "스크립트를 실행하면 결과만 나오고 파라미터를 바꾸려면 코드를 수정해야 한다"
        ↓  "파라미터를 실시간으로 조작하고 결과를 즉시 확인하는 인터랙티브 튜너를 만들자"

Brainstorming (목적·범위·구조 확정)
  - 인터랙티브 파라미터 튜너로 결정
  - 4개 Lab 모두 포함, 탭 구조
  - 백그라운드 학습 스레드 + 실시간 Loss 그래프
  - Matplotlib 앱 내 임베드
        ↓

Design Spec 작성  (무엇을 만들까)
  - 전체 레이아웃: 왼쪽 파라미터 패널 + 오른쪽 Loss/Result 캔버스
  - 아키텍처: models/ + lab_widgets/ + training_worker.py 분리
  - 공통 파라미터 + Lab별 전용 파라미터 명세
        ↓

Implementation Plan 작성  (어떻게 만들까)
  - 12개 Task, TDD 방식
  - models/ 레이어 → TrainingWorker → BaseLabWidget → 각 LabWidget → MainWindow
```

---

## 2. Week 4 실습 내용 요약

| Lab | 파일 | 내용 | 핵심 개념 |
|-----|------|------|----------|
| Lab 1 | `01perfect1d.py` | 1D 함수 근사 | Universal Approximation Theorem |
| Lab 2 | `02projectile.py` | 포물선 운동 회귀 | 물리 법칙을 데이터로 학습 |
| Lab 3 | `03overfitting.py` | 과적합 vs 과소적합 | 모델 복잡도와 일반화 성능 |
| Lab 4 | `04pendulum.py` | 진자 주기 예측 | 비선형 물리 법칙 + RK4 시뮬레이션 |

---

## 3. 앱 설계 (Design Spec)

### 3-1. 전체 레이아웃

```
┌─────────────────────────────────────────────────────────────┐
│  Week 4: Physics Neural Network Tuner                        │
├──────────┬──────────┬──────────┬──────────────────────────┤
│ Lab 1    │ Lab 2    │ Lab 3    │ Lab 4                      │ ← QTabWidget
├──────────┴──────────┴──────────┴──────────────────────────┤
│  [좌: 파라미터 패널 260px]   │  [우: 그래프 영역]           │
│                               │  ┌──────────────────────┐  │
│  공통 파라미터                 │  │  Training Loss 그래프  │  │
│  • Epochs (100~5000)          │  │  (실시간 업데이트)      │  │
│  • Learning Rate              │  └──────────────────────┘  │
│  • 모델 크기                  │  ┌──────────────────────┐  │
│                               │  │  Prediction Result   │  │
│  Lab 전용 파라미터             │  │  (학습 완료 후 표시)   │  │
│  (각 Lab마다 다름)             │  └──────────────────────┘  │
│                               │                              │
│  [▶ Train]  [Reset]           │                              │
├───────────────────────────────┴──────────────────────────────┤
│  Epoch 1847 / 3000  —  Loss: 0.00234  —  Val Loss: 0.00289  │ ← StatusBar
└─────────────────────────────────────────────────────────────┘
```

### 3-2. Lab별 전용 파라미터

| Lab | 전용 파라미터 |
|-----|-------------|
| Lab 1 (1D 함수 근사) | 대상 함수 (sin/cos+sin/x·sin), x 범위, Activation (tanh/relu) |
| Lab 2 (포물선 운동) | 초기 속력 v₀ (m/s), 발사 각도 θ (°), 노이즈 레벨 |
| Lab 3 (과적합) | 모델 복잡도 (Underfit/Good/Overfit), Dropout 비율, 학습 데이터 수 |
| Lab 4 (진자) | 진자 길이 L (m), 초기 각도 θ₀ (°), RK4 시뮬레이션 표시 여부 |

---

## 4. 아키텍처 설계

### 4-1. 파일 구조

```
week4/
├── app.py                    # 진입점, MainWindow (QMainWindow + QTabWidget)
├── training_worker.py        # TrainingWorker (QThread) + EpochSignalCallback
├── lab_widgets/
│   ├── __init__.py
│   ├── base_lab.py           # BaseLabWidget — 공통 UI + 추상 인터페이스
│   ├── lab1_widget.py        # Lab1: 1D 함수 근사 전용 파라미터
│   ├── lab2_widget.py        # Lab2: 포물선 운동 전용 파라미터
│   ├── lab3_widget.py        # Lab3: 과적합 전용 파라미터
│   └── lab4_widget.py        # Lab4: 진자 전용 파라미터
└── models/
    ├── __init__.py
    ├── lab1_model.py         # create_model, generate_1d_data (01perfect1d.py에서 추출)
    ├── lab2_model.py         # generate_projectile_data, predict_trajectory
    ├── lab3_model.py         # generate_overfit_data, create_complexity_model
    └── lab4_model.py         # generate_pendulum_data, simulate_pendulum_rk4
```

### 4-2. 신호 흐름 (Signal/Slot)

```
사용자가 파라미터 설정 → [Train] 버튼 클릭
        ↓
BaseLabWidget.on_train_clicked()
  params 수집 (공통 + Lab전용)
  TrainingWorker 생성 및 시작
        ↓
TrainingWorker.run()  ← QThread (백그라운드)
  generate_data(params) 호출
  build_model(params) 호출
  model.fit() + EpochSignalCallback
        ↓
  epoch_progress(epoch, loss, val_loss)  →  Loss 그래프 실시간 업데이트
  training_done(history, result)          →  Result 그래프 렌더링
  training_error(msg)                     →  QMessageBox 에러 표시
```

### 4-3. 설계 원칙

- **기존 스크립트 무수정**: `01~04_*.py`는 그대로 유지, `models/`에서 TF 로직만 추출
- **책임 분리**: models(TF 연산) / lab_widgets(UI) / training_worker(스레드) 완전 분리
- **BaseLabWidget 상속**: 공통 UI를 한 곳에서 관리, 각 Lab은 3개 메서드만 오버라이드

---

## 5. 구현 계획 (12개 Task)

| Task | 내용 | 산출물 |
|------|------|--------|
| 1 | 프로젝트 설정 | `pyproject.toml`에 `pyside6`, `pytest` 추가 |
| 2 | Lab1 모델 레이어 | `models/lab1_model.py` + 테스트 6개 |
| 3 | Lab2 모델 레이어 | `models/lab2_model.py` + 테스트 6개 |
| 4 | Lab3 모델 레이어 | `models/lab3_model.py` + 테스트 7개 |
| 5 | Lab4 모델 레이어 | `models/lab4_model.py` + 테스트 6개 |
| 6 | TrainingWorker | `training_worker.py` + 테스트 3개 |
| 7 | BaseLabWidget | `lab_widgets/base_lab.py` |
| 8 | Lab1Widget | `lab_widgets/lab1_widget.py` |
| 9 | Lab2Widget | `lab_widgets/lab2_widget.py` |
| 10 | Lab3Widget | `lab_widgets/lab3_widget.py` |
| 11 | Lab4Widget | `lab_widgets/lab4_widget.py` |
| 12 | MainWindow | `app.py` + 최종 실행 확인 |

**총 테스트**: ~28개 (모델 레이어 25개 + 워커 3개), TDD 방식

---

## 6. 핵심 학습 내용

### Neural Network로 물리 법칙 학습

이번 과제를 통해 Neural Network가 단순한 분류 문제를 넘어 **물리 법칙을 데이터로부터 학습**할 수 있음을 확인했다.

| 물리 현상 | 입력 | 출력 | 핵심 발견 |
|----------|------|------|----------|
| 1D 함수 근사 | x | f(x) | Universal Approximation: tanh 활성화가 주기 함수에 유리 |
| 포물선 운동 | v₀, θ, t | x, y | 물리 공식 없이도 데이터만으로 궤적 학습 가능 |
| 과적합 데모 | x | y | 모델 복잡도 > 데이터 복잡도이면 노이즈까지 학습 |
| 진자 주기 | L, θ₀ | T | 비선형 √L 관계와 큰 각도 보정항을 데이터로 학습 |

### 인터랙티브 앱의 교육적 가치

- 파라미터를 바꿀 때마다 코드를 수정하지 않아도 되어 **실험 속도 향상**
- 실시간 Loss 그래프로 **학습 과정을 직관적으로 이해**
- 과적합/과소적합 경계를 슬라이더로 탐색하며 **모델 복잡도의 영향을 체감**

---

## 7. 참고 문서

| 문서 | 경로 | 내용 |
|------|------|------|
| 설계 스펙 | `docs/superpowers/specs/2026-04-01-week4-pyside6-tuner-design.md` | 컴포넌트·데이터 흐름·파라미터 명세 |
| 구현 플랜 | `docs/superpowers/plans/2026-04-01-week4-pyside6-tuner.md` | 12개 Task 상세 코드 및 테스트 |
| Week4 실습 문서 | `week4/week4.md` | 4개 Lab 이론 및 실험 내용 |

---

---

## 8. 구현 결과

### 실행 방법

```bash
cd C:/Users/USER/AIandMLcourse
uv run python week4/app.py
```

### 테스트 결과

```bash
uv run pytest week4/tests/ -v
# 결과: 28 passed
```

| 테스트 파일 | 통과 수 |
|------------|--------|
| test_lab1_model.py | 6/6 |
| test_lab2_model.py | 6/6 |
| test_lab3_model.py | 7/7 |
| test_lab4_model.py | 6/6 |
| test_training_worker.py | 3/3 |
| **합계** | **28/28** |

### 구현된 파일 목록

```
week4/
├── app.py                        ← 진입점 (MainWindow)
├── training_worker.py            ← QThread 백그라운드 학습
├── lab_widgets/
│   ├── base_lab.py               ← 공통 UI (파라미터 패널 + 캔버스)
│   ├── lab1_widget.py            ← 1D 함수 근사
│   ├── lab2_widget.py            ← 포물선 운동
│   ├── lab3_widget.py            ← 과적합 데모
│   └── lab4_widget.py            ← 진자 주기
├── models/
│   ├── lab1_model.py             ← TF 모델 + 데이터 생성
│   ├── lab2_model.py
│   ├── lab3_model.py
│   └── lab4_model.py
└── tests/                        ← 28개 단위 테스트
```

---

**작성자**: 202212577 박성안
**버전**: 2.0 (구현 완료)
**최종 수정**: 2026-04-03
