# Week 3 — NeuralNetLab 제작 결과 정리

> **작성일**: 2026-03-31
> **프로젝트**: NeuralNetLab — 신경망 기초 학습 시각화 앱
> **경로**: `week3/NeuralNetLab/`

---

## 1. 제작 배경 및 전체 흐름

### NeuralNetLab이란?

> Week 3의 Python 스크립트 5개를 **PRD/TRD로 설계**하고 **PySide6로 구현**한 결과물

```
Week 3 Python 스크립트 5개
(01~05_*.py — NumPy로 직접 구현한 신경망 기초 실습)
        ↓  "코드 실행 결과만 보는 것은 한계가 있다"
        ↓  "인터랙티브 데스크톱 앱으로 만들자"

PRD.md 작성  (무엇을 만들까)
  - 5탭 구조 정의
  - 슬라이더·애니메이션·실시간 시각화 기능 명세
  - 우선순위 P0 / P1 / P2 분류
        ↓

TRD.md 작성  (어떻게 만들까)
  - PySide6 클래스 설계 (QTimer, QThread, QPainter)
  - 파일 구조 설계 (core / ui / widgets 분리)
  - 구현 순서 Phase 1~5 계획
        ↓

PySide6 + matplotlib + NumPy로 구현
        ↓

NeuralNetLab  ← 최종 결과물
(week3/NeuralNetLab/)
```

### PRD와 TRD의 역할

| 문서 | 질문 | 주요 내용 |
|------|------|----------|
| **PRD** (Product Requirements) | **무엇을** 만들까? | 기능 목록, 탭 구성, 인터랙션, 우선순위 |
| **TRD** (Technical Requirements) | **어떻게** 만들까? | 클래스 설계, 파일 구조, 알고리즘, 구현 순서 |

### 기존 방식과의 차이

```
기존 방식:
01_perceptron.py          → 실행 → PNG 저장 (인터랙션 없음)
02_activation_functions.py → 실행 → PNG 저장
03_forward_propagation.py  → 실행 → PNG 저장
04_mlp_numpy.py            → 실행 → PNG 저장
05_universal_approximation.py → 실행 → PNG 저장

NeuralNetLab:
하나의 앱 → 5개 탭 통합 → 슬라이더·애니메이션·실시간 시각화
```

**목표**: 파라미터를 직접 조작하면서 신경망이 어떻게 작동하는지 눈으로 배운다

---

## 2. 실행 방법

```bash
cd week3/NeuralNetLab
uv run main.py
```

**의존성 설치 (최초 1회):**
```bash
uv add pyside6 matplotlib numpy
```

---

## 3. 전체 파일 구조

```
week3/NeuralNetLab/
│
├── main.py                      # 앱 진입점 (QApplication 초기화, QSS 로드)
│
├── core/                        # 순수 수학 로직 (UI 없음)
│   ├── activation.py            # 활성화 함수 4종 + 도함수
│   ├── perceptron.py            # 단층 퍼셉트론 + GATE_DATA
│   ├── mlp.py                   # XOR 전용 2층 MLP
│   └── neural_net.py            # 범용 n층 신경망
│
├── widgets/                     # 재사용 가능한 공통 위젯
│   ├── mpl_canvas.py            # matplotlib → Qt 임베딩 래퍼
│   └── param_slider.py          # 레이블 + 슬라이더 + 값 표시 복합 위젯
│
├── ui/                          # 화면 구성
│   ├── main_window.py           # QMainWindow + QTabWidget (5탭)
│   ├── tab_perceptron.py        # Tab 1: 퍼셉트론 학습 애니메이션
│   ├── tab_activation.py        # Tab 2: 활성화 함수 비교
│   ├── tab_forward_prop.py      # Tab 3: 순전파 단계별 실행
│   ├── tab_mlp.py               # Tab 4: MLP + 역전파 시각화
│   └── tab_universal.py         # Tab 5: 보편 근사 정리
│
├── styles/
│   └── dark_theme.qss           # 다크 테마 스타일시트
│
├── outputs/                     # PNG 저장 폴더 (💾 버튼으로 저장)
├── PRD.md                       # 기능 요구사항 문서
└── TRD.md                       # 기술 설계 문서
```

---

## 4. 탭별 기능 설명

### Tab 1 — 퍼셉트론 (Perceptron)

**원본 스크립트**: `01_perceptron.py`

| 기능 | 설명 |
|------|------|
| 게이트 선택 | AND / OR / XOR 버튼 |
| 학습 애니메이션 | QTimer 120ms 간격으로 1 epoch씩 실행 |
| 결정 경계 | 학습 진행에 따라 실시간으로 직선 업데이트 |
| 학습 로그 | Epoch \| w1 \| w2 \| b \| 오류 수 테이블 |
| 오류 감소 그래프 | Epoch별 오류 수 꺾은선 그래프 |
| XOR 경고 | "⚠ XOR cannot be separated by a single line!" |
| PNG 저장 | `outputs/perceptron_AND_xxxxxx.png` |

**핵심 코드 흐름:**
```
▶ Train 클릭
  → QTimer.start()
  → 120ms마다 train_one_epoch() 호출
  → contourf로 배경 분류 영역 갱신
  → 오류 == 0 또는 최대 epoch → 타이머 종료
```

---

### Tab 2 — 활성화 함수 (Activation Functions)

**원본 스크립트**: `02_activation_functions.py`

| 기능 | 설명 |
|------|------|
| 2×2 그래프 | [함수 비교] [도함수 비교] [Vanishing 데모] [값 테이블] |
| 체크박스 ON/OFF | 각 함수를 개별적으로 표시/숨김 |
| x 범위 슬라이더 | 1~10 조정 시 그래프 즉시 갱신 |
| Leaky α 슬라이더 | 0.001~0.5, 실시간 곡선 반영 |
| x값 입력 | 특정 x에서 각 함수의 출력값·기울기 테이블 표시 |

**Vanishing Gradient 데모:**
```
x 범위를 -10~10으로 넓히면
Sigmoid/Tanh의 기울기가 거의 0에 수렴하는 것을 시각적으로 확인 가능
→ 이것이 ReLU가 현대 신경망의 표준이 된 이유
```

---

### Tab 3 — 순전파 (Forward Propagation)

**원본 스크립트**: `03_forward_propagation.py`

| 기능 | 설명 |
|------|------|
| 네트워크 다이어그램 | QPainter로 직접 렌더링 (뉴런, 가중치 선) |
| 가중치 선 두께/색상 | 양수=파랑, 음수=빨강, 두께=\|가중치\|에 비례 |
| 단계별 실행 | ◀ Prev / Next ▶ 버튼으로 층 하이라이트 이동 |
| 수식 패널 | 현재 단계의 z값, a값, 활성화 함수 표시 |
| 활성화 바 차트 | 각 층의 활성화 값을 막대 그래프로 표시 |
| 구조 선택 | 2→3→1 / 2→4→2→1 / 2→8→4→1 |
| 입력 슬라이더 | x1, x2 변경 시 전체 순전파 즉시 재계산 |

**단계별 실행 순서:**
```
[Input] → [Layer1 linear z=Wx+b] → [Layer1 relu(z)]
        → [Layer2 linear] → [Layer2 sigmoid(z)] → [Output]
```

---

### Tab 4 — MLP + 역전파 (MLP + Backpropagation)

**원본 스크립트**: `04_mlp_numpy.py`

| 기능 | 설명 |
|------|------|
| 결정 경계 (2D) | XOR 학습 진행에 따른 컬러맵 실시간 업데이트 |
| Loss 곡선 | 로그 스케일 y축, epoch별 MSE 감소 |
| 은닉층 히트맵 | 4개 XOR 입력에 대한 각 뉴런의 활성화값 |
| 역전파 스냅샷 | 현재 δ2, dW1, dW2 평균 절댓값 표시 |
| XOR 예측 테이블 | 4개 입력의 실시간 예측값 (정답=초록, 오답=빨강) |
| 은닉 뉴런 수 | 슬라이더 변경 시 즉시 네트워크 재초기화 |

**성능 최적화:**
```
타이머 틱당 20 epoch 처리 (30ms 간격)
→ 약 660 epoch/sec
→ 5000 epoch를 ~7.5초 안에 완료
```

**XOR 해결 확인:**
```
학습 전:  결정 경계 = 직선 (XOR 분리 불가)
학습 후:  결정 경계 = 곡선/비선형 (XOR 100% 분리)
```

---

### Tab 5 — 보편 근사 (Universal Approximation)

**원본 스크립트**: `05_universal_approximation.py`

| 기능 | 설명 |
|------|------|
| 함수 선택 | Sine / Step / Complex / Custom (텍스트 입력) |
| 뉴런 수 슬라이더 | 1~100개 선택 |
| Epoch 슬라이더 | 1000~20000 (100 단위) |
| ▶ Train | 선택한 뉴런 수로 단일 학습 |
| Compare 3/10/50 | 3가지 뉴런 수 동시 병렬 학습 + 비교 |
| MSE 바 차트 | 뉴런 수별 최종 Loss 비교 |

**QThread 비동기 처리:**
```
학습 중에도 UI 완전히 반응 → 프리징 없음
3개 worker 동시 실행 → 병렬 비교 학습
진행 상황을 Signal로 메인 스레드에 전달
```

**Custom 함수 예시:**
```python
sin(x) + cos(2*x)
abs(x - 0.5)
exp(-x) * sin(10*x)
```

---

## 5. 핵심 기술 구현

### 공통 위젯: MplCanvas

```python
# 단일 그래프
canvas = MplCanvas.single(figsize=(6, 4))

# 격자 그래프 (Tab2, Tab4에서 사용)
canvas = MplCanvas.grid(rows=2, cols=2, figsize=(10, 8))
```

matplotlib을 Qt 위젯 안에 임베딩하는 래퍼.
다크 테마 색상 자동 적용, tight_layout 자동 처리.

### 공통 위젯: ParamSlider

```python
slider = ParamSlider("Learning Rate", min=0.01, max=1.0, default=0.1, decimals=2)
slider.valueChanged.connect(lambda v: model.lr = v)
```

레이블 + 슬라이더 + 값 표시를 하나의 위젯으로 통합.
정수/실수 모두 지원 (decimals=0이면 정수).

### NeuralNet (범용 n층 신경망)

```python
# Tab3 순전파용
net = NeuralNet(layers=[2, 4, 1], activations=['relu', 'sigmoid'])

# Tab5 보편근사용
net = NeuralNet(layers=[1, 50, 1], activations=['relu', 'linear'])

# 순전파 후 각 층 값 조회
net.forward(X)
values = net.get_layer_values()  # {'input':[], 'layer1':{'z':[], 'a':[]}, ...}
```

---

## 6. Week 3 스크립트 → NeuralNetLab 매핑

| 원본 스크립트 | NeuralNetLab 탭 | 추가된 기능 |
|--------------|----------------|------------|
| `01_perceptron.py` | Tab 1 | 애니메이션, 실시간 결정 경계, 로그 테이블 |
| `02_activation_functions.py` | Tab 2 | ON/OFF 토글, α 슬라이더, x값 테이블 |
| `03_forward_propagation.py` | Tab 3 | QPainter 다이어그램, 단계별 실행, 구조 선택 |
| `04_mlp_numpy.py` | Tab 4 | 실시간 결정 경계, 히트맵, 역전파 스냅샷 |
| `05_universal_approximation.py` | Tab 5 | QThread 비동기, 병렬 비교, Custom 함수 |

---

## 7. 설계 원칙

### core와 ui 분리

```
core/ ─── 순수 NumPy 수학 로직
           테스트 가능, UI 없음
           재사용 가능

ui/   ─── Qt 위젯과 연결
           core를 가져다 씀
           시각화 담당
```

### 성능 고려

| 상황 | 해결책 |
|------|--------|
| 학습 중 UI 프리징 | QThread 백그라운드 실행 (Tab5) |
| 매 프레임 결정 경계 계산 비용 | 100ms 간격 + 저해상도 mesh (120×120) |
| 탭 전환 시 기존 학습 유지 | 각 탭이 독립적인 모델 인스턴스 보유 |

---

## 8. 학습 효과 비교

| 방식 | 퍼셉트론 이해 | XOR 한계 이해 | 역전파 이해 |
|------|:---:|:---:|:---:|
| 스크립트 실행 후 PNG | △ | △ | △ |
| NeuralNetLab 탭 조작 | ✅ | ✅ | ✅ |

- AND/OR는 바로 수렴하고, XOR는 아무리 학습해도 실패하는 걸 **직접 눈으로** 확인
- 학습률을 높이면 결정 경계가 **불안정하게 흔들리는** 것을 실시간으로 관찰
- 은닉 뉴런 수가 많을수록 복잡한 함수를 **더 정확히** 근사하는 것을 비교

---

## 9. PySide6 주요 사용 요소

NeuralNetLab은 PySide6의 다양한 기능을 적극 활용했다.

| PySide6 요소 | 사용 위치 | 역할 |
|-------------|----------|------|
| `QMainWindow` | main_window.py | 앱 최상위 창 |
| `QTabWidget` | main_window.py | 5개 탭 컨테이너 |
| `QTimer` | tab_perceptron, tab_mlp | 학습 애니메이션 (120ms/틱) |
| `QThread` + `Signal` | tab_universal.py | 비동기 학습 (UI 프리징 방지) |
| `QPainter` | tab_forward_prop.py | 신경망 다이어그램 직접 렌더링 |
| `QSlider` | param_slider.py | 파라미터 실시간 조작 |
| `QTableWidget` | tab_perceptron, tab_mlp | 학습 로그, 예측값 테이블 |
| `QProgressBar` | tab_perceptron, tab_universal | 학습 진행률 표시 |
| `QButtonGroup` | tab_perceptron, tab_universal | 게이트/함수 라디오 버튼 |
| `FigureCanvasQTAgg` | mpl_canvas.py | matplotlib 그래프를 Qt에 임베딩 |
| QSS 스타일시트 | dark_theme.qss | 다크 테마 전체 적용 |

### PySide6 핵심 패턴: Signal-Slot

```python
# 슬라이더 값 변경 → 모델 파라미터 즉시 반영
slider_lr.valueChanged.connect(lambda v: setattr(self.perceptron, 'lr', v))

# QTimer → 학습 한 스텝씩 실행
self.timer = QTimer()
self.timer.setInterval(120)          # 120ms 간격
self.timer.timeout.connect(self._training_step)

# QThread → 백그라운드 학습, 완료 시 UI 업데이트
worker.progress.connect(self._on_progress)   # 진행 중 업데이트
worker.finished.connect(self._on_done)       # 완료 시 그래프 갱신
```

---

## 10. 기술 스택

| 역할 | 라이브러리 | 버전 |
|------|-----------|------|
| GUI | PySide6 | 6.11.0 |
| 그래프 | matplotlib | 3.x |
| 수학 | NumPy | 1.x / 2.x |
| 패키지 관리 | uv | — |
| Python | CPython | 3.12 |

---

## 10. 파일 라인 수 요약

| 파일 | 역할 | 라인 수 (약) |
|------|------|:-----------:|
| `core/activation.py` | 활성화 함수 수학 | 50 |
| `core/perceptron.py` | 퍼셉트론 모델 | 65 |
| `core/mlp.py` | XOR MLP | 90 |
| `core/neural_net.py` | 범용 신경망 | 80 |
| `widgets/mpl_canvas.py` | 캔버스 래퍼 | 75 |
| `widgets/param_slider.py` | 슬라이더 위젯 | 50 |
| `styles/dark_theme.qss` | 다크 테마 | 110 |
| `ui/main_window.py` | 메인 창 | 45 |
| `ui/tab_perceptron.py` | Tab 1 | 185 |
| `ui/tab_activation.py` | Tab 2 | 145 |
| `ui/tab_forward_prop.py` | Tab 3 | 270 |
| `ui/tab_mlp.py` | Tab 4 | 210 |
| `ui/tab_universal.py` | Tab 5 | 230 |
| `main.py` | 진입점 | 25 |
| **합계** | | **~1,630줄** |
