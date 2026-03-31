# TRD — NeuralNetLab
**Technical Requirements Document**

> 버전: 1.0 | 작성일: 2026-03-28

---

## 1. 기술 스택

| 역할 | 라이브러리 | 용도 |
|------|-----------|------|
| GUI | PySide6 6.x | 창, 위젯, 레이아웃, 시그널 |
| 그래프 | matplotlib 3.x | FigureCanvasQTAgg 임베딩 |
| 수학 | NumPy 1.x | 행렬 연산, 신경망 계산 |
| 패키지 | uv | 의존성 관리 |

```bash
cd week3/NeuralNetLab
uv add pyside6 matplotlib numpy
uv run main.py
```

---

## 2. 프로젝트 구조

```
week3/NeuralNetLab/
├── main.py                        # 진입점
├── ui/
│   ├── main_window.py             # QMainWindow + QTabWidget
│   ├── tab_perceptron.py          # Tab1: 퍼셉트론
│   ├── tab_activation.py          # Tab2: 활성화 함수
│   ├── tab_forward_prop.py        # Tab3: 순전파
│   ├── tab_mlp.py                 # Tab4: MLP + 역전파
│   └── tab_universal.py           # Tab5: 보편 근사
├── core/
│   ├── perceptron.py              # Perceptron 모델
│   ├── mlp.py                     # 2층 MLP (XOR용)
│   ├── activation.py              # 활성화 함수 수학
│   └── neural_net.py              # 범용 n층 신경망
├── widgets/
│   ├── mpl_canvas.py              # matplotlib Qt 캔버스
│   └── param_slider.py            # 레이블+슬라이더 복합 위젯
├── styles/
│   └── dark_theme.qss
├── outputs/
├── PRD.md
└── TRD.md
```

---

## 3. 핵심 클래스 명세

### core/perceptron.py — `Perceptron`
```python
class Perceptron:
    weights: np.ndarray      # (2,)
    bias: float
    lr: float
    history: list[dict]      # {epoch, w1, w2, b, errors}

    def predict(inputs) -> int
    def train_one_epoch(X, y) -> int          # 오류 수 반환
    def train(X, y, epochs) -> list
    def reset()
    def get_decision_boundary() -> (w1, w2, b)
    def evaluate(X, y) -> (predictions, accuracy)

GATE_DATA = {'AND': ..., 'OR': ..., 'XOR': ...}
```

### core/activation.py — 활성화 함수 모음
```python
def sigmoid(x)       -> np.ndarray
def sigmoid_d(x)     -> np.ndarray   # 도함수
def tanh(x)          -> np.ndarray
def tanh_d(x)        -> np.ndarray
def relu(x)          -> np.ndarray
def relu_d(x)        -> np.ndarray
def leaky_relu(x, alpha=0.01) -> np.ndarray
def leaky_relu_d(x, alpha=0.01) -> np.ndarray

FUNCTIONS = {
    'Sigmoid': (sigmoid, sigmoid_d, '1/(1+e⁻ˣ)', '#FF6B6B'),
    'Tanh':    (tanh,    tanh_d,    'tanh(x)',    '#4FC3F7'),
    'ReLU':    (relu,    relu_d,    'max(0,x)',   '#69F0AE'),
    'Leaky':   (leaky_relu, leaky_relu_d, 'max(αx,x)', '#FFD740'),
}
```

### core/neural_net.py — `NeuralNet` (범용)
```python
class NeuralNet:
    """Tab3(순전파), Tab5(보편근사) 공용"""
    layers: list[int]         # e.g. [2, 4, 1]
    activations: list[str]    # e.g. ['relu', 'sigmoid']
    lr: float
    loss_history: list[float]

    def forward(X) -> np.ndarray          # 모든 z, a 저장
    def backward(X, y)
    def train(X, y, epochs, callback=None)  # callback(epoch, loss)
    def get_layer_values() -> dict          # Tab3 시각화용
    def predict(X) -> np.ndarray
```

### core/mlp.py — `SimpleMLP`
```python
class SimpleMLP:
    """Tab4 전용 XOR MLP, hidden_size 동적 변경 지원"""
    def reset(hidden_size)
    def train_step() -> float             # 1 epoch, loss 반환
    def get_decision_boundary_mesh(res=200) -> (xx, yy, Z)
    def get_hidden_activations(X) -> np.ndarray
```

### widgets/mpl_canvas.py — `MplCanvas`
```python
class MplCanvas(FigureCanvasQTAgg):
    fig: Figure
    axes: list[Axes]    # 다중 서브플롯 지원

    @classmethod
    def single(cls) -> 'MplCanvas'          # 단일 축
    @classmethod
    def grid(cls, rows, cols) -> 'MplCanvas'  # 격자 축

    def clear_all()
    def redraw()
```

### widgets/param_slider.py — `ParamSlider`
```python
class ParamSlider(QWidget):
    """레이블 + QSlider + 값 표시 통합 위젯"""
    valueChanged = Signal(float)

    def __init__(label, min_val, max_val, default, decimals=2, step=0.01)
    def value() -> float
    def set_value(v: float)
```

---

## 4. 탭별 핵심 구현

### Tab1 — 퍼셉트론 (`tab_perceptron.py`)

**레이아웃**
```
QHBoxLayout
├── 좌측 컨트롤 (width=300)
│   ├── 게이트 버튼 그룹 [AND] [OR] [XOR]
│   ├── ParamSlider: 학습률 (0.01~1.0)
│   ├── QSpinBox: Epoch (10~500)
│   ├── QPushButton: ▶ / ⏸ / ↺
│   ├── QProgressBar
│   ├── QLabel: 정확도
│   └── QTableWidget: 학습 로그
└── 우측 시각화
    ├── MplCanvas (결정 경계)  — 위
    └── MplCanvas (오류 감소)  — 아래
```

**애니메이션 로직**
```python
self.timer = QTimer()
self.timer.setInterval(100)
self.timer.timeout.connect(self._training_step)

def _training_step(self):
    errors = self.perceptron.train_one_epoch(X, y)
    self._update_log(errors)
    self._draw_boundary()
    if self.epoch >= self.max_epochs or errors == 0:
        self.timer.stop()
```

**결정 경계 시각화**
```python
def _draw_boundary(self):
    ax = self.canvas_boundary.axes[0]
    ax.clear()
    # 배경 메쉬
    xx, yy = np.meshgrid(np.linspace(-0.5,1.5,200), np.linspace(-0.5,1.5,200))
    Z = np.array([self.perceptron.predict(np.array([x,y]))
                  for x,y in zip(xx.ravel(), yy.ravel())]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
    # 결정 경계 직선: w1*x1 + w2*x2 + b = 0
    w1, w2, b = self.perceptron.get_decision_boundary()
    if abs(w2) > 1e-6:
        x1r = np.array([-0.5, 1.5])
        ax.plot(x1r, -(w1*x1r + b)/w2, 'w--', lw=2)
    self.canvas_boundary.redraw()
```

---

### Tab2 — 활성화 함수 (`tab_activation.py`)

**레이아웃**
```
QVBoxLayout
├── 상단 컨트롤
│   ├── 체크박스 그룹 (Sigmoid / Tanh / ReLU / Leaky)
│   ├── ParamSlider: x 범위 (1~10)
│   ├── ParamSlider: Leaky α (0.001~0.5)
│   └── QDoubleSpinBox: 현재 x값
└── MplCanvas (2×2 격자)
    ├── [0,0] 함수 비교       [0,1] 도함수 비교
    └── [1,0] Vanishing demo  [1,1] 현재 x 값 테이블
```

**실시간 업데이트**
```python
def _update_plots(self):
    x = np.linspace(-self.x_range, self.x_range, 400)
    for name, (fn, fn_d, formula, color) in FUNCTIONS.items():
        if self.enabled[name]:
            self.axes[0,0].plot(x, fn(x), color=color, label=name)
            self.axes[0,1].plot(x, fn_d(x), color=color, label=f"{name}'")
```

---

### Tab3 — 순전파 (`tab_forward_prop.py`)

**네트워크 다이어그램** (QPainter)
```python
class NetworkDiagram(QWidget):
    def paintEvent(self, event):
        p = QPainter(self)
        # 뉴런: fillEllipse
        # 가중치 선: drawLine (두께 = |weight|에 비례)
        # 활성값: 뉴런 안에 텍스트
        # 현재 하이라이트 층: 테두리 강조
```

**단계별 실행**
```python
STEPS = ['입력', 'Layer1 선형', 'Layer1 ReLU', 'Layer2 선형', 'Layer2 Sigmoid', '출력']
self.step_idx = 0

def _next_step(self):
    self.step_idx = (self.step_idx + 1) % len(STEPS)
    self.diagram.highlight_layer(self.step_idx)
    self._update_formula_panel()
```

---

### Tab4 — MLP + 역전파 (`tab_mlp.py`)

**레이아웃**
```
QHBoxLayout
├── 좌측 컨트롤
│   ├── ParamSlider: hidden neurons (2~16)
│   ├── ParamSlider: 학습률 (0.01~2.0)
│   ├── QSpinBox: epochs (100~20000)
│   ├── ▶ / ⏸ / ↺ 버튼
│   ├── QLabel: 현재 Loss
│   ├── QLabel: 정확도
│   └── 예측값 테이블 (4행: 입력→예측→정답)
└── 우측 MplCanvas (2×2)
    ├── [0,0] 결정 경계 (컬러맵)
    ├── [0,1] Loss 곡선
    ├── [1,0] 은닉층 활성화 히트맵
    └── [1,1] 역전파 수식 현황
```

**결정 경계 실시간 업데이트**
```python
# 100 epoch마다 갱신 (성능)
if self.epoch % 100 == 0:
    xx, yy, Z = self.mlp.get_decision_boundary_mesh()
    ax.contourf(xx, yy, Z, levels=50, cmap='RdBu', alpha=0.8)
```

---

### Tab5 — 보편 근사 (`tab_universal.py`)

**함수 정의**
```python
TARGET_FUNCTIONS = {
    'Sine':    lambda x: np.sin(2 * np.pi * x),
    'Step':    lambda x: np.where(x > 0.5, 1.0, 0.0),
    'Complex': lambda x: np.sin(2*np.pi*x) + 0.5*np.sin(6*np.pi*x),
    'Custom':  None,   # 텍스트 입력으로 eval
}
```

**뉴런 수 슬라이더 → 즉시 재학습**
```python
def _on_neuron_changed(self, n):
    self.net = NeuralNet(layers=[1, n, 1], activations=['relu','linear'])
    self._run_training()   # QThread로 백그라운드 실행
```

**QThread 학습 (UI 블로킹 방지)**
```python
class TrainWorker(QThread):
    progress = Signal(int, float)   # epoch, loss
    finished = Signal(np.ndarray)   # 최종 예측값

    def run(self):
        for epoch in range(self.epochs):
            loss = self.net.train_step(self.X, self.y)
            if epoch % 100 == 0:
                self.progress.emit(epoch, loss)
        self.finished.emit(self.net.predict(self.X_plot))
```

---

## 5. 시그널 연결 요약

```python
# Tab1
btn_gate.clicked        → load_gate(name)
timer.timeout           → _training_step()
slider_lr.valueChanged  → perceptron.lr = v

# Tab2
cb_sigmoid.toggled      → enabled['Sigmoid'] = v; _update_plots()
slider_alpha.valueChanged → leaky_alpha = v; _update_plots()
spinbox_x.valueChanged  → _update_current_x(v)

# Tab3
btn_next.clicked        → _next_step()
slider_x1.valueChanged  → net.forward([x1,x2]); _update_diagram()

# Tab4
timer.timeout           → _mlp_train_step()
slider_hidden.valueChanged → mlp.reset(int(v)); _restart_training()

# Tab5
worker.progress         → progressbar.setValue(); _update_partial()
worker.finished         → _draw_final(y_pred)
```

---

## 6. 다크 테마 핵심 색상

```
배경:      #1a1a2e
패널:      #16213e
강조:      #4fc3f7  (파랑)
성공:      #69f0ae  (초록)
경고:      #ffd740  (노랑)
오류:      #ff6b6b  (빨강)
텍스트:    #e0e0e0
서브텍스트: #9e9e9e
```

---

## 7. 구현 순서

```
Phase 1 — 기반 (공통)
  core/activation.py
  core/neural_net.py
  core/perceptron.py (기존 활용)
  core/mlp.py        (기존 활용)
  widgets/mpl_canvas.py
  widgets/param_slider.py
  styles/dark_theme.qss
  main.py + ui/main_window.py

Phase 2 — P0 탭
  ui/tab_perceptron.py
  ui/tab_activation.py

Phase 3 — P1 탭
  ui/tab_forward_prop.py
  ui/tab_mlp.py

Phase 4 — P2 탭
  ui/tab_universal.py

Phase 5 — 마무리
  PNG 저장 버튼
  상태바 메시지
  전체 테스트
```

---

## 8. 성능 목표

| 항목 | 목표 |
|------|------|
| 앱 시작 | < 3초 |
| 슬라이더 반응 | < 100ms |
| 학습 애니메이션 | 10 fps (100ms/frame) |
| MLP 결정경계 갱신 | 100 epoch마다 |
| 메모리 | < 300 MB |
