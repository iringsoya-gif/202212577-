# TRD — Physics Neural Network Tuner

> **Technical Requirements Document**
> **작성일**: 2026-04-03
> **학번/이름**: 202212577 박성안
> **프로젝트**: Week 4 PySide6 인터랙티브 Neural Network 튜너

---

## 1. 기술 스택 (Tech Stack)

| 분류 | 기술 | 용도 |
|------|------|------|
| GUI 프레임워크 | PySide6 6.6+ | 메인 UI, 위젯, 시그널/슬롯 |
| 그래프 | matplotlib (FigureCanvasQTAgg) | 앱 내 Loss/Result 캔버스 임베드 |
| ML 프레임워크 | TensorFlow/Keras 2.15+ | Neural Network 정의·학습 |
| 수치 계산 | NumPy | 데이터 생성·변환 |
| 멀티스레딩 | QThread | 백그라운드 학습 (UI 블로킹 방지) |
| 테스트 | pytest + pytest-qt | 단위 테스트 (~28개) |

---

## 2. 파일 구조 (File Architecture)

```
week4/
├── app.py                      # 진입점 — MainWindow, QTabWidget 조립
├── training_worker.py          # TrainingWorker(QThread) + EpochSignalCallback
├── lab_widgets/
│   ├── __init__.py
│   ├── base_lab.py             # BaseLabWidget — 공통 UI + 추상 인터페이스
│   ├── lab1_widget.py          # Lab1: 1D 함수 근사
│   ├── lab2_widget.py          # Lab2: 포물선 운동
│   ├── lab3_widget.py          # Lab3: 과적합/과소적합
│   └── lab4_widget.py          # Lab4: 진자 주기
├── models/
│   ├── __init__.py
│   ├── lab1_model.py           # create_model, generate_1d_data, predict_1d
│   ├── lab2_model.py           # generate_projectile_data, predict_trajectory
│   ├── lab3_model.py           # generate_overfit_data, create_complexity_model
│   └── lab4_model.py           # generate_pendulum_data, simulate_pendulum_rk4
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_lab1_model.py
    ├── test_lab2_model.py
    ├── test_lab3_model.py
    ├── test_lab4_model.py
    └── test_training_worker.py
```

> 기존 `01~04_*.py` 스크립트는 **수정하지 않음**.
> `models/` 레이어가 각 스크립트의 TF 로직(모델 생성·데이터 생성)만 추출해 재사용.

---

## 3. 컴포넌트 설계

### 3-1. TrainingWorker (training_worker.py)

```
TrainingWorker(QThread)
├── 시그널
│   ├── epoch_progress(int, float, float)  # epoch, loss, val_loss
│   ├── training_done(dict, object)         # history_dict, result
│   └── training_error(str)                 # 에러 메시지
├── 생성자: __init__(run_fn, params)
│   └── run_fn(params, progress_cb) → (history_dict, result)
└── run(): run_fn 호출 → 시그널 emit

EpochSignalCallback(keras.callbacks.Callback)
└── on_epoch_end(epoch, logs): 매 N epoch마다 progress_cb(epoch, loss, val_loss) 호출
```

### 3-2. BaseLabWidget (lab_widgets/base_lab.py)

```
BaseLabWidget(QWidget)
├── 레이아웃
│   ├── 좌: 파라미터 패널 (260px 고정)
│   │   ├── 공통 파라미터 (Epochs, LR, 모델 크기)
│   │   ├── Lab 전용 파라미터 (build_specific_params() 반환)
│   │   └── Train / Reset 버튼
│   └── 우: 그래프 영역
│       ├── _loss_canvas (FigureCanvasQTAgg) — 상단
│       └── _result_canvas (FigureCanvasQTAgg) — 하단
│
├── 추상 메서드 (서브클래스 구현 필수)
│   ├── build_specific_params() → QWidget
│   ├── get_specific_params() → dict
│   ├── run_training(params, progress_cb) → (history_dict, result)
│   └── plot_result(ax, result) → None
│
└── 시그널 핸들러
    ├── _on_epoch_progress(epoch, loss, val_loss): Loss 캔버스 갱신
    ├── _on_training_done(history, result): Result 캔버스 갱신
    └── _on_training_error(msg): QMessageBox 표시
```

### 3-3. LabNWidget (lab_widgets/labN_widget.py)

| 클래스 | 오버라이드 메서드 | 특이사항 |
|--------|-----------------|---------|
| `Lab1Widget` | 전부 | FUNCTIONS dict로 함수 선택, x_range 옵션 |
| `Lab2Widget` | 전부 | `_labeled_slider` 헬퍼 정의 (Lab4도 재사용) |
| `Lab3Widget` | 전부 | COMPLEXITY_CONFIGS로 3가지 모델 분기 |
| `Lab4Widget` | 전부 | `_labeled_slider` Lab2에서 임포트, RK4 체크박스 |

### 3-4. MainWindow (app.py)

```
MainWindow(QMainWindow)
├── QTabWidget (중앙 위젯)
│   ├── Tab 0: Lab1Widget
│   ├── Tab 1: Lab2Widget
│   ├── Tab 2: Lab3Widget
│   └── Tab 3: Lab4Widget
└── QStatusBar
    └── 각 LabWidget에 set_status_bar()로 주입
```

---

## 4. 데이터 흐름 (Data Flow)

```
[사용자] 파라미터 설정 → Train 클릭
        │
        ▼
BaseLabWidget._on_train()
  common_params = _get_common_params()      # epochs, lr, hidden_layers
  specific_params = get_specific_params()   # lab별 고유 파라미터
  params = {**common_params, **specific_params}
        │
        ▼
TrainingWorker(run_fn=self.run_training, params=params).start()
        │
        │  [QThread — 백그라운드]
        ▼
run_training(params, progress_cb):
  1. generate_data(params) → X_train, y_train
  2. build_model(params) → Keras model
  3. model.fit(..., callbacks=[EpochSignalCallback])
     └─ 매 N epoch → progress_cb(epoch, loss, val_loss)
                         └─ emit epoch_progress
                              └─ _on_epoch_progress() → Loss 캔버스 갱신
  4. 예측 수행 → result dict
  5. return (history.history, result)
        │
        ▼ [메인 스레드로 복귀]
training_done.emit(history_dict, result)
  └─ _on_training_done() → plot_result(ax, result) → Result 캔버스 갱신
```

---

## 5. 인터페이스 명세

### models/ 레이어 공통 규칙

| 함수 | 반환 타입 | 설명 |
|------|----------|------|
| `create_*_model(hidden_layers, lr)` | `keras.Model` | 컴파일된 모델 |
| `generate_*_data(n_samples, noise_level)` | `(X: ndarray, Y: ndarray)` | 학습 데이터 |

### run_training 반환 타입

| Lab | result dict 키 |
|-----|---------------|
| Lab 1 | `x_plot, y_pred, x_test, y_true` |
| Lab 2 | `x_true, y_true, x_pred, y_pred, v0, theta` |
| Lab 3 | `x_test, y_test, y_pred, x_train, y_train, complexity` |
| Lab 4 | `angles, T_true, T_pred, L, theta0, rk4` |

---

## 6. 의존성 및 환경

```toml
# pyproject.toml 추가 의존성
pyside6 >= 6.6.0
pytest >= 8.0.0
pytest-qt >= 4.4.0
```

### 환경 설정

```bash
uv sync          # 의존성 설치
uv run python week4/app.py    # 앱 실행
uv run pytest week4/tests/    # 테스트 실행
```

---

## 7. 구현 단계 (Phase)

| Phase | 내용 | Task |
|-------|------|------|
| Phase 1 | 기반 설정 | Task 1: 의존성, 디렉토리 구조 |
| Phase 2 | 모델 레이어 | Task 2~5: 4개 models/*.py + 단위 테스트 |
| Phase 3 | 워커 | Task 6: TrainingWorker + 시그널 테스트 |
| Phase 4 | UI 기반 | Task 7: BaseLabWidget |
| Phase 5 | Lab 위젯 | Task 8~11: 4개 LabNWidget |
| Phase 6 | 조립 | Task 12: MainWindow + 최종 실행 검증 |

---

**작성자**: 202212577 박성안
