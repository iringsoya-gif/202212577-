import os
import numpy as np
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QSpinBox, QGroupBox, QProgressBar, QLineEdit,
    QButtonGroup, QSizePolicy,
)
from PySide6.QtCore import Qt, QThread, Signal

from core.neural_net import NeuralNet
from widgets.mpl_canvas import MplCanvas
from widgets.param_slider import ParamSlider


TARGET_FUNCTIONS = {
    'Sine':    lambda x: np.sin(2 * np.pi * x),
    'Step':    lambda x: np.where(x > 0.5, 1.0, 0.0),
    'Complex': lambda x: np.sin(2 * np.pi * x) + 0.5 * np.sin(6 * np.pi * x),
    'Custom':  None,
}

COMPARE_NEURONS = [3, 10, 50]
COMPARE_COLORS  = ['#ff6b6b', '#ffd740', '#69f0ae']


class TrainWorker(QThread):
    progress = Signal(int, float)      # epoch, loss
    finished = Signal(object, int, float)  # y_pred, n_neurons, final_loss

    def __init__(self, n_neurons: int, X, y, epochs: int):
        super().__init__()
        self.n_neurons = n_neurons
        self.X = X
        self.y = y
        self.epochs = epochs
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        net = NeuralNet(layers=[1, self.n_neurons, 1],
                        activations=['relu', 'linear'], lr=0.005)
        loss = 1.0
        for ep in range(self.epochs):
            if self._stop:
                return
            loss = net.train_step(self.X, self.y)
            if ep % max(1, self.epochs // 100) == 0:
                self.progress.emit(ep, loss)
        y_pred = net.predict(self.X)
        self.finished.emit(y_pred, self.n_neurons, loss)


class TabUniversal(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_fn_name = 'Sine'
        self.custom_expr = 'sin(x) + cos(2*x)'
        self.n_neurons = 20
        self.epochs = 3000
        self._workers: list[TrainWorker] = []
        self._compare_results: dict[int, np.ndarray] = {}
        self._compare_losses: dict[int, float] = {}
        self._pending_jobs: int = 0

        # Training data
        self.X_train = np.linspace(0, 1, 200).reshape(-1, 1)
        self.X_plot  = np.linspace(0, 1, 400).reshape(-1, 1)
        self._y_target = None
        self._y_pred: np.ndarray | None = None

        self._build_ui()
        self._refresh_target()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setSpacing(10)

        # ── 좌측 컨트롤 ─────────────────────────────────────────
        ctrl = QVBoxLayout()
        ctrl.setSpacing(8)

        # 함수 선택
        fn_box = QGroupBox("Target Function")
        fn_layout = QVBoxLayout(fn_box)
        self.fn_btns = QButtonGroup(self)
        for name in TARGET_FUNCTIONS:
            btn = QPushButton(name)
            btn.setCheckable(True)
            self.fn_btns.addButton(btn)
            fn_layout.addWidget(btn)
            btn.clicked.connect(lambda checked, n=name: self._select_fn(n))
        self.fn_btns.buttons()[0].setChecked(True)

        self.custom_input = QLineEdit(self.custom_expr)
        self.custom_input.setPlaceholderText("e.g. sin(x)+cos(2*x)")
        self.custom_input.returnPressed.connect(self._on_custom_expr)
        fn_layout.addWidget(QLabel("Custom expression:"))
        fn_layout.addWidget(self.custom_input)
        ctrl.addWidget(fn_box)

        # 뉴런 수 슬라이더
        self.slider_neurons = ParamSlider("Neurons", 1, 100, 20, decimals=0, step=1)
        self.slider_neurons.valueChanged.connect(self._on_neurons_changed)
        ctrl.addWidget(self.slider_neurons)

        # Epoch 슬라이더
        self.slider_epochs = ParamSlider("Epochs (x100)", 10, 200, 30, decimals=0, step=5)
        self.slider_epochs.valueChanged.connect(lambda v: setattr(self, 'epochs', int(v) * 100))
        ctrl.addWidget(self.slider_epochs)

        # 버튼
        self.btn_train = QPushButton("▶ Train")
        self.btn_compare = QPushButton("Compare 3/10/50")
        self.btn_stop = QPushButton("■ Stop")
        self.btn_stop.setEnabled(False)
        ctrl.addWidget(self.btn_train)
        ctrl.addWidget(self.btn_compare)
        ctrl.addWidget(self.btn_stop)

        self.btn_train.clicked.connect(self._train_single)
        self.btn_compare.clicked.connect(self._train_compare)
        self.btn_stop.clicked.connect(self._stop_all)

        # 진행 바
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        ctrl.addWidget(self.progress)

        # 상태 레이블
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setWordWrap(True)
        ctrl.addWidget(self.lbl_status)

        ctrl.addStretch()

        self.btn_save = QPushButton("💾 Save PNG")
        self.btn_save.clicked.connect(self._save_png)
        ctrl.addWidget(self.btn_save)

        ctrl_widget = QWidget()
        ctrl_widget.setLayout(ctrl)
        ctrl_widget.setFixedWidth(280)
        root.addWidget(ctrl_widget)

        # ── 우측: 캔버스 ─────────────────────────────────────────
        self.canvas_main = MplCanvas.single(figsize=(8, 5))
        self.canvas_bar  = MplCanvas.single(figsize=(8, 3))

        right = QVBoxLayout()
        right.addWidget(self.canvas_main, 3)
        right.addWidget(self.canvas_bar, 2)

        right_widget = QWidget()
        right_widget.setLayout(right)
        root.addWidget(right_widget, 1)

    # ── 함수 선택 ─────────────────────────────────────────────────
    def _select_fn(self, name: str):
        self.current_fn_name = name
        for btn in self.fn_btns.buttons():
            btn.setChecked(btn.text() == name)
        self._refresh_target()
        self._compare_results.clear()
        self._compare_losses.clear()
        self._y_pred = None
        self._draw_main()

    def _on_custom_expr(self):
        self.custom_expr = self.custom_input.text()
        if self.current_fn_name == 'Custom':
            self._refresh_target()
            self._draw_main()

    def _refresh_target(self):
        fn = TARGET_FUNCTIONS.get(self.current_fn_name)
        if fn is None:
            # Custom
            try:
                x = self.X_train.flatten()
                self._y_target = eval(self.custom_expr,  # noqa: S307
                                       {'x': x, 'np': np,
                                        '__builtins__': {}},
                                       {'sin': np.sin, 'cos': np.cos, 'pi': np.pi,
                                        'exp': np.exp, 'sqrt': np.sqrt, 'abs': np.abs})
                self._y_target = self._y_target.reshape(-1, 1)
            except Exception as e:
                self.lbl_status.setText(f"Expr error: {e}")
                self._y_target = np.zeros_like(self.X_train)
        else:
            self._y_target = fn(self.X_train).reshape(-1, 1)

    def _on_neurons_changed(self, v: float):
        self.n_neurons = max(1, int(round(v)))

    # ── 학습 제어 ─────────────────────────────────────────────────
    def _stop_all(self):
        for w in self._workers:
            w.stop()
        self._workers.clear()
        self.btn_stop.setEnabled(False)
        self.btn_train.setEnabled(True)
        self.btn_compare.setEnabled(True)
        self.lbl_status.setText("Stopped.")

    def _train_single(self):
        self._stop_all()
        self._y_pred = None
        self._refresh_target()
        self.progress.setValue(0)
        self.btn_train.setEnabled(False)
        self.btn_compare.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_status.setText(f"Training {self.n_neurons} neurons x {self.epochs} epochs...")

        w = TrainWorker(self.n_neurons, self.X_train, self._y_target, self.epochs)
        w.progress.connect(self._on_progress)
        w.finished.connect(self._on_single_done)
        self._workers.append(w)
        w.start()

    def _train_compare(self):
        self._stop_all()
        self._compare_results.clear()
        self._compare_losses.clear()
        self._refresh_target()
        self.progress.setValue(0)
        self.btn_train.setEnabled(False)
        self.btn_compare.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._pending_jobs = len(COMPARE_NEURONS)
        self.lbl_status.setText(f"Training {COMPARE_NEURONS} neurons x {self.epochs} epochs...")

        for n in COMPARE_NEURONS:
            w = TrainWorker(n, self.X_train, self._y_target, self.epochs)
            w.progress.connect(self._on_progress)
            w.finished.connect(self._on_compare_done)
            self._workers.append(w)
            w.start()

    def _on_progress(self, ep: int, loss: float):
        pct = int(ep / self.epochs * 100)
        self.progress.setValue(pct)
        self.lbl_status.setText(f"Epoch {ep}/{self.epochs} | Loss {loss:.5f}")

    def _on_single_done(self, y_pred, n_neurons: int, final_loss: float):
        self._y_pred = y_pred
        self.progress.setValue(100)
        self.lbl_status.setText(f"Done! n={n_neurons}, Loss={final_loss:.5f}")
        self.btn_train.setEnabled(True)
        self.btn_compare.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._draw_main()

    def _on_compare_done(self, y_pred, n_neurons: int, final_loss: float):
        self._compare_results[n_neurons] = y_pred
        self._compare_losses[n_neurons] = final_loss
        self._pending_jobs -= 1
        if self._pending_jobs == 0:
            self.progress.setValue(100)
            self.btn_train.setEnabled(True)
            self.btn_compare.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.lbl_status.setText("Compare done!")
            self._draw_main()
            self._draw_bar()

    # ── 시각화 ────────────────────────────────────────────────────
    def _draw_main(self):
        ax = self.canvas_main.axes[0]
        ax.cla()
        self.canvas_main._style_ax(ax)

        # Target function
        x_plot = self.X_plot.flatten()
        fn = TARGET_FUNCTIONS.get(self.current_fn_name)
        if fn is not None:
            y_true = fn(x_plot)
        else:
            try:
                y_true = eval(self.custom_expr,  # noqa: S307
                              {'x': x_plot, 'np': np, '__builtins__': {}},
                              {'sin': np.sin, 'cos': np.cos, 'pi': np.pi,
                               'exp': np.exp, 'sqrt': np.sqrt, 'abs': np.abs})
            except Exception:
                y_true = np.zeros_like(x_plot)

        ax.plot(x_plot, y_true, color='white', lw=2.5, label='Target', alpha=0.9)

        # Compare results (3/10/50)
        for n, color in zip(COMPARE_NEURONS, COMPARE_COLORS):
            if n in self._compare_results:
                y_p = self._compare_results[n]
                loss = self._compare_losses.get(n, 0)
                ax.plot(x_plot, y_p.flatten(), color=color, lw=1.8,
                        label=f'{n} neurons (MSE={loss:.4f})', alpha=0.85)

        # Single result
        if self._y_pred is not None:
            ax.plot(x_plot, self._y_pred.flatten(), color='#4fc3f7', lw=2,
                    label=f'{self.n_neurons} neurons', ls='--')

        ax.legend(fontsize=9, loc='upper right',
                  facecolor='#1a1a2e', edgecolor='#333', labelcolor='#e0e0e0')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title(f'Universal Approximation — {self.current_fn_name}')
        ax.set_xlim(0, 1)
        self.canvas_main.redraw()

    def _draw_bar(self):
        ax = self.canvas_bar.axes[0]
        ax.cla()
        self.canvas_bar._style_ax(ax)

        if not self._compare_losses:
            self.canvas_bar.redraw()
            return

        ns = sorted(self._compare_losses.keys())
        losses = [self._compare_losses[n] for n in ns]
        bars = ax.bar(range(len(ns)), losses,
                      color=COMPARE_COLORS[:len(ns)], alpha=0.85, width=0.5)
        ax.set_xticks(range(len(ns)))
        ax.set_xticklabels([f'{n} neurons' for n in ns])
        ax.set_ylabel('Final MSE Loss')
        ax.set_title('MSE Comparison by Neuron Count')
        for bar, loss in zip(bars, losses):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05,
                    f'{loss:.4f}', ha='center', va='bottom', color='#e0e0e0', fontsize=9)
        self.canvas_bar.redraw()

    def _save_png(self):
        out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime('%H%M%S')
        path = os.path.join(out_dir, f'universal_{self.current_fn_name}_{ts}.png')
        self.canvas_main.fig.savefig(path, dpi=150, bbox_inches='tight',
                                     facecolor=self.canvas_main.fig.get_facecolor())
        self.lbl_status.setText(f"Saved: {os.path.basename(path)}")
