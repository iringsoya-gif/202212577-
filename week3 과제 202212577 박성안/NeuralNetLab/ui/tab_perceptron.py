import os
import numpy as np
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QSpinBox, QProgressBar, QTableWidget, QTableWidgetItem,
    QButtonGroup, QGroupBox, QFrame, QSizePolicy,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor

from core.perceptron import Perceptron, GATE_DATA
from widgets.mpl_canvas import MplCanvas
from widgets.param_slider import ParamSlider


class TabPerceptron(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.perceptron = Perceptron(learning_rate=0.1)
        self.current_gate = 'AND'
        self.epoch = 0
        self.max_epochs = 100
        self.timer = QTimer(self)
        self.timer.setInterval(120)
        self.timer.timeout.connect(self._training_step)
        self._build_ui()
        self._load_gate('AND')

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setSpacing(10)

        # ── 좌측 컨트롤 ────────────────────────────────────────
        ctrl = QVBoxLayout()
        ctrl.setSpacing(8)

        # 게이트 선택
        gate_box = QGroupBox("Logic Gate")
        gate_layout = QHBoxLayout(gate_box)
        self.btn_group = QButtonGroup(self)
        for name in ('AND', 'OR', 'XOR'):
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setFixedWidth(60)
            self.btn_group.addButton(btn)
            gate_layout.addWidget(btn)
            btn.clicked.connect(lambda checked, n=name: self._load_gate(n))
        self.btn_group.buttons()[0].setChecked(True)
        ctrl.addWidget(gate_box)

        # 학습률 슬라이더
        self.slider_lr = ParamSlider("Learning Rate", 0.01, 1.0, 0.1, decimals=2, step=0.01)
        self.slider_lr.valueChanged.connect(lambda v: setattr(self.perceptron, 'lr', v))
        ctrl.addWidget(self.slider_lr)

        # Epoch 스핀박스
        epoch_row = QHBoxLayout()
        epoch_row.addWidget(QLabel("Max Epochs:"))
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(10, 500)
        self.spin_epochs.setValue(100)
        self.spin_epochs.valueChanged.connect(lambda v: setattr(self, 'max_epochs', v))
        epoch_row.addWidget(self.spin_epochs)
        epoch_row.addStretch()
        ctrl.addLayout(epoch_row)

        # 버튼 행
        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("▶ Train")
        self.btn_pause = QPushButton("⏸ Pause")
        self.btn_reset = QPushButton("↺ Reset")
        self.btn_pause.setEnabled(False)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_pause)
        btn_row.addWidget(self.btn_reset)
        ctrl.addLayout(btn_row)

        self.btn_start.clicked.connect(self._start)
        self.btn_pause.clicked.connect(self._pause)
        self.btn_reset.clicked.connect(self._reset)

        # 진행바
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        ctrl.addWidget(self.progress)

        # 정확도 레이블
        self.lbl_accuracy = QLabel("Accuracy: --")
        self.lbl_accuracy.setObjectName("lbl_accuracy")
        self.lbl_accuracy.setAlignment(Qt.AlignCenter)
        ctrl.addWidget(self.lbl_accuracy)

        # 경고 레이블 (XOR용)
        self.lbl_warning = QLabel("")
        self.lbl_warning.setObjectName("lbl_warning")
        self.lbl_warning.setWordWrap(True)
        self.lbl_warning.setAlignment(Qt.AlignCenter)
        ctrl.addWidget(self.lbl_warning)

        # 구분선
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        ctrl.addWidget(line)

        # 학습 로그 테이블
        log_label = QLabel("Training Log")
        log_label.setObjectName("lbl_title")
        ctrl.addWidget(log_label)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(['Epoch', 'w1', 'w2', 'b', 'Errors'])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 48)
        self.table.setColumnWidth(1, 54)
        self.table.setColumnWidth(2, 54)
        self.table.setColumnWidth(3, 54)
        self.table.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        ctrl.addWidget(self.table, 1)

        # PNG 저장 버튼
        self.btn_save = QPushButton("💾 Save PNG")
        self.btn_save.clicked.connect(self._save_png)
        ctrl.addWidget(self.btn_save)

        ctrl_widget = QWidget()
        ctrl_widget.setLayout(ctrl)
        ctrl_widget.setFixedWidth(310)
        root.addWidget(ctrl_widget)

        # ── 우측 시각화 ─────────────────────────────────────────
        viz = QVBoxLayout()
        self.canvas_boundary = MplCanvas.single(figsize=(6, 4))
        self.canvas_error = MplCanvas.single(figsize=(6, 3))
        viz.addWidget(self.canvas_boundary, 3)
        viz.addWidget(self.canvas_error, 2)

        viz_widget = QWidget()
        viz_widget.setLayout(viz)
        root.addWidget(viz_widget, 1)

    # ── 게이트 로드 ──────────────────────────────────────────────
    def _load_gate(self, name: str):
        self.timer.stop()
        self.current_gate = name
        self.perceptron.reset()
        self.epoch = 0
        self.table.setRowCount(0)
        self.lbl_accuracy.setText("Accuracy: --")
        self.lbl_warning.setText("")
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.progress.setValue(0)

        for btn in self.btn_group.buttons():
            btn.setChecked(btn.text() == name)

        if name == 'XOR':
            self.lbl_warning.setText("⚠ XOR cannot be separated by a single line!")

        self._draw_boundary()
        self._draw_error()

    # ── 학습 제어 ─────────────────────────────────────────────────
    def _start(self):
        if self.epoch >= self.max_epochs:
            self._reset()
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.timer.start()

    def _pause(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn_pause.setText("▶ Resume")
            self.btn_start.setEnabled(False)
        else:
            self.timer.start()
            self.btn_pause.setText("⏸ Pause")

    def _reset(self):
        self.timer.stop()
        self._load_gate(self.current_gate)
        self.btn_pause.setText("⏸ Pause")

    # ── 학습 스텝 (QTimer) ────────────────────────────────────────
    def _training_step(self):
        gate = GATE_DATA[self.current_gate]
        X, y = gate['X'], gate['y']
        errors = self.perceptron.train_one_epoch(X, y)
        self.epoch += 1

        # 로그 테이블 추가
        row = self.table.rowCount()
        self.table.insertRow(row)
        w1, w2, b = self.perceptron.get_decision_boundary()
        for col, val in enumerate([self.epoch, w1, w2, b, errors]):
            item = QTableWidgetItem(f'{val:.4f}' if isinstance(val, float) else str(val))
            item.setTextAlignment(Qt.AlignCenter)
            if col == 4 and errors > 0:
                item.setForeground(QColor('#ff6b6b'))
            self.table.setItem(row, col, item)
        self.table.scrollToBottom()

        # 진행바
        pct = int(self.epoch / self.max_epochs * 100)
        self.progress.setValue(pct)

        # 시각화 갱신
        self._draw_boundary()
        if self.epoch % 3 == 0:
            self._draw_error()

        # 종료 조건
        if self.epoch >= self.max_epochs or errors == 0:
            self.timer.stop()
            self.btn_start.setEnabled(True)
            self.btn_pause.setEnabled(False)
            self.progress.setValue(100)
            _, acc = self.perceptron.evaluate(X, y)
            self.lbl_accuracy.setText(f"Accuracy: {acc:.0f}%")
            self._draw_error()

    # ── 시각화 ───────────────────────────────────────────────────
    def _draw_boundary(self):
        ax = self.canvas_boundary.axes[0]
        ax.cla()
        self.canvas_boundary._style_ax(ax)

        gate = GATE_DATA[self.current_gate]
        X, y = gate['X'], gate['y']

        xx, yy = np.meshgrid(
            np.linspace(-0.5, 1.5, 150),
            np.linspace(-0.5, 1.5, 150),
        )
        Z = np.array([self.perceptron.predict(np.array([xi, yi]))
                      for xi, yi in zip(xx.ravel(), yy.ravel())]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.25, cmap='RdBu', levels=1)

        # 결정 경계 직선
        w1, w2, b = self.perceptron.get_decision_boundary()
        if abs(w2) > 1e-6:
            x1r = np.array([-0.5, 1.5])
            ax.plot(x1r, -(w1 * x1r + b) / w2, 'w--', lw=2, alpha=0.9)

        # 데이터 포인트
        colors = ['#ff6b6b' if yi == 0 else '#4fc3f7' for yi in y]
        ax.scatter(X[:, 0], X[:, 1], c=colors, s=120, zorder=5, edgecolors='white', lw=1.5)
        for xi, yi_val in zip(X, y):
            ax.annotate(f'({int(xi[0])},{int(xi[1])})', xi,
                        textcoords='offset points', xytext=(6, 6),
                        color='#e0e0e0', fontsize=8)

        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title(f'{self.current_gate} Gate — Decision Boundary (Epoch {self.epoch})')
        self.canvas_boundary.redraw()

    def _draw_error(self):
        ax = self.canvas_error.axes[0]
        ax.cla()
        self.canvas_error._style_ax(ax)

        history = self.perceptron.history
        if not history:
            ax.set_title('Error Count per Epoch')
            self.canvas_error.redraw()
            return

        epochs = [h['epoch'] for h in history]
        errors = [h['errors'] for h in history]
        ax.plot(epochs, errors, color='#ff6b6b', lw=2)
        ax.fill_between(epochs, errors, alpha=0.2, color='#ff6b6b')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Errors')
        ax.set_title('Error Count per Epoch')
        ax.set_ylim(bottom=0)
        self.canvas_error.redraw()

    def _save_png(self):
        out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime('%H%M%S')
        path = os.path.join(out_dir, f'perceptron_{self.current_gate}_{ts}.png')
        self.canvas_boundary.fig.savefig(path, dpi=150, bbox_inches='tight',
                                          facecolor=self.canvas_boundary.fig.get_facecolor())
        self.lbl_warning.setText(f"Saved: {os.path.basename(path)}")
