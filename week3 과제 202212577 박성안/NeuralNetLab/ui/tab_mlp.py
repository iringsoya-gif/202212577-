import os
import numpy as np
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QSpinBox, QGroupBox, QTableWidget, QTableWidgetItem,
    QSizePolicy,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor

from core.mlp import SimpleMLP, XOR_X, XOR_y
from widgets.mpl_canvas import MplCanvas
from widgets.param_slider import ParamSlider


class TabMLP(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.mlp = SimpleMLP(hidden_size=4, lr=0.1)
        self.epoch = 0
        self.max_epochs = 5000
        self.timer = QTimer(self)
        self.timer.setInterval(30)   # ~33fps
        self.timer.timeout.connect(self._train_step)
        self._build_ui()
        self._draw_all()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setSpacing(10)

        # ── 좌측 컨트롤 ─────────────────────────────────────────
        ctrl = QVBoxLayout()
        ctrl.setSpacing(8)

        # 파라미터
        params_box = QGroupBox("Parameters")
        params_layout = QVBoxLayout(params_box)
        self.slider_hidden = ParamSlider("Hidden neurons", 2, 16, 4, decimals=0, step=1)
        self.slider_lr = ParamSlider("Learning rate", 0.01, 2.0, 0.1, decimals=3, step=0.01)
        self.slider_hidden.valueChanged.connect(self._on_hidden_changed)
        self.slider_lr.valueChanged.connect(lambda v: setattr(self.mlp, 'lr', v))
        params_layout.addWidget(self.slider_hidden)
        params_layout.addWidget(self.slider_lr)

        epoch_row = QHBoxLayout()
        epoch_row.addWidget(QLabel("Max Epochs:"))
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(100, 20000)
        self.spin_epochs.setValue(5000)
        self.spin_epochs.setSingleStep(100)
        self.spin_epochs.valueChanged.connect(lambda v: setattr(self, 'max_epochs', v))
        epoch_row.addWidget(self.spin_epochs)
        params_layout.addLayout(epoch_row)
        ctrl.addWidget(params_box)

        # 버튼
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

        # 현재 Loss
        self.lbl_loss = QLabel("Loss: --")
        self.lbl_loss.setObjectName("lbl_accuracy")
        self.lbl_loss.setAlignment(Qt.AlignCenter)
        ctrl.addWidget(self.lbl_loss)

        self.lbl_accuracy = QLabel("Accuracy: --")
        self.lbl_accuracy.setAlignment(Qt.AlignCenter)
        ctrl.addWidget(self.lbl_accuracy)

        self.lbl_epoch = QLabel("Epoch: 0")
        self.lbl_epoch.setAlignment(Qt.AlignCenter)
        ctrl.addWidget(self.lbl_epoch)

        # XOR 예측 테이블
        tbl_label = QLabel("XOR Predictions")
        tbl_label.setObjectName("lbl_title")
        ctrl.addWidget(tbl_label)
        self.pred_table = QTableWidget(4, 4)
        self.pred_table.setHorizontalHeaderLabels(['x1', 'x2', 'Pred', 'Target'])
        self.pred_table.verticalHeader().setVisible(False)
        self.pred_table.setFixedHeight(130)
        for r, (xi, yi) in enumerate(zip(XOR_X, XOR_y)):
            for c, v in enumerate([int(xi[0]), int(xi[1]), '--', int(yi[0])]):
                item = QTableWidgetItem(str(v))
                item.setTextAlignment(Qt.AlignCenter)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.pred_table.setItem(r, c, item)
        ctrl.addWidget(self.pred_table)

        ctrl.addStretch()

        # 저장
        self.btn_save = QPushButton("💾 Save PNG")
        self.btn_save.clicked.connect(self._save_png)
        ctrl.addWidget(self.btn_save)

        ctrl_widget = QWidget()
        ctrl_widget.setLayout(ctrl)
        ctrl_widget.setFixedWidth(300)
        root.addWidget(ctrl_widget)

        # ── 우측: 2×2 캔버스 ────────────────────────────────────
        self.canvas = MplCanvas.grid(2, 2, figsize=(10, 8))
        root.addWidget(self.canvas, 1)

    # ── 이벤트 ────────────────────────────────────────────────────
    def _on_hidden_changed(self, v: float):
        self.timer.stop()
        n = max(2, int(round(v)))
        self.mlp.reset(n)
        self.epoch = 0
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self._draw_all()

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
        else:
            self.timer.start()
            self.btn_pause.setText("⏸ Pause")

    def _reset(self):
        self.timer.stop()
        n = max(2, int(round(self.slider_hidden.value())))
        self.mlp = SimpleMLP(hidden_size=n, lr=self.slider_lr.value())
        self.epoch = 0
        self.lbl_loss.setText("Loss: --")
        self.lbl_accuracy.setText("Accuracy: --")
        self.lbl_epoch.setText("Epoch: 0")
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("⏸ Pause")
        self._draw_all()

    # ── 학습 스텝 ─────────────────────────────────────────────────
    _STEPS_PER_TICK = 20   # 한 타이머 틱당 20 epoch

    def _train_step(self):
        for _ in range(self._STEPS_PER_TICK):
            loss = self.mlp.train_step()
            self.epoch += 1
            if self.epoch >= self.max_epochs:
                break

        self.lbl_loss.setText(f"Loss: {loss:.6f}")
        self.lbl_epoch.setText(f"Epoch: {self.epoch}")
        self._update_pred_table()
        self._draw_all()

        if self.epoch >= self.max_epochs:
            self.timer.stop()
            self.btn_start.setEnabled(True)
            self.btn_pause.setEnabled(False)
            preds = self.mlp.predict_xor()
            acc = np.mean(preds == XOR_y.astype(int)) * 100
            self.lbl_accuracy.setText(f"Accuracy: {acc:.0f}%")

    # ── 시각화 ────────────────────────────────────────────────────
    def _draw_all(self):
        axes = self.canvas.axes
        self.canvas.clear_all()

        ax_boundary = axes[0][0]
        ax_loss     = axes[0][1]
        ax_hidden   = axes[1][0]
        ax_backprop = axes[1][1]

        # [0,0] 결정 경계
        self._draw_boundary(ax_boundary)

        # [0,1] Loss 곡선
        self._draw_loss(ax_loss)

        # [1,0] 은닉층 히트맵
        self._draw_hidden(ax_hidden)

        # [1,1] 역전파 수식
        self._draw_backprop(ax_backprop)

        self.canvas.redraw()

    def _draw_boundary(self, ax):
        xx, yy, Z = self.mlp.get_decision_boundary_mesh(res=120)
        ax.contourf(xx, yy, Z, levels=50, cmap='RdBu', alpha=0.85)
        colors = ['#ff6b6b' if y == 0 else '#4fc3f7' for y in XOR_y.flatten()]
        ax.scatter(XOR_X[:, 0], XOR_X[:, 1], c=colors, s=100, zorder=5,
                   edgecolors='white', lw=1.5)
        for xi, yi in zip(XOR_X, XOR_y):
            ax.annotate(f'({int(xi[0])},{int(xi[1])})', xi,
                        textcoords='offset points', xytext=(5, 5),
                        color='white', fontsize=8)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_title(f'XOR Decision Boundary (epoch {self.epoch})')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')

    def _draw_loss(self, ax):
        hist = self.mlp.loss_history
        if hist:
            ax.plot(hist, color='#ff6b6b', lw=1.5)
            ax.fill_between(range(len(hist)), hist, alpha=0.15, color='#ff6b6b')
            ax.set_yscale('log')
        ax.set_title('Loss Curve')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')

    def _draw_hidden(self, ax):
        acts = self.mlp.get_hidden_activations()   # (4, hidden_size)
        im = ax.imshow(acts.T, aspect='auto', cmap='RdBu', vmin=0, vmax=1,
                       interpolation='nearest')
        ax.set_title('Hidden Layer Activations')
        ax.set_xlabel('XOR sample index')
        ax.set_ylabel('Hidden neuron')
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(['(0,0)', '(0,1)', '(1,0)', '(1,1)'], fontsize=8)

    def _draw_backprop(self, ax):
        ax.axis('off')
        bp = self.mlp.get_backprop_info()
        if not bp:
            ax.text(0.5, 0.5, 'Train to see backprop info',
                    ha='center', va='center', color='#9e9e9e', fontsize=11,
                    transform=ax.transAxes)
            return

        dW1 = bp.get('dW1', np.zeros((2, 2)))
        dW2 = bp.get('dW2', np.zeros((2, 1)))
        d2  = bp.get('delta2', np.zeros((4, 1)))

        lines = [
            "Backpropagation Info",
            "",
            f"delta2 (output error):",
            "  " + "  ".join([f"{v:.3f}" for v in d2.flatten()[:4]]),
            "",
            f"dW2 (output weight grad):",
            "  " + "  ".join([f"{v:.3f}" for v in dW2.flatten()[:4]]),
            "",
            f"|dW1| mean: {np.abs(dW1).mean():.4f}",
            f"|dW2| mean: {np.abs(dW2).mean():.4f}",
        ]
        ax.text(0.05, 0.95, '\n'.join(lines),
                transform=ax.transAxes, va='top', ha='left',
                color='#e0e0e0', fontsize=9,
                fontfamily='Consolas',
                bbox=dict(facecolor='#16213e', edgecolor='#333344', boxstyle='round,pad=0.5'))
        ax.set_title('Backprop Snapshot')

    def _update_pred_table(self):
        preds = self.mlp.forward(XOR_X)
        for r in range(4):
            v = float(preds[r, 0])
            item = QTableWidgetItem(f'{v:.3f}')
            item.setTextAlignment(Qt.AlignCenter)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            target = int(XOR_y[r, 0])
            correct = (round(v) == target)
            item.setForeground(QColor('#69f0ae') if correct else QColor('#ff6b6b'))
            self.pred_table.setItem(r, 2, item)

    def _save_png(self):
        out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime('%H%M%S')
        path = os.path.join(out_dir, f'mlp_{ts}.png')
        self.canvas.fig.savefig(path, dpi=150, bbox_inches='tight',
                                facecolor=self.canvas.fig.get_facecolor())
