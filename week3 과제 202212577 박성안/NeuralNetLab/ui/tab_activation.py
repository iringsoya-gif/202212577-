import os
import numpy as np
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox,
    QDoubleSpinBox, QGroupBox, QPushButton, QGridLayout,
)
from PySide6.QtCore import Qt

from core.activation import DISPLAY_FUNCTIONS, leaky_relu, leaky_relu_d
from widgets.mpl_canvas import MplCanvas
from widgets.param_slider import ParamSlider


class TabActivation(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.enabled = {name: True for name in DISPLAY_FUNCTIONS}
        self.x_range = 5.0
        self.leaky_alpha = 0.1
        self.current_x = 0.0
        self._build_ui()
        self._update_plots()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)

        # ── 상단 컨트롤 ─────────────────────────────────────────
        ctrl = QHBoxLayout()
        ctrl.setSpacing(12)

        # 함수 ON/OFF 체크박스
        cb_box = QGroupBox("Functions")
        cb_layout = QHBoxLayout(cb_box)
        self.checkboxes: dict[str, QCheckBox] = {}
        for name, (_, _, formula, color) in DISPLAY_FUNCTIONS.items():
            cb = QCheckBox(f"{name}")
            cb.setChecked(True)
            cb.setStyleSheet(f'color: {color}; font-weight: bold;')
            cb.stateChanged.connect(lambda state, n=name: self._toggle(n, state))
            self.checkboxes[name] = cb
            cb_layout.addWidget(cb)
        ctrl.addWidget(cb_box)

        # x 범위 슬라이더
        slider_box = QGroupBox("Display Range")
        slider_layout = QVBoxLayout(slider_box)
        self.slider_range = ParamSlider("x range", 1.0, 10.0, 5.0, decimals=1, step=0.5)
        self.slider_range.valueChanged.connect(self._on_range_changed)
        self.slider_alpha = ParamSlider("Leaky alpha", 0.001, 0.5, 0.1, decimals=3, step=0.005)
        self.slider_alpha.valueChanged.connect(self._on_alpha_changed)
        slider_layout.addWidget(self.slider_range)
        slider_layout.addWidget(self.slider_alpha)
        ctrl.addWidget(slider_box, 1)

        # 현재 x 값 입력
        x_box = QGroupBox("Point x")
        x_layout = QVBoxLayout(x_box)
        self.spin_x = QDoubleSpinBox()
        self.spin_x.setRange(-10.0, 10.0)
        self.spin_x.setSingleStep(0.1)
        self.spin_x.setValue(0.0)
        self.spin_x.valueChanged.connect(self._on_x_changed)
        x_layout.addWidget(self.spin_x)
        ctrl.addWidget(x_box)

        # 저장 버튼
        self.btn_save = QPushButton("💾 Save PNG")
        self.btn_save.clicked.connect(self._save_png)
        ctrl.addWidget(self.btn_save)

        root.addLayout(ctrl)

        # ── 2×2 캔버스 ─────────────────────────────────────────
        self.canvas = MplCanvas.grid(2, 2, figsize=(11, 7))
        root.addWidget(self.canvas, 1)

    def _toggle(self, name: str, state):
        self.enabled[name] = bool(state)
        self._update_plots()

    def _on_range_changed(self, v: float):
        self.x_range = v
        self._update_plots()

    def _on_alpha_changed(self, v: float):
        self.leaky_alpha = v
        self._update_plots()

    def _on_x_changed(self, v: float):
        self.current_x = v
        self._update_plots()

    def _update_plots(self):
        axes = self.canvas.axes   # 2D array [row][col]
        self.canvas.clear_all()

        x = np.linspace(-self.x_range, self.x_range, 500)

        ax_fn   = axes[0][0]
        ax_grad = axes[0][1]
        ax_vg   = axes[1][0]
        ax_tbl  = axes[1][1]

        for name, (fn, fn_d, formula, color) in DISPLAY_FUNCTIONS.items():
            if not self.enabled[name]:
                continue
            if name == 'Leaky':
                y_fn = leaky_relu(x, self.leaky_alpha)
                y_d  = leaky_relu_d(x, self.leaky_alpha)
                lbl  = f'Leaky(α={self.leaky_alpha:.3f})'
            else:
                y_fn = fn(x)
                y_d  = fn_d(x)
                lbl  = name

            ax_fn.plot(x, y_fn, color=color, lw=2, label=f'{lbl}: {formula}')
            ax_grad.plot(x, y_d, color=color, lw=2, label=f"{lbl}'")

        # [0,0] 함수 비교
        ax_fn.axhline(0, color='#444', lw=0.8)
        ax_fn.axvline(0, color='#444', lw=0.8)
        ax_fn.set_title('Activation Functions')
        ax_fn.set_xlabel('x')
        ax_fn.set_ylabel('f(x)')
        ax_fn.legend(fontsize=8, loc='upper left',
                     facecolor='#1a1a2e', edgecolor='#333', labelcolor='#e0e0e0')

        # [0,1] 도함수 비교
        ax_grad.axhline(0, color='#444', lw=0.8)
        ax_grad.axvline(0, color='#444', lw=0.8)
        ax_grad.set_title('Derivatives (Gradients)')
        ax_grad.set_xlabel('x')
        ax_grad.set_ylabel("f'(x)")
        ax_grad.legend(fontsize=8, loc='upper left',
                       facecolor='#1a1a2e', edgecolor='#333', labelcolor='#e0e0e0')

        # [1,0] Vanishing Gradient 데모
        x_vg = np.linspace(-10, 10, 500)
        from core.activation import sigmoid_d, tanh_d
        if self.enabled.get('Sigmoid'):
            ax_vg.plot(x_vg, sigmoid_d(x_vg), color='#FF6B6B', lw=2, label='Sigmoid grad')
        if self.enabled.get('Tanh'):
            ax_vg.plot(x_vg, tanh_d(x_vg), color='#4FC3F7', lw=2, label='Tanh grad')
        ax_vg.axhline(0.01, color='#FFD740', lw=1.2, ls='--', label='0.01 threshold')
        ax_vg.set_title('Vanishing Gradient Demo (x in [-10,10])')
        ax_vg.set_xlabel('x')
        ax_vg.set_ylabel("gradient")
        ax_vg.set_ylim(-0.02, 0.32)
        ax_vg.legend(fontsize=8, facecolor='#1a1a2e', edgecolor='#333', labelcolor='#e0e0e0')

        # [1,1] 현재 x 값 테이블
        ax_tbl.axis('off')
        rows = []
        for name, (fn, fn_d, formula, color) in DISPLAY_FUNCTIONS.items():
            if not self.enabled[name]:
                continue
            if name == 'Leaky':
                fv = float(leaky_relu(np.array([self.current_x]), self.leaky_alpha)[0])
                dv = float(leaky_relu_d(np.array([self.current_x]), self.leaky_alpha)[0])
            else:
                fv = float(fn(np.array([self.current_x]))[0])
                dv = float(fn_d(np.array([self.current_x]))[0])
            rows.append([name, f'{fv:.4f}', f'{dv:.4f}'])

        if rows:
            col_labels = ['Function', f'f({self.current_x:.2f})', "f'(x)"]
            tbl = ax_tbl.table(
                cellText=rows,
                colLabels=col_labels,
                loc='center',
                cellLoc='center',
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(10)
            for (r, c), cell in tbl.get_celld().items():
                cell.set_facecolor('#16213e')
                cell.set_edgecolor('#333344')
                cell.set_text_props(color='#e0e0e0')
                if r == 0:
                    cell.set_facecolor('#0f3460')
                    cell.set_text_props(color='#4fc3f7', fontweight='bold')
        ax_tbl.set_title(f'Values at x = {self.current_x:.2f}')

        self.canvas.redraw()

    def _save_png(self):
        out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime('%H%M%S')
        path = os.path.join(out_dir, f'activation_{ts}.png')
        self.canvas.fig.savefig(path, dpi=150, bbox_inches='tight',
                                facecolor=self.canvas.fig.get_facecolor())
