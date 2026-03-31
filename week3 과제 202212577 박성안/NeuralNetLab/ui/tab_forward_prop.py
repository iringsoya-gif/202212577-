import os
import math
import numpy as np
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QGroupBox, QComboBox, QSizePolicy, QFrame,
)
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPainterPath

from core.neural_net import NeuralNet
from widgets.mpl_canvas import MplCanvas
from widgets.param_slider import ParamSlider


NETWORK_PRESETS = {
    '2 → 3 → 1':   [2, 3, 1],
    '2 → 4 → 2 → 1': [2, 4, 2, 1],
    '2 → 8 → 4 → 1': [2, 8, 4, 1],
}

ACTIVATION_OPTS = ['relu', 'tanh', 'sigmoid']

STEP_NAMES = ['Input', 'Layer 1 linear (z=Wx+b)', 'Layer 1 activation (a=f(z))',
              'Layer 2 linear', 'Layer 2 activation', 'Output']


class NetworkDiagram(QWidget):
    """QPainter 직접 렌더링으로 신경망 구조 다이어그램 그리기"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layers: list[int] = [2, 3, 1]
        self.activations: list[float] = []   # flattened layer activation values
        self.highlight_layer = -1
        self.weights: list[np.ndarray] = []
        self.setMinimumSize(300, 260)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_network(self, layers, weights, act_values, highlight=-1):
        self.layers = layers
        self.weights = weights
        self.activations = act_values
        self.highlight_layer = highlight
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # Background
        p.fillRect(0, 0, w, h, QColor('#0d1117'))

        n_layers = len(self.layers)
        if n_layers < 2:
            return

        margin_x = 50
        margin_y = 30
        layer_x = [margin_x + i * (w - 2 * margin_x) / (n_layers - 1) for i in range(n_layers)]

        # Neuron positions
        positions: list[list[tuple]] = []
        max_n = max(self.layers)
        for li, n in enumerate(self.layers):
            col = []
            for ni in range(n):
                offset = (max_n - n) / 2
                y = margin_y + (ni + offset) * (h - 2 * margin_y) / max_n
                col.append((layer_x[li], y))
            positions.append(col)

        # Draw weights (connections)
        for li in range(n_layers - 1):
            if li < len(self.weights):
                W = self.weights[li]
                max_w = max(abs(W.max()), abs(W.min()), 1e-6)
                for ni, (x1, y1) in enumerate(positions[li]):
                    for nj, (x2, y2) in enumerate(positions[li + 1]):
                        if ni < W.shape[0] and nj < W.shape[1]:
                            w_val = W[ni, nj]
                            alpha = int(min(255, abs(w_val) / max_w * 200 + 30))
                            color = QColor(79, 195, 247, alpha) if w_val > 0 else QColor(255, 107, 107, alpha)
                            thickness = max(0.5, abs(w_val) / max_w * 2.5)
                            pen = QPen(color, thickness)
                            p.setPen(pen)
                            p.drawLine(int(x1), int(y1), int(x2), int(y2))
            else:
                for (x1, y1) in positions[li]:
                    for (x2, y2) in positions[li + 1]:
                        p.setPen(QPen(QColor('#333355'), 1))
                        p.drawLine(int(x1), int(y1), int(x2), int(y2))

        # Draw neurons
        radius = min(18, (h - 2 * margin_y) / max_n / 2 - 2)
        radius = max(8, radius)

        # Flatten activation values per layer
        flat_acts: list[list[float]] = []
        if self.activations:
            idx = 0
            for n in self.layers:
                flat_acts.append(self.activations[idx:idx + n])
                idx += n
        else:
            flat_acts = [[] for _ in self.layers]

        for li, col in enumerate(positions):
            is_hl = (li == self.highlight_layer or
                     (self.highlight_layer == len(self.layers) and li == len(self.layers) - 1))
            for ni, (cx, cy) in enumerate(col):
                # Fill based on activation value
                act_vals = flat_acts[li] if li < len(flat_acts) else []
                if act_vals and ni < len(act_vals):
                    v = float(act_vals[ni])
                    v_norm = max(0.0, min(1.0, (v + 1) / 2))
                    r = int(30 + v_norm * 80)
                    g = int(60 + v_norm * 130)
                    b_c = int(80 + v_norm * 150)
                    fill = QColor(r, g, b_c)
                else:
                    fill = QColor('#16213e')

                # Border
                if is_hl:
                    p.setPen(QPen(QColor('#4fc3f7'), 2.5))
                elif li == 0:
                    p.setPen(QPen(QColor('#69f0ae'), 1.5))
                elif li == len(self.layers) - 1:
                    p.setPen(QPen(QColor('#ffd740'), 1.5))
                else:
                    p.setPen(QPen(QColor('#555577'), 1.2))

                p.setBrush(QBrush(fill))
                p.drawEllipse(QRectF(cx - radius, cy - radius, 2 * radius, 2 * radius))

                # Activation text inside neuron
                if act_vals and ni < len(act_vals):
                    v = float(act_vals[ni])
                    p.setPen(QColor('#e0e0e0'))
                    p.setFont(QFont('Segoe UI', max(6, int(radius * 0.55))))
                    p.drawText(QRectF(cx - radius, cy - radius, 2 * radius, 2 * radius),
                               Qt.AlignCenter, f'{v:.2f}')

        # Layer labels
        layer_labels = ['Input'] + [f'H{i}' for i in range(1, n_layers - 1)] + ['Output']
        p.setPen(QColor('#9e9e9e'))
        p.setFont(QFont('Segoe UI', 9))
        for li, lx in enumerate(layer_x):
            label = layer_labels[li]
            n_count = self.layers[li]
            p.drawText(QRectF(lx - 30, h - 22, 60, 20), Qt.AlignCenter,
                       f'{label} ({n_count})')

        p.end()


class TabForwardProp(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_preset = '2 → 3 → 1'
        self.current_act = 'relu'
        self.step_idx = 0
        self._init_network()
        self._build_ui()
        self._run_forward()

    def _init_network(self):
        layers = NETWORK_PRESETS[self.current_preset]
        n_hidden = len(layers) - 2
        acts = [self.current_act] * n_hidden + ['sigmoid']
        self.net = NeuralNet(layers=layers, activations=acts, lr=0.01)

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setSpacing(10)

        # ── 좌측 컨트롤 ─────────────────────────────────────────
        ctrl = QVBoxLayout()
        ctrl.setSpacing(8)

        # 입력값 슬라이더
        inp_box = QGroupBox("Input Values")
        inp_layout = QVBoxLayout(inp_box)
        self.slider_x1 = ParamSlider("x1", -2.0, 2.0, 0.5, decimals=2, step=0.05)
        self.slider_x2 = ParamSlider("x2", -2.0, 2.0, 0.8, decimals=2, step=0.05)
        self.slider_x1.valueChanged.connect(self._run_forward)
        self.slider_x2.valueChanged.connect(self._run_forward)
        inp_layout.addWidget(self.slider_x1)
        inp_layout.addWidget(self.slider_x2)
        ctrl.addWidget(inp_box)

        # 네트워크 구조 선택
        net_box = QGroupBox("Network Architecture")
        net_layout = QVBoxLayout(net_box)
        self.combo_arch = QComboBox()
        self.combo_arch.addItems(list(NETWORK_PRESETS.keys()))
        self.combo_arch.currentTextChanged.connect(self._on_arch_changed)
        net_layout.addWidget(self.combo_arch)
        ctrl.addWidget(net_box)

        # 활성화 함수 선택
        act_box = QGroupBox("Activation Function")
        act_layout = QVBoxLayout(act_box)
        self.combo_act = QComboBox()
        self.combo_act.addItems(ACTIVATION_OPTS)
        self.combo_act.currentTextChanged.connect(self._on_act_changed)
        act_layout.addWidget(self.combo_act)
        ctrl.addWidget(act_box)

        # 가중치 초기화
        self.btn_reinit = QPushButton("↺ Randomize Weights")
        self.btn_reinit.clicked.connect(self._reinit)
        ctrl.addWidget(self.btn_reinit)

        # 단계별 실행
        step_box = QGroupBox("Step-by-Step")
        step_layout = QVBoxLayout(step_box)
        self.lbl_step = QLabel("Step: Input")
        self.lbl_step.setObjectName("lbl_title")
        self.lbl_step.setWordWrap(True)
        step_btn_row = QHBoxLayout()
        self.btn_prev = QPushButton("◀ Prev")
        self.btn_next = QPushButton("Next ▶")
        step_btn_row.addWidget(self.btn_prev)
        step_btn_row.addWidget(self.btn_next)
        self.btn_prev.clicked.connect(self._prev_step)
        self.btn_next.clicked.connect(self._next_step)
        step_layout.addWidget(self.lbl_step)
        step_layout.addLayout(step_btn_row)
        ctrl.addWidget(step_box)

        ctrl.addStretch()

        # 저장
        self.btn_save = QPushButton("💾 Save PNG")
        self.btn_save.clicked.connect(self._save_png)
        ctrl.addWidget(self.btn_save)

        ctrl_widget = QWidget()
        ctrl_widget.setLayout(ctrl)
        ctrl_widget.setFixedWidth(290)
        root.addWidget(ctrl_widget)

        # ── 중앙: 네트워크 다이어그램 ────────────────────────────
        self.diagram = NetworkDiagram()
        root.addWidget(self.diagram, 2)

        # ── 우측: 수식 패널 + 그래프 ─────────────────────────────
        right = QVBoxLayout()
        self.lbl_formula = QLabel()
        self.lbl_formula.setWordWrap(True)
        self.lbl_formula.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.lbl_formula.setStyleSheet(
            'background:#16213e; border:1px solid #333344; padding:10px; '
            'font-family: Consolas, monospace; font-size: 12px; color: #e0e0e0;'
        )
        self.lbl_formula.setMinimumHeight(180)
        right.addWidget(self.lbl_formula)

        self.canvas = MplCanvas.single(figsize=(5, 3))
        right.addWidget(self.canvas, 1)

        right_widget = QWidget()
        right_widget.setLayout(right)
        root.addWidget(right_widget, 2)

    # ── 네트워크 갱신 ─────────────────────────────────────────────
    def _on_arch_changed(self, text: str):
        self.current_preset = text
        self._init_network()
        self.step_idx = 0
        self._run_forward()

    def _on_act_changed(self, text: str):
        self.current_act = text
        self._init_network()
        self.step_idx = 0
        self._run_forward()

    def _reinit(self):
        self._init_network()
        self._run_forward()

    def _run_forward(self):
        x1 = self.slider_x1.value()
        x2 = self.slider_x2.value()
        X = np.array([[x1, x2]])
        self.net.forward(X)
        self._update_diagram()
        self._update_formula()
        self._draw_activations()

    # ── 단계별 실행 ───────────────────────────────────────────────
    def _next_step(self):
        n_steps = len(self.net.layers) * 2 - 1
        self.step_idx = (self.step_idx + 1) % n_steps
        self._update_diagram()
        self._update_formula()

    def _prev_step(self):
        n_steps = len(self.net.layers) * 2 - 1
        self.step_idx = (self.step_idx - 1) % n_steps
        self._update_diagram()
        self._update_formula()

    def _update_diagram(self):
        values = self.net.get_layer_values()
        # Flatten all layer activation values
        all_acts = list(values.get('input', []))
        for i in range(1, len(self.net.layers)):
            all_acts += values.get(f'layer{i}', {}).get('a', [0] * self.net.layers[i])

        highlight = self.step_idx // 2
        self.diagram.set_network(
            self.net.layers,
            self.net.weights,
            all_acts,
            highlight=highlight,
        )

    def _update_formula(self):
        values = self.net.get_layer_values()
        x1 = self.slider_x1.value()
        x2 = self.slider_x2.value()
        n_layers = len(self.net.layers)

        step = self.step_idx
        layer_idx = step // 2  # which layer we're looking at
        is_activation = (step % 2 == 1)

        lines = [f"Step {step + 1}/{n_layers * 2 - 1}"]

        if layer_idx == 0 and not is_activation:
            lines += [
                "── Input ──",
                f"  x1 = {x1:.4f}",
                f"  x2 = {x2:.4f}",
            ]
            step_name = "Input"
        elif layer_idx < n_layers - 1:
            k = layer_idx  # 1-based layer number
            key = f'layer{k}'
            ldata = values.get(key, {})
            act_name = ldata.get('act', '?')
            zvals = ldata.get('z', [])
            avals = ldata.get('a', [])

            if not is_activation:
                lines += [f"── Layer {k}: Linear z = W·a_prev + b ──"]
                for i, z in enumerate(zvals[:6]):
                    lines.append(f"  z{i+1} = {z:.4f}")
                if len(zvals) > 6:
                    lines.append(f"  ... ({len(zvals)} neurons)")
                step_name = f"Layer {k} linear"
            else:
                lines += [f"── Layer {k}: {act_name.upper()}(z) ──"]
                for i, (z, a) in enumerate(zip(zvals[:6], avals[:6])):
                    lines.append(f"  a{i+1} = {act_name}({z:.3f}) = {a:.4f}")
                if len(avals) > 6:
                    lines.append(f"  ... ({len(avals)} neurons)")
                step_name = f"Layer {k} {act_name}"
        else:
            out_key = f'layer{n_layers-1}'
            out = values.get(out_key, {})
            a_out = out.get('a', [])
            lines += ["── Output ──"]
            for i, v in enumerate(a_out):
                lines.append(f"  output{i+1} = {v:.4f}")
            step_name = "Output"

        self.lbl_step.setText(f"Step: {step_name}")
        self.lbl_formula.setText('\n'.join(lines))

    def _draw_activations(self):
        """오른쪽 하단: 각 층의 activation 값 bar chart"""
        ax = self.canvas.axes[0]
        ax.cla()
        self.canvas._style_ax(ax)

        values = self.net.get_layer_values()
        colors = ['#69f0ae', '#4fc3f7', '#ffd740', '#ff6b6b', '#ce93d8']
        x_offset = 0
        ticks = []
        tick_labels = []

        input_vals = values.get('input', [])
        for i, v in enumerate(input_vals):
            ax.bar(x_offset, v, color=colors[0], width=0.7, alpha=0.8)
            ticks.append(x_offset)
            tick_labels.append(f'x{i+1}')
            x_offset += 1
        x_offset += 0.5

        for li in range(1, len(self.net.layers)):
            ldata = values.get(f'layer{li}', {})
            avals = ldata.get('a', [])
            c = colors[li % len(colors)]
            for i, v in enumerate(avals[:8]):
                ax.bar(x_offset, v, color=c, width=0.7, alpha=0.8)
                ticks.append(x_offset)
                tick_labels.append(f'L{li}[{i}]')
                x_offset += 1
            if len(avals) > 8:
                ax.text(x_offset - 0.5, 0.5, f'+{len(avals)-8}', color='#9e9e9e', fontsize=8)
            x_offset += 0.5

        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, rotation=45, fontsize=7)
        ax.set_ylabel('Activation')
        ax.set_title(f'Layer Activations — input=[{self.slider_x1.value():.2f}, {self.slider_x2.value():.2f}]')
        ax.axhline(0, color='#444', lw=0.8)
        self.canvas.redraw()

    def _save_png(self):
        out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime('%H%M%S')
        path = os.path.join(out_dir, f'forward_prop_{ts}.png')
        self.canvas.fig.savefig(path, dpi=150, bbox_inches='tight',
                                facecolor=self.canvas.fig.get_facecolor())
