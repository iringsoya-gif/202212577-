from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QSlider
from PySide6.QtCore import Qt, Signal


class ParamSlider(QWidget):
    """레이블 + QSlider + 값 표시 통합 위젯"""

    valueChanged = Signal(float)

    def __init__(self, label: str, min_val: float, max_val: float,
                 default: float, decimals: int = 2, step: float = 0.01,
                 parent=None):
        super().__init__(parent)
        self._min = min_val
        self._max = max_val
        self._decimals = decimals
        self._step = step
        self._steps = max(1, round((max_val - min_val) / step))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._label = QLabel(label)
        self._label.setFixedWidth(110)
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, self._steps)
        self._val_label = QLabel()
        self._val_label.setFixedWidth(46)
        self._val_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        layout.addWidget(self._label)
        layout.addWidget(self._slider, 1)
        layout.addWidget(self._val_label)

        self.set_value(default)
        self._slider.valueChanged.connect(self._on_slider)

    def _on_slider(self, int_val: int):
        v = self._min + int_val * (self._max - self._min) / self._steps
        self._val_label.setText(f'{v:.{self._decimals}f}')
        self.valueChanged.emit(v)

    def value(self) -> float:
        int_val = self._slider.value()
        return self._min + int_val * (self._max - self._min) / self._steps

    def set_value(self, v: float):
        int_val = round((v - self._min) / (self._max - self._min) * self._steps)
        int_val = max(0, min(self._steps, int_val))
        self._slider.blockSignals(True)
        self._slider.setValue(int_val)
        self._slider.blockSignals(False)
        self._val_label.setText(f'{v:.{self._decimals}f}')
