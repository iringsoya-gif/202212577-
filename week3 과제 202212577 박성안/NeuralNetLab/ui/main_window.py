from PySide6.QtWidgets import QMainWindow, QTabWidget, QStatusBar
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

from .tab_perceptron import TabPerceptron
from .tab_activation import TabActivation
from .tab_forward_prop import TabForwardProp
from .tab_mlp import TabMLP
from .tab_universal import TabUniversal


TAB_DESCRIPTIONS = [
    "Tab 1 - Perceptron: AND/OR/XOR gate learning with decision boundary animation",
    "Tab 2 - Activation Functions: Sigmoid / Tanh / ReLU / Leaky ReLU comparison",
    "Tab 3 - Forward Propagation: step-by-step layer computation visualization",
    "Tab 4 - MLP + Backpropagation: XOR training with real-time decision boundary",
    "Tab 5 - Universal Approximation: neuron count vs function approximation quality",
]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuralNetLab - Neural Network Basics Visualizer")
        self.setMinimumSize(1280, 820)

        self._tab_widget = QTabWidget()
        self._tab_widget.setDocumentMode(True)
        self.setCentralWidget(self._tab_widget)

        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)

        self._build_tabs()
        self._tab_widget.currentChanged.connect(self._on_tab_changed)
        self._on_tab_changed(0)

    def _build_tabs(self):
        tabs = [
            ("Perceptron", TabPerceptron()),
            ("Activation Fn", TabActivation()),
            ("Forward Prop", TabForwardProp()),
            ("MLP + Backprop", TabMLP()),
            ("Universal Approx", TabUniversal()),
        ]
        for name, widget in tabs:
            self._tab_widget.addTab(widget, name)

    def _on_tab_changed(self, idx: int):
        if 0 <= idx < len(TAB_DESCRIPTIONS):
            self._status_bar.showMessage(TAB_DESCRIPTIONS[idx])
