import sys
import os

# Ensure UTF-8 output
os.environ.setdefault('PYTHONUTF8', '1')

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("NeuralNetLab")

    # Load dark theme
    qss_path = os.path.join(os.path.dirname(__file__), 'styles', 'dark_theme.qss')
    if os.path.exists(qss_path):
        with open(qss_path, 'r', encoding='utf-8') as f:
            app.setStyleSheet(f.read())

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
