import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']


class MplCanvas(FigureCanvasQTAgg):
    """matplotlib Figure Qt 임베딩 래퍼"""

    def __init__(self, figsize=(6, 4), dpi=90):
        self.fig = Figure(figsize=figsize, dpi=dpi, facecolor='#0d1117')
        super().__init__(self.fig)
        self.axes: list = []
        self._apply_style()

    def _apply_style(self):
        self.fig.patch.set_facecolor('#0d1117')

    def _style_ax(self, ax):
        ax.set_facecolor('#0d1117')
        ax.tick_params(colors='#9e9e9e', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#333333')
        ax.title.set_color('#e0e0e0')
        ax.xaxis.label.set_color('#9e9e9e')
        ax.yaxis.label.set_color('#9e9e9e')

    @classmethod
    def single(cls, figsize=(6, 4), dpi=90) -> 'MplCanvas':
        c = cls(figsize=figsize, dpi=dpi)
        ax = c.fig.add_subplot(111)
        c._style_ax(ax)
        c.axes = [ax]
        c.fig.tight_layout(pad=1.0)
        return c

    @classmethod
    def grid(cls, rows: int, cols: int, figsize=(10, 8), dpi=90) -> 'MplCanvas':
        c = cls(figsize=figsize, dpi=dpi)
        axes_2d = c.fig.subplots(rows, cols)
        if rows == 1 and cols == 1:
            c.axes = [axes_2d]
        elif rows == 1 or cols == 1:
            c.axes = list(axes_2d.flatten())
        else:
            c.axes = axes_2d  # 2D array kept as-is for [row][col] indexing
            for row in axes_2d:
                for ax in row:
                    c._style_ax(ax)
            c.fig.tight_layout(pad=1.2)
            return c
        for ax in c.axes:
            c._style_ax(ax)
        c.fig.tight_layout(pad=1.2)
        return c

    def clear_all(self):
        if hasattr(self.axes, '__iter__') and not isinstance(self.axes, list):
            for row in self.axes:
                for ax in row:
                    ax.cla()
                    self._style_ax(ax)
        else:
            for ax in (self.axes if isinstance(self.axes, list) else [self.axes]):
                ax.cla()
                self._style_ax(ax)

    def redraw(self):
        try:
            self.fig.tight_layout(pad=1.0)
        except Exception:
            pass
        self.draw()
