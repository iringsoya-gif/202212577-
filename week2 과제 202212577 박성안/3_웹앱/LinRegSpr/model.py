"""
Hooke's Law Neural Network — TensorFlow Linear Regression Model
Physics: F = kx  →  x = (m * g) / k
Spring constant k = 50 N/m, gravity g = 9.81 m/s²
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
import tensorflow as tf

# ── Physical constants ───────────────────────────────────────────────────────
SPRING_CONSTANT = 50.0   # N/m  (true k)
GRAVITY         = 9.81   # m/s²
OUTPUT_DIR      = "output"

# ── Dark-theme colour palette ────────────────────────────────────────────────
C = {
    "bg":     "#0f172a",
    "card":   "#1e293b",
    "border": "#334155",
    "dim":    "#475569",
    "sky":    "#38bdf8",
    "violet": "#818cf8",
    "green":  "#4ade80",
    "amber":  "#fbbf24",
    "red":    "#fb7185",
    "pink":   "#f472b6",
    "white":  "#f8fafc",
    "gray":   "#94a3b8",
}


def _dark(fig, axes):
    """Apply dark-theme styling to a figure and its axes list."""
    fig.patch.set_facecolor(C["bg"])
    for ax in axes:
        ax.set_facecolor(C["card"])
        ax.tick_params(colors=C["gray"], labelsize=10)
        for spine in ax.spines.values():
            spine.set_color(C["border"])
        ax.grid(True, alpha=0.18, color=C["dim"], linestyle="--", linewidth=0.8)


# ── Model class ───────────────────────────────────────────────────────────────
class HookesLawModel:
    """
    TensorFlow single-layer linear regression that learns Hooke's Law.
    Architecture: Dense(1, input_shape=[1])  — only 2 trainable params: w, b
    """

    def __init__(self):
        self.model: tf.keras.Model | None = None
        self.trained = False
        self.history = None
        self.X_train = None
        self.y_train = None
        self.w: float = 0.0   # learned weight  (≈ g/k)
        self.b: float = 0.0   # learned bias    (≈ 0)
        self.k_learned: float = 0.0
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Data generation ───────────────────────────────────────────────────
    def generate_data(self, n_samples: int = 200, noise_std: float = 0.005):
        """Generate (mass, extension) pairs obeying Hooke's Law + Gaussian noise."""
        rng = np.random.default_rng(42)
        masses = np.linspace(0.1, 5.0, n_samples, dtype=np.float32)
        extensions = (masses * GRAVITY / SPRING_CONSTANT).astype(np.float32)
        extensions += rng.normal(0, noise_std, n_samples).astype(np.float32)
        return masses, extensions

    # ── Model builder ─────────────────────────────────────────────────────
    def _build(self, learning_rate: float = 0.01) -> tf.keras.Model:
        model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=[1], name="mass_input"),
                tf.keras.layers.Dense(1, name="hookes_dense"),
            ],
            name="HookesLawNet",
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="mse",
            metrics=["mae"],
        )
        return model

    # ── Training ──────────────────────────────────────────────────────────
    def train(
        self,
        epochs: int = 500,
        learning_rate: float = 0.01,
        n_samples: int = 200,
    ) -> dict:
        masses, extensions = self.generate_data(n_samples=n_samples)
        self.X_train = masses
        self.y_train = extensions

        self.model = self._build(learning_rate=learning_rate)

        history = self.model.fit(
            masses,
            extensions,
            epochs=epochs,
            validation_split=0.2,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    patience=60,
                    restore_best_weights=True,
                    monitor="val_loss",
                )
            ],
        )
        self.history = history
        self.trained = True

        # Extract learned parameters
        weights = self.model.layers[0].get_weights()
        self.w = float(weights[0][0][0])
        self.b = float(weights[1][0])
        self.k_learned = GRAVITY / self.w if abs(self.w) > 1e-9 else float("inf")

        # Metrics
        preds = self.model.predict(masses, verbose=0).flatten()
        ss_res = float(np.sum((extensions - preds) ** 2))
        ss_tot = float(np.sum((extensions - np.mean(extensions)) ** 2))
        r2  = 1.0 - ss_res / ss_tot
        mae = float(np.mean(np.abs(extensions - preds)))

        actual_epochs = len(history.history["loss"])
        final_loss     = float(history.history["loss"][-1])
        final_val_loss = float(history.history["val_loss"][-1])

        # Save all plots
        self._plot_loss_curve(history, actual_epochs)
        self._plot_analysis(masses, extensions, preds)
        self._plot_summary(r2, mae, final_loss)

        return {
            "status": "success",
            "epochs_requested": epochs,
            "epochs_actual": actual_epochs,
            "final_loss": final_loss,
            "final_val_loss": final_val_loss,
            "r_squared": r2,
            "mae": mae,
            "learned_spring_constant": self.k_learned,
            "true_spring_constant": SPRING_CONSTANT,
            "weight_w": self.w,
            "bias_b": self.b,
            "plots": {
                "loss_curve": "/output/loss_curve.png",
                "predictions": "/output/predictions.png",
                "spring_diagram": "/output/spring_diagram.png",
            },
        }

    # ── Prediction ────────────────────────────────────────────────────────
    def predict(self, mass: float) -> dict:
        if not self.trained or self.model is None:
            self.train()

        arr  = np.array([mass], dtype=np.float32)
        pred = float(self.model.predict(arr, verbose=0).flatten()[0])
        true = float(mass * GRAVITY / SPRING_CONSTANT)
        err  = abs(pred - true) / true * 100 if true != 0 else 0.0

        self._plot_prediction(mass, pred, true)

        return {
            "mass": mass,
            "predicted_extension": pred,
            "true_extension": true,
            "predicted_cm": pred * 100,
            "true_cm": true * 100,
            "error_percent": err,
            "plot": "/output/single_prediction.png",
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Private plot helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _plot_loss_curve(self, history, actual_epochs: int):
        """Loss Curve — linear + log scale side by side."""
        loss     = history.history["loss"]
        val_loss = history.history["val_loss"]
        ep = range(1, actual_epochs + 1)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        _dark(fig, axes)

        best_ep  = int(np.argmin(val_loss)) + 1
        best_val = float(min(val_loss))

        for i, (ax, yscale, title) in enumerate(
            zip(axes, ["linear", "log"], ["Linear Scale", "Log Scale"])
        ):
            if yscale == "linear":
                ax.plot(ep, loss,     color=C["sky"],  lw=2.2, label="Train Loss", alpha=0.9)
                ax.plot(ep, val_loss, color=C["red"],  lw=2.2, label="Val Loss", ls="--", alpha=0.9)
                ax.fill_between(ep, loss, alpha=0.08, color=C["sky"])
                ax.fill_between(ep, val_loss, alpha=0.06, color=C["red"])
            else:
                ax.semilogy(ep, loss,     color=C["sky"], lw=2.2, label="Train Loss", alpha=0.9)
                ax.semilogy(ep, val_loss, color=C["red"], lw=2.2, label="Val Loss", ls="--", alpha=0.9)

            # Best epoch marker
            ax.axvline(x=best_ep, color=C["amber"], lw=1.2, ls=":", alpha=0.7)
            ax.scatter([best_ep], [best_val], s=80, color=C["amber"],
                       zorder=6, label=f"Best @ ep {best_ep}")

            ax.set_xlabel("Epoch",   color=C["gray"], fontsize=12)
            ax.set_ylabel("MSE Loss", color=C["gray"], fontsize=12)
            ax.set_title(f"Loss Curve ({title})", color=C["white"], fontsize=14, fontweight="bold", pad=10)
            ax.legend(facecolor=C["card"], edgecolor=C["border"],
                      labelcolor=C["white"], fontsize=10)

        fig.suptitle("Hooke's Law Neural Network — Training History",
                     color=C["white"], fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"),
                    dpi=150, bbox_inches="tight", facecolor=C["bg"])
        plt.close(fig)

    def _plot_analysis(self, masses, extensions, predictions):
        """4-panel analysis: main fit, residuals, predicted vs actual, histogram."""
        fig = plt.figure(figsize=(18, 12))
        fig.patch.set_facecolor(C["bg"])
        gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

        ax_main = fig.add_subplot(gs[0, :])
        ax_res  = fig.add_subplot(gs[1, 0])
        ax_scat = fig.add_subplot(gs[1, 1])
        ax_hist = fig.add_subplot(gs[1, 2])
        _dark(fig, [ax_main, ax_res, ax_scat, ax_hist])

        true_line = (masses * GRAVITY / SPRING_CONSTANT)
        residuals_mm = (extensions - predictions) * 1_000

        # ── Main fit ──
        ax_main.scatter(masses, extensions * 100,
                        color=C["sky"], alpha=0.5, s=20, label="Training data (+ noise)", zorder=3)
        ax_main.plot(masses, predictions * 100,
                     color=C["amber"], lw=2.5,
                     label=f"TF Model: x = {self.w:.5f}·m + {self.b:.5f}", zorder=4)
        ax_main.plot(masses, true_line * 100,
                     color=C["green"], lw=1.8, ls="--", alpha=0.85,
                     label=f"True Hooke's Law: x = (m×9.81)/50", zorder=5)
        ax_main.set_xlabel("Mass (kg)", color=C["gray"], fontsize=13)
        ax_main.set_ylabel("Extension (cm)", color=C["gray"], fontsize=13)
        ax_main.set_title("Mass vs Spring Extension — TF Linear Regression",
                          color=C["white"], fontsize=15, fontweight="bold")
        ax_main.legend(facecolor=C["card"], edgecolor=C["border"],
                       labelcolor=C["white"], fontsize=11)
        info = (f"Learned:  x = {self.w:.6f} × m + {self.b:.6f}\n"
                f"True:     x = {GRAVITY/SPRING_CONSTANT:.6f} × m  (k={SPRING_CONSTANT} N/m)\n"
                f"k_learned = {self.k_learned:.4f} N/m")
        ax_main.text(0.02, 0.97, info, transform=ax_main.transAxes,
                     fontsize=9.5, va="top", color=C["gray"], fontfamily="monospace",
                     bbox=dict(boxstyle="round,pad=0.4", fc=C["bg"], ec=C["dim"], alpha=0.9))

        # ── Residuals vs mass ──
        ax_res.scatter(masses, residuals_mm, color=C["violet"], alpha=0.7, s=18)
        ax_res.axhline(0, color=C["red"], lw=1.5, ls="--", alpha=0.8)
        ax_res.set_xlabel("Mass (kg)", color=C["gray"], fontsize=11)
        ax_res.set_ylabel("Residual (mm)", color=C["gray"], fontsize=11)
        ax_res.set_title("Residuals", color=C["white"], fontsize=13, fontweight="bold")

        # ── Predicted vs Actual ──
        ax_scat.scatter(extensions * 100, predictions * 100,
                        color=C["green"], alpha=0.6, s=18)
        lo = min(extensions.min(), predictions.min()) * 100
        hi = max(extensions.max(), predictions.max()) * 100
        ax_scat.plot([lo, hi], [lo, hi], color=C["red"], lw=2, ls="--", label="Perfect fit")
        ax_scat.set_xlabel("Actual Extension (cm)", color=C["gray"], fontsize=11)
        ax_scat.set_ylabel("Predicted Extension (cm)", color=C["gray"], fontsize=11)
        ax_scat.set_title("Predicted vs Actual", color=C["white"], fontsize=13, fontweight="bold")
        ax_scat.legend(facecolor=C["card"], edgecolor=C["border"],
                       labelcolor=C["white"], fontsize=9)

        # ── Residual histogram ──
        ax_hist.hist(residuals_mm, bins=25, color=C["pink"], alpha=0.85,
                     edgecolor=C["bg"], linewidth=0.6)
        ax_hist.axvline(0, color=C["amber"], lw=1.5, ls="--")
        ax_hist.set_xlabel("Residual (mm)", color=C["gray"], fontsize=11)
        ax_hist.set_ylabel("Count", color=C["gray"], fontsize=11)
        ax_hist.set_title("Residual Distribution", color=C["white"], fontsize=13, fontweight="bold")

        fig.suptitle("Hooke's Law — TF Model Analysis",
                     color=C["white"], fontsize=17, fontweight="bold")
        fig.savefig(os.path.join(OUTPUT_DIR, "predictions.png"),
                    dpi=150, bbox_inches="tight", facecolor=C["bg"])
        plt.close(fig)

    def _plot_summary(self, r2: float, mae: float, final_loss: float):
        """Spring-diagram infographic + model metrics table."""
        fig, (ax_spring, ax_info) = plt.subplots(1, 2, figsize=(14, 8))
        _dark(fig, [ax_spring, ax_info])
        ax_spring.axis("off")
        ax_info.axis("off")

        # ── Left: spring diagram ──────────────────────────────────────────
        ax_spring.set_xlim(-2.5, 2.5)
        ax_spring.set_ylim(-1, 5)

        # Ceiling
        ax_spring.add_patch(plt.Rectangle((-2, 4.4), 4, 0.35,
                                          fc=C["border"], ec=C["dim"], zorder=2))
        for xi in np.linspace(-1.8, 1.8, 9):
            ax_spring.plot([xi, xi - 0.25], [4.4, 4.75],
                           color=C["dim"], lw=1.5, zorder=2)

        # Ceiling pin
        ax_spring.plot([0, 0], [4.4, 4.0], color=C["dim"], lw=3, zorder=3)
        ax_spring.add_patch(plt.Circle((0, 4.0), 0.08, fc=C["gray"], zorder=4))

        # Spring coils (sine wave = coil appearance)
        t = np.linspace(0, 9 * np.pi, 500)
        sx = 0.55 * np.sin(t)
        sy = np.linspace(4.0, 1.85, 500)
        ax_spring.plot(sx, sy, color=C["sky"], lw=2.8, zorder=4)

        # Mass block
        ax_spring.add_patch(FancyBboxPatch(
            (-0.85, 1.0), 1.7, 0.85,
            boxstyle="round,pad=0.06",
            fc="#1d4ed8", ec=C["sky"], lw=2, zorder=5,
        ))
        ax_spring.text(0, 1.425, "m  (kg)", ha="center", va="center",
                       color="white", fontsize=14, fontweight="bold", zorder=6)

        # Extension arrow
        ax_spring.annotate("", xy=(2.0, 1.85), xytext=(2.0, 4.0),
                           arrowprops=dict(arrowstyle="<->", color=C["amber"], lw=2))
        ax_spring.text(2.35, 2.9, "x", ha="center", va="center",
                       color=C["amber"], fontsize=20, fontweight="bold")

        # Force arrow
        ax_spring.annotate("", xy=(0, 0.7), xytext=(0, 1.0),
                           arrowprops=dict(arrowstyle="->", color=C["red"], lw=2))
        ax_spring.text(0.35, 0.82, "F=mg", ha="left", va="center",
                       color=C["red"], fontsize=11, fontfamily="monospace")

        # Formula
        ax_spring.text(0, -0.15, "F = k · x", ha="center", va="center",
                       color=C["sky"], fontsize=22, fontweight="bold", fontfamily="monospace")
        ax_spring.text(0, -0.55, "x = (m × g) / k", ha="center", va="center",
                       color=C["gray"], fontsize=13, fontfamily="monospace")
        ax_spring.set_title("Hooke's Law Spring Model",
                            color=C["white"], fontsize=14, fontweight="bold", pad=12)

        # ── Right: metrics table ──────────────────────────────────────────
        k_err = abs(self.k_learned - SPRING_CONSTANT) / SPRING_CONSTANT * 100

        ax_info.text(0.5, 0.97, "Model Performance Report",
                     ha="center", va="top", transform=ax_info.transAxes,
                     color=C["white"], fontsize=16, fontweight="bold")
        # header separator (use Line2D since axis is off)
        from matplotlib.lines import Line2D as _L2D
        ax_info.add_line(_L2D([0.04, 0.96], [0.93, 0.93],
                              transform=ax_info.transAxes,
                              color=C["border"], linewidth=1.0, alpha=0.7))

        rows = [
            ("R² Score",          f"{r2:.6f}",                     C["green"],  "≥ 0.990"),
            ("Final MSE Loss",    f"{final_loss:.3e}",              C["sky"],    "< 1e-4"),
            ("Mean Abs Error",    f"{mae * 100:.4f} cm",            C["violet"], "< 0.1 cm"),
            ("True k",            f"{SPRING_CONSTANT:.1f} N/m",    C["amber"],  "—"),
            ("Learned k",         f"{self.k_learned:.4f} N/m",     C["pink"],   f"≈ {SPRING_CONSTANT}"),
            ("k Rel. Error",      f"{k_err:.4f} %",                 C["red"],    "< 0.5 %"),
            ("Learned weight w",  f"{self.w:.6f}",                  C["sky"],    f"≈ {GRAVITY/SPRING_CONSTANT:.4f}"),
            ("Learned bias b",    f"{self.b:.6f}",                  C["violet"], "≈ 0"),
        ]

        y_start = 0.88
        for i, (label, value, color, target) in enumerate(rows):
            y = y_start - i * 0.105
            ax_info.text(0.04, y, label, transform=ax_info.transAxes,
                         color=C["gray"], fontsize=11, va="center")
            ax_info.text(0.55, y, value, transform=ax_info.transAxes,
                         color=color, fontsize=11, va="center",
                         fontweight="bold", fontfamily="monospace")
            ax_info.text(0.85, y, target, transform=ax_info.transAxes,
                         color=C["dim"], fontsize=9.5, va="center")
            # separator
            from matplotlib.lines import Line2D
            line = Line2D([0.04, 0.96], [y - 0.042, y - 0.042],
                          transform=ax_info.transAxes,
                          color=C["border"], linewidth=0.7, alpha=0.5)
            ax_info.add_line(line)

        # Column headers
        for xpos, label in [(0.04, "Metric"), (0.55, "Value"), (0.85, "Target")]:
            ax_info.text(xpos, 0.92, label, transform=ax_info.transAxes,
                         color=C["dim"], fontsize=9, fontweight="bold")

        ax_info.set_title("TensorFlow Linear Regression — Summary",
                          color=C["white"], fontsize=14, fontweight="bold", pad=12)

        fig.suptitle("Hooke's Law Neural Network — Complete Report",
                     color=C["white"], fontsize=16, fontweight="bold")
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "spring_diagram.png"),
                    dpi=150, bbox_inches="tight", facecolor=C["bg"])
        plt.close(fig)

    def _plot_prediction(self, mass: float, prediction: float, true_value: float):
        """Single-prediction plot: fit curve + bar comparison."""
        max_m = max(5.5, mass * 1.35)
        m_range = np.linspace(0.1, max_m, 400, dtype=np.float32)
        p_range = self.model.predict(m_range, verbose=0).flatten()
        t_range = m_range * GRAVITY / SPRING_CONSTANT

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        _dark(fig, [ax1, ax2])

        # ── Fit curve with prediction highlighted ──
        ax1.plot(m_range, p_range * 100, color=C["amber"], lw=2.5,
                 label="TF Model", zorder=3)
        ax1.plot(m_range, t_range * 100, color=C["green"], lw=1.8,
                 ls="--", alpha=0.8, label="True Hooke's Law", zorder=2)
        ax1.scatter([mass], [prediction * 100], color=C["sky"], s=220, zorder=5,
                    edgecolors="white", linewidths=1.8,
                    label=f"Prediction (m={mass} kg)")
        ax1.axvline(x=mass, color=C["sky"], ls=":", lw=1.5, alpha=0.55)
        ax1.axhline(y=prediction * 100, color=C["sky"], ls=":", lw=1.5, alpha=0.55)
        ax1.annotate(
            f"m = {mass} kg\n"
            f"x̂ = {prediction * 100:.3f} cm\n"
            f"x  = {true_value * 100:.3f} cm",
            xy=(mass, prediction * 100),
            xytext=(mass + max_m * 0.12, prediction * 100 + 1.5),
            arrowprops=dict(arrowstyle="->", color=C["sky"], lw=1.5),
            fontsize=11, color=C["white"], fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc=C["bg"],
                      ec=C["sky"], alpha=0.9),
        )
        ax1.set_xlabel("Mass (kg)", color=C["gray"], fontsize=13)
        ax1.set_ylabel("Extension (cm)", color=C["gray"], fontsize=13)
        ax1.set_title("Spring Extension Prediction",
                      color=C["white"], fontsize=14, fontweight="bold")
        ax1.legend(facecolor=C["card"], edgecolor=C["border"],
                   labelcolor=C["white"], fontsize=11)

        # ── Bar comparison ──
        categories = ["Predicted\n(TF Model)", "True\n(Hooke's Law)"]
        values = [prediction * 100, true_value * 100]
        colors = [C["sky"], C["green"]]
        bars = ax2.bar(categories, values, color=colors, alpha=0.85,
                       width=0.4, edgecolor=C["border"], linewidth=1.5)
        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.04,
                     f"{val:.3f} cm",
                     ha="center", va="bottom",
                     color=C["white"], fontsize=13,
                     fontweight="bold", fontfamily="monospace")

        err_pct = abs(prediction - true_value) / true_value * 100
        ax2.set_ylabel("Extension (cm)", color=C["gray"], fontsize=13)
        ax2.set_title(f"Model vs True  (m = {mass} kg)",
                      color=C["white"], fontsize=14, fontweight="bold")
        ax2.text(0.5, 0.04, f"Error: {err_pct:.4f} %",
                 transform=ax2.transAxes, ha="center",
                 color=C["amber"], fontsize=13, fontweight="bold",
                 bbox=dict(boxstyle="round", fc=C["card"], ec=C["amber"], alpha=0.85))
        ax2.tick_params(axis="x", colors=C["white"], labelsize=12)

        fig.suptitle(f"Hooke's Law — Prediction for m = {mass} kg",
                     color=C["white"], fontsize=16, fontweight="bold")
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "single_prediction.png"),
                    dpi=150, bbox_inches="tight", facecolor=C["bg"])
        plt.close(fig)
