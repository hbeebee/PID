"""
app.py — Streamlit веб-интерфейс для Process Model Analyzer
============================================================
Запуск: streamlit run app.py
Зависимости: см. requirements.txt
"""

import io
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter

# ══════════════════════════════════════════════════════════════════
#  КОПИЯ МОДУЛЕЙ ИЗ process_analyzer.py
#  (Streamlit запускает один файл, поэтому модули встроены сюда.
#   Если хотите — вынесите их в отдельный models.py и сделайте
#   from models import ...)
# ══════════════════════════════════════════════════════════════════

# ── Шумоподавление ────────────────────────────────────────────────

def denoise(signal: np.ndarray, window: int = 5) -> np.ndarray:
    """Savitzky-Golay фильтр. Если мало точек — возвращает копию."""
    n = len(signal)
    if n < window + 2:
        return signal.copy()
    wl = window if window % 2 == 1 else window + 1
    wl = min(wl, n - 1 if (n - 1) % 2 == 1 else n - 2)
    try:
        return savgol_filter(signal, window_length=wl, polyorder=2)
    except Exception:
        return signal.copy()


# ── StepInfo ──────────────────────────────────────────────────────

class StepInfo:
    """Хранит информацию о найденном ступенчатом воздействии."""
    def __init__(self, step_idx, end_idx, tStep,
                 mv0, delta_mv, pv0, pv_final, delta_pv,
                 t, mv, pv):
        self.step_idx = step_idx
        self.end_idx  = end_idx
        self.tStep    = tStep
        self.mv0      = mv0
        self.delta_mv = delta_mv
        self.pv0      = pv0
        self.pv_final = pv_final
        self.delta_pv = delta_pv
        self.t = t
        self.mv = mv
        self.pv = pv
        self.n = len(t)


def find_step(t: np.ndarray, mv: np.ndarray,
              pv: np.ndarray) -> StepInfo:
    """
    Ищет наибольший скачок MV — это и есть момент
    ступенчатого воздействия.
    Поднимает ValueError если ступенька не найдена.
    """
    n = len(mv)
    if n < 5:
        raise ValueError("Слишком мало точек данных (< 5)")

    diffs  = np.abs(np.diff(mv))
    si     = int(np.argmax(diffs)) + 1
    step_v = diffs[si - 1]

    if step_v < 1e-9:
        raise ValueError(
            "Ступенчатое изменение MV не найдено. "
            "Проверьте назначение колонок MV / PV."
        )

    mv0      = mv[si - 1]
    delta_mv = mv[si] - mv0

    # PV до ступеньки — среднее по 5 точкам
    pre  = pv[max(0, si - 5):si]
    pv0  = float(np.mean(pre)) if len(pre) > 0 else pv[si]

    # Конец плато: следующий скачок MV > 30 % от первого
    end_idx = n
    for i in range(si + 1, n):
        if abs(mv[i] - mv[si]) > step_v * 0.3:
            end_idx = i
            break

    # Финальное PV — среднее последних 15 % плато
    plateau  = pv[si:end_idx]
    sz       = max(3, int(len(plateau) * 0.15))
    pv_final = float(np.mean(plateau[-sz:]))
    delta_pv = pv_final - pv0

    return StepInfo(si, end_idx, t[si],
                    mv0, delta_mv, pv0, pv_final, delta_pv,
                    t, mv, pv)


def compute_metrics(actual: np.ndarray,
                    predicted: np.ndarray,
                    start: int, end: int):
    """Считает R² и RMSE на участке [start:end]."""
    a = actual[start:end]
    p = predicted[start:end]
    if len(a) < 2:
        return -999.0, 999.0
    ss_res = np.sum((a - p) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    r2   = 1 - ss_res / ss_tot if ss_tot > 1e-12 else -999.0
    rmse = float(np.sqrt(np.mean((a - p) ** 2)))
    return float(r2), rmse


# ── FOPDT ─────────────────────────────────────────────────────────

class FOPDTModel:
    """
    Модель 1-го порядка с запаздыванием.
    G(s) = K · exp(-θ·s) / (τ·s + 1)
    Метод идентификации: две точки 28.3 % и 63.2 %.
    """
    label = "FOPDT"
    color = "#38bdf8"

    def __init__(self):
        self.K = self.tau = self.theta = None
        self.pv0 = self.delta_mv = self.tStep = None
        self.pv_final = None
        self.r2 = self.rmse = None
        self.sim_data = None
        self.error = None
        self.t283 = self.t632 = None

    def identify(self, s: StepInfo):
        if abs(s.delta_pv) < 1e-9:
            self.error = "PV не изменяется"
            return

        self.K        = s.delta_pv / s.delta_mv
        self.pv0      = s.pv0
        self.pv_final = s.pv_final
        self.delta_mv = s.delta_mv
        self.tStep    = s.tStep

        p283 = s.pv0 + 0.283 * s.delta_pv
        p632 = s.pv0 + 0.632 * s.delta_pv
        t283 = t632 = None

        for i in range(s.step_idx, s.end_idx - 1):
            if t283 is None:
                cond = (s.pv[i] >= p283 if s.delta_pv > 0
                        else s.pv[i] <= p283)
                if cond:
                    frac = ((p283 - s.pv[i]) /
                            (s.pv[i+1] - s.pv[i] + 1e-12))
                    t283 = s.t[i] + frac * (s.t[i+1] - s.t[i])
            if t632 is None:
                cond = (s.pv[i] >= p632 if s.delta_pv > 0
                        else s.pv[i] <= p632)
                if cond:
                    frac = ((p632 - s.pv[i]) /
                            (s.pv[i+1] - s.pv[i] + 1e-12))
                    t632 = s.t[i] + frac * (s.t[i+1] - s.t[i])
            if t283 is not None and t632 is not None:
                break

        if t283 is None or t632 is None:
            self.error = (
                f"PV не достигает "
                f"{'28.3 %' if t283 is None else '63.2 %'} порога"
            )
            return

        self.t283  = t283
        self.t632  = t632
        self.tau   = 1.5 * (t632 - t283)
        self.theta = t632 - self.tau

    def simulate(self, t: np.ndarray) -> np.ndarray:
        if self.error:
            return np.zeros(len(t))
        dt = t - self.tStep - self.theta
        return np.where(
            dt <= 0,
            self.pv0,
            self.pv0 + self.K * self.delta_mv *
            (1 - np.exp(-dt / max(self.tau, 1e-9)))
        )

    def tf_string(self) -> str:
        if self.error:
            return "Ошибка идентификации"
        return (f"G(s) = {self.K:.4f} · exp(-{self.theta:.4f}·s)"
                f" / ({self.tau:.4f}·s + 1)")

    def params_dict(self) -> dict:
        if self.error:
            return {}
        return {"K": self.K, "τ": self.tau, "θ": self.theta}


# ── SOPDT ─────────────────────────────────────────────────────────

class SOPDTModel:
    """
    Модель 2-го порядка с запаздыванием.
    G(s) = K · exp(-θ·s) / ((τ₁·s+1)(τ₂·s+1))
    """
    label = "SOPDT"
    color = "#a78bfa"

    def __init__(self):
        self.K = self.tau1 = self.tau2 = self.theta = None
        self.zeta = 1.0
        self.pv0 = self.delta_mv = self.tStep = None
        self.pv_final = None
        self.r2 = self.rmse = None
        self.sim_data = None
        self.error = None

    def identify(self, s: StepInfo):
        if abs(s.delta_pv) < 1e-9:
            self.error = "PV не изменяется"
            return

        self.K        = s.delta_pv / s.delta_mv
        self.pv0      = s.pv0
        self.pv_final = s.pv_final
        self.delta_mv = s.delta_mv
        self.tStep    = s.tStep

        plateau  = s.pv[s.step_idx:s.end_idx]
        pv_ext   = (np.max(plateau) if s.delta_pv > 0
                    else np.min(plateau))
        overshoot = abs((pv_ext - s.pv_final) /
                        (s.delta_pv + 1e-12))

        p264 = s.pv0 + 0.264 * s.delta_pv
        p632 = s.pv0 + 0.632 * s.delta_pv
        t264 = t632 = None

        for i in range(s.step_idx, s.end_idx - 1):
            def _cross(pct, idx=i):
                return (s.pv[idx] >= pct if s.delta_pv > 0
                        else s.pv[idx] <= pct)
            if t264 is None and _cross(p264):
                frac = ((p264 - s.pv[i]) /
                        (s.pv[i+1] - s.pv[i] + 1e-12))
                t264 = s.t[i] + frac * (s.t[i+1] - s.t[i])
            if t632 is None and _cross(p632):
                frac = ((p632 - s.pv[i]) /
                        (s.pv[i+1] - s.pv[i] + 1e-12))
                t632 = s.t[i] + frac * (s.t[i+1] - s.t[i])
            if t264 is not None and t632 is not None:
                break

        if t264 is None or t632 is None:
            self.error = "Недостаточно данных для SOPDT"
            return

        t1 = t264 - s.tStep
        t2 = t632 - s.tStep
        if t1 <= 0 or t2 <= t1:
            self.error = "Не удалось определить параметры SOPDT"
            return

        if overshoot > 0.02:
            ln_os     = np.log(max(overshoot, 1e-6))
            self.zeta = abs(ln_os) / np.sqrt(np.pi**2 + ln_os**2)
            wn        = np.pi / t2
            self.tau1 = self.tau2 = 1 / max(wn, 1e-9)
            self.theta = max(0.0, t1 * 0.5)
        else:
            self.tau2  = t2 * 0.6
            self.tau1  = t2 * 0.2
            self.theta = max(0.0, t1 - self.tau1 * 0.1)
            self.zeta  = 1.0

    def simulate(self, t: np.ndarray) -> np.ndarray:
        if self.error:
            return np.zeros(len(t))
        dt  = t - self.tStep - self.theta
        out = np.zeros(len(t))
        for i, d in enumerate(dt):
            if d <= 0:
                out[i] = self.pv0
            elif abs(self.zeta - 1.0) < 0.01:
                wn = 1 / max(self.tau1, 1e-9)
                out[i] = self.pv0 + self.K * self.delta_mv * (
                    1 - np.exp(-wn * d) * (1 + wn * d))
            elif self.zeta > 1:
                a = -1 / max(self.tau1, 1e-9)
                b = -1 / max(self.tau2, 1e-9)
                if abs(a - b) < 1e-9:
                    a *= 1.001
                out[i] = self.pv0 + self.K * self.delta_mv * (
                    1 + (b * np.exp(a * d) - a * np.exp(b * d))
                    / (a - b))
            else:
                wd = (1 / max(self.tau1, 1e-9)) * np.sqrt(
                    1 - self.zeta**2)
                out[i] = self.pv0 + self.K * self.delta_mv * (
                    1 - np.exp(
                        -self.zeta / max(self.tau1, 1e-9) * d) * (
                        np.cos(wd * d) +
                        self.zeta / max(
                            np.sqrt(1 - self.zeta**2), 1e-9) *
                        np.sin(wd * d)))
        return out

    def tf_string(self) -> str:
        if self.error:
            return "Ошибка идентификации"
        return (f"G(s) = {self.K:.4f} · exp(-{self.theta:.4f}·s)"
                f" / (({self.tau1:.4f}·s+1)·({self.tau2:.4f}·s+1))")

    def params_dict(self) -> dict:
        if self.error:
            return {}
        return {"K": self.K, "τ₁": self.tau1, "τ₂": self.tau2,
                "θ": self.theta, "ζ": self.zeta}


# ── Интегрирующий процесс ─────────────────────────────────────────

class IntegratingModel:
    """
    Интегрирующий процесс.
    G(s) = Ki · exp(-θ·s) / s
    Идентификация: МНК наклона на участке отклика.
    """
    label = "Интегрирующий"
    color = "#fbbf24"

    def __init__(self):
        self.Ki = self.theta = None
        self.pv0 = self.delta_mv = self.tStep = None
        self.r2 = self.rmse = None
        self.sim_data = None
        self.error = None

    def identify(self, s: StepInfo):
        if abs(s.delta_mv) < 1e-9:
            self.error = "delta_mv = 0"
            return

        self.pv0      = s.pv0
        self.delta_mv = s.delta_mv
        self.tStep    = s.tStep

        t_r  = s.t[s.step_idx:s.end_idx]
        pv_r = s.pv[s.step_idx:s.end_idx]
        n    = len(t_r)
        if n < 3:
            self.error = "Мало точек"
            return

        t_mean  = np.mean(t_r)
        pv_mean = np.mean(pv_r)
        num     = np.sum((t_r - t_mean) * (pv_r - pv_mean))
        den     = np.sum((t_r - t_mean) ** 2)
        slope   = num / max(den, 1e-12)
        self.Ki = slope / s.delta_mv

        self.theta = 0.0
        for i in range(s.step_idx, s.end_idx):
            if abs(s.pv[i] - s.pv0) > abs(s.delta_pv) * 0.05:
                self.theta = s.t[i] - s.tStep
                break

    def simulate(self, t: np.ndarray) -> np.ndarray:
        if self.error:
            return np.zeros(len(t))
        dt = t - self.tStep - self.theta
        return np.where(
            dt <= 0,
            self.pv0,
            self.pv0 + self.Ki * self.delta_mv * dt
        )

    def tf_string(self) -> str:
        if self.error:
            return "Ошибка идентификации"
        return (f"G(s) = {self.Ki:.4f} · exp(-{self.theta:.4f}·s)"
                f" / s")

    def params_dict(self) -> dict:
        if self.error:
            return {}
        return {"Ki": self.Ki, "θ": self.theta}


# ── PID Calculator ────────────────────────────────────────────────

class PIDCalculator:
    """
    Расчёт PID по методу Desired Closed Loop Response.

    Формулы:
      Kc = (2T + d) / (K · (2e + d))
      Ti = T + d/2
      Td = T·d / (2T + d)   [только PID]

    где e — desired closed loop response time,
    начальное значение: e = T + d
    """

    @staticmethod
    def calculate(K: float, T: float, d: float,
                  e: float = None,
                  controller: str = "PID") -> dict:
        if e is None:
            e = T + d
        gain  = (2 * T + d) / max(K * (2 * e + d), 1e-12)
        reset = T + d / 2.0
        deriv = ((T * d) / max(2 * T + d, 1e-12)
                 if controller == "PID" else None)
        return {
            "Gain (Kc)":       round(gain,  4),
            "Reset (Ti)":      round(reset, 4),
            "Derivative (Td)": (round(deriv, 4)
                                if deriv is not None else None),
            "e (response time)": round(e, 4),
            "controller": controller,
        }

    @staticmethod
    def simulate_pid(t: np.ndarray,
                     pv: np.ndarray,
                     setpoint: float,
                     Kc: float, Ti: float, Td: float = 0.0,
                     mv_init: float = 50.0,
                     mv_min: float = 0.0,
                     mv_max: float = 100.0):
        """Симуляция дискретного PID регулятора."""
        n          = len(t)
        mv         = np.zeros(n)
        err        = np.zeros(n)
        integral   = 0.0
        prev_error = 0.0
        mv[0]      = mv_init

        for i in range(1, n):
            dt = max(t[i] - t[i-1], 1e-6)
            err[i]   = setpoint - pv[i]
            integral += err[i] * dt
            deriv    = (err[i] - prev_error) / dt if Td > 0 else 0.0
            output   = Kc * (err[i] +
                             integral / max(Ti, 1e-9) +
                             Td * deriv)
            mv[i]      = np.clip(mv_init + output, mv_min, mv_max)
            prev_error = err[i]

        return mv, err


# ══════════════════════════════════════════════════════════════════
#  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ══════════════════════════════════════════════════════════════════

def guess_col(columns: list, patterns: list) -> str:
    """Угадывает нужную колонку по шаблонам регулярных выражений."""
    for pat in patterns:
        match = [c for c in columns
                 if re.search(pat, str(c), re.IGNORECASE)]
        if match:
            return match[0]
    return columns[0] if columns else ""


def detect_best_model(step: StepInfo,
                      fopdt: FOPDTModel,
                      sopdt: SOPDTModel,
                      integr: IntegratingModel) -> str:
    """
    Автовыбор модели по трём критериям:
    1. PV не стабилизируется → интегрирующий
    2. Перерегулирование > 5 % → SOPDT
    3. R²(SOPDT) >> R²(FOPDT) → SOPDT
    Иначе → FOPDT
    """
    pv      = step.pv
    plateau = pv[step.step_idx:step.end_idx]

    # Проверка интегрирующего: хвост плато всё ещё движется
    if len(plateau) > 5 and step.end_idx == step.n:
        tail       = plateau[-max(3, len(plateau) // 7):]
        tail_trend = abs(float(tail[-1]) - float(tail[0]))
        if tail_trend / (abs(step.delta_pv) + 1e-12) > 0.15:
            return "integrating"

    # Перерегулирование
    pv_ext    = (np.max(plateau) if step.delta_pv > 0
                 else np.min(plateau))
    overshoot = abs((pv_ext - step.pv_final) /
                    (step.delta_pv + 1e-12))
    if overshoot > 0.05:
        return "sopdt"

    # Сравнение R²
    r2_f = fopdt.r2 if (not fopdt.error and
                        fopdt.r2 is not None) else -999.0
    r2_s = sopdt.r2 if (not sopdt.error and
                        sopdt.r2 is not None) else -999.0
    if r2_s - r2_f > 0.01:
        return "sopdt"

    return "fopdt"


# ══════════════════════════════════════════════════════════════════
#  ЦВЕТОВАЯ ТЕМА PLOTLY
# ══════════════════════════════════════════════════════════════════

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0f1520",
    plot_bgcolor="#141c2b",
    font=dict(color="#e2e8f0", family="monospace", size=11),
    xaxis=dict(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f"),
    yaxis=dict(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f"),
    legend=dict(bgcolor="#0f1520", bordercolor="#1e3a5f",
                borderwidth=1),
    margin=dict(l=50, r=20, t=40, b=40),
)


def apply_theme(fig: go.Figure,
                rows: int = 1) -> go.Figure:
    """Применяет тёмную тему ко всем осям фигуры."""
    fig.update_layout(**PLOTLY_LAYOUT)
    for i in range(1, rows + 1):
        fig.update_xaxes(gridcolor="#1e3a5f",
                         zerolinecolor="#1e3a5f", row=i)
        fig.update_yaxes(gridcolor="#1e3a5f",
                         zerolinecolor="#1e3a5f", row=i)
    return fig


# ══════════════════════════════════════════════════════════════════
#  STREAMLIT APP
# ══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Process Model Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Кастомный CSS (тёмная тема + шрифт) ──────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
  html, body, [class*="css"] { font-family: 'JetBrains Mono', monospace; }
  .main { background: #0a0e14; }
  .block-container { padding-top: 1.5rem; }
  .stMetric { background: #0f1520; border: 1px solid #1e3a5f;
              border-radius: 8px; padding: 0.75rem; }
  div[data-testid="metric-container"] label {
      font-size: 11px; color: #7d93b2; letter-spacing: 0.1em;
      text-transform: uppercase; }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
      font-size: 1.6rem; font-weight: 700; color: #e2e8f0; }
  .stAlert { border-radius: 8px; }
  code { background: #141c2b !important; color: #38bdf8 !important; }
</style>
""", unsafe_allow_html=True)

# ── Заголовок ─────────────────────────────────────────────────────
st.markdown(
    "<h1 style='color:#38bdf8;letter-spacing:-.02em;"
    "font-family:JetBrains Mono,monospace;'>"
    "⚡ Process Model Analyzer</h1>",
    unsafe_allow_html=True,
)
st.caption(
    "FOPDT · SOPDT · Интегрирующий процесс · "
    "PID расчёт (Desired Response Method)"
)

# ══════════════════════════════════════════════════════════════════
#  БОКОВАЯ ПАНЕЛЬ
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚙️ Параметры")

    # ── Загрузка файла ─────────────────────────────────────────────
    st.markdown("#### 📂 Данные")
    uploaded = st.file_uploader(
        "CSV или Excel",
        type=["csv", "xlsx", "xls"],
        help="Файл должен содержать колонки MV (выход регулятора) "
             "и PV (значение процесса)"
    )

    st.divider()

    # ── Шкалы приборов ─────────────────────────────────────────────
    st.markdown("#### 📏 Шкалы приборов")
    st.caption("Используются для перевода K в % к %")

    c1, c2 = st.columns(2)
    mv_min = c1.number_input("MV min", value=0.0, step=1.0,
                              key="mv_min")
    mv_max = c2.number_input("MV max", value=100.0, step=1.0,
                              key="mv_max")
    c3, c4 = st.columns(2)
    pv_min = c3.number_input("PV min", value=0.0, step=1.0,
                              key="pv_min")
    pv_max = c4.number_input("PV max", value=100.0, step=1.0,
                              key="pv_max")

    st.divider()

    # ── Сглаживание ────────────────────────────────────────────────
    st.markdown("#### 🔇 Шумоподавление")
    smooth_window = st.slider(
        "Окно фильтра (Savitzky-Golay)",
        min_value=1, max_value=51, value=1, step=2,
        help="1 = без сглаживания. Увеличивайте при сильном шуме."
    )

    st.divider()

    # ── PID настройки ──────────────────────────────────────────────
    st.markdown("#### 🎛️ PID настройки")
    pid_type = st.radio("Тип регулятора", ["PI", "PID"],
                        horizontal=True)
    pid_e_val = st.number_input(
        "e — desired response time",
        value=0.0, step=0.1, format="%.4f",
        help="Желаемое время отклика замкнутого контура. "
             "Если 0 — рассчитывается как T + d автоматически."
    )
    pid_sp = st.number_input("Setpoint (%)", value=60.0,
                              step=1.0)

    st.divider()

    # ── Корректировка PID ──────────────────────────────────────────
    st.markdown("#### ✏️ Корректировка PID")
    st.caption("Перезапишите рекомендованные значения")
    pid_kc_override = st.number_input("Gain Kc",  value=0.0,
                                       step=0.01, format="%.4f")
    pid_ti_override = st.number_input("Reset Ti", value=0.0,
                                       step=0.01, format="%.4f")
    pid_td_override = st.number_input("Deriv Td", value=0.0,
                                       step=0.01, format="%.4f")
    st.caption("Оставьте 0 для использования рекомендованных")

# ══════════════════════════════════════════════════════════════════
#  ОСНОВНАЯ ОБЛАСТЬ
# ══════════════════════════════════════════════════════════════════

if uploaded is None:
    # ── Заглушка при отсутствии данных ────────────────────────────
    st.info(
        "👈 Загрузите файл CSV или Excel в боковой панели слева.\n\n"
        "Файл должен содержать минимум две числовые колонки:\n"
        "- **MV** — выход регулятора (управляющее воздействие)\n"
        "- **PV** — значение процесса (отклик)\n\n"
        "Опционально: колонка времени."
    )
    st.stop()

# ── Чтение файла ──────────────────────────────────────────────────
try:
    if uploaded.name.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(uploaded.read()),
                         sep=None, engine="python")
    else:
        df = pd.read_excel(io.BytesIO(uploaded.read()))
except Exception as ex:
    st.error(f"Ошибка чтения файла: {ex}")
    st.stop()

cols = list(df.columns)

# ── Выбор колонок ─────────────────────────────────────────────────
st.markdown("### 1️⃣ Назначение колонок")
c1, c2, c3 = st.columns(3)

default_mv = guess_col(cols, [r"\.MV$", r"^MV$", r"output",
                               r"выход", r"регул", r"mv"])
default_pv = guess_col(cols, [r"\.PV$", r"^PV$", r"process",
                               r"процесс", r"pv"])
default_t  = next(
    (c for c in cols
     if re.match(r"^(t|time|время|timestamp|index)$",
                 str(c), re.I)),
    None
)

mv_col = c1.selectbox("MV — выход регулятора",  cols,
                       index=cols.index(default_mv)
                       if default_mv in cols else 0)
pv_col = c2.selectbox("PV — значение процесса", cols,
                       index=cols.index(default_pv)
                       if default_pv in cols else
                       min(1, len(cols)-1))
t_options = ["— индекс строки —"] + cols
t_col  = c3.selectbox("Время (опционально)", t_options,
                       index=(t_options.index(default_t)
                              if default_t in t_options else 0))

# ── Предпросмотр данных ───────────────────────────────────────────
with st.expander("🔍 Предпросмотр данных", expanded=False):
    st.dataframe(df[[mv_col, pv_col] +
                    ([t_col] if t_col != "— индекс строки —"
                     and t_col in df.columns else [])
                    ].head(20),
                 use_container_width=True)
    st.caption(f"Всего строк: {len(df)}")

# ── Кнопка анализа ────────────────────────────────────────────────
st.markdown("### 2️⃣ Идентификация")
run_btn = st.button("▶ Запустить анализ",
                    type="primary", use_container_width=False)

if not run_btn:
    st.caption("Нажмите кнопку после настройки параметров")
    st.stop()

# ══════════════════════════════════════════════════════════════════
#  ОБРАБОТКА ДАННЫХ И ИДЕНТИФИКАЦИЯ
# ══════════════════════════════════════════════════════════════════

with st.spinner("Анализируем данные..."):

    # Извлечение числовых массивов
    try:
        mv_raw = pd.to_numeric(df[mv_col], errors="coerce").values
        pv_raw = pd.to_numeric(df[pv_col], errors="coerce").values
        t_raw  = (
            np.arange(len(mv_raw), dtype=float)
            if t_col == "— индекс строки —"
            else pd.to_numeric(df[t_col], errors="coerce").values
        )
    except KeyError as ex:
        st.error(f"Колонка не найдена: {ex}")
        st.stop()

    # Удаление NaN
    mask  = ~(np.isnan(mv_raw) | np.isnan(pv_raw) |
              np.isnan(t_raw))
    t_arr  = t_raw[mask].astype(float)
    mv_arr = mv_raw[mask].astype(float)
    pv_arr = pv_raw[mask].astype(float)

    if len(t_arr) < 10:
        st.error("Слишком мало валидных строк после удаления NaN "
                 f"(осталось {len(t_arr)}).")
        st.stop()

    # Сглаживание
    if smooth_window > 1:
        mv_arr = denoise(mv_arr, smooth_window)
        pv_arr = denoise(pv_arr, smooth_window)

    # Нормировка в % относительно шкал приборов
    mv_span = max(mv_max - mv_min, 1e-6)
    pv_span = max(pv_max - pv_min, 1e-6)
    mv_pct  = (mv_arr - mv_min) / mv_span * 100.0
    pv_pct  = (pv_arr - pv_min) / pv_span * 100.0

    # Поиск ступеньки
    try:
        step = find_step(t_arr, mv_pct, pv_pct)
    except ValueError as ex:
        st.error(f"Ошибка поиска ступеньки: {ex}")
        st.stop()

    # Идентификация трёх моделей
    fopdt_m = FOPDTModel()
    sopdt_m = SOPDTModel()
    integr_m = IntegratingModel()

    models_dict = {
        "fopdt":      fopdt_m,
        "sopdt":      sopdt_m,
        "integrating": integr_m,
    }

    for m in models_dict.values():
        m.identify(step)
        if not m.error:
            m.sim_data = m.simulate(t_arr)
            m.r2, m.rmse = compute_metrics(
                pv_pct, m.sim_data,
                step.step_idx, step.end_idx)

    # Автовыбор лучшей модели
    best_key = detect_best_model(step, fopdt_m, sopdt_m, integr_m)

    # Расчёт PID (рекомендованные параметры)
    best_m   = models_dict[best_key]
    p        = best_m.params_dict()
    K_pid    = p.get("K", p.get("Ki", 1.0))
    T_pid    = p.get("τ", p.get("τ₁", 1.0))
    d_pid    = p.get("θ", 0.0)
    e_pid    = pid_e_val if pid_e_val > 0 else None
    pid_rec  = PIDCalculator.calculate(K_pid, T_pid, d_pid,
                                       e_pid, pid_type)

    # Параметры для применения (корректировка или рекомендованные)
    Kc_use = pid_kc_override if pid_kc_override != 0 else pid_rec["Gain (Kc)"]
    Ti_use = pid_ti_override if pid_ti_override != 0 else pid_rec["Reset (Ti)"]
    Td_use = pid_td_override if pid_td_override != 0 else (pid_rec.get("Derivative (Td)") or 0.0)

# ══════════════════════════════════════════════════════════════════
#  ВЫВОД РЕЗУЛЬТАТОВ
# ══════════════════════════════════════════════════════════════════

st.success(
    f"✓ Анализ завершён · {len(t_arr)} точек · "
    f"Ступенька в t = {step.tStep:.3f} · "
    f"Рекомендована модель: **{best_key.upper()}**"
)

# ── Метрики лучшей модели ─────────────────────────────────────────
st.markdown("### 3️⃣ Параметры модели")

model_labels = {
    "fopdt":       "FOPDT",
    "sopdt":       "SOPDT",
    "integrating": "Интегрирующий",
}

# Выбор отображаемой модели
display_key = st.radio(
    "Показать модель:",
    list(models_dict.keys()),
    format_func=lambda k: (
        f"{model_labels[k]}"
        + (" ✓ авто" if k == best_key else "")
        + (f"  R²={models_dict[k].r2:.3f}"
           if not models_dict[k].error and
           models_dict[k].r2 is not None else "  (ошибка)")
    ),
    horizontal=True,
    index=list(models_dict.keys()).index(best_key),
)
display_m = models_dict[display_key]

if display_m.error:
    st.warning(f"Ошибка модели {display_key}: {display_m.error}")
else:
    # Метрики
    params = display_m.params_dict()
    metric_cols = st.columns(len(params) + 2)
    for idx, (k, v) in enumerate(params.items()):
        metric_cols[idx].metric(k, f"{v:.4f}")
    metric_cols[len(params)].metric("R²",   f"{display_m.r2:.4f}")
    metric_cols[len(params)+1].metric("RMSE", f"{display_m.rmse:.4f}")

    st.code(display_m.tf_string(), language=None)

# ── График идентификации ──────────────────────────────────────────
st.markdown("### 4️⃣ Графики идентификации")

fig_id = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    row_heights=[0.25, 0.45, 0.30],
    subplot_titles=[
        "MV, %",
        "PV — данные и модели, %",
        f"Сравнение моделей (R²)",
    ],
    vertical_spacing=0.08,
)

# MV
fig_id.add_trace(
    go.Scatter(x=t_arr, y=mv_pct, name="MV",
               line=dict(color="#fbbf24", width=1.5),
               mode="lines"),
    row=1, col=1,
)
fig_id.add_vline(x=step.tStep, line_dash="dash",
                 line_color="#f87171", line_width=1,
                 annotation_text=f"t_step={step.tStep:.2f}",
                 annotation_font_color="#f87171", row=1)

# PV + все модели
fig_id.add_trace(
    go.Scatter(x=t_arr, y=pv_pct, name="PV (данные)",
               line=dict(color="#e2e8f0", width=1.5),
               mode="lines"),
    row=2, col=1,
)

model_colors = {
    "fopdt":       "#38bdf8",
    "sopdt":       "#a78bfa",
    "integrating": "#fbbf24",
}
for key, m in models_dict.items():
    if not m.error and m.sim_data is not None:
        r2_label = (f"  R²={m.r2:.3f}"
                    if m.r2 is not None else "")
        fig_id.add_trace(
            go.Scatter(
                x=t_arr, y=m.sim_data,
                name=f"{model_labels[key]}{r2_label}",
                line=dict(color=model_colors[key],
                          width=2, dash="dash"),
                mode="lines",
            ),
            row=2, col=1,
        )

# Маркеры 28.3 % и 63.2 % для FOPDT
if not fopdt_m.error and fopdt_m.t283 and fopdt_m.t632:
    p283_val = fopdt_m.pv0 + 0.283 * step.delta_pv
    p632_val = fopdt_m.pv0 + 0.632 * step.delta_pv
    fig_id.add_trace(
        go.Scatter(
            x=[fopdt_m.t283, fopdt_m.t632],
            y=[p283_val,     p632_val],
            name="28.3% / 63.2%",
            mode="markers",
            marker=dict(color="#34d399", size=10,
                        symbol="circle-open", line=dict(width=2)),
        ),
        row=2, col=1,
    )

# Сравнение R²
valid_models = {k: m for k, m in models_dict.items()
                if not m.error and m.r2 is not None}
fig_id.add_trace(
    go.Bar(
        x=[model_labels[k] for k in valid_models],
        y=[m.r2 for m in valid_models.values()],
        marker_color=[model_colors[k] for k in valid_models],
        name="R²",
        text=[f"{m.r2:.4f}" for m in valid_models.values()],
        textposition="outside",
    ),
    row=3, col=1,
)
fig_id.update_yaxes(range=[
    min(0.9, min(m.r2 for m in valid_models.values()) - 0.05)
    if valid_models else 0,
    1.02
], row=3)

fig_id.update_layout(height=700, showlegend=True, **PLOTLY_LAYOUT)
apply_theme(fig_id, rows=3)
st.plotly_chart(fig_id, use_container_width=True)

# ── Таблица сравнения моделей ─────────────────────────────────────
st.markdown("### 5️⃣ Сравнение моделей")

cmp_rows = []
for key, m in models_dict.items():
    row = {"Модель": model_labels[key]}
    if m.error:
        row.update({"R²": "—", "RMSE": "—",
                    "Параметры": m.error,
                    "Ф-ция передачи": "—"})
    else:
        params_str = "  ".join(
            f"{k}={v:.4f}" for k, v in m.params_dict().items()
        )
        row.update({
            "R²":   f"{m.r2:.4f}",
            "RMSE": f"{m.rmse:.4f}",
            "Параметры": params_str,
            "Ф-ция передачи": m.tf_string(),
        })
    if key == best_key:
        row["Модель"] += " ✓"
    cmp_rows.append(row)

st.dataframe(
    pd.DataFrame(cmp_rows).set_index("Модель"),
    use_container_width=True,
)

# ══════════════════════════════════════════════════════════════════
#  PID РАЗДЕЛ
# ══════════════════════════════════════════════════════════════════

st.markdown("### 6️⃣ PID параметры")

if best_m.error:
    st.warning(
        "Лучшая модель содержит ошибку. "
        "PID расчёт невозможен."
    )
else:
    # Рекомендованные параметры
    st.markdown("#### Рекомендованные параметры "
                "(Desired Response Method)")

    pid_cols = st.columns(4)
    pid_cols[0].metric("Gain Kc",   f"{pid_rec['Gain (Kc)']:.4f}")
    pid_cols[1].metric("Reset Ti",  f"{pid_rec['Reset (Ti)']:.4f}")
    if pid_rec.get("Derivative (Td)") is not None:
        pid_cols[2].metric("Deriv Td",
                           f"{pid_rec['Derivative (Td)']:.4f}")
    pid_cols[3].metric("e (response time)",
                       f"{pid_rec['e (response time)']:.4f}")

    with st.expander("📐 Формулы расчёта"):
        st.markdown(f"""
**Метод:** Desired Closed Loop Response

| Параметр | Формула | Значение |
|----------|---------|---------|
| Gain (Kc) | `(2T + d) / (K · (2e + d))` | `{pid_rec['Gain (Kc)']:.4f}` |
| Reset (Ti) | `T + d/2` | `{pid_rec['Reset (Ti)']:.4f}` |
| Deriv (Td) | `T·d / (2T + d)` *(только PID)* | `{pid_rec.get('Derivative (Td)') or '—'}` |

**Параметры процесса:**
- K = `{K_pid:.4f}` (усиление, % к %)
- T = `{T_pid:.4f}` (постоянная времени)
- d = `{d_pid:.4f}` (запаздывание)
- e = `{pid_rec['e (response time)']:.4f}` (desired response time)
        """)

    # Применённые параметры (с учётом корректировки)
    if any([pid_kc_override != 0,
            pid_ti_override != 0,
            pid_td_override != 0]):
        st.markdown("#### Применённые параметры (скорректированные)")
        a1, a2, a3 = st.columns(3)
        a1.metric("Gain Kc",  f"{Kc_use:.4f}",
                  delta=f"{Kc_use - pid_rec['Gain (Kc)']:+.4f}")
        a2.metric("Reset Ti", f"{Ti_use:.4f}",
                  delta=f"{Ti_use - pid_rec['Reset (Ti)']:+.4f}")
        if pid_type == "PID":
            a3.metric("Deriv Td", f"{Td_use:.4f}",
                      delta=f"{Td_use - (pid_rec.get('Derivative (Td)') or 0):+.4f}")

    # ── График PID симуляции ───────────────────────────────────────
    st.markdown("#### Визуализация PID регулятора")

    # Симуляция с рекомендованными параметрами
    mv_rec_sim, err_rec = PIDCalculator.simulate_pid(
        t_arr, pv_pct, pid_sp,
        Kc=pid_rec["Gain (Kc)"],
        Ti=pid_rec["Reset (Ti)"],
        Td=pid_rec.get("Derivative (Td)") or 0.0,
        mv_init=float(np.mean(mv_pct[:max(5, step.step_idx)])),
    )

    # Симуляция с применёнными параметрами
    mv_user_sim, err_user = PIDCalculator.simulate_pid(
        t_arr, pv_pct, pid_sp,
        Kc=Kc_use, Ti=Ti_use, Td=Td_use,
        mv_init=float(np.mean(mv_pct[:max(5, step.step_idx)])),
    )

    fig_pid = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.40, 0.35, 0.25],
        subplot_titles=[
            "PV процесса и setpoint, %",
            "MV — выход регулятора, %",
            "Ошибка регулирования e(t), %",
        ],
        vertical_spacing=0.09,
    )

    # PV
    fig_pid.add_trace(
        go.Scatter(x=t_arr, y=pv_pct, name="PV (исходные данные)",
                   line=dict(color="#7d93b2", width=1, dash="dot"),
                   mode="lines"),
        row=1, col=1,
    )
    fig_pid.add_hline(
        y=pid_sp, line_dash="dash", line_color="#f87171",
        line_width=1.5,
        annotation_text=f"SP = {pid_sp:.1f} %",
        annotation_font_color="#f87171",
    )

    # MV рекомендованные и пользовательские
    fig_pid.add_trace(
        go.Scatter(
            x=t_arr, y=mv_rec_sim,
            name=(f"MV рекоменд. "
                  f"(Kc={pid_rec['Gain (Kc)']:.3f}, "
                  f"Ti={pid_rec['Reset (Ti)']:.3f})"),
            line=dict(color="#fbbf24", width=2),
            mode="lines",
        ),
        row=2, col=1,
    )
    if any([pid_kc_override != 0,
            pid_ti_override != 0,
            pid_td_override != 0]):
        fig_pid.add_trace(
            go.Scatter(
                x=t_arr, y=mv_user_sim,
                name=(f"MV скоррект. "
                      f"(Kc={Kc_use:.3f}, Ti={Ti_use:.3f})"),
                line=dict(color="#34d399", width=2, dash="dash"),
                mode="lines",
            ),
            row=2, col=1,
        )

    # Ошибка
    fig_pid.add_trace(
        go.Scatter(x=t_arr, y=pid_sp - pv_pct,
                   name="e(t) = SP − PV",
                   line=dict(color="#f87171", width=1.5),
                   fill="tozeroy",
                   fillcolor="rgba(248,113,113,0.08)",
                   mode="lines"),
        row=3, col=1,
    )
    fig_pid.add_hline(y=0, line_color="#3d5a7a",
                      line_width=0.8, row=3)

    fig_pid.update_layout(height=680, showlegend=True,
                          **PLOTLY_LAYOUT)
    apply_theme(fig_pid, rows=3)
    st.plotly_chart(fig_pid, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
#  ПОДВАЛ
# ══════════════════════════════════════════════════════════════════

st.divider()
st.caption(
    "Process Model Analyzer · "
    "FOPDT · SOPDT · Integrating · "
    "Desired Response PID Method · v2.0"
)
