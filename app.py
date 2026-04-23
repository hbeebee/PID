"""
app.py — Streamlit веб-интерфейс Process Model Analyzer v3.0
=============================================================
Изменения v3:
  - Анализ ВСЕХ ступенчатых воздействий в файле
  - Усреднённые параметры модели по всем ступенькам
  - Переключение модели без перезапуска анализа (st.session_state)
  - Белый читаемый текст метрик
  - Линии моделей строятся как реакция на КАЖДОЕ воздействие
  - Исправлена цветовая схема текста

Запуск: streamlit run app.py
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
#  МОДУЛИ ИДЕНТИФИКАЦИИ
# ══════════════════════════════════════════════════════════════════

def denoise(signal: np.ndarray, window: int = 5) -> np.ndarray:
    """Savitzky-Golay сглаживание сигнала."""
    n = len(signal)
    if n < window + 2:
        return signal.copy()
    wl = window if window % 2 == 1 else window + 1
    wl = min(wl, n - 1 if (n - 1) % 2 == 1 else n - 2)
    try:
        return savgol_filter(signal, window_length=wl, polyorder=2)
    except Exception:
        return signal.copy()


class StepInfo:
    """Одно ступенчатое воздействие в данных."""
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


def find_all_steps(t: np.ndarray, mv: np.ndarray,
                   pv: np.ndarray,
                   min_step_frac: float = 0.1) -> list:
    """
    Находит ВСЕ ступенчатые изменения MV в данных.
    min_step_frac — минимальный размер скачка как доля от
    максимального скачка (фильтрует шум).
    Возвращает список StepInfo.
    """
    diffs   = np.abs(np.diff(mv))
    max_d   = np.max(diffs)
    if max_d < 1e-9:
        raise ValueError("Ступенчатые изменения MV не найдены")

    threshold = max_d * min_step_frac
    step_indices = [i + 1 for i, d in enumerate(diffs)
                    if d >= threshold]

    if not step_indices:
        raise ValueError("Нет значимых изменений MV")

    # Формируем участки: от ступеньки i до ступеньки i+1
    steps = []
    for k, si in enumerate(step_indices):
        end_idx = (step_indices[k + 1]
                   if k + 1 < len(step_indices)
                   else len(t))

        mv0      = mv[si - 1]
        delta_mv = mv[si] - mv0

        pre  = pv[max(0, si - 5):si]
        pv0  = float(np.mean(pre)) if len(pre) > 0 else pv[si]

        # Финальное PV — последние 15 % плато
        plateau  = pv[si:end_idx]
        sz       = max(3, int(len(plateau) * 0.15))
        pv_final = float(np.mean(plateau[-sz:]))
        delta_pv = pv_final - pv0

        steps.append(StepInfo(
            si, end_idx, t[si],
            mv0, delta_mv, pv0, pv_final, delta_pv,
            t, mv, pv
        ))

    return steps


def compute_metrics(actual: np.ndarray,
                    predicted: np.ndarray,
                    start: int, end: int):
    """R² и RMSE на участке [start:end]."""
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
    G(s) = K · exp(-θ·s) / (τ·s + 1)
    Метод: две точки 28.3 % и 63.2 %.
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
                f"{'28.3%' if t283 is None else '63.2%'} порога"
            )
            return

        self.t283  = t283
        self.t632  = t632
        self.tau   = 1.5 * (t632 - t283)
        self.theta = t632 - self.tau

    def simulate_step(self, t_local: np.ndarray) -> np.ndarray:
        """
        Симуляция отклика на ОДНУ ступеньку.
        t_local — время начиная с момента ступеньки (t - tStep).
        """
        if self.error:
            return np.full(len(t_local), self.pv0 or 0.0)
        out = np.where(
            t_local - self.theta <= 0,
            self.pv0,
            self.pv0 + self.K * self.delta_mv *
            (1 - np.exp(-(t_local - self.theta) /
                        max(self.tau, 1e-9)))
        )
        return out

    def simulate(self, t: np.ndarray) -> np.ndarray:
        if self.error:
            return np.zeros(len(t))
        return self.simulate_step(t - self.tStep)

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
    """G(s) = K·exp(-θ·s) / ((τ₁·s+1)(τ₂·s+1))"""
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

        plateau   = s.pv[s.step_idx:s.end_idx]
        pv_ext    = (np.max(plateau) if s.delta_pv > 0
                     else np.min(plateau))
        overshoot = abs((pv_ext - s.pv_final) /
                        (s.delta_pv + 1e-12))

        p264 = s.pv0 + 0.264 * s.delta_pv
        p632 = s.pv0 + 0.632 * s.delta_pv
        t264 = t632 = None

        for i in range(s.step_idx, s.end_idx - 1):
            def _c(pct, ii=i):
                return (s.pv[ii] >= pct if s.delta_pv > 0
                        else s.pv[ii] <= pct)
            if t264 is None and _c(p264):
                frac = ((p264 - s.pv[i]) /
                        (s.pv[i+1] - s.pv[i] + 1e-12))
                t264 = s.t[i] + frac * (s.t[i+1] - s.t[i])
            if t632 is None and _c(p632):
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

    def _step_response(self, d: float) -> float:
        """Единичный отклик в момент d после ступеньки."""
        if d <= 0:
            return 0.0
        if abs(self.zeta - 1.0) < 0.01:
            wn = 1 / max(self.tau1, 1e-9)
            return 1 - np.exp(-wn * d) * (1 + wn * d)
        elif self.zeta > 1:
            a = -1 / max(self.tau1, 1e-9)
            b = -1 / max(self.tau2, 1e-9)
            if abs(a - b) < 1e-9:
                a *= 1.001
            return 1 + (b * np.exp(a * d) - a * np.exp(b * d)) / (a - b)
        else:
            wd = (1 / max(self.tau1, 1e-9)) * np.sqrt(1 - self.zeta**2)
            return 1 - np.exp(-self.zeta / max(self.tau1, 1e-9) * d) * (
                np.cos(wd * d) +
                self.zeta / max(np.sqrt(1 - self.zeta**2), 1e-9) *
                np.sin(wd * d))

    def simulate_step(self, t_local: np.ndarray) -> np.ndarray:
        if self.error:
            return np.full(len(t_local), self.pv0 or 0.0)
        out = np.array([
            self.pv0 + self.K * self.delta_mv *
            self._step_response(d - self.theta)
            for d in t_local
        ])
        return out

    def simulate(self, t: np.ndarray) -> np.ndarray:
        if self.error:
            return np.zeros(len(t))
        return self.simulate_step(t - self.tStep)

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


# ── Интегрирующий ─────────────────────────────────────────────────

class IntegratingModel:
    """G(s) = Ki · exp(-θ·s) / s"""
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
        if len(t_r) < 3:
            self.error = "Мало точек"
            return

        t_m = np.mean(t_r); pv_m = np.mean(pv_r)
        num = np.sum((t_r - t_m) * (pv_r - pv_m))
        den = np.sum((t_r - t_m) ** 2)
        self.Ki = (num / max(den, 1e-12)) / s.delta_mv

        self.theta = 0.0
        for i in range(s.step_idx, s.end_idx):
            if abs(s.pv[i] - s.pv0) > abs(s.delta_pv) * 0.05:
                self.theta = s.t[i] - s.tStep
                break

    def simulate_step(self, t_local: np.ndarray) -> np.ndarray:
        if self.error:
            return np.full(len(t_local), self.pv0 or 0.0)
        dt = t_local - self.theta
        return np.where(dt <= 0, self.pv0,
                        self.pv0 + self.Ki * self.delta_mv * dt)

    def simulate(self, t: np.ndarray) -> np.ndarray:
        if self.error:
            return np.zeros(len(t))
        return self.simulate_step(t - self.tStep)

    def tf_string(self) -> str:
        if self.error:
            return "Ошибка идентификации"
        return (f"G(s) = {self.Ki:.4f} · exp(-{self.theta:.4f}·s)"
                f" / s")

    def params_dict(self) -> dict:
        if self.error:
            return {}
        return {"Ki": self.Ki, "θ": self.theta}


# ── PID ───────────────────────────────────────────────────────────

class PIDCalculator:
    """
    Desired Response Method:
      Kc = (2T + d) / (K·(2e + d))
      Ti = T + d/2
      Td = T·d / (2T + d)
    """
    @staticmethod
    def calculate(K, T, d, e=None, controller="PID"):
        if e is None or e <= 0:
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
    def simulate_pid(t, pv, setpoint, Kc, Ti, Td=0.0,
                     mv_init=50.0, mv_min=0.0, mv_max=100.0):
        n = len(t)
        mv = np.zeros(n); err = np.zeros(n)
        integral = 0.0; prev_err = 0.0
        mv[0] = mv_init
        for i in range(1, n):
            dt = max(t[i] - t[i-1], 1e-6)
            err[i]   = setpoint - pv[i]
            integral += err[i] * dt
            deriv    = (err[i] - prev_err) / dt if Td > 0 else 0.0
            output   = Kc * (err[i] + integral / max(Ti, 1e-9) +
                             Td * deriv)
            mv[i]    = np.clip(mv_init + output, mv_min, mv_max)
            prev_err = err[i]
        return mv, err


# ══════════════════════════════════════════════════════════════════
#  МУЛЬТИ-СТУПЕНЧАТЫЙ АНАЛИЗ
# ══════════════════════════════════════════════════════════════════

MODEL_CLASSES = {
    "fopdt":       FOPDTModel,
    "sopdt":       SOPDTModel,
    "integrating": IntegratingModel,
}
MODEL_LABELS = {
    "fopdt":       "FOPDT",
    "sopdt":       "SOPDT",
    "integrating": "Интегрирующий",
}
MODEL_COLORS = {
    "fopdt":       "#38bdf8",
    "sopdt":       "#a78bfa",
    "integrating": "#fbbf24",
}


def run_multi_step_analysis(t, mv_pct, pv_pct):
    """
    Анализирует ВСЕ ступенчатые воздействия.
    Возвращает:
      steps        — список StepInfo
      per_step     — {step_idx: {model_key: model_instance}}
      avg_params   — {model_key: усреднённые параметры}
      avg_models   — {model_key: модель с усреднёнными параметрами}
      full_sim     — {model_key: np.ndarray симуляции на всём ряду}
      best_key     — ключ лучшей модели
    """
    steps = find_all_steps(t, mv_pct, pv_pct)

    per_step = {}
    # Для каждой ступеньки идентифицируем все три модели
    for s in steps:
        si = s.step_idx
        step_models = {}
        for key, Cls in MODEL_CLASSES.items():
            m = Cls()
            m.identify(s)
            if not m.error:
                m.sim_data = m.simulate(t)
                m.r2, m.rmse = compute_metrics(
                    pv_pct, m.sim_data, s.step_idx, s.end_idx)
            step_models[key] = m
        per_step[si] = step_models

    # Усреднение параметров по всем ступенькам
    avg_params = {}
    for key in MODEL_CLASSES:
        all_p = [per_step[s.step_idx][key].params_dict()
                 for s in steps
                 if not per_step[s.step_idx][key].error]
        if not all_p:
            avg_params[key] = None
            continue
        merged = {}
        for pk in all_p[0]:
            vals = [p[pk] for p in all_p if pk in p]
            merged[pk] = float(np.mean(vals))
        avg_params[key] = merged

    # Строим «усреднённую» полную симуляцию —
    # для каждой ступеньки применяем усреднённые параметры
    full_sim = {}
    avg_models = {}

    for key, Cls in MODEL_CLASSES.items():
        p_avg = avg_params.get(key)
        if p_avg is None:
            full_sim[key]   = np.zeros(len(t))
            avg_models[key] = Cls()
            avg_models[key].error = "Нет данных"
            continue

        sim = np.full(len(t), np.nan)
        # Сначала заполняем сегмент ДО первой ступеньки
        sim[:steps[0].step_idx] = pv_pct[:steps[0].step_idx]

        for k, s in enumerate(steps):
            # Создаём модель с усреднёнными параметрами
            m = Cls()
            # Устанавливаем параметры через params_dict-совместимые атрибуты
            _set_avg_params(m, key, p_avg, s)
            t_local = t[s.step_idx:s.end_idx] - s.tStep
            seg     = m.simulate_step(t_local)
            sim[s.step_idx:s.end_idx] = seg

        # Заполняем NaN (если вдруг остались)
        mask = np.isnan(sim)
        sim[mask] = pv_pct[mask]

        full_sim[key] = sim

        # Создаём «представительскую» модель для отображения параметров
        rep_m = Cls()
        _set_avg_params(rep_m, key, p_avg, steps[0])
        rep_m.r2, rep_m.rmse = compute_metrics(
            pv_pct, sim, steps[0].step_idx, steps[-1].end_idx)
        rep_m.sim_data = sim
        avg_models[key] = rep_m

    # Автовыбор лучшей модели по среднему R²
    r2_scores = {}
    for key in MODEL_CLASSES:
        m = avg_models.get(key)
        if m and not m.error and m.r2 is not None:
            r2_scores[key] = m.r2
    best_key = max(r2_scores, key=r2_scores.get) if r2_scores else "fopdt"

    return steps, per_step, avg_params, avg_models, full_sim, best_key


def _set_avg_params(model, key: str, p_avg: dict, s: StepInfo):
    """Устанавливает усреднённые параметры в экземпляр модели."""
    model.pv0      = s.pv0
    model.pv_final = s.pv_final
    model.delta_mv = s.delta_mv
    model.tStep    = s.tStep

    if key == "fopdt":
        model.K     = p_avg.get("K", 1.0)
        model.tau   = p_avg.get("τ", 1.0)
        model.theta = p_avg.get("θ", 0.0)
    elif key == "sopdt":
        model.K     = p_avg.get("K", 1.0)
        model.tau1  = p_avg.get("τ₁", 1.0)
        model.tau2  = p_avg.get("τ₂", 1.0)
        model.theta = p_avg.get("θ", 0.0)
        model.zeta  = p_avg.get("ζ", 1.0)
    elif key == "integrating":
        model.Ki    = p_avg.get("Ki", 1.0)
        model.theta = p_avg.get("θ", 0.0)


def guess_col(columns, patterns):
    for pat in patterns:
        m = [c for c in columns
             if re.search(pat, str(c), re.IGNORECASE)]
        if m:
            return m[0]
    return columns[0] if columns else ""


# ══════════════════════════════════════════════════════════════════
#  ТЕМА PLOTLY
# ══════════════════════════════════════════════════════════════════

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0f1520",
    plot_bgcolor="#141c2b",
    font=dict(color="#e2e8f0", family="monospace", size=11),
    legend=dict(bgcolor="#0f1520", bordercolor="#1e3a5f",
                borderwidth=1, font=dict(color="#e2e8f0")),
    margin=dict(l=55, r=20, t=40, b=40),
)


def apply_theme(fig, rows=1):
    fig.update_layout(**PLOTLY_LAYOUT)
    for i in range(1, rows + 1):
        fig.update_xaxes(gridcolor="#1e3a5f",
                         zerolinecolor="#1e3a5f",
                         tickfont=dict(color="#e2e8f0"),
                         title_font=dict(color="#e2e8f0"), row=i)
        fig.update_yaxes(gridcolor="#1e3a5f",
                         zerolinecolor="#1e3a5f",
                         tickfont=dict(color="#e2e8f0"),
                         title_font=dict(color="#e2e8f0"), row=i)
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

# ── CSS: белый текст везде, читаемые метрики ──────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    color: #e2e8f0 !important;
}

/* Метрики — белый текст, тёмный фон, цветная рамка */
div[data-testid="metric-container"] {
    background: #0f1520;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 0.75rem 1rem;
}
div[data-testid="metric-container"] label,
div[data-testid="metric-container"] [data-testid="stMetricLabel"] p {
    color: #7dd3fc !important;
    font-size: 11px !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"],
div[data-testid="metric-container"] [data-testid="stMetricValue"] * {
    color: #ffffff !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
}
div[data-testid="metric-container"] [data-testid="stMetricDelta"],
div[data-testid="metric-container"] [data-testid="stMetricDelta"] * {
    color: #34d399 !important;
}

/* Таблицы */
div[data-testid="stDataFrame"] { color: #e2e8f0 !important; }

/* Текст в expander и caption */
.streamlit-expanderContent p,
.streamlit-expanderHeader p,
[data-testid="stCaptionContainer"] p {
    color: #7d93b2 !important;
}

/* Код */
code, pre { color: #38bdf8 !important; background: #141c2b !important; }

/* Успех/предупреждение/ошибка */
.stAlert { border-radius: 8px; }
.stSuccess { background: rgba(52,211,153,0.1) !important;
             border-color: #34d399 !important; color: #e2e8f0 !important; }
.stWarning { background: rgba(251,191,36,0.1) !important; }
.stError   { background: rgba(248,113,113,0.1) !important; }

/* Radio кнопки */
.stRadio label { color: #e2e8f0 !important; }
.stRadio > div > label > div > p { color: #e2e8f0 !important; }

/* Основной заголовок */
h1 { color: #38bdf8 !important; }
h2, h3 { color: #e2e8f0 !important; }

/* Selectbox / input labels */
label[data-testid="stWidgetLabel"] p { color: #7dd3fc !important; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #0a0e14 !important; }
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] label p { color: #7dd3fc !important; }
</style>
""", unsafe_allow_html=True)

# ── Заголовок ─────────────────────────────────────────────────────
st.markdown(
    "<h1>⚡ Process Model Analyzer</h1>",
    unsafe_allow_html=True,
)
st.caption(
    "Мульти-ступенчатый анализ · "
    "FOPDT · SOPDT · Интегрирующий · "
    "PID (Desired Response Method) v3.0"
)

# ── Session state — хранит результаты анализа ─────────────────────
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done   = False
    st.session_state.display_key     = "fopdt"
    st.session_state.results         = None
    st.session_state.arrays          = None

# ══════════════════════════════════════════════════════════════════
#  БОКОВАЯ ПАНЕЛЬ
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚙️ Параметры")

    st.markdown("#### 📂 Данные")
    uploaded = st.file_uploader(
        "CSV или Excel",
        type=["csv", "xlsx", "xls"],
    )

    st.divider()

    st.markdown("#### 📏 Шкалы приборов")
    st.caption("Для перевода K в % к %")
    c1, c2 = st.columns(2)
    mv_min = c1.number_input("MV min", value=0.0, step=1.0)
    mv_max = c2.number_input("MV max", value=100.0, step=1.0)
    c3, c4 = st.columns(2)
    pv_min = c3.number_input("PV min", value=0.0, step=1.0)
    pv_max = c4.number_input("PV max", value=100.0, step=1.0)

    st.divider()

    st.markdown("#### 🔇 Шумоподавление")
    smooth_window = st.slider(
        "Окно фильтра (Savitzky-Golay)",
        min_value=1, max_value=51, value=1, step=2,
    )

    st.divider()

    st.markdown("#### 🎛️ PID настройки")
    pid_type  = st.radio("Тип", ["PI", "PID"], horizontal=True)
    pid_e_val = st.number_input(
        "e — response time (0 = авто T+d)",
        value=0.0, step=0.1, format="%.4f",
    )
    pid_sp = st.number_input("Setpoint (%)", value=60.0, step=1.0)

    st.divider()

    st.markdown("#### ✏️ Корректировка PID")
    pid_kc_ov = st.number_input("Gain Kc",  value=0.0,
                                 step=0.001, format="%.4f")
    pid_ti_ov = st.number_input("Reset Ti", value=0.0,
                                 step=0.001, format="%.4f")
    pid_td_ov = st.number_input("Deriv Td", value=0.0,
                                 step=0.001, format="%.4f")
    st.caption("0 = использовать рекомендованные")

# ══════════════════════════════════════════════════════════════════
#  ОСНОВНАЯ ОБЛАСТЬ
# ══════════════════════════════════════════════════════════════════

if uploaded is None:
    st.info(
        "👈 Загрузите CSV или Excel в боковой панели.\n\n"
        "Файл должен содержать колонки **MV** и **PV**. "
        "Приложение автоматически найдёт все ступенчатые "
        "воздействия и усреднит параметры модели."
    )
    st.stop()

# ── Чтение файла ──────────────────────────────────────────────────
try:
    raw = uploaded.read()
    if uploaded.name.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
    else:
        df = pd.read_excel(io.BytesIO(raw))
except Exception as ex:
    st.error(f"Ошибка чтения файла: {ex}")
    st.stop()

cols = list(df.columns)

# ── Выбор колонок ─────────────────────────────────────────────────
st.markdown("### 1️⃣ Назначение колонок")
c1, c2, c3 = st.columns(3)

def_mv = guess_col(cols, [r"\.MV$", r"^MV$", r"output",
                           r"выход", r"регул", r"mv"])
def_pv = guess_col(cols, [r"\.PV$", r"^PV$", r"process",
                           r"процесс", r"pv"])
def_t  = next(
    (c for c in cols
     if re.match(r"^(t|time|время|timestamp|index)$",
                 str(c), re.I)),
    None
)
t_opts = ["— индекс строки —"] + cols

mv_col = c1.selectbox("MV — выход регулятора",  cols,
                       index=cols.index(def_mv) if def_mv in cols else 0)
pv_col = c2.selectbox("PV — значение процесса", cols,
                       index=min(cols.index(def_pv), len(cols)-1)
                       if def_pv in cols else min(1, len(cols)-1))
t_col  = c3.selectbox("Время (опционально)", t_opts,
                       index=(t_opts.index(def_t)
                              if def_t in t_opts else 0))

with st.expander("🔍 Предпросмотр данных"):
    show_cols = [mv_col, pv_col] + (
        [t_col] if t_col != "— индекс строки —"
        and t_col in df.columns else [])
    st.dataframe(df[show_cols].head(20),
                 use_container_width=True)
    st.caption(f"Всего строк: {len(df)}")

# ── Кнопка анализа ────────────────────────────────────────────────
st.markdown("### 2️⃣ Анализ")

if st.button("▶ Запустить анализ", type="primary"):
    with st.spinner("Анализируем данные…"):
        try:
            mv_raw = pd.to_numeric(df[mv_col], errors="coerce").values
            pv_raw = pd.to_numeric(df[pv_col], errors="coerce").values
            t_raw  = (
                np.arange(len(mv_raw), dtype=float)
                if t_col == "— индекс строки —"
                else pd.to_numeric(df[t_col],
                                   errors="coerce").values
            )
            mask  = ~(np.isnan(mv_raw) | np.isnan(pv_raw) |
                      np.isnan(t_raw))
            t_a   = t_raw[mask].astype(float)
            mv_a  = mv_raw[mask].astype(float)
            pv_a  = pv_raw[mask].astype(float)

            if len(t_a) < 10:
                st.error(f"Мало точек после фильтрации: {len(t_a)}")
                st.stop()

            if smooth_window > 1:
                mv_a = denoise(mv_a, smooth_window)
                pv_a = denoise(pv_a, smooth_window)

            mv_span = max(mv_max - mv_min, 1e-6)
            pv_span = max(pv_max - pv_min, 1e-6)
            mv_pct  = (mv_a - mv_min) / mv_span * 100.0
            pv_pct  = (pv_a - pv_min) / pv_span * 100.0

            (steps, per_step, avg_params,
             avg_models, full_sim, best_key) = \
                run_multi_step_analysis(t_a, mv_pct, pv_pct)

            st.session_state.analysis_done = True
            st.session_state.display_key   = best_key
            st.session_state.results = {
                "steps":      steps,
                "per_step":   per_step,
                "avg_params": avg_params,
                "avg_models": avg_models,
                "full_sim":   full_sim,
                "best_key":   best_key,
            }
            st.session_state.arrays = {
                "t":      t_a,
                "mv_pct": mv_pct,
                "pv_pct": pv_pct,
            }

        except Exception as ex:
            st.error(f"Ошибка анализа: {ex}")
            import traceback
            st.code(traceback.format_exc())

# ── Если анализ ещё не запускался ────────────────────────────────
if not st.session_state.analysis_done:
    st.caption("Нажмите кнопку для запуска анализа")
    st.stop()

# ══════════════════════════════════════════════════════════════════
#  РЕЗУЛЬТАТЫ (используем session_state — без сброса)
# ══════════════════════════════════════════════════════════════════

res    = st.session_state.results
arrays = st.session_state.arrays
t_a, mv_pct, pv_pct = arrays["t"], arrays["mv_pct"], arrays["pv_pct"]
steps      = res["steps"]
per_step   = res["per_step"]
avg_params = res["avg_params"]
avg_models = res["avg_models"]
full_sim   = res["full_sim"]
best_key   = res["best_key"]

n_steps = len(steps)
st.success(
    f"✓ Анализ завершён · {len(t_a)} точек · "
    f"{n_steps} ступенчатых воздействий · "
    f"Рекомендована модель: **{best_key.upper()}**"
)

# ══════════════════════════════════════════════════════════════════
#  РАЗДЕЛ 3: ПАРАМЕТРЫ МОДЕЛИ (переключение без сброса)
# ══════════════════════════════════════════════════════════════════

st.markdown("### 3️⃣ Параметры модели")

# Переключатель модели — через session_state, без перезапуска
def _on_model_change():
    st.session_state.display_key = st.session_state._model_radio

current_key = st.session_state.display_key

radio_opts  = list(MODEL_CLASSES.keys())
radio_labels = []
for k in radio_opts:
    m = avg_models.get(k)
    r2_str = (f"  R²={m.r2:.3f}"
              if m and not m.error and m.r2 is not None else "  (ошибка)")
    suffix = "  ✓ авто" if k == best_key else ""
    radio_labels.append(f"{MODEL_LABELS[k]}{suffix}{r2_str}")

st.radio(
    "Выбрать модель для отображения:",
    options=radio_opts,
    format_func=lambda k: radio_labels[radio_opts.index(k)],
    index=radio_opts.index(current_key),
    key="_model_radio",
    on_change=_on_model_change,
    horizontal=True,
)

display_key = st.session_state.display_key
display_m   = avg_models.get(display_key)

if display_m and display_m.error:
    st.warning(f"Модель {display_key}: {display_m.error}")
elif display_m:
    params = avg_models[display_key].params_dict()

    # Метрики с белым текстом
    n_p = len(params)
    metric_cols = st.columns(n_p + 2)
    for idx, (k, v) in enumerate(params.items()):
        metric_cols[idx].metric(k, f"{v:.4f}")
    metric_cols[n_p].metric("R²",   f"{display_m.r2:.4f}")
    metric_cols[n_p + 1].metric("RMSE", f"{display_m.rmse:.4f}")

    st.code(display_m.tf_string(), language=None)

    # Таблица параметров по каждой ступеньке
    with st.expander(
        f"📊 Параметры по каждой из {n_steps} ступенек", expanded=False
    ):
        rows = []
        for s in steps:
            si = s.step_idx
            m  = per_step[si][display_key]
            row = {
                "t ступеньки": f"{s.tStep:.1f}",
                "ΔMV":         f"{s.delta_mv:+.3f}",
                "PV₀":         f"{s.pv0:.3f}",
                "PV_fin":      f"{s.pv_final:.3f}",
            }
            if not m.error:
                for pk, pv_v in m.params_dict().items():
                    row[pk] = f"{pv_v:.4f}"
                row["R²"]   = f"{m.r2:.4f}"
                row["RMSE"] = f"{m.rmse:.4f}"
            else:
                row["Ошибка"] = m.error
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ══════════════════════════════════════════════════════════════════
#  РАЗДЕЛ 4: ГРАФИКИ ИДЕНТИФИКАЦИИ
# ══════════════════════════════════════════════════════════════════

st.markdown("### 4️⃣ Графики идентификации")

fig_id = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.30, 0.70],
    subplot_titles=["MV, %", "PV — данные и модели, %"],
    vertical_spacing=0.08,
)

# MV
fig_id.add_trace(
    go.Scatter(x=t_a, y=mv_pct, name="MV",
               line=dict(color="#fbbf24", width=1.5),
               mode="lines"),
    row=1, col=1,
)

# Вертикальные линии ступенек
for s in steps:
    fig_id.add_vline(
        x=s.tStep, line_dash="dash",
        line_color="rgba(248,113,113,0.4)", line_width=1,
        row=1,
    )

# PV данные
fig_id.add_trace(
    go.Scatter(x=t_a, y=pv_pct, name="PV (данные)",
               line=dict(color="#e2e8f0", width=1.8),
               mode="lines"),
    row=2, col=1,
)

# Линии ВСЕХ моделей
for key in MODEL_CLASSES:
    m = avg_models.get(key)
    if m and not m.error:
        sim = full_sim[key]
        r2_label = (f"  R²={m.r2:.3f}"
                    if m.r2 is not None else "")
        # Активная модель — жирнее
        lw   = 2.5 if key == display_key else 1.2
        dash = "solid" if key == display_key else "dot"
        fig_id.add_trace(
            go.Scatter(
                x=t_a, y=sim,
                name=f"{MODEL_LABELS[key]}{r2_label}",
                line=dict(color=MODEL_COLORS[key],
                          width=lw, dash=dash),
                mode="lines",
            ),
            row=2, col=1,
        )

# Маркеры 28.3 % / 63.2 % для FOPDT (первая ступенька)
if not avg_models["fopdt"].error:
    fopdt_pts_x, fopdt_pts_y = [], []
    for s in steps:
        m_step = per_step[s.step_idx]["fopdt"]
        if not m_step.error and m_step.t283 and m_step.t632:
            p283_v = s.pv0 + 0.283 * s.delta_pv
            p632_v = s.pv0 + 0.632 * s.delta_pv
            fopdt_pts_x += [m_step.t283, m_step.t632]
            fopdt_pts_y += [p283_v, p632_v]
    if fopdt_pts_x:
        fig_id.add_trace(
            go.Scatter(
                x=fopdt_pts_x, y=fopdt_pts_y,
                name="28.3% / 63.2% (FOPDT)",
                mode="markers",
                marker=dict(color="#34d399", size=9,
                            symbol="circle-open",
                            line=dict(width=2)),
            ),
            row=2, col=1,
        )

# Вертикальные линии на PV
for s in steps:
    fig_id.add_vline(
        x=s.tStep, line_dash="dash",
        line_color="rgba(248,113,113,0.3)", line_width=1,
        row=2,
    )

fig_id.update_layout(height=620, showlegend=True, **PLOTLY_LAYOUT)
apply_theme(fig_id, rows=2)
st.plotly_chart(fig_id, use_container_width=True)

# ── Сравнение моделей ──────────────────────────────────────────────
st.markdown("### 5️⃣ Сравнение моделей")

cmp_rows = []
for key in MODEL_CLASSES:
    m   = avg_models.get(key)
    row = {"Модель": MODEL_LABELS[key] +
           (" ✓ авто" if key == best_key else "")}
    if m and not m.error:
        params_str = "  ".join(
            f"{k}={v:.4f}" for k, v in m.params_dict().items())
        row.update({
            "R²":   f"{m.r2:.4f}",
            "RMSE": f"{m.rmse:.4f}",
            "Параметры (среднее)": params_str,
            "Передаточная функция": m.tf_string(),
        })
    else:
        row.update({"R²": "—", "RMSE": "—",
                    "Параметры (среднее)": (m.error if m else "—"),
                    "Передаточная функция": "—"})
    cmp_rows.append(row)

st.dataframe(
    pd.DataFrame(cmp_rows).set_index("Модель"),
    use_container_width=True,
)

# ══════════════════════════════════════════════════════════════════
#  РАЗДЕЛ 6: PID
# ══════════════════════════════════════════════════════════════════

st.markdown("### 6️⃣ PID параметры")

best_m = avg_models.get(best_key)
if not best_m or best_m.error:
    st.warning("Лучшая модель содержит ошибку — PID расчёт недоступен.")
else:
    p     = best_m.params_dict()
    K_pid = p.get("K", p.get("Ki", 1.0))
    T_pid = p.get("τ", p.get("τ₁", 1.0))
    d_pid = p.get("θ", 0.0)

    pid_rec = PIDCalculator.calculate(
        K_pid, T_pid, d_pid, pid_e_val or None, pid_type)

    # Рекомендованные
    st.markdown("#### Рекомендованные параметры")
    pc = st.columns(4)
    pc[0].metric("Gain Kc",  f"{pid_rec['Gain (Kc)']:.4f}")
    pc[1].metric("Reset Ti", f"{pid_rec['Reset (Ti)']:.4f}")
    if pid_rec.get("Derivative (Td)") is not None:
        pc[2].metric("Deriv Td",
                     f"{pid_rec['Derivative (Td)']:.4f}")
    pc[3].metric("e", f"{pid_rec['e (response time)']:.4f}")

    with st.expander("📐 Формулы расчёта"):
        st.markdown(f"""
**Метод:** Desired Closed Loop Response

| Параметр | Формула | Значение |
|----------|---------|---------|
| Gain Kc  | `(2T + d) / (K·(2e + d))` | **{pid_rec['Gain (Kc)']:.4f}** |
| Reset Ti | `T + d/2` | **{pid_rec['Reset (Ti)']:.4f}** |
| Deriv Td | `T·d / (2T + d)` | **{pid_rec.get('Derivative (Td)') or '—'}** |

**Параметры процесса (усреднённые по {n_steps} ступенькам):**

| K (% к %) | T (постоянная) | d (запаздывание) | e (response time) |
|-----------|---------------|-----------------|------------------|
| {K_pid:.4f} | {T_pid:.4f} | {d_pid:.4f} | {pid_rec['e (response time)']:.4f} |
        """)

    # Параметры для симуляции
    Kc_use = pid_kc_ov if pid_kc_ov != 0 else pid_rec["Gain (Kc)"]
    Ti_use = pid_ti_ov if pid_ti_ov != 0 else pid_rec["Reset (Ti)"]
    Td_use = (pid_td_ov if pid_td_ov != 0
              else (pid_rec.get("Derivative (Td)") or 0.0))

    if any([pid_kc_ov != 0, pid_ti_ov != 0, pid_td_ov != 0]):
        st.markdown("#### Применённые параметры (скорректированные)")
        ac = st.columns(3)
        ac[0].metric("Gain Kc",  f"{Kc_use:.4f}",
                     delta=f"{Kc_use - pid_rec['Gain (Kc)']:+.4f}")
        ac[1].metric("Reset Ti", f"{Ti_use:.4f}",
                     delta=f"{Ti_use - pid_rec['Reset (Ti)']:+.4f}")
        td_rec = pid_rec.get("Derivative (Td)") or 0.0
        if pid_type == "PID":
            ac[2].metric("Deriv Td", f"{Td_use:.4f}",
                         delta=f"{Td_use - td_rec:+.4f}")

    # ── График PID ──────────────────────────────────────────────────
    st.markdown("#### Визуализация PID регулятора")

    pv_sim = full_sim.get(best_key, pv_pct)
    mv_init_v = float(np.mean(mv_pct[:max(5, steps[0].step_idx)]))

    mv_rec_sim, _ = PIDCalculator.simulate_pid(
        t_a, pv_sim, pid_sp,
        pid_rec["Gain (Kc)"],
        pid_rec["Reset (Ti)"],
        pid_rec.get("Derivative (Td)") or 0.0,
        mv_init=mv_init_v,
    )
    mv_usr_sim, _ = PIDCalculator.simulate_pid(
        t_a, pv_sim, pid_sp,
        Kc_use, Ti_use, Td_use,
        mv_init=mv_init_v,
    )

    fig_pid = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.40, 0.35, 0.25],
        subplot_titles=[
            "PV и уставка, %",
            "MV — выход регулятора, %",
            "Ошибка e(t) = SP − PV, %",
        ],
        vertical_spacing=0.09,
    )

    fig_pid.add_trace(
        go.Scatter(x=t_a, y=pv_pct, name="PV (исходные данные)",
                   line=dict(color="#7d93b2", width=1, dash="dot")),
        row=1, col=1,
    )
    fig_pid.add_hline(
        y=pid_sp, line_dash="dash",
        line_color="#f87171", line_width=1.5,
        annotation_text=f"SP={pid_sp:.1f}%",
        annotation_font_color="#f87171",
    )

    fig_pid.add_trace(
        go.Scatter(
            x=t_a, y=mv_rec_sim,
            name=(f"MV рекоменд. "
                  f"Kc={pid_rec['Gain (Kc)']:.3f} "
                  f"Ti={pid_rec['Reset (Ti)']:.3f}"),
            line=dict(color="#fbbf24", width=2)),
        row=2, col=1,
    )
    if any([pid_kc_ov != 0, pid_ti_ov != 0, pid_td_ov != 0]):
        fig_pid.add_trace(
            go.Scatter(
                x=t_a, y=mv_usr_sim,
                name=(f"MV скоррект. "
                      f"Kc={Kc_use:.3f} Ti={Ti_use:.3f}"),
                line=dict(color="#34d399", width=2, dash="dash")),
            row=2, col=1,
        )

    fig_pid.add_trace(
        go.Scatter(
            x=t_a, y=pid_sp - pv_pct,
            name="e(t)", fill="tozeroy",
            fillcolor="rgba(248,113,113,0.08)",
            line=dict(color="#f87171", width=1.5)),
        row=3, col=1,
    )
    fig_pid.add_hline(y=0, line_color="#3d5a7a",
                      line_width=0.8, row=3)

    fig_pid.update_layout(height=650, showlegend=True,
                          **PLOTLY_LAYOUT)
    apply_theme(fig_pid, rows=3)
    st.plotly_chart(fig_pid, use_container_width=True)

# ── Подвал ─────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Process Model Analyzer v3.0 · "
    f"Мульти-ступенчатый анализ ({n_steps} ступенек) · "
    "FOPDT · SOPDT · Integrating · "
    "Desired Response PID Method"
)
