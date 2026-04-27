"""
АналитикПро  v4.0
=============================
Ключевые изменения v4:
  - Идентификация FOPDT через ARX МНК только на ЗАВЕРШЁННЫХ ступеньках
    y_k = a·y_(k-1) + b·u_(k-d)  →  K=b/(1-a), τ=-dt/ln(a), θ=(d-1)·dt
  - Симуляция суперпозицией всех ступенек
  - Фильтрация незавершённых ступенек по std плато
  - SOPDT только при реальном перерегулировании > 3%
  - Чёрный фон, белый текст
  - Переключение модели без сброса анализа
"""

import io, re, warnings, base64
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter
from datetime import datetime

# PID step display: SV jumps from 0 to Setpoint at 3 seconds
PID_STEP_TIME = 3.0

# ══════════════════════════════════════════════════════════════════
#  УТИЛИТЫ
# ══════════════════════════════════════════════════════════════════

def denoise(sig, w=5):
    n=len(sig)
    if n<w+2: return sig.copy()
    wl=w if w%2==1 else w+1
    wl=min(wl,n-2 if (n-2)%2==1 else n-3)
    try: return savgol_filter(sig,window_length=max(wl,3),polyorder=2)
    except: return sig.copy()

def guess_col(cols,patterns):
    for p in patterns:
        m=[c for c in cols if re.search(p,str(c),re.I)]
        if m: return m[0]
    return cols[0] if cols else ""

def r2_rmse(a,p,i0=0,i1=None):
    a=np.asarray(a[i0:i1],float); p=np.asarray(p[i0:i1],float)
    if len(a)<2: return -999.,999.
    ss_r=np.sum((a-p)**2); ss_t=np.sum((a-np.mean(a))**2)
    return float(1-ss_r/ss_t if ss_t>1e-12 else -999.), float(np.sqrt(np.mean((a-p)**2)))

# ══════════════════════════════════════════════════════════════════
#  ПОИСК СТУПЕНЕК
# ══════════════════════════════════════════════════════════════════

class StepInfo:
    def __init__(self,si,ei,tStep,mv0,dMV,pv0,pvF,dPV,t,mv,pv):
        self.step_idx=si; self.end_idx=ei; self.tStep=tStep
        self.mv0=mv0; self.delta_mv=dMV; self.pv0=pv0
        self.pv_final=pvF; self.delta_pv=dPV
        self.t=t; self.mv=mv; self.pv=pv; self.n=len(t)

def find_all_steps(t,mv,pv,frac=0.10):
    diffs=np.abs(np.diff(mv)); maxd=np.max(diffs)
    if maxd<1e-9: raise ValueError("Ступенчатые изменения MV не найдены")
    idxs=[i+1 for i,d in enumerate(diffs) if d>=maxd*frac]
    steps=[]
    for k,si in enumerate(idxs):
        ei=idxs[k+1] if k+1<len(idxs) else len(t)
        dMV=mv[si]-mv[si-1]
        pre=pv[max(0,si-5):si]; pv0=float(np.mean(pre)) if len(pre)>0 else pv[si]
        plat=pv[si:ei]; sz=max(3,int(len(plat)*0.15))
        pvF=float(np.mean(plat[-sz:])); dPV=pvF-pv0
        steps.append(StepInfo(si,ei,t[si],mv[si-1],dMV,pv0,pvF,dPV,t,mv,pv))
    return steps

def step_quality(s: StepInfo) -> float:
    """
    Оценка завершённости переходного процесса.
    Возвращает std хвоста плато нормированный на размах сигнала PV.
    Низкое значение = ступенька завершена.
    Нормировка позволяет корректно работать с зашумлёнными данными.
    """
    plat = s.pv[s.step_idx:s.end_idx]
    sz   = max(3, int(len(plat) * 0.15))
    tail_std = float(np.std(plat[-sz:]))
    # Нормируем на амплитуду скачка PV — делает порог адаптивным к шуму
    scale = max(abs(s.delta_pv), 1e-6)
    return tail_std / scale   # безразмерная величина: 0.0 = идеально, >0.5 = незавершённо

# ══════════════════════════════════════════════════════════════════
#  FOPDT — идентификация ARX МНК
# ══════════════════════════════════════════════════════════════════

class FOPDTModel:
    label="FOPDT"; color="#38bdf8"

    def __init__(self):
        self.K=self.tau=self.theta=None
        self.a=self.b=self.delay=None
        self.pv0=self.delta_mv=self.tStep=self.pv_final=None
        self.r2=self.rmse=None; self.sim_data=None; self.error=None
        self.t283=self.t632=None; self.quality="—"

    def identify_arx(self, steps_good, dt=1.0):
        """
        МНК по всем хорошим ступенькам совместно.
        Для зашумлённых данных: перед МНК применяем лёгкое сглаживание
        только для идентификации параметров (симуляция остаётся по реальным данным).
        Ищем a, b, delay минимизирующие суммарный MSE.
        """
        if not steps_good:
            self.error="Нет завершённых ступенек для идентификации"
            return

        best_mse=np.inf; best_a=best_b=best_d=None

        for d in range(1, 9):
            rows=[]; tgt=[]
            for s in steps_good:
                # Сглаживаем PV для МНК — устойчивость к шуму
                y_raw = s.pv[s.step_idx:s.end_idx]
                u_raw = s.mv[s.step_idx:s.end_idx]
                N = len(y_raw)
                if N < d + 4: continue
                # Лёгкое сглаживание (окно 3) только для матрицы регрессии
                y = denoise(y_raw, min(5, N//3*2+1 if N//3*2+1>=3 else 3))
                u = u_raw   # MV не сглаживаем — ступеньки должны быть чёткими
                for k in range(d+1, N):
                    rows.append([y[k-1], u[k-d]])
                    tgt.append(y[k])
            if len(rows) < 6: continue
            try:
                ab = np.linalg.lstsq(np.array(rows), np.array(tgt), rcond=None)[0]
            except: continue
            a_c, b_c = ab[0], ab[1]
            if not (0.01 < a_c < 0.9999): continue
            K_c = b_c / (1 - a_c)
            if K_c <= 0: continue
            # MSE считаем на RAW данных (не сглаженных)
            total_mse=0.; cnt=0
            for s in steps_good:
                y=s.pv[s.step_idx:s.end_idx]; u=s.mv[s.step_idx:s.end_idx]; N=len(y)
                if N < d+4: continue
                pred=np.zeros(N); pred[0]=y[0]
                for k in range(1,N):
                    ud=u[k-d] if k>=d else 0.; pred[k]=a_c*pred[k-1]+b_c*ud
                total_mse+=float(np.sum((y-pred)**2)); cnt+=N
            if cnt==0: continue
            mse=total_mse/cnt
            if mse<best_mse:
                best_mse=mse; best_a=a_c; best_b=b_c; best_d=d

        if best_a is None:
            # Резерв: попробовать без ограничения на знак K (интегрирующий?)
            self.error="ARX МНК не сошёлся — проверьте данные или попробуйте модель Интегрирующий"
            return

        self.a=best_a; self.b=best_b; self.delay=best_d
        self.tau=-dt/np.log(best_a)
        self.K=best_b/(1-best_a)
        self.theta=max(0.0,(best_d-1)*dt)
        s0=steps_good[0]
        self.pv0=s0.pv0; self.pv_final=s0.pv_final
        self.delta_mv=s0.delta_mv; self.tStep=s0.tStep
        self.quality=f"{len(steps_good)} хор. ступенек"

    def simulate_superposition(self, t, mv, steps, pv0_init=0.0):
        """
        Симуляция суперпозицией: вклад каждого скачка MV.
        Работает правильно даже при незавершённых предыдущих переходных процессах.
        """
        if self.error: return np.zeros(len(t))
        sim=np.full(len(t), pv0_init)
        # Накапливаем вклады всех ступенек
        for s in steps:
            dMV=s.delta_mv
            for i,ti in enumerate(t):
                dt_l=ti-s.tStep-self.theta
                if dt_l>0:
                    sim[i]+=self.K*dMV*(1-np.exp(-dt_l/max(self.tau,1e-9)))
        return sim

    def simulate_segment(self, t_seg):
        """Одиночный сегмент (для вычисления R² по ступеньке)."""
        if self.error: return np.full(len(t_seg),self.pv0 or 0.)
        dt_l=t_seg-self.tStep-self.theta
        return np.where(dt_l<=0,self.pv0,
            self.pv0+self.K*self.delta_mv*(1-np.exp(-dt_l/max(self.tau,1e-9))))

    def tf_string(self):
        if self.error: return "Ошибка идентификации"
        return (f"G(s) = {self.K:.4f} · exp(−{self.theta:.4f}·s) / ({self.tau:.4f}·s + 1)")

    def discrete_string(self):
        if self.error or self.a is None: return ""
        return (f"y_k = {self.a:.4f}·y_(k−1) + {self.b:.4f}·u_(k−{self.delay})"
                f"   [K={self.K:.4f}, τ={self.tau:.4f}, θ={self.theta:.4f}]")

    def params_dict(self):
        if self.error: return {}
        return {"K":self.K,"τ":self.tau,"θ":self.theta}


# ══════════════════════════════════════════════════════════════════
#  SOPDT
# ══════════════════════════════════════════════════════════════════

class SOPDTModel:
    label="SOPDT"; color="#a78bfa"

    def __init__(self):
        self.K=self.tau1=self.tau2=self.theta=None; self.zeta=1.0
        self.pv0=self.delta_mv=self.tStep=self.pv_final=None
        self.r2=self.rmse=None; self.sim_data=None; self.error=None

    def identify(self, steps_good, dt=1.0):
        # Проверяем перерегулирование хотя бы в одной ступеньке
        max_os=0.
        best_step=None
        for s in steps_good:
            plat=s.pv[s.step_idx:s.end_idx]
            pvExt=np.max(plat) if s.delta_pv>0 else np.min(plat)
            os=abs((pvExt-s.pv_final)/(s.delta_pv+1e-12))
            if os>max_os: max_os=os; best_step=s

        if max_os<0.03:
            self.error=(f"Нет перерегулирования ({max_os*100:.1f}%) — "
                        "процесс 1-го порядка, используйте FOPDT")
            return

        s=best_step
        self.K=s.delta_pv/s.delta_mv; self.pv0=s.pv0
        self.pv_final=s.pv_final; self.delta_mv=s.delta_mv; self.tStep=s.tStep
        ln_os=np.log(max(max_os,1e-6))
        self.zeta=abs(ln_os)/np.sqrt(np.pi**2+ln_os**2)
        plat=s.pv[s.step_idx:s.end_idx]
        pk_i=(np.argmax(plat) if s.delta_pv>0 else np.argmin(plat))
        t_pk=s.t[s.step_idx+pk_i]-s.tStep
        wd=np.pi/max(t_pk,0.01); wn=wd/max(np.sqrt(1-self.zeta**2),1e-6)
        self.tau1=self.tau2=1/max(wn,1e-9)
        self.theta=0.
        for i in range(s.step_idx,s.end_idx):
            if abs(s.pv[i]-s.pv0)>abs(s.delta_pv)*0.02:
                self.theta=max(0.,s.t[i]-s.tStep-dt); break

    def _r(self,d):
        if d<=0: return 0.
        if abs(self.zeta-1)<0.01:
            wn=1/max(self.tau1,1e-9); return 1-np.exp(-wn*d)*(1+wn*d)
        elif self.zeta>1:
            a=-1/max(self.tau1,1e-9); b=-1/max(self.tau2,1e-9)
            if abs(a-b)<1e-9: a*=1.001
            return 1+(b*np.exp(a*d)-a*np.exp(b*d))/(a-b)
        else:
            wd=(1/max(self.tau1,1e-9))*np.sqrt(1-self.zeta**2)
            return 1-np.exp(-self.zeta/max(self.tau1,1e-9)*d)*(
                np.cos(wd*d)+self.zeta/max(np.sqrt(1-self.zeta**2),1e-9)*np.sin(wd*d))

    def simulate_segment(self,t_seg):
        if self.error: return np.full(len(t_seg),self.pv0 or 0.)
        return np.array([self.pv0+self.K*self.delta_mv*self._r(ti-self.tStep-self.theta)
                         for ti in t_seg])

    def simulate_superposition(self,t,mv,steps,pv0_init=0.):
        if self.error: return np.zeros(len(t))
        sim=np.full(len(t),pv0_init)
        for s in steps:
            dMV=s.delta_mv
            for i,ti in enumerate(t):
                d=ti-s.tStep-self.theta
                if d>0: sim[i]+=self.K*dMV*self._r(d)
        return sim

    def tf_string(self):
        if self.error: return self.error
        return (f"G(s)={self.K:.4f}·exp(−{self.theta:.4f}s)"
                f"/(({self.tau1:.4f}s+1)·({self.tau2:.4f}s+1))")

    def params_dict(self):
        if self.error: return {}
        return {"K":self.K,"τ₁":self.tau1,"τ₂":self.tau2,"θ":self.theta,"ζ":self.zeta}


# ══════════════════════════════════════════════════════════════════
#  ИНТЕГРИРУЮЩИЙ
# ══════════════════════════════════════════════════════════════════

class IntegratingModel:
    label="Интегрирующий"; color="#fbbf24"

    def __init__(self):
        self.Ki=self.theta=None
        self.pv0=self.delta_mv=self.tStep=None
        self.r2=self.rmse=None; self.sim_data=None; self.error=None

    def identify(self,steps_good,dt=1.):
        if not steps_good: self.error="Нет данных"; return
        Kis=[]; thetas=[]
        for s in steps_good:
            if abs(s.delta_mv)<1e-9: continue
            tr=s.t[s.step_idx:s.end_idx]; pr=s.pv[s.step_idx:s.end_idx]
            if len(tr)<3: continue
            tm=np.mean(tr); pm=np.mean(pr)
            num=np.sum((tr-tm)*(pr-pm)); den=np.sum((tr-tm)**2)
            Kis.append((num/max(den,1e-12))/s.delta_mv)
            theta=0.
            for i in range(s.step_idx,s.end_idx):
                if abs(s.pv[i]-s.pv0)>abs(s.delta_pv)*0.05:
                    theta=s.t[i]-s.tStep; break
            thetas.append(theta)
        if not Kis: self.error="МНК не сошёлся"; return
        self.Ki=float(np.mean(Kis)); self.theta=float(np.mean(thetas))
        s0=steps_good[0]
        self.pv0=s0.pv0; self.delta_mv=s0.delta_mv; self.tStep=s0.tStep

    def simulate_superposition(self,t,mv,steps,pv0_init=0.):
        if self.error: return np.zeros(len(t))
        sim=np.full(len(t),pv0_init)
        for s in steps:
            dMV=s.delta_mv
            for i,ti in enumerate(t):
                d=ti-s.tStep-self.theta
                if d>0: sim[i]+=self.Ki*dMV*d
        return sim

    def tf_string(self):
        if self.error: return "Ошибка"
        return f"G(s)={self.Ki:.4f}·exp(−{self.theta:.4f}s)/s"

    def params_dict(self):
        if self.error: return {}
        return {"Ki":self.Ki,"θ":self.theta}


# ══════════════════════════════════════════════════════════════════
#  PID
# ══════════════════════════════════════════════════════════════════

class PIDCalculator:
    @staticmethod
    def calculate(K,T,d,e=None,ctrl="PID"):
        """Desired Response PID tuning with preserved process-gain sign."""
        if e is None or e<=0:
            e=T+d
        den = K*(2*e+d)
        if abs(den) < 1e-12:
            den = 1e-12 if den >= 0 else -1e-12
        Kc=(2*T+d)/den
        Ti=T+d/2.
        Td=(T*d)/max(2*T+d,1e-12) if ctrl=="PID" else None
        return {"Gain (Kc)":round(Kc,4),"Reset (Ti)":round(Ti,4),
                "Derivative (Td)":(round(Td,4) if Td else None),"e":round(e,4),"controller":ctrl}

    @staticmethod
    def simulate_pid(t,pv,sp,Kc,Ti,Td=0.,mv_init=50.,lo=0.,hi=100.):
        n=len(t); mv=np.zeros(n); err=np.zeros(n); integ=0.; prev=0.; mv[0]=mv_init
        for i in range(1,n):
            dt=max(t[i]-t[i-1],1e-6); err[i]=sp-pv[i]; integ+=err[i]*dt
            der=(err[i]-prev)/dt if Td>0 else 0.
            mv[i]=np.clip(mv_init+Kc*(err[i]+integ/max(Ti,1e-9)+Td*der),lo,hi); prev=err[i]
        return mv,err

    @staticmethod
    def simulate_closed_loop_step(t, sp_final, K, T, d, Kc, Ti, Td=0., mv_bias=0., lo=0., hi=100.):
        """Closed-loop PID response to a setpoint step 0 → sp_final at PID_STEP_TIME.

        Controller:
            e(t) = SV(t) - PV(t)
            MV = bias + Kc * (e + 1/Ti * integral(e) - Td * dPV/dt)

        The derivative is calculated on PV, not on error, to avoid derivative kick
        at the setpoint step. The sign of Kc is preserved for reverse/direct action.
        """
        t = np.asarray(t, float)
        if len(t) < 2:
            ts = np.linspace(0.0, max(PID_STEP_TIME + 10.0, 20.0), 300)
        else:
            src_ts = t - t[0]
            duration = float(max(src_ts[-1], PID_STEP_TIME + max(20.0, 5.0 * max(T, 1.0))))
            src_dt = float(np.median(np.diff(src_ts))) if len(src_ts) > 1 else 1.0
            dt_sim = max(min(src_dt, 0.1), 0.02)
            ts = np.arange(0.0, duration + dt_sim / 2.0, dt_sim)

        n = len(ts)
        dt0 = float(np.median(np.diff(ts))) if n > 1 else 0.1
        dt0 = max(dt0, 1e-6)

        delay_steps = max(0, int(round(max(d, 0.) / dt0)))
        sv = np.where(ts >= PID_STEP_TIME, float(sp_final), 0.0)

        pv = np.zeros(n, float)
        mv = np.zeros(n, float)
        err = np.zeros(n, float)

        integ = 0.0
        T = max(float(T), dt0)
        Ti = max(float(Ti), 1e-9)
        Td = max(float(Td or 0.), 0.0)
        Kc = float(Kc)

        for i in range(1, n):
            dt = max(float(ts[i] - ts[i-1]), 1e-6)

            if ts[i] < PID_STEP_TIME:
                sv[i] = 0.0
                pv[i] = 0.0
                mv[i] = 0.0
                err[i] = 0.0
                integ = 0.0
                continue

            err[i] = float(sv[i] - pv[i-1])

            # Derivative on measurement: avoids artificial kick when SV jumps.
            d_pv = (pv[i-1] - pv[i-2]) / dt if i >= 2 else 0.0
            derivative_term = -Td * d_pv

            u_unsat = mv_bias + Kc * (err[i] + integ / Ti + derivative_term)
            u = float(np.clip(u_unsat, lo, hi))

            # Anti-windup. Works for positive and negative Kc.
            push = Kc * err[i]
            saturated_high = (u >= hi) and (push > 0)
            saturated_low = (u <= lo) and (push < 0)

            if not (saturated_high or saturated_low):
                integ += err[i] * dt
                u_unsat = mv_bias + Kc * (err[i] + integ / Ti + derivative_term)
                u = float(np.clip(u_unsat, lo, hi))

            mv[i] = u

            j = max(0, i - delay_steps)
            u_delayed = mv[j]
            pv[i] = pv[i-1] + dt * ((K * u_delayed - pv[i-1]) / T)

        err = sv - pv
        return ts, sv, pv, mv, err


# ══════════════════════════════════════════════════════════════════
#  АНАЛИЗ
# ══════════════════════════════════════════════════════════════════

MODEL_CLASSES={"fopdt":FOPDTModel,"sopdt":SOPDTModel,"integrating":IntegratingModel}
MODEL_LABELS={"fopdt":"FOPDT","sopdt":"SOPDT","integrating":"Интегрирующий"}
MODEL_COLORS={"fopdt":"#2563eb","sopdt":"#7c3aed","integrating":"#d97706"}

STD_THRESHOLD = 0.30   # нормированный порог (std/delta_pv): < 0.30 = ступенька завершена
                       # 0.30 соответствует ~30% шума от амплитуды — работает с зашумлёнными данными


def run_analysis(t,mv_pct,pv_pct):
    dt=float(np.median(np.diff(t))) if len(t)>1 else 1.
    steps=find_all_steps(t,mv_pct,pv_pct)

    # Оцениваем уровень шума по всем данным (медианное абсолютное отклонение)
    noise_est = float(np.median(np.abs(np.diff(pv_pct)))) * 1.5
    # Адаптивный порог: если данные зашумлены, порог выше
    # Порог 0.30 нормированный — работает для шума до ~30% от амплитуды ступеньки
    adaptive_threshold = max(STD_THRESHOLD, noise_est / max(
        np.max(np.abs(np.diff(pv_pct))), 1e-6) * 2)
    adaptive_threshold = min(adaptive_threshold, 0.6)  # но не более 60%

    # Разделяем ступеньки на хорошие и незавершённые
    good_steps=[]; partial_steps=[]
    for s in steps:
        q=step_quality(s)
        if q<=adaptive_threshold and len(s.pv[s.step_idx:s.end_idx])>=6:
            good_steps.append(s)
        else:
            partial_steps.append(s)

    # Идентификация моделей
    fopdt=FOPDTModel()
    sopdt=SOPDTModel()
    integr=IntegratingModel()

    if good_steps:
        fopdt.identify_arx(good_steps, dt=dt)
        sopdt.identify(good_steps, dt=dt)
        integr.identify(good_steps, dt=dt)
    else:
        for m in [fopdt,sopdt,integr]:
            m.error="Нет завершённых ступенек (std плато > порога)"

    pv0_init=float(pv_pct[0])

    # Полная симуляция суперпозицией
    full_sim={}
    for key,m in [("fopdt",fopdt),("sopdt",sopdt),("integrating",integr)]:
        if m.error:
            full_sim[key]=np.zeros(len(t)); continue
        if key=="fopdt":
            sim=m.simulate_superposition(t,mv_pct,steps,pv0_init)
        elif key=="sopdt":
            sim=m.simulate_superposition(t,mv_pct,steps,pv0_init)
        else:
            sim=m.simulate_superposition(t,mv_pct,steps,pv0_init)
        full_sim[key]=sim

    # R² считаем только по хорошим ступенькам
    for key,m in [("fopdt",fopdt),("sopdt",sopdt),("integrating",integr)]:
        if m.error: continue
        sim=full_sim[key]
        if good_steps:
            all_actual=[]; all_pred=[]
            for s in good_steps:
                all_actual.extend(pv_pct[s.step_idx:s.end_idx])
                all_pred.extend(sim[s.step_idx:s.end_idx])
            m.r2,m.rmse=r2_rmse(np.array(all_actual),np.array(all_pred))
        else:
            m.r2,m.rmse=r2_rmse(pv_pct,sim)
        m.sim_data=sim

    # Лучшая модель
    scores={k:MODEL_CLASSES[k] for k in MODEL_CLASSES}
    score_r2={}
    for k,m in [("fopdt",fopdt),("sopdt",sopdt),("integrating",integr)]:
        if not m.error and m.r2 is not None: score_r2[k]=m.r2
    best_key=max(score_r2,key=score_r2.get) if score_r2 else "fopdt"

    models={"fopdt":fopdt,"sopdt":sopdt,"integrating":integr}
    return steps,good_steps,partial_steps,models,full_sim,best_key


# ══════════════════════════════════════════════════════════════════
#  PLOTLY ТЕМА — тёмная
# ══════════════════════════════════════════════════════════════════

BG      = "#06111f"
BG_P    = "#0f172a"
GRID    = "#334155"
TXT     = "#f8fafc"
TXT2    = "#cbd5e1"
BORDER  = "#475569"

PLOTLY_BASE = dict(
    paper_bgcolor=BG,
    plot_bgcolor=BG_P,
    font=dict(color=TXT, family="IBM Plex Mono, monospace", size=11),
    legend=dict(bgcolor="rgba(15,23,42,0.72)", bordercolor=BORDER,
                borderwidth=1, font=dict(color=TXT, size=10)),
    margin=dict(l=55, r=20, t=40, b=36),
)


# Runtime light/dark Plotly override
if "ui_theme" in st.session_state and st.session_state.ui_theme == "light":
    BG = "#f8fafc"
    BG_P = "#ffffff"
    GRID = "#cbd5e1"
    TXT = "#0f172a"
    TXT2 = "#475569"
    BORDER = "#cbd5e1"
    PLOTLY_BASE.update(
        paper_bgcolor=BG,
        plot_bgcolor=BG_P,
        font=dict(color=TXT, family="IBM Plex Mono, monospace", size=11),
        legend=dict(bgcolor="rgba(255,255,255,0.94)", bordercolor=BORDER,
                    borderwidth=1, font=dict(color=TXT, size=10)),
    )

def apply_theme(fig, rows=1):
    """
    Applies Plotly theme safely.
    Works both for make_subplots figures and simple go.Figure charts.
    """
    fig.update_layout(**PLOTLY_BASE)
    for i in range(1, rows + 1):
        try:
            fig.update_xaxes(
                gridcolor=GRID, zerolinecolor=GRID, zerolinewidth=1,
                tickfont=dict(color=TXT2), title_font=dict(color=TXT),
                linecolor=BORDER, showline=True, mirror=False, row=i
            )
            fig.update_yaxes(
                gridcolor=GRID, zerolinecolor=GRID, zerolinewidth=1,
                tickfont=dict(color=TXT2), title_font=dict(color=TXT),
                linecolor=BORDER, showline=True, mirror=False, row=i
            )
        except Exception:
            fig.update_xaxes(
                gridcolor=GRID, zerolinecolor=GRID, zerolinewidth=1,
                tickfont=dict(color=TXT2), title_font=dict(color=TXT),
                linecolor=BORDER, showline=True, mirror=False
            )
            fig.update_yaxes(
                gridcolor=GRID, zerolinecolor=GRID, zerolinewidth=1,
                tickfont=dict(color=TXT2), title_font=dict(color=TXT),
                linecolor=BORDER, showline=True, mirror=False
            )
            break
    return fig



# ══════════════════════════════════════════════════════════════════
#  ONLINE IDENTIFICATION + EXPORT HELPERS
# ══════════════════════════════════════════════════════════════════

def online_rls_identification(t, mv_pct, pv_pct, max_delay=6, lam=0.985):
    """
    Онлайн-идентификация ARX/FOPDT:
        y_k = a*y_(k-1) + b*u_(k-d)
    Для каждого delay считается RLS, выбирается delay с минимальной ошибкой.
    Возвращает DataFrame с K, tau, theta во времени.
    """
    t = np.asarray(t, dtype=float)
    u = np.asarray(mv_pct, dtype=float)
    y = np.asarray(pv_pct, dtype=float)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0

    best = None

    for d in range(1, max_delay + 1):
        theta = np.array([0.5, 0.5], dtype=float)
        P = np.eye(2) * 1000.0

        rows = []
        sq_err = []

        for k in range(max(d + 1, 2), len(y)):
            phi = np.array([y[k - 1], u[k - d]], dtype=float)
            y_hat = float(phi @ theta)
            err = float(y[k] - y_hat)

            denom = lam + float(phi.T @ P @ phi)
            gain = (P @ phi) / max(denom, 1e-12)
            theta = theta + gain * err
            P = (P - np.outer(gain, phi) @ P) / lam

            a, b = float(theta[0]), float(theta[1])

            if 0.001 < a < 0.9999:
                K = b / max(1.0 - a, 1e-12)
                tau = -dt / np.log(a)
            else:
                K = np.nan
                tau = np.nan

            theta_dead = max(0.0, (d - 1) * dt)

            rows.append({
                "Time": t[k],
                "delay": d,
                "a": a,
                "b": b,
                "K_online": K,
                "tau_online": tau,
                "theta_online": theta_dead,
                "y_hat": y_hat,
                "error": err,
            })
            sq_err.append(err * err)

        mse = float(np.mean(sq_err)) if sq_err else np.inf
        if best is None or mse < best[0]:
            best = (mse, d, rows)

    if best is None:
        return pd.DataFrame()

    return pd.DataFrame(best[2])


def build_results_dataframe(t, mv_pct, pv_pct, full_sim, models):
    """Единая таблица результатов для экспорта."""
    out = pd.DataFrame({
        "Time": t,
        "MV_percent": mv_pct,
        "PV_percent": pv_pct,
    })
    for key, sim in full_sim.items():
        if key in models and not getattr(models[key], "error", None):
            out[f"{MODEL_LABELS[key]}_model"] = sim
            out[f"{MODEL_LABELS[key]}_error"] = pv_pct - sim
    return out


def build_model_summary_dataframe(models, best_key):
    rows = []
    for key, m in models.items():
        if m is None:
            continue
        row = {
            "model": MODEL_LABELS.get(key, key),
            "best": "YES" if key == best_key else "",
            "r2": getattr(m, "r2", None),
            "rmse": getattr(m, "rmse", None),
            "transfer_function": m.tf_string() if hasattr(m, "tf_string") else "",
            "error": getattr(m, "error", None),
        }
        if not getattr(m, "error", None):
            for p, v in m.params_dict().items():
                row[p] = v
        rows.append(row)
    return pd.DataFrame(rows)


def export_to_excel_bytes(result_df, summary_df, online_df=None):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        result_df.to_excel(writer, index=False, sheet_name="Model_Results")
        summary_df.to_excel(writer, index=False, sheet_name="Model_Summary")
        if online_df is not None and not online_df.empty:
            online_df.to_excel(writer, index=False, sheet_name="Online_RLS")
    buffer.seek(0)
    return buffer.getvalue()


def export_to_html_report(summary_df, result_df):
    """Build a safe standalone HTML report."""
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_html = summary_df.to_html(index=False, escape=False)
    results_html = result_df.head(80).to_html(index=False, escape=False)

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="utf-8">
    <title>АналитикПро — отчет</title>
    <style>
        body {{
            margin: 32px;
            font-family: Arial, sans-serif;
            background: #0f172a;
            color: #e5e7eb;
        }}
        h1 {{
            color: #22d3ee;
            margin-bottom: 4px;
        }}
        h2 {{
            color: #f8fafc;
            margin-top: 28px;
        }}
        .subtitle {{
            color: #94a3b8;
            margin-bottom: 24px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin-top: 12px;
            background: #111827;
            border: 1px solid #334155;
        }}
        th, td {{
            border: 1px solid #334155;
            padding: 8px 10px;
            font-size: 13px;
            text-align: left;
        }}
        th {{
            background: #1e293b;
            color: #cbd5e1;
        }}
        td {{
            color: #e5e7eb;
        }}
    </style>
</head>
<body>
    <h1>АналитикПро</h1>
    <div class="subtitle">Анализ · Моделирование · Оптимизация · Generated: {generated}</div>

    <h2>Сводка моделей</h2>
    {summary_html}

    <h2>Результаты моделирования</h2>
    {results_html}
</body>
</html>"""
    return html.encode("utf-8")


# ══════════════════════════════════════════════════════════════════
#  APP
# ══════════════════════════════════════════════════════════════════

st.set_page_config(page_title="АналитикПро", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

# ── Theme switch ─────────────────────────────────────────────────
if "ui_theme" not in st.session_state:
    st.session_state.ui_theme = "dark"

_theme_cols = st.columns([0.78, 0.22])
with _theme_cols[1]:
    _light_mode = st.toggle(
        "☀️ Светлая тема",
        value=(st.session_state.ui_theme == "light"),
        key="theme_toggle"
    )
st.session_state.ui_theme = "light" if _light_mode else "dark"
_is_light = st.session_state.ui_theme == "light"









st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;600;700&display=swap');

/* ═════════════════════════════════════════════════════════════════
   GLASSMORPHISM UI — trading dashboard style
   Цветовая логика:
   - фон: глубокий тёмно-синий
   - панели: полупрозрачное стекло
   - input: тёмные читаемые поля
   - акценты: cyan / blue / red
   ═════════════════════════════════════════════════════════════════ */

:root {
    --bg-main: #06111f;
    --bg-deep: #020617;
    --glass: rgba(15, 23, 42, 0.62);
    --glass-strong: rgba(17, 24, 39, 0.78);
    --glass-soft: rgba(30, 41, 59, 0.52);
    --input-bg: rgba(15, 23, 42, 0.92);
    --border: rgba(148, 163, 184, 0.22);
    --border-strong: rgba(56, 189, 248, 0.42);
    --text-main: #f8fafc;
    --text-soft: #cbd5e1;
    --text-muted: #94a3b8;
    --cyan: #22d3ee;
    --blue: #3b82f6;
    --blue-soft: #60a5fa;
    --red: #ef4444;
    --red-dark: #b91c1c;
    --green: #22c55e;
    --yellow: #f59e0b;
}

/* Base */
html, body, [class*="css"], [data-testid] {
    font-family: 'IBM Plex Sans', sans-serif !important;
    color: var(--text-main) !important;
}

[data-testid="stAppViewContainer"],
[data-testid="stMainBlockContainer"],
.main, .block-container {
    background:
        radial-gradient(circle at 12% 8%, rgba(34, 211, 238, 0.18), transparent 32rem),
        radial-gradient(circle at 85% 12%, rgba(59, 130, 246, 0.16), transparent 34rem),
        radial-gradient(circle at 70% 92%, rgba(239, 68, 68, 0.08), transparent 28rem),
        linear-gradient(145deg, #020617 0%, #06111f 42%, #0f172a 100%) !important;
    color: var(--text-main) !important;
}

.block-container {
    padding-top: 2rem !important;
    max-width: 1520px !important;
}

/* Global text */
h1, h2, h3, h4, h5, h6 {
    color: var(--text-main) !important;
    letter-spacing: -0.02em !important;
}
p, span, li, td, th, label {
    color: var(--text-main) !important;
}
small {
    color: var(--text-muted) !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header {
    visibility: hidden;
}

/* Sidebar glass panel */
section[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, rgba(2, 6, 23, 0.92) 0%, rgba(15, 23, 42, 0.78) 100%) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: 10px 0 40px rgba(0,0,0,0.34) !important;
    backdrop-filter: blur(18px) saturate(150%) !important;
    -webkit-backdrop-filter: blur(18px) saturate(150%) !important;
}
section[data-testid="stSidebar"] * {
    color: var(--text-main) !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
    font-weight: 800 !important;
}
section[data-testid="stSidebar"] label p {
    color: var(--text-soft) !important;
    font-size: 13px !important;
}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: var(--text-muted) !important;
}

/* Captions / dividers */
[data-testid="stCaptionContainer"] p {
    color: var(--text-muted) !important;
}
hr {
    border-color: var(--border) !important;
}

/* Glass cards / metrics */
div[data-testid="metric-container"] {
    background:
        linear-gradient(145deg, rgba(30,41,59,0.72) 0%, rgba(15,23,42,0.56) 100%) !important;
    border: 1px solid var(--border) !important;
    border-top: 2px solid rgba(34,211,238,0.78) !important;
    border-radius: 20px !important;
    padding: 1rem 1.25rem !important;
    box-shadow:
        0 18px 45px rgba(0,0,0,0.28),
        inset 0 1px 0 rgba(255,255,255,0.08) !important;
    backdrop-filter: blur(16px) saturate(160%) !important;
    -webkit-backdrop-filter: blur(16px) saturate(160%) !important;
}
div[data-testid="stMetricLabel"] p,
div[data-testid="stMetricLabel"] label {
    color: var(--text-muted) !important;
    font-size: 11px !important;
    letter-spacing: 0.10em !important;
    text-transform: uppercase !important;
    font-weight: 800 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] * {
    color: #ffffff !important;
    font-size: 1.45rem !important;
    font-weight: 800 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
div[data-testid="stMetricDelta"] * {
    color: var(--green) !important;
}

/* Tables / dataframe */
[data-testid="stDataFrame"] {
    border-radius: 18px !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
    background: var(--glass) !important;
    box-shadow: 0 18px 45px rgba(0,0,0,0.24) !important;
}
[data-testid="stDataFrame"] * {
    color: var(--text-main) !important;
}
.dataframe th {
    background: rgba(2, 6, 23, 0.92) !important;
    color: var(--text-soft) !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    border-color: rgba(148,163,184,0.14) !important;
}
.dataframe td {
    background: rgba(15, 23, 42, 0.82) !important;
    color: var(--text-main) !important;
    border-color: rgba(148,163,184,0.10) !important;
}

/* Alerts */
[data-testid="stAlert"] {
    background: rgba(15, 23, 42, 0.78) !important;
    border: 1px solid var(--border) !important;
    border-radius: 18px !important;
    box-shadow: 0 12px 30px rgba(0,0,0,0.22) !important;
    backdrop-filter: blur(14px) !important;
}
[data-testid="stAlert"] * {
    color: var(--text-main) !important;
}

/* Inputs: readable dark, no white blocks */
.stSelectbox select,
.stNumberInput input,
.stTextInput input,
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background: var(--input-bg) !important;
    color: var(--text-main) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    min-height: 42px !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.04) !important;
}
input {
    color: var(--text-main) !important;
    caret-color: var(--cyan) !important;
}
input::placeholder {
    color: var(--text-muted) !important;
}
.stNumberInput button {
    background: rgba(30,41,59,0.95) !important;
    color: var(--text-main) !important;
    border: 1px solid rgba(148,163,184,0.22) !important;
}
.stNumberInput button svg {
    color: var(--text-main) !important;
    fill: var(--text-main) !important;
}
div[data-baseweb="select"] svg {
    fill: var(--text-soft) !important;
}
[data-baseweb="popover"] {
    background: #0f172a !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    box-shadow: 0 20px 45px rgba(0,0,0,0.35) !important;
}
[data-baseweb="menu"] {
    background: #0f172a !important;
}
[data-baseweb="menu"] li {
    color: var(--text-main) !important;
}

/* Slider / radio */
[data-testid="stSlider"] * {
    color: var(--text-soft) !important;
}
[data-testid="stRadio"] label p {
    color: var(--text-soft) !important;
}
[data-testid="stRadio"] div {
    color: var(--text-main) !important;
}

/* Tabs */
[data-testid="stTabs"] [role="tablist"] {
    gap: 0.55rem !important;
    border-bottom: 1px solid var(--border) !important;
    padding-bottom: 0 !important;
}
[data-testid="stTabs"] [role="tab"] {
    background: rgba(15, 23, 42, 0.58) !important;
    border: 1px solid var(--border) !important;
    border-bottom: none !important;
    border-radius: 16px 16px 0 0 !important;
    padding: 0.78rem 1.15rem !important;
    font-weight: 800 !important;
    color: var(--text-soft) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.06) !important;
    backdrop-filter: blur(14px) !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background:
        linear-gradient(145deg, rgba(30,41,59,0.85) 0%, rgba(15,23,42,0.78) 100%) !important;
    color: #ffffff !important;
    border-top: 3px solid var(--cyan) !important;
    box-shadow: 0 12px 28px rgba(34,211,238,0.12) !important;
}

/* Code */
code, pre {
    background: rgba(2,6,23,0.92) !important;
    color: #93c5fd !important;
    border: 1px solid rgba(56,189,248,0.22) !important;
    border-radius: 10px !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* File uploader glass */
[data-testid="stFileUploader"] {
    background:
        linear-gradient(145deg, rgba(15,23,42,0.78), rgba(30,41,59,0.42)) !important;
    border: 1px dashed rgba(148,163,184,0.38) !important;
    border-radius: 20px !important;
    box-shadow:
        0 18px 40px rgba(0,0,0,0.24),
        inset 0 1px 0 rgba(255,255,255,0.06) !important;
    backdrop-filter: blur(14px) !important;
}
[data-testid="stFileUploader"] section {
    background: transparent !important;
}
[data-testid="stFileUploader"] * {
    color: var(--text-soft) !important;
}
[data-testid="stFileUploaderDropzone"] small {
    color: var(--text-muted) !important;
}
[data-testid="stFileUploader"] button div span,
[data-testid="stFileUploader"] button p,
[data-testid="stFileUploader"] button span {
    display: none !important;
}
[data-testid="stFileUploader"] button::after {
    content: "Upload";
    display: inline-block;
    color: #ffffff !important;
    font-size: 16px !important;
    font-weight: 800 !important;
}
[data-testid="stFileUploader"] button {
    background: linear-gradient(135deg, #2563eb 0%, #06b6d4 100%) !important;
    border: 1px solid rgba(125,211,252,0.30) !important;
    border-radius: 15px !important;
    padding: 0.62rem 1.45rem !important;
    box-shadow: 0 12px 28px rgba(37,99,235,0.32) !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] > div:first-child {
    background: rgba(15,23,42,0.82) !important;
    color: var(--text-soft) !important;
    border-radius: 14px !important;
}

/* Buttons */
div[data-testid="stButton"],
div[data-testid="stButton"] > div,
div[data-testid="stButton"] div,
div[data-testid="stDownloadButton"],
div[data-testid="stDownloadButton"] > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}
div[data-testid="stButton"] button,
div[data-testid="stDownloadButton"] button,
button[kind="primary"] {
    background:
        linear-gradient(135deg, #ef4444 0%, #dc2626 52%, #991b1b 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(252,165,165,0.28) !important;
    border-radius: 18px !important;
    padding: 0.78rem 1.95rem !important;
    min-height: 3.05rem !important;
    font-size: 16px !important;
    font-weight: 800 !important;
    box-shadow:
        0 16px 34px rgba(220,38,38,0.32),
        inset 0 1px 0 rgba(255,255,255,0.12) !important;
    transition: all 0.15s ease !important;
}
div[data-testid="stButton"] button:hover,
div[data-testid="stDownloadButton"] button:hover,
button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    filter: brightness(1.08) !important;
    box-shadow: 0 18px 38px rgba(220,38,38,0.42) !important;
}
div[data-testid="stButton"] button *,
div[data-testid="stDownloadButton"] button *,
button[kind="primary"] * {
    color: #ffffff !important;
    font-weight: 800 !important;
}
button:focus,
button:focus-visible {
    outline: none !important;
}

/* Custom pro blocks */
.pro-card {
    background:
        linear-gradient(145deg, rgba(30,41,59,0.72), rgba(15,23,42,0.58));
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 20px 22px;
    box-shadow:
        0 22px 55px rgba(0,0,0,0.32),
        inset 0 1px 0 rgba(255,255,255,0.08);
    margin: 10px 0 18px 0;
    backdrop-filter: blur(18px) saturate(160%);
}
.pro-subtitle {
    color: var(--text-soft);
    font-size: 13px;
    margin-top: -6px;
}

/* Plotly containers */
[data-testid="stPlotlyChart"] {
    background: rgba(15, 23, 42, 0.42) !important;
    border: 1px solid rgba(148,163,184,0.16) !important;
    border-radius: 20px !important;
    padding: 8px !important;
    box-shadow: 0 18px 45px rgba(0,0,0,0.22) !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}
::-webkit-scrollbar-track {
    background: #020617;
}
::-webkit-scrollbar-thumb {
    background: rgba(148,163,184,0.35);
    border-radius: 20px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(34,211,238,0.55);
}

/* Fix dropdown (BaseWeb select) readability */
[data-baseweb="popover"] {

    border: 1px solid rgba(148,163,184,0.25) !important;
    border-radius: 14px !important;
}
[data-baseweb="menu"] {

}
[data-baseweb="menu"] li {
    color: #e5e7eb !important;
    background: transparent !important;
}
[data-baseweb="menu"] li:hover {
    background: rgba(59,130,246,0.18) !important;
    color: #ffffff !important;
}
[data-baseweb="menu"] li[aria-selected="true"] {
    background: rgba(59,130,246,0.28) !important;
    color: #ffffff !important;
}


/* Stronger selected highlight for dropdown */
li[aria-selected="true"],
div[aria-selected="true"],
li[role="option"][aria-selected="true"],
div[role="option"][aria-selected="true"] {
    background: rgba(15,23,42,0.85) !important;
    color: #ffffff !important;
}


/* Selectbox visible value */
div[data-baseweb="select"] span,
div[data-baseweb="select"] div {{
    color: {_dropdown_text} !important;
    opacity: 1 !important;
}}

/* ── Dropdown cyan text fix, app UI only ── */
div[data-baseweb="popover"],
div[data-baseweb="popover"] > div,
div[data-baseweb="menu"],
ul[role="listbox"],
div[role="listbox"] {
    background: #0f172a !important;
    background-color: #0f172a !important;
    border: 1px solid rgba(148,163,184,0.35) !important;
    border-radius: 14px !important;
    box-shadow: 0 20px 45px rgba(0,0,0,0.35) !important;
}

/* обычные пункты списка */
div[data-baseweb="menu"] li,
ul[role="listbox"] li,
li[role="option"],
div[role="option"] {
    background: #0f172a !important;
    background-color: #0f172a !important;
    color: #38bdf8 !important;
    -webkit-text-fill-color: #38bdf8 !important;
    opacity: 1 !important;
}

/* текст внутри пунктов */
div[data-baseweb="menu"] li *,
ul[role="listbox"] li *,
li[role="option"] *,
div[role="option"] * {
    color: #38bdf8 !important;
    -webkit-text-fill-color: #38bdf8 !important;
    opacity: 1 !important;
}

/* выбранный пункт */
li[aria-selected="true"],
div[aria-selected="true"],
li[role="option"][aria-selected="true"],
div[role="option"][aria-selected="true"] {
    background: #1e293b !important;
    background-color: #1e293b !important;
    color: #67e8f9 !important;
    -webkit-text-fill-color: #67e8f9 !important;
}

/* hover */
div[data-baseweb="menu"] li:hover,
ul[role="listbox"] li:hover,
li[role="option"]:hover,
div[role="option"]:hover {
    background: #334155 !important;
    background-color: #334155 !important;
    color: #e0f7ff !important;
    -webkit-text-fill-color: #e0f7ff !important;
}

/* закрытое поле select: только текст голубой, фон оставляем как в основной теме */
div[data-baseweb="select"] span,
div[data-baseweb="select"] div {
    color: #38bdf8 !important;
    -webkit-text-fill-color: #38bdf8 !important;
    opacity: 1 !important;
}


/* ── Expander arrow text fix for app UI only ── */
[data-testid="stExpander"] summary svg,
[data-testid="stExpander"] summary [data-testid="stIconMaterial"],
[data-testid="stExpander"] summary span[class*="material"],
[data-testid="stExpander"] summary i,
details summary svg,
details summary [data-testid="stIconMaterial"],
details summary span[class*="material"],
details summary i {
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
    min-width: 0 !important;
    max-width: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
}

[data-testid="stExpander"] summary,
details summary {
    padding-left: 0.9rem !important;
}

/* ══════════════════════════════════════════════════════════════════
   SIDEBAR COLLAPSE BUTTON — скрываем нативный "keyboard_double_arrow"
   Гамбургер-кнопка добавляется отдельным markdown-блоком ниже
   ══════════════════════════════════════════════════════════════════ */


/* ── Dark theme only: PID expander title cyan ── */
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary *,
details summary,
details summary * {
    color: #38bdf8 !important;
    -webkit-text-fill-color: #38bdf8 !important;
}

</style>
""", unsafe_allow_html=True)




# ── Light theme runtime override ─────────────────────────────────
if _is_light:
    st.markdown("""
    <style>
    :root {
        --bg-main: #f6f8fb;
        --bg-deep: #eef2f7;
        --glass: rgba(255, 255, 255, 0.84);
        --glass-strong: rgba(255, 255, 255, 0.94);
        --glass-soft: rgba(241, 245, 249, 0.88);
        --input-bg: #ffffff;
        --border: rgba(15, 23, 42, 0.16);
        --border-strong: rgba(14, 165, 233, 0.42);
        --text-main: #0f172a;
        --text-soft: #334155;
        --text-muted: #64748b;
        --cyan: #0284c7;
        --blue: #2563eb;
        --blue-soft: #0284c7;
        --red: #dc2626;
        --red-dark: #991b1b;
        --green: #15803d;
        --yellow: #b45309;
    }

    html, body, [class*="css"], [data-testid] {
        color: #0f172a !important;
    }

    [data-testid="stAppViewContainer"],
    [data-testid="stMainBlockContainer"],
    .main, .block-container {
        background:
            radial-gradient(circle at 12% 8%, rgba(14, 165, 233, 0.12), transparent 32rem),
            radial-gradient(circle at 85% 12%, rgba(37, 99, 235, 0.10), transparent 34rem),
            linear-gradient(145deg, #f8fafc 0%, #eef2f7 48%, #e2e8f0 100%) !important;
        color: #0f172a !important;
    }

    h1, h2, h3, h4, h5, h6,
    p, span, li, td, th, label,
    section[data-testid="stSidebar"] * {
        color: #0f172a !important;
    }

    small,
    [data-testid="stCaptionContainer"] p,
    label[data-testid="stWidgetLabel"] p {
        color: #64748b !important;
    }

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, rgba(255,255,255,0.96) 0%, rgba(241,245,249,0.92) 100%) !important;
        border-right: 1px solid rgba(15,23,42,0.14) !important;
        box-shadow: 10px 0 40px rgba(15,23,42,0.08) !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #0f172a !important;
    }

    div[data-testid="metric-container"],
    .pro-card,
    [data-testid="stAlert"],
    [data-testid="stPlotlyChart"],
    [data-testid="stDataFrame"] {
        background: rgba(255,255,255,0.86) !important;
        border: 1px solid rgba(15,23,42,0.12) !important;
        box-shadow: 0 18px 45px rgba(15,23,42,0.10) !important;
    }

    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] * {
        color: #0f172a !important;
    }

    div[data-testid="stMetricLabel"] p,
    div[data-testid="stMetricLabel"] label {
        color: #475569 !important;
    }

    .dataframe th {
        background: #e2e8f0 !important;
        color: #334155 !important;
        border-color: rgba(15,23,42,0.10) !important;
    }

    .dataframe td {
        background: #ffffff !important;
        color: #0f172a !important;
        border-color: rgba(15,23,42,0.08) !important;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    .stNumberInput input,
    .stTextInput input,
    input {
        background: #ffffff !important;
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        border: 1px solid rgba(15,23,42,0.18) !important;
        box-shadow: inset 0 1px 0 rgba(15,23,42,0.03) !important;
    }

    input::placeholder {
        color: #94a3b8 !important;
        -webkit-text-fill-color: #94a3b8 !important;
    }

    div[data-baseweb="select"] svg {
        fill: #475569 !important;
    }

    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
    }

    /* Dropdown for light theme */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    div[data-baseweb="menu"],
    ul[role="listbox"],
    div[role="listbox"] {
        background: #ffffff !important;
        background-color: #ffffff !important;
        border: 1px solid rgba(15,23,42,0.18) !important;
        box-shadow: 0 18px 42px rgba(15,23,42,0.16) !important;
    }

    div[data-baseweb="menu"] li,
    ul[role="listbox"] li,
    li[role="option"],
    div[role="option"] {
        background: #ffffff !important;
        background-color: #ffffff !important;
        color: #075985 !important;
        -webkit-text-fill-color: #075985 !important;
        opacity: 1 !important;
    }

    div[data-baseweb="menu"] li *,
    ul[role="listbox"] li *,
    li[role="option"] *,
    div[role="option"] * {
        color: #075985 !important;
        -webkit-text-fill-color: #075985 !important;
        opacity: 1 !important;
    }

    li[aria-selected="true"],
    div[aria-selected="true"],
    li[role="option"][aria-selected="true"],
    div[role="option"][aria-selected="true"] {
        background: #dbeafe !important;
        background-color: #dbeafe !important;
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
    }

    div[data-baseweb="menu"] li:hover,
    ul[role="listbox"] li:hover,
    li[role="option"]:hover,
    div[role="option"]:hover {
        background: #e0f2fe !important;
        background-color: #e0f2fe !important;
        color: #0c4a6e !important;
        -webkit-text-fill-color: #0c4a6e !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.82) !important;
        border: 1px dashed rgba(15,23,42,0.24) !important;
        box-shadow: 0 18px 40px rgba(15,23,42,0.08) !important;
    }

    [data-testid="stFileUploader"] * {
        color: #334155 !important;
    }

    [data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, #2563eb 0%, #0284c7 100%) !important;
        color: #ffffff !important;
    }

    [data-testid="stFileUploader"] button::after {
        color: #ffffff !important;
    }

    
    /* Light theme: softer number input +/- buttons */
    .stNumberInput button,
    [data-testid="stNumberInput"] button,
    button[aria-label="Increment"],
    button[aria-label="Decrement"] {
        background: #e2e8f0 !important;
        background-color: #e2e8f0 !important;
        color: #0f172a !important;
        border: 1px solid rgba(15,23,42,0.16) !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.55) !important;
    }

    .stNumberInput button:hover,
    [data-testid="stNumberInput"] button:hover,
    button[aria-label="Increment"]:hover,
    button[aria-label="Decrement"]:hover {
        background: #cbd5e1 !important;
        background-color: #cbd5e1 !important;
        color: #0f172a !important;
        border-color: rgba(14,165,233,0.35) !important;
    }

    .stNumberInput button svg,
    [data-testid="stNumberInput"] button svg,
    button[aria-label="Increment"] svg,
    button[aria-label="Decrement"] svg {
        color: #0f172a !important;
        fill: #0f172a !important;
        stroke: #0f172a !important;
        opacity: 1 !important;
    }

    .stNumberInput button *,
    [data-testid="stNumberInput"] button *,
    button[aria-label="Increment"] *,
    button[aria-label="Decrement"] * {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        opacity: 1 !important;
    }


    /* Buttons remain expressive and readable */
    div[data-testid="stButton"] button,
    div[data-testid="stDownloadButton"] button,
    button[kind="primary"] {
        color: #ffffff !important;
    }

    /* Tabs */
    [data-testid="stTabs"] [role="tab"] {
        background: rgba(255,255,255,0.72) !important;
        border: 1px solid rgba(15,23,42,0.14) !important;
        color: #475569 !important;
    }

    [data-testid="stTabs"] [aria-selected="true"] {
        background: #ffffff !important;
        color: #0f172a !important;
        border-top: 3px solid #0284c7 !important;
        box-shadow: 0 12px 28px rgba(14,165,233,0.12) !important;
    }

    code, pre {
        background: #f1f5f9 !important;
        color: #1d4ed8 !important;
        border: 1px solid rgba(37,99,235,0.20) !important;
    }

    hr {
        border-color: rgba(15,23,42,0.16) !important;
    }

    ::-webkit-scrollbar-track {
        background: #e2e8f0;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(71,85,105,0.35);
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(14,165,233,0.55);
    }
/* keep expander title aligned after hiding icon */
[data-testid="stExpander"] summary,
details summary {
    padding-left: 0.9rem !important;
}

    /* Light theme: keep expander title readable dark */
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary *,
    details summary,
    details summary * {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
    }

    /* Light theme: hamburger button */
    #hamburger-btn {
        background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%) !important;
        border-color: rgba(15,23,42,0.20) !important;
        box-shadow: 0 4px 16px rgba(15,23,42,0.14) !important;
    }
    #hamburger-btn .hb-line {
        background: #334155 !important;
    }
    #hamburger-btn:hover {
        border-color: rgba(14,165,233,0.50) !important;
        box-shadow: 0 6px 20px rgba(14,165,233,0.20) !important;
    }
    #hamburger-btn:hover .hb-line {
        background: #0284c7 !important;
    }

    /* Light theme: sidebar close button */
    .sidebar-close-btn {
        background: rgba(241,245,249,0.92) !important;
        border-color: rgba(15,23,42,0.20) !important;
        color: #334155 !important;
    }
    .sidebar-close-btn:hover {
        background: #e2e8f0 !important;
        border-color: rgba(14,165,233,0.45) !important;
        color: #0f172a !important;
    }

</style>
    """, unsafe_allow_html=True)



# ── Dynamic Plotly colors for current theme ──────────────────────
if _is_light:
    BG = "#f8fafc"
    BG_P = "#ffffff"
    GRID = "#cbd5e1"
    TXT = "#0f172a"
    TXT2 = "#475569"
    BORDER = "#cbd5e1"
    PLOTLY_BASE.update(
        paper_bgcolor=BG,
        plot_bgcolor=BG_P,
        font=dict(color=TXT, family="IBM Plex Mono, monospace", size=11),
        legend=dict(bgcolor="rgba(255,255,255,0.94)", bordercolor=BORDER,
                    borderwidth=1, font=dict(color=TXT, size=10)),
    )
else:
    BG = "#06111f"
    BG_P = "#0f172a"
    GRID = "#334155"
    TXT = "#f8fafc"
    TXT2 = "#cbd5e1"
    BORDER = "#475569"
    PLOTLY_BASE.update(
        paper_bgcolor=BG,
        plot_bgcolor=BG_P,
        font=dict(color=TXT, family="IBM Plex Mono, monospace", size=11),
        legend=dict(bgcolor="rgba(15,23,42,0.72)", bordercolor=BORDER,
                    borderwidth=1, font=dict(color=TXT, size=10)),
    )

# Session state — короткие ключи без спецсимволов
for _k,_v in [("ok",False),("model","fopdt"),("result",None),("signals",None)]:
    if _k not in st.session_state: st.session_state[_k]=_v

# ══════════════════════════════════════════════════════════════════
#  ГАМБУРГЕР-КНОПКА — фиксированная, кликает нативный Streamlit toggle
#  Это единственный рабочий способ управлять sidebar без перезагрузки.
#  Нативная кнопка Streamlit скрыта CSS, но остаётся в DOM и кликается JS.
# ══════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── Скрыть нативную Streamlit кнопку визуально, но оставить в DOM ── */
[data-testid="stSidebarCollapseButton"] {
    opacity: 0 !important;
    pointer-events: none !important;
    position: absolute !important;
    width: 1px !important;
    height: 1px !important;
    overflow: hidden !important;
}
[data-testid="collapsedControl"] {
    opacity: 0 !important;
    pointer-events: none !important;
    position: absolute !important;
    width: 1px !important;
    height: 1px !important;
    overflow: hidden !important;
}

/* ── Фиксированная гамбургер-кнопка ── */
#hamburger-btn {
    position: fixed;
    top: 16px;
    left: 16px;
    z-index: 99999;
    width: 46px;
    height: 46px;
    border-radius: 12px;
    border: 1px solid rgba(148,163,184,0.30);
    background: linear-gradient(135deg,rgba(30,41,59,0.92) 0%,rgba(15,23,42,0.82) 100%);
    box-shadow: 0 6px 22px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.08);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    cursor: pointer;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 5px;
    transition: all 0.18s ease;
    user-select: none;
}
#hamburger-btn:hover {
    border-color: rgba(34,211,238,0.55);
    box-shadow: 0 8px 28px rgba(34,211,238,0.22), inset 0 1px 0 rgba(255,255,255,0.12);
    transform: scale(1.05);
}
.hb-line {
    width: 22px;
    height: 2.5px;
    background: #cbd5e1;
    border-radius: 2px;
    transition: background 0.15s;
}
#hamburger-btn:hover .hb-line {
    background: #22d3ee;
}

/* Кнопка Свернуть внутри sidebar */
.sidebar-close-row {
    display: flex;
    justify-content: flex-end;
    padding: 10px 8px 4px 8px;
    margin-bottom: 4px;
}
.sidebar-close-btn {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 6px 14px;
    border-radius: 10px;
    border: 1px solid rgba(148,163,184,0.28);
    background: rgba(30,41,59,0.75);
    color: #94a3b8;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    font-family: 'IBM Plex Sans', sans-serif;
    transition: all 0.15s;
    letter-spacing: 0.02em;
}
.sidebar-close-btn:hover {
    border-color: rgba(34,211,238,0.45);
    color: #f8fafc;
    background: rgba(51,65,85,0.88);
}
/* Light theme override */
body.light-theme #hamburger-btn,
[data-theme="light"] #hamburger-btn {
    background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
    border-color: rgba(15,23,42,0.18);
    box-shadow: 0 4px 16px rgba(15,23,42,0.14);
}
[data-theme="light"] .hb-line { background: #334155; }
</style>

<div id="hamburger-btn" onclick="toggleSidebar()" title="Меню параметров">
  <div class="hb-line"></div>
  <div class="hb-line"></div>
  <div class="hb-line"></div>
</div>

<script>
function toggleSidebar() {
    // Ищем нативную кнопку Streamlit в разных состояниях
    var btn =
        document.querySelector('[data-testid="stSidebarCollapseButton"] button') ||
        document.querySelector('[data-testid="collapsedControl"] button') ||
        document.querySelector('button[aria-label="Close sidebar"]') ||
        document.querySelector('button[aria-label="Open sidebar"]') ||
        document.querySelector('button[kind="header"]');

    if (btn) {
        btn.click();
    } else {
        // Fallback: ищем все header-кнопки
        var btns = document.querySelectorAll('button');
        for (var i = 0; i < btns.length; i++) {
            var label = btns[i].getAttribute('aria-label') || '';
            if (label.toLowerCase().includes('sidebar') ||
                label.toLowerCase().includes('collapse') ||
                label.toLowerCase().includes('expand')) {
                btns[i].click();
                break;
            }
        }
    }
}
// Убедимся что обработчик доступен глобально
window.toggleSidebar = toggleSidebar;
</script>
""", unsafe_allow_html=True)

# Заголовок
_DARK_LOGO_B64 = "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"
_LIGHT_LOGO_B64 = "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"

if _is_light:
    _logo_b64 = _LIGHT_LOGO_B64
    _logo_wrap_bg = "linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(226,242,255,0.94) 100%)"
    _logo_wrap_border = "1px solid rgba(14,165,233,0.34)"
    _logo_wrap_shadow = "0 12px 28px rgba(14,165,233,0.16), inset 0 1px 0 rgba(255,255,255,0.82)"
    _logo_filter = "contrast(1.12) saturate(1.14) drop-shadow(0 2px 3px rgba(15,23,42,0.12)) drop-shadow(0 0 6px rgba(34,211,238,0.18))"
    _subtitle_color = "#94a3b8"
    _glow_std = "0.55"
else:
    _logo_b64 = _DARK_LOGO_B64
    _logo_wrap_bg = "transparent"
    _logo_wrap_border = "none"
    _logo_wrap_shadow = "none"
    _logo_filter = "drop-shadow(0 0 10px rgba(34,211,238,0.24))"
    _subtitle_color = "#94a3b8"
    _glow_std = "1.2"

_header_html = f"""
<div style="display:flex;flex-direction:row;align-items:center;justify-content:flex-start;gap:18px;margin:8px 0 18px 20px;padding:0;background:transparent;border:none;box-shadow:none;">
  <div style="width:84px;height:84px;min-width:84px;max-width:84px;display:flex;align-items:center;justify-content:center;background:{_logo_wrap_bg};border:{_logo_wrap_border};border-radius:22px;box-shadow:{_logo_wrap_shadow};overflow:hidden;">
    <img src="data:image/png;base64,{_logo_b64}" alt="АналитикПро logo" style="width:84px;height:84px;min-width:84px;max-width:84px;max-height:84px;object-fit:contain;display:block;background:transparent;border:none;border-radius:22px;box-shadow:none;filter:{_logo_filter};image-rendering:-webkit-optimize-contrast;image-rendering:crisp-edges;"/>
  </div>
  <div style="display:flex;flex-direction:column;justify-content:center;align-items:flex-start;padding:0;margin:0;">
    <svg width="640" height="62" viewBox="0 0 640 62" xmlns="http://www.w3.org/2000/svg" style="display:block;overflow:visible;">
      <defs><filter id="apSoftGlow" x="-20%" y="-40%" width="140%" height="180%"><feGaussianBlur stdDeviation="{_glow_std}" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>
      <text x="0" y="49" font-family="IBM Plex Sans, Segoe UI, Arial, sans-serif" font-size="54" font-weight="900" letter-spacing="-1.5" fill="#22d3ee" filter="url(#apSoftGlow)">Аналитик</text>
      <text x="242" y="49" font-family="IBM Plex Sans, Segoe UI, Arial, sans-serif" font-size="54" font-weight="900" letter-spacing="-1.5" fill="#38bdf8" filter="url(#apSoftGlow)">Про</text>
    </svg>
    <div style="width:600px;height:2px;margin:4px 0 8px 0;border-radius:999px;background:linear-gradient(90deg,rgba(34,211,238,0.95) 0%,rgba(59,130,246,0.55) 55%,rgba(37,99,235,0.00) 100%);box-shadow:0 0 12px rgba(34,211,238,0.18);"></div>
    <div style="font-size:13px;color:{_subtitle_color};font-weight:600;white-space:nowrap;letter-spacing:0.25px;">ARX МНК · FOPDT · SOPDT · Интегрирующий · PID v4.0</div>
  </div>
</div>
"""
st.markdown(_header_html, unsafe_allow_html=True)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    # Кнопка Свернуть — в правом верхнем углу sidebar
    st.markdown("""
<div class="sidebar-close-row">
  <div class="sidebar-close-btn" onclick="toggleSidebar()" title="Свернуть меню">
    ✕&nbsp; Свернуть
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("## ⚙️ Параметры")
    st.markdown("### 📂 Данные")
    uploaded=st.file_uploader("CSV или Excel",type=["csv","xlsx","xls"])
    st.markdown("### 🔇 Шумоподавление")
    smooth=st.slider("Окно фильтра",1,51,1,step=2)

# ── Основная область ──────────────────────────────────────────────
if uploaded is None:
    cols=st.columns(3)
    cols[0].markdown("**Загрузите файл** CSV/Excel\nс колонками MV и PV.\nПриложение найдёт все ступеньки.")
    cols[1].markdown("**Метод:** ARX МНК\n`y_k = a·y(k-1) + b·u(k-d)`\n→ K, τ, θ  (R²≈1 для 1-го порядка)")
    cols[2].markdown("**Результат:** K, τ, θ\nГрафики PV + модель\nPID (Desired Response)")
    st.stop()

try:
    raw=uploaded.read()
    df=(pd.read_csv(io.BytesIO(raw),sep=None,engine="python")
        if uploaded.name.lower().endswith(".csv") else pd.read_excel(io.BytesIO(raw)))
except Exception as ex: st.error(f"Ошибка чтения: {ex}"); st.stop()

cols=list(df.columns)
st.markdown("### 1️⃣ Колонки данных")
c1,c2,c3=st.columns(3)
def_mv=guess_col(cols,[r"\.MV$",r"^MV$",r"output",r"выход",r"mv"])
def_pv=guess_col(cols,[r"\.PV$",r"^PV$",r"process",r"процесс",r"pv"])
def_t=next((c for c in cols if re.match(r"^(t|time|время|timestamp)$",str(c),re.I)),None)
t_opts=["— индекс строки —"]+cols
mv_col=c1.selectbox("MV",cols,index=cols.index(def_mv) if def_mv in cols else 0)
pv_col=c2.selectbox("PV",cols,index=min(cols.index(def_pv),len(cols)-1) if def_pv in cols else min(1,len(cols)-1))
t_col=c3.selectbox("Время",t_opts,index=t_opts.index(def_t) if def_t in t_opts else 0)

# ── Шкалы и PID-параметры после выбора колонок ───────────────────
# MV предустановлен как 0…100.
# PV автоматически берётся как min/max из выбранной PV-колонки загруженной таблицы.
_pv_for_scale = pd.to_numeric(df[pv_col], errors="coerce").dropna()
if _pv_for_scale.empty:
    pv_auto_min, pv_auto_max = 0.0, 1.0
else:
    pv_auto_min = float(_pv_for_scale.min())
    pv_auto_max = float(_pv_for_scale.max())
    if abs(pv_auto_max - pv_auto_min) < 1e-12:
        pv_auto_max = pv_auto_min + 1.0

with st.sidebar:
    st.markdown("### 📏 Шкалы приборов")
    st.caption("K считается по нормированным шкалам: MV% и PV%")
    c1,c2=st.columns(2)
    mv_min=c1.number_input("MV min",value=0.0,step=1.,key="mv_min_scale")
    mv_max=c2.number_input("MV max",value=100.0,step=1.,key="mv_max_scale")

    c3s,c4s=st.columns(2)
    pv_min=c3s.number_input("PV min",value=float(pv_auto_min),step=1.,key=f"pv_min_scale_{pv_col}")
    pv_max=c4s.number_input("PV max",value=float(pv_auto_max),step=1.,key=f"pv_max_scale_{pv_col}")
    st.caption("PV min/max подставлены из выбранной PV-колонки, но их можно изменить вручную.")

    st.markdown("### 🎛️ PID")
    pid_type=st.radio("Тип",["PI","PID"],horizontal=True)
    pid_sv_eng=st.number_input("SV / Setpoint, инженерные ед.", value=float((pv_min+pv_max)/2), step=1.0)
    st.markdown("### ✏️ Корректировка PID")
    st.caption("0 = использовать рекомендованные")
    kc_ov=st.number_input("Kc",value=0.0,step=0.001,format="%.4f")
    ti_ov=st.number_input("Ti",value=0.0,step=0.001,format="%.4f")
    td_ov=st.number_input("Td",value=0.0,step=0.001,format="%.4f")

st.markdown("#### 🔍 Предпросмотр (20 строк)")
with st.container():
    show=[mv_col,pv_col]+([t_col] if t_col!="— индекс строки —" and t_col in df.columns else [])
    st.dataframe(df[show].head(20),use_container_width=True)
    st.caption(f"Строк: {len(df)}")

st.markdown("### 2️⃣ Анализ")

_current_analysis_signature = (
    uploaded.name if uploaded is not None else None,
    mv_col,
    pv_col,
    t_col,
    float(mv_min),
    float(mv_max),
    float(pv_min),
    float(pv_max),
    int(smooth),
)

_run_analysis_clicked = st.button("▶  Запустить анализ", type="primary")
_scale_or_column_changed = (
    st.session_state.get("ok", False)
    and st.session_state.get("analysis_signature") is not None
    and st.session_state.get("analysis_signature") != _current_analysis_signature
)

if _scale_or_column_changed:
    st.info("Шкала или выбранные колонки изменились — коэффициенты пересчитываются автоматически.")

if _run_analysis_clicked or _scale_or_column_changed:
    with st.spinner("Идентификация…"):
        try:
            mv_r=pd.to_numeric(df[mv_col],errors="coerce").values
            pv_r=pd.to_numeric(df[pv_col],errors="coerce").values
            t_r=(np.arange(len(mv_r),dtype=float) if t_col=="— индекс строки —"
                 else pd.to_numeric(df[t_col],errors="coerce").values)
            mask=~(np.isnan(mv_r)|np.isnan(pv_r)|np.isnan(t_r))
            ta=t_r[mask].astype(float); mva=mv_r[mask].astype(float); pva=pv_r[mask].astype(float)
            if len(ta)<10: st.error(f"Мало точек: {len(ta)}"); st.stop()
            if smooth>1: mva=denoise(mva,smooth); pva=denoise(pva,smooth)

            # ВАЖНО: идентификация модели всегда выполняется в нормированных шкалах %/%
            # с учетом шкал, заданных в меню "Параметры".
            # Поэтому K = ΔPV_% / ΔMV_%, а не ΔPV_инж / ΔMV_инж.
            mv_sp=max(float(mv_max)-float(mv_min),1e-6)
            pv_sp=max(float(pv_max)-float(pv_min),1e-6)
            mv_pct=(mva-float(mv_min))/mv_sp*100.0
            pv_pct=(pva-float(pv_min))/pv_sp*100.0

            # Контроль диапазонов: если шкала задана неверно, K будет физически некорректным.
            scale_warnings=[]
            if np.nanmin(mva) < float(mv_min)-1e-9 or np.nanmax(mva) > float(mv_max)+1e-9:
                scale_warnings.append(
                    f"MV выходит за заданную шкалу [{mv_min:g}; {mv_max:g}]. "
                    "Проверьте MV min/max."
                )
            if np.nanmin(pva) < float(pv_min)-1e-9 or np.nanmax(pva) > float(pv_max)+1e-9:
                scale_warnings.append(
                    f"PV выходит за заданную шкалу [{pv_min:g}; {pv_max:g}]. "
                    "Проверьте PV min/max."
                )

            steps,good_steps,partial_steps,models,full_sim,best_key=run_analysis(ta,mv_pct,pv_pct)
            st.session_state.ok=True; st.session_state.model=best_key
            st.session_state.result=dict(steps=steps,good=good_steps,partial=partial_steps,
                models=models,full_sim=full_sim,best_key=best_key)
            st.session_state.signals=dict(
                t=ta,
                mv=mv_pct,
                pv=pv_pct,
                mv_raw=mva,
                pv_raw=pva,
                mv_min=float(mv_min),
                mv_max=float(mv_max),
                pv_min=float(pv_min),
                pv_max=float(pv_max),
                mv_span=mv_sp,
                pv_span=pv_sp,
                scale_warnings=scale_warnings,
            )
            st.session_state.analysis_signature = _current_analysis_signature
        except Exception as ex:
            if _scale_or_column_changed:
                st.session_state.ok = False
            import traceback; st.error(f"Ошибка: {ex}"); st.code(traceback.format_exc())

if not st.session_state.ok: st.caption("Нажмите кнопку"); st.stop()

R=st.session_state.result; A=st.session_state.signals
ta=A["t"]; mv_pct=A["mv"]; pv_pct=A["pv"]
mv_span=float(A.get("mv_span", max(mv_max-mv_min,1e-6)))
pv_span=float(A.get("pv_span", max(pv_max-pv_min,1e-6)))
mv_scale_min=float(A.get("mv_min", mv_min)); mv_scale_max=float(A.get("mv_max", mv_max))
pv_scale_min=float(A.get("pv_min", pv_min)); pv_scale_max=float(A.get("pv_max", pv_max))
pid_sp_eng=float(pid_sv_eng)
pid_sp=(pid_sp_eng-pv_scale_min)/pv_span*100.0
steps=R["steps"]; good_steps=R["good"]; partial_steps=R["partial"]
models=R["models"]; full_sim=R["full_sim"]; best_key=R["best_key"]
n_steps=len(steps); n_good=len(good_steps)

bm=models[best_key]
st.success(
    f"✓  {len(ta)} точек · {n_steps} ступенек "
    f"({n_good} завершённых для идентификации) · "
    f"Лучшая модель: **{best_key.upper()}**"
    +(f"  Точность: {bm.r2:.4f}" if bm.r2 else "")
)
if A.get("scale_warnings"):
    for _msg in A["scale_warnings"]:
        st.warning("⚠️ " + _msg)

st.caption(
    f"Масштабирование для идентификации: MV {mv_scale_min:g}…{mv_scale_max:g} → 0…100%, "
    f"PV {pv_scale_min:g}…{pv_scale_max:g} → 0…100%. "
    "Коэффициент K рассчитывается как ΔPV% / ΔMV%."
)

if partial_steps:
    st.info(f"ℹ️ {len(partial_steps)} ступенек с незавершённым переходным процессом "
            f"исключены из идентификации (учитываются только в симуляции суперпозицией).")

# ── Разделы результатов: профессиональные вкладки ──────────────────
st.markdown("### 📌 Результаты анализа")
tab_overview, tab_models, tab_online, tab_pid, tab_export = st.tabs([
    "📊 Обзор",
    "🔬 Сравнение моделей",
    " Онлайн-идентификация",
    "🎛 PID",
    "📤 Экспорт"
])

# Общие таблицы для вкладок и экспорта
result_df = build_results_dataframe(ta, mv_pct, pv_pct, full_sim, models)
summary_df = build_model_summary_dataframe(models, best_key)

with tab_overview:
    st.markdown("<div class='pro-card'>", unsafe_allow_html=True)
    st.markdown("#### Лучшая модель")
    bm = models.get(best_key)

    if bm and not bm.error:
        params = bm.params_dict()
        mc = st.columns(len(params) + 2)
        for i, (k, v) in enumerate(params.items()):
            mc[i].metric("K, %/%" if k=="K" else k, f"{v:.4f}")
        mc[len(params)].metric("Точность R²", f"{bm.r2:.4f}" if bm.r2 is not None else "—")
        mc[len(params)+1].metric("RMSE", f"{bm.rmse:.4f}" if bm.rmse is not None else "—")

        st.markdown(
            f"<div style='background:#0b1220;border:1px solid #334155;border-left:4px solid #38bdf8;"
            f"border-radius:10px;padding:12px 16px;margin:10px 0;"
            f"font-family:IBM Plex Mono,monospace;font-size:14px;"
            f"color:#93c5fd'>{bm.tf_string()}</div>",
            unsafe_allow_html=True
        )
        if hasattr(bm, "discrete_string") and bm.discrete_string():
            st.code(bm.discrete_string(), language="text")
    else:
        st.warning("Нет корректной модели для отображения.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("#### Графики идентификации")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.28, 0.72],
        subplot_titles=["MV, %", "PV — данные и модели, %"],
        vertical_spacing=0.07)

    fig.add_trace(go.Scatter(x=ta, y=mv_pct, name="MV",
        line=dict(color="#d97706", width=1.8), mode="lines",
        fill="tozeroy", fillcolor="rgba(217,119,6,0.08)"), row=1, col=1)

    for s in steps:
        clr="rgba(22,163,74,0.5)" if s in good_steps else "rgba(220,38,38,0.3)"
        fig.add_vline(x=s.tStep, line_dash="dash", line_color=clr, line_width=1, row=1)

    fig.add_trace(go.Scatter(x=ta, y=pv_pct, name="PV (данные)",
        line=dict(color="#facc15", width=3.0), mode="lines"), row=2, col=1)

    for key in MODEL_CLASSES:
        m=models.get(key)
        if m and not m.error:
            lw=3.0 if key==best_key else 1.4
            dash="solid" if key==best_key else "dot"
            r2s=f"  R²={m.r2:.3f}" if m.r2 is not None else ""
            fig.add_trace(go.Scatter(x=ta, y=full_sim[key],
                name=f"{MODEL_LABELS[key]}{r2s}",
                line=dict(color=MODEL_COLORS[key], width=lw, dash=dash),
                mode="lines"), row=2, col=1)

    for s in steps:
        clr="rgba(22,163,74,0.3)" if s in good_steps else "rgba(220,38,38,0.2)"
        fig.add_vline(x=s.tStep, line_dash="dash", line_color=clr, line_width=1, row=2)

    for s in steps:
        lbl="✓" if s in good_steps else "⚠"
        fig.add_annotation(x=s.tStep, y=max(mv_pct)*0.9, text=f"{lbl}t={s.tStep:.0f}",
            showarrow=False, font=dict(color="#64748b", size=8), row=1, col=1)

    fig.update_layout(height=650, showlegend=True, **PLOTLY_BASE)
    apply_theme(fig, rows=2)
    for ann in fig.layout.annotations:
        ann.font.color="#475569"
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"#### 📊 Детали по {n_steps} ступенькам")
    rows=[]
    for s in steps:
        q=step_quality(s)
        status="✓ хорошая" if s in good_steps else f"⚠ исключена"
        row={"t":f"{s.tStep:.0f}", "ΔMV":f"{s.delta_mv:+.2f}%",
             "PV₀":f"{s.pv0:.3f}", "PV_fin":f"{s.pv_final:.3f}",
             "quality":f"{q:.3f}", "Статус":status}
        sim=full_sim.get(best_key, np.zeros(len(ta)))
        r2s,rmses=r2_rmse(pv_pct, sim, s.step_idx, s.end_idx)
        row["R² сегмента"]=f"{r2s:.4f}"
        row["RMSE сегмента"]=f"{rmses:.4f}"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

with tab_models:
    st.markdown("#### Сравнение моделей")
    st.dataframe(summary_df.set_index("model"), use_container_width=True)

    cmp_fig = go.Figure()
    valid = summary_df.dropna(subset=["r2"])
    if not valid.empty:
        cmp_fig.add_trace(go.Bar(
            x=valid["model"],
            y=valid["r2"],
            name="R²",
            text=[f"{v:.3f}" for v in valid["r2"]],
            textposition="outside"
        ))
        cmp_fig.update_layout(
            title="Сравнение моделей по R²",
            yaxis_title="R²",
            height=420,
            **PLOTLY_BASE
        )
        apply_theme(cmp_fig, rows=1)
        st.plotly_chart(cmp_fig, use_container_width=True)

    st.markdown("#### Как выбирать модель")
    st.markdown("""
- **FOPDT** — лучший выбор для плавных монотонных процессов без перерегулирования.
- **SOPDT** — нужен, если есть выраженное перерегулирование или второй динамический режим.
- **Интегрирующая** — подходит, если PV не выходит на установившееся значение, а продолжает расти/падать.
""")

with tab_online:
    st.markdown("#### Онлайн-идентификация RLS")
    st.markdown("<p class='pro-subtitle'>Параметры K, τ, θ обновляются по мере поступления новых точек данных.</p>", unsafe_allow_html=True)

    c_online1, c_online2 = st.columns(2)
    lam = c_online1.slider("Коэффициент забывания λ", 0.90, 0.999, 0.985, step=0.001)
    max_delay = c_online2.slider("Максимальная задержка delay", 1, 10, 6, step=1)

    online_df = online_rls_identification(ta, mv_pct, pv_pct, max_delay=max_delay, lam=lam)

    if online_df.empty:
        st.warning("Онлайн-идентификация не смогла построить оценки.")
    else:
        last = online_df.dropna(subset=["K_online", "tau_online"]).tail(1)
        if not last.empty:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("K online", f"{float(last['K_online'].iloc[0]):.4f}")
            c2.metric("τ online", f"{float(last['tau_online'].iloc[0]):.4f}")
            c3.metric("θ online", f"{float(last['theta_online'].iloc[0]):.4f}")
            c4.metric("delay", f"{int(last['delay'].iloc[0])}")

        ofig = make_subplots(rows=3, cols=1, shared_xaxes=True,
            subplot_titles=["K online", "τ online", "Ошибка прогноза"],
            vertical_spacing=0.08)

        ofig.add_trace(go.Scatter(x=online_df["Time"], y=online_df["K_online"],
            name="K online", line=dict(width=2, color="#2563eb")), row=1, col=1)
        ofig.add_trace(go.Scatter(x=online_df["Time"], y=online_df["tau_online"],
            name="τ online", line=dict(width=2, color="#7c3aed")), row=2, col=1)
        ofig.add_trace(go.Scatter(x=online_df["Time"], y=online_df["error"],
            name="prediction error", line=dict(width=1.5, color="#dc2626"),
            fill="tozeroy", fillcolor="rgba(220,38,38,0.06)"), row=3, col=1)

        ofig.update_layout(height=650, showlegend=True, **PLOTLY_BASE)
        apply_theme(ofig, rows=3)
        for ann in ofig.layout.annotations:
            ann.font.color="#475569"
        st.plotly_chart(ofig, use_container_width=True)

        st.dataframe(online_df.tail(40), use_container_width=True)
        st.session_state["online_df_export"] = online_df

with tab_pid:
    st.markdown("#### PID параметры")
    bm=models.get(best_key)
    if not bm or bm.error:
        st.warning("Нет корректной модели для PID")
    else:
        p=bm.params_dict()
        # K_p уже нормирован в %/%: модель идентифицируется по mv_pct и pv_pct
        K_p=p.get("K",p.get("Ki",1.)); T_p=p.get("τ",p.get("τ₁",1.)); d_p=p.get("θ",0.)
        pid_rec=PIDCalculator.calculate(K_p,T_p,d_p,None,pid_type)

        st.markdown("##### Рекомендованные")
        Kc_rec = pid_rec["Gain (Kc)"]
        Ti_rec = pid_rec["Reset (Ti)"]
        Td_rec = pid_rec.get("Derivative (Td)")
        PB_rec = round(100.0 / abs(Kc_rec), 4) if Kc_rec else None

        pc=st.columns(5 if Td_rec else 4)
        pc[0].metric("Gain Kc",  f"{Kc_rec:.4f}")
        pc[1].metric("PB = 100/|Kc|", f"{PB_rec:.2f}" if PB_rec else "—")
        pc[2].metric("Reset Ti", f"{Ti_rec:.4f}")
        if Td_rec is not None:
            pc[3].metric("Deriv Td", f"{Td_rec:.4f}")
            pc[4].metric("e", f"{pid_rec['e']:.4f}")
        else:
            pc[3].metric("e", f"{pid_rec['e']:.4f}")

        st.markdown("##### 📐 Формулы (Desired Response)")
        formula_df = pd.DataFrame([
            {"Параметр": "Kc", "Формула": "(2T+d) / (K·(2e+d))", "Значение": f"{Kc_rec:.4f}"},
            {"Параметр": "PB", "Формула": "100 / abs(Kc)", "Значение": f"{PB_rec:.2f}" if PB_rec else "—"},
            {"Параметр": "Ti", "Формула": "T + d/2", "Значение": f"{Ti_rec:.4f}"},
            {"Параметр": "Td", "Формула": "T·d / (2T+d)", "Значение": f"{Td_rec:.4f}" if Td_rec else "—"},
        ])
        st.dataframe(formula_df, use_container_width=True, hide_index=True)
        st.markdown(f"Параметры процесса: K={K_p:.4f} · T={T_p:.4f} · d={d_p:.4f} · e={pid_rec['e']:.4f}")
        st.caption("SV вводится в инженерных единицах и пересчитывается в % по шкале PV min/max. PID-имитация: e=SV−PV, D-составляющая считается по PV без derivative kick, знак Kc сохраняется по знаку K процесса.")

        Kc_u=kc_ov if kc_ov!=0 else pid_rec["Gain (Kc)"]
        Ti_u=ti_ov if ti_ov!=0 else pid_rec["Reset (Ti)"]
        Td_u=td_ov if td_ov!=0 else (pid_rec.get("Derivative (Td)") or 0.)

        # Имитация замкнутого контура: SV считается ступенькой 0 → Setpoint.
        # PV рассчитывается по идентифицированной модели процесса, MV — выход PID.
        sim_t, sv_step, pv_cl_rec, mv_cl_rec, err_cl_rec = PIDCalculator.simulate_closed_loop_step(
            ta, pid_sp, K_p, T_p, d_p,
            pid_rec["Gain (Kc)"], pid_rec["Reset (Ti)"], pid_rec.get("Derivative (Td)") or 0.,
            mv_bias=0., lo=0., hi=100.)

        sim_t, sv_step, pv_cl_usr, mv_cl_usr, err_cl_usr = PIDCalculator.simulate_closed_loop_step(
            ta, pid_sp, K_p, T_p, d_p,
            Kc_u, Ti_u, Td_u,
            mv_bias=0., lo=0., hi=100.)

        use_user_pid = any([kc_ov!=0, ti_ov!=0, td_ov!=0])
        pv_main = pv_cl_usr if use_user_pid else pv_cl_rec
        mv_main = mv_cl_usr if use_user_pid else mv_cl_rec
        err_main = err_cl_usr if use_user_pid else err_cl_rec

        # PID внутри рассчитывается в %, но на графике SV/PV показываем
        # обратно в инженерных единицах PV, потому что уставка вводится в инженерных единицах.
        sv_step_eng = pv_scale_min + sv_step / 100.0 * pv_span
        pv_main_eng = pv_scale_min + pv_main / 100.0 * pv_span
        pv_cl_rec_eng = pv_scale_min + pv_cl_rec / 100.0 * pv_span
        pid_sp_eng_line = pv_scale_min + pid_sp / 100.0 * pv_span

        pid_label = f"настроенные Kc={Kc_u:.3f}, Ti={Ti_u:.3f}, Td={Td_u:.3f}"
        if not use_user_pid:
            pid_label = f"рекомендованные Kc={pid_rec['Gain (Kc)']:.3f}, Ti={pid_rec['Reset (Ti)']:.3f}"

        st.caption(f"Имитация реакции замкнутого контура на ступеньку SV: 0 → {pid_sp_eng:.3g} инженерных ед. ({pid_sp:.1f}%) · Используются {pid_label}")

        fp=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[.58,.42],
            subplot_titles=["SV (ступень с 3 c) и PV замкнутого контура, инженерные ед.","MV выход PID-регулятора, %"],
            vertical_spacing=0.10)

        fp.add_trace(go.Scatter(x=sim_t,y=sv_step_eng,name=f"SV: 0→{pid_sp_eng:.3g} ед. в 3 c",
            line=dict(color="#dc2626",width=2.2,dash="dash")),row=1,col=1)
        fp.add_trace(go.Scatter(x=sim_t,y=pv_main_eng,name="PV реакция процесса",
            line=dict(color="#22d3ee",width=3.0)),row=1,col=1)

        # Если введены ручные PID-параметры — пунктиром показываем рекомендованную реакцию для сравнения.
        if use_user_pid:
            fp.add_trace(go.Scatter(x=sim_t,y=pv_cl_rec_eng,name="PV с рекоменд. PID",
                line=dict(color="#94a3b8",width=1.8,dash="dot")),row=1,col=1)

        fp.add_trace(go.Scatter(x=sim_t,y=mv_main,name="MV PID",
            line=dict(color="#f59e0b",width=2.5)),row=2,col=1)
        if use_user_pid:
            fp.add_trace(go.Scatter(x=sim_t,y=mv_cl_rec,name="MV рекоменд. PID",
                line=dict(color="#94a3b8",width=1.8,dash="dot")),row=2,col=1)

        fp.add_hline(y=pid_sp_eng_line,line_dash="dot",line_color="#dc2626",line_width=1,row=1,col=1)
        fp.add_vline(x=PID_STEP_TIME,line_dash="dash",line_color="#64748b",line_width=1,row=1,col=1)
        fp.add_vline(x=PID_STEP_TIME,line_dash="dash",line_color="#64748b",line_width=1,row=2,col=1)
        fp.add_annotation(x=PID_STEP_TIME,y=pid_sp_eng_line,text="ступень SV, 3 c",
            showarrow=True,arrowhead=2,ax=35,ay=-28,row=1,col=1,
            font=dict(size=11,color="#64748b"))
        fp.add_hline(y=0,line_color="#cbd5e1",line_width=0.8,row=2,col=1)
        fp.update_yaxes(title_text="SV / PV, инженерные ед.", row=1, col=1)
        fp.update_yaxes(title_text="MV, %", range=[-3,103], row=2, col=1)
        fp.update_xaxes(title_text="Время", row=2, col=1)
        fp.update_layout(height=640,showlegend=True,**PLOTLY_BASE)
        apply_theme(fp,rows=2)
        for ann in fp.layout.annotations: ann.font.color="#475569"
        st.plotly_chart(fp,use_container_width=True)

        with st.expander("Численные данные PID-имитации", expanded=False):
            pid_sim_df = pd.DataFrame({
                "Time": sim_t,
                "SV_%": sv_step,
                "PV_%": pv_main,
                "MV_%": mv_main,
                "Error_%": err_main,
            })
            st.dataframe(pid_sim_df.tail(80), use_container_width=True)

with tab_export:
    st.markdown("#### Экспорт результатов")
    online_export_df = st.session_state.get("online_df_export", None)

    excel_bytes = export_to_excel_bytes(result_df, summary_df, online_export_df)
    html_bytes = export_to_html_report(summary_df, result_df)

    cexp1, cexp2, cexp3 = st.columns(3)
    cexp1.download_button(
        "⬇️ Скачать Excel",
        data=excel_bytes,
        file_name="process_model_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
    cexp2.download_button(
        "⬇️ Скачать CSV",
        data=result_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="process_model_results.csv",
        mime="text/csv",
        use_container_width=True
    )
    cexp3.download_button(
        "⬇️ Скачать HTML-отчёт",
        data=html_bytes,
        file_name="process_model_report.html",
        mime="text/html",
        use_container_width=True
    )

    st.markdown("#### Что входит в экспорт")
    st.markdown("""
- исходные MV/PV в процентах;
- рассчитанные кривые моделей;
- ошибки моделей;
- сводка по FOPDT/SOPDT/интегрирующей модели;
- онлайн-оценки RLS, если вкладка онлайн-идентификации была открыта.
""")


st.divider()
st.caption(f"АналитикПро v4.0 · ARX МНК · {n_good}/{n_steps} ступенек · Desired Response PID")
