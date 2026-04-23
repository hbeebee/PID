"""
Process Model Analyzer  v4.0
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

import io, re, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter

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
    Возвращает std хвоста плато (% от диапазона).
    Низкое значение = ступенька завершена.
    """
    plat=s.pv[s.step_idx:s.end_idx]
    sz=max(3,int(len(plat)*0.15))
    tail_std=float(np.std(plat[-sz:]))
    return tail_std

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
        Ищем a, b, delay минимизирующие суммарный MSE.
        """
        if not steps_good:
            self.error="Нет завершённых ступенек для идентификации"
            return

        best_mse=np.inf; best_a=best_b=best_d=None

        for d in range(2, 8):          # delay от 2 (минимум 1 шаг dead time)
            rows=[]; tgt=[]
            for s in steps_good:
                y=s.pv[s.step_idx:s.end_idx]
                u=s.mv[s.step_idx:s.end_idx]
                N=len(y)
                if N<d+3: continue
                for k in range(d+1,N):
                    rows.append([y[k-1], u[k-d]])
                    tgt.append(y[k])
            if len(rows)<6: continue
            try: ab=np.linalg.lstsq(np.array(rows),np.array(tgt),rcond=None)[0]
            except: continue
            a_c,b_c=ab[0],ab[1]
            if not (0.01<a_c<0.9999): continue
            # Проверяем знак K
            K_c=b_c/(1-a_c)
            if K_c<=0: continue
            # Общий MSE
            total_mse=0.; cnt=0
            for s in steps_good:
                y=s.pv[s.step_idx:s.end_idx]; u=s.mv[s.step_idx:s.end_idx]; N=len(y)
                if N<d+3: continue
                pred=np.zeros(N); pred[0]=y[0]
                for k in range(1,N):
                    ud=u[k-d] if k>=d else 0.; pred[k]=a_c*pred[k-1]+b_c*ud
                total_mse+=float(np.sum((y-pred)**2)); cnt+=N
            if cnt==0: continue
            mse=total_mse/cnt
            if mse<best_mse:
                best_mse=mse; best_a=a_c; best_b=b_c; best_d=d

        if best_a is None:
            self.error="ARX МНК не сошёлся"
            return

        self.a=best_a; self.b=best_b; self.delay=best_d
        self.tau=-dt/np.log(best_a)
        self.K=best_b/(1-best_a)
        self.theta=max(0.0,(best_d-1)*dt)
        # Берём pv0/delta_mv из первой хорошей ступеньки для совместимости
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
        if e is None or e<=0: e=T+d
        Kc=(2*T+d)/max(K*(2*e+d),1e-12); Ti=T+d/2.
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


# ══════════════════════════════════════════════════════════════════
#  АНАЛИЗ
# ══════════════════════════════════════════════════════════════════

MODEL_CLASSES={"fopdt":FOPDTModel,"sopdt":SOPDTModel,"integrating":IntegratingModel}
MODEL_LABELS={"fopdt":"FOPDT","sopdt":"SOPDT","integrating":"Интегрирующий"}
MODEL_COLORS={"fopdt":"#2563eb","sopdt":"#7c3aed","integrating":"#d97706"}

STD_THRESHOLD = 0.15   # % — порог для "хорошей" ступеньки


def run_analysis(t,mv_pct,pv_pct):
    dt=float(np.median(np.diff(t))) if len(t)>1 else 1.
    steps=find_all_steps(t,mv_pct,pv_pct)

    # Разделяем ступеньки на хорошие и незавершённые
    good_steps=[]; partial_steps=[]
    for s in steps:
        q=step_quality(s)
        if q<=STD_THRESHOLD and len(s.pv[s.step_idx:s.end_idx])>=6:
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
#  PLOTLY ТЕМА — светлая
# ══════════════════════════════════════════════════════════════════

BG      = "#ffffff"
BG_P    = "#f8fafc"
GRID    = "#e2e8f0"
TXT     = "#1e293b"
TXT2    = "#64748b"
BORDER  = "#cbd5e1"

PLOTLY_BASE = dict(
    paper_bgcolor=BG,
    plot_bgcolor=BG_P,
    font=dict(color=TXT, family="IBM Plex Mono, monospace", size=11),
    legend=dict(bgcolor="rgba(255,255,255,0.95)", bordercolor=BORDER,
                borderwidth=1, font=dict(color=TXT, size=10)),
    margin=dict(l=55, r=20, t=40, b=36),
)

def apply_theme(fig, rows=1):
    fig.update_layout(**PLOTLY_BASE)
    for i in range(1, rows+1):
        fig.update_xaxes(
            gridcolor=GRID, zerolinecolor=GRID, zerolinewidth=1,
            tickfont=dict(color=TXT2), title_font=dict(color=TXT),
            linecolor=BORDER, showline=True, mirror=False, row=i)
        fig.update_yaxes(
            gridcolor=GRID, zerolinecolor=GRID, zerolinewidth=1,
            tickfont=dict(color=TXT2), title_font=dict(color=TXT),
            linecolor=BORDER, showline=True, mirror=False, row=i)
    return fig


# ══════════════════════════════════════════════════════════════════
#  APP
# ══════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Process Model Analyzer", page_icon="⚡",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;600&display=swap');

/* ── Base ── */
html, body, [class*="css"], [data-testid] {
    font-family: 'IBM Plex Sans', sans-serif !important;
    background-color: #f1f5f9 !important;
    color: #1e293b !important;
}
.main, .block-container,
[data-testid="stAppViewContainer"],
[data-testid="stMainBlockContainer"] {
    background: #f1f5f9 !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
    box-shadow: 2px 0 8px rgba(0,0,0,0.04) !important;
}
section[data-testid="stSidebar"] * { color: #1e293b !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #0f172a !important; font-weight: 600 !important; }
section[data-testid="stSidebar"] label p { color: #475569 !important; font-size: 12px !important; }
section[data-testid="stSidebar"] hr { border-color: #e2e8f0 !important; }

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-top: 3px solid #2563eb !important;
    border-radius: 10px !important;
    padding: 1rem 1.25rem !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
}
div[data-testid="stMetricLabel"] p,
div[data-testid="stMetricLabel"] label {
    color: #64748b !important;
    font-size: 11px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] * {
    color: #0f172a !important;
    font-size: 1.55rem !important;
    font-weight: 600 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
div[data-testid="stMetricDelta"] * { color: #16a34a !important; }
div[data-testid="stMetricDelta"][data-direction="down"] * { color: #dc2626 !important; }

/* ── Content cards (expander) ── */
[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
}
[data-testid="stExpander"] * { color: #1e293b !important; }
[data-testid="stExpander"] summary { color: #0f172a !important; font-weight: 500 !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] * { color: #1e293b !important; }
.dataframe th { background: #f8fafc !important; color: #475569 !important;
                font-size: 11px !important; text-transform: uppercase !important;
                letter-spacing: 0.06em !important; }
.dataframe td { color: #1e293b !important; }

/* ── Code ── */
code, pre {
    background: #f0f4ff !important;
    color: #1d4ed8 !important;
    border: 1px solid #bfdbfe !important;
    border-radius: 6px !important;
    padding: 0.25em 0.6em !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] * { color: #1e293b !important; }

/* ── Radio ── */
[data-testid="stRadio"] label p { color: #334155 !important; }
[data-testid="stRadio"] div { color: #1e293b !important; }

/* ── Labels / inputs ── */
label[data-testid="stWidgetLabel"] p { color: #475569 !important; font-size: 13px !important; }
p, span, li, td, th { color: #1e293b !important; }
h1 { color: #0f172a !important; }
h2, h3 { color: #1e293b !important; }

/* ── Caption ── */
[data-testid="stCaptionContainer"] p { color: #94a3b8 !important; font-size: 12px !important; }

/* ── Divider ── */
hr { border-color: #e2e8f0 !important; }

/* ── File uploader — убираем наложение текста ── */
[data-testid="stFileUploader"] {
    background: #ffffff !important;
    border: 2px dashed #cbd5e1 !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"] * { color: #475569 !important; }
/* Скрываем дублирующийся текст Browse/Upload внутри кнопки загрузчика */
[data-testid="stFileUploader"] section > button > div > span:first-child {
    display: none !important;
}
[data-testid="stFileUploaderDropzone"] small { display: none !important; }

/* ── Buttons — убираем все фоны/обводки вокруг кнопки ── */
.stButton {
    background: none !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}
.stButton > button {
    background: #dc2626 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 0.55rem 1.75rem !important;
    letter-spacing: 0.02em !important;
    box-shadow: none !important;
    outline: none !important;
    transition: background 0.15s !important;
}
.stButton > button:hover {
    background: #b91c1c !important;
    box-shadow: none !important;
    outline: none !important;
}
.stButton > button:focus,
.stButton > button:focus-visible,
.stButton > button:active {
    outline: none !important;
    box-shadow: none !important;
    border: none !important;
}
/* Убираем любой фон-обёртку вокруг кнопки */
.stButton > div,
[data-testid="baseButton-secondary"],
[data-testid="baseButton-primary"] {
    background: none !important;
    box-shadow: none !important;
    border: none !important;
}

/* ── Select / number input ── */
.stSelectbox select, .stNumberInput input, .stTextInput input {
    background: #ffffff !important;
    color: #1e293b !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 6px !important;
}

/* ── Success / info message styling ── */
div[data-testid="stAlert"][data-baseweb="notification"] {
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# Session state
for _k,_v in [("done",False),("mk","fopdt"),("res",None),("data_arrays",None)]:
    if _k not in st.session_state: st.session_state[_k]=_v

# Заголовок
st.markdown(
    "<h1 style='color:#1e3a8a;letter-spacing:-.02em;font-family:IBM Plex Sans,sans-serif;'>"
    "⚡ Process Model Analyzer</h1>",
    unsafe_allow_html=True
)
st.caption("ARX МНК · FOPDT · SOPDT · Интегрирующий · PID  v4.0")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Параметры")
    st.markdown("### 📂 Данные")
    uploaded=st.file_uploader("CSV или Excel",type=["csv","xlsx","xls"])
    st.markdown("### 📏 Шкалы приборов")
    st.caption("Для перевода K в % к %")
    c1,c2=st.columns(2)
    mv_min=c1.number_input("MV min",value=0.0,step=1.); mv_max=c2.number_input("MV max",value=10.0,step=1.)
    c3,c4=st.columns(2)
    pv_min=c3.number_input("PV min",value=0.0,step=1.); pv_max=c4.number_input("PV max",value=10.0,step=1.)
    st.markdown("### 🔇 Шумоподавление")
    smooth=st.slider("Окно фильтра",1,51,1,step=2)
    st.markdown("### 🎛️ PID")
    pid_type=st.radio("Тип",["PI","PID"],horizontal=True)
    pid_e=st.number_input("e (0=авто T+d)",value=0.0,step=0.1,format="%.4f")
    pid_sp=st.number_input("Setpoint (%)",value=60.0,step=1.)
    st.markdown("### ✏️ Корректировка PID")
    st.caption("0 = использовать рекомендованные")
    kc_ov=st.number_input("Kc",value=0.0,step=0.001,format="%.4f")
    ti_ov=st.number_input("Ti",value=0.0,step=0.001,format="%.4f")
    td_ov=st.number_input("Td",value=0.0,step=0.001,format="%.4f")

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

with st.expander("🔍 Предпросмотр (20 строк)"):
    show=[mv_col,pv_col]+([t_col] if t_col!="— индекс строки —" and t_col in df.columns else [])
    st.dataframe(df[show].head(20),use_container_width=True)
    st.caption(f"Строк: {len(df)}")

st.markdown("### 2️⃣ Анализ")
if st.button("▶  Запустить анализ",type="primary"):
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
            mv_sp=max(mv_max-mv_min,1e-6); pv_sp=max(pv_max-pv_min,1e-6)
            mv_pct=(mva-mv_min)/mv_sp*100.; pv_pct=(pva-pv_min)/pv_sp*100.
            steps,good_steps,partial_steps,models,full_sim,best_key=run_analysis(ta,mv_pct,pv_pct)
            st.session_state.done=True; st.session_state.mk=best_key
            st.session_state.res=dict(steps=steps,good=good_steps,partial=partial_steps,
                models=models,full_sim=full_sim,best_key=best_key)
            st.session_state.data_arrays=dict(t=ta,mv=mv_pct,pv=pv_pct)
        except Exception as ex:
            import traceback; st.error(f"Ошибка: {ex}"); st.code(traceback.format_exc())

if not st.session_state.done: st.caption("Нажмите кнопку"); st.stop()

R=st.session_state.res; A=st.session_state.data_arrays
ta=A["t"]; mv_pct=A["mv"]; pv_pct=A["pv"]
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
if partial_steps:
    st.info(f"ℹ️ {len(partial_steps)} ступенек с незавершённым переходным процессом "
            f"исключены из идентификации (учитываются только в симуляции суперпозицией).")

# ── Раздел 3: Параметры ───────────────────────────────────────────
st.markdown("### 3️⃣ Параметры модели")

def _sw(): st.session_state.mk=st.session_state["model_selector"]
opts=list(MODEL_CLASSES.keys())
labels=[]
for k in opts:
    m=models.get(k)
    fit_str=(f"  Точность: {m.r2:.4f}" if m and not m.error and m.r2 else "  (н/д)")
    sfx="  ✓" if k==best_key else ""
    labels.append(f"{MODEL_LABELS[k]}{sfx}{fit_str}")

st.radio("Модель:",opts,format_func=lambda k:labels[opts.index(k)],
         index=opts.index(st.session_state.mk),key="model_selector",on_change=_sw,horizontal=True)

dk=st.session_state.mk; dm=models.get(dk)

if dm and dm.error:
    st.warning(f"**{MODEL_LABELS[dk]}**: {dm.error}")
elif dm:
    params=dm.params_dict()
    mc=st.columns(len(params)+2)
    for i,(k,v) in enumerate(params.items()): mc[i].metric(k,f"{v:.4f}")
    mc[len(params)].metric("Точность подгонки",f"{dm.r2:.4f}" if dm.r2 else "—")
    mc[len(params)+1].metric("Ошибка (RMSE)",f"{dm.rmse:.4f}" if dm.rmse else "—")

    st.markdown(
        f"<div style='background:#eff6ff;border:1px solid #bfdbfe;border-left:4px solid #2563eb;"
        f"border-radius:8px;padding:12px 16px;margin:8px 0;"
        f"font-family:IBM Plex Mono,monospace;font-size:14px;"
        f"color:#1d4ed8'>{dm.tf_string()}</div>",
        unsafe_allow_html=True)

    with st.expander(f"📊 Детали по {n_steps} ступенькам"):
        rows=[]
        for s in steps:
            q=step_quality(s)
            status="✓ хорошая" if q<=STD_THRESHOLD else f"⚠ незаверш. (std={q:.2f})"
            row={"t":f"{s.tStep:.0f}","ΔMV":f"{s.delta_mv:+.2f}%",
                 "PV₀":f"{s.pv0:.3f}","PV_fin":f"{s.pv_final:.3f}","Статус":status}
            sim=full_sim.get(dk,np.zeros(len(ta)))
            r2s,rmses=r2_rmse(pv_pct,sim,s.step_idx,s.end_idx)
            row["Точность"]=f"{r2s:.4f}"; row["RMSE"]=f"{rmses:.4f}"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows),use_container_width=True)

# ── Раздел 4: Графики ─────────────────────────────────────────────
st.markdown("### 4️⃣ Графики идентификации")

fig=make_subplots(rows=2,cols=1,shared_xaxes=True,
    row_heights=[0.28,0.72],
    subplot_titles=["MV, %","PV — данные и модели, %"],
    vertical_spacing=0.07)

fig.add_trace(go.Scatter(x=ta,y=mv_pct,name="MV",
    line=dict(color="#d97706",width=1.8),mode="lines",
    fill="tozeroy",fillcolor="rgba(217,119,6,0.08)"),row=1,col=1)

for s in steps:
    clr="rgba(22,163,74,0.5)" if s in good_steps else "rgba(220,38,38,0.3)"
    fig.add_vline(x=s.tStep,line_dash="dash",line_color=clr,line_width=1,row=1)

fig.add_trace(go.Scatter(x=ta,y=pv_pct,name="PV (данные)",
    line=dict(color="#0f172a",width=2.0),mode="lines"),row=2,col=1)

for key in MODEL_CLASSES:
    m=models.get(key)
    if m and not m.error:
        lw=2.5 if key==dk else 1.2; dash="solid" if key==dk else "dot"
        r2s=f"  Точность={m.r2:.3f}" if m.r2 else ""
        fig.add_trace(go.Scatter(x=ta,y=full_sim[key],
            name=f"{MODEL_LABELS[key]}{r2s}",
            line=dict(color=MODEL_COLORS[key],width=lw,dash=dash),mode="lines"),row=2,col=1)

for s in steps:
    clr="rgba(22,163,74,0.3)" if s in good_steps else "rgba(220,38,38,0.2)"
    fig.add_vline(x=s.tStep,line_dash="dash",line_color=clr,line_width=1,row=2)

# Метки ступенек
for s in steps:
    lbl="✓" if s in good_steps else "⚠"
    fig.add_annotation(x=s.tStep,y=max(mv_pct)*0.9,text=f"{lbl}t={s.tStep:.0f}",
        showarrow=False,font=dict(color="#64748b",size=8),row=1,col=1)

fig.update_layout(height=620,showlegend=True,**PLOTLY_BASE)
apply_theme(fig,rows=2)
for ann in fig.layout.annotations: ann.font.color="#475569"
st.plotly_chart(fig,use_container_width=True)

# ── Раздел 5: Сравнение ───────────────────────────────────────────
st.markdown("### 5️⃣ Сравнение моделей")
cmp=[]
for key in MODEL_CLASSES:
    m=models.get(key)
    row={"Модель":MODEL_LABELS[key]+(" ✓" if key==best_key else "")}
    if m and not m.error:
        ps=" · ".join(f"{k}={v:.4f}" for k,v in m.params_dict().items())
        row.update({"Точность":f"{m.r2:.4f}" if m.r2 else "—",
                    "RMSE":f"{m.rmse:.4f}" if m.rmse else "—",
                    "Параметры":ps,"G(s)":m.tf_string()})
    else:
        row.update({"Точность":"—","RMSE":"—","Параметры":m.error if m else "—","G(s)":"—"})
    cmp.append(row)
st.dataframe(pd.DataFrame(cmp).set_index("Модель"),use_container_width=True)

# ── Раздел 6: PID ─────────────────────────────────────────────────
st.markdown("### 6️⃣ PID параметры")
bm=models.get(best_key)
if not bm or bm.error:
    st.warning("Нет корректной модели для PID")
else:
    p=bm.params_dict()
    K_p=p.get("K",p.get("Ki",1.)); T_p=p.get("τ",p.get("τ₁",1.)); d_p=p.get("θ",0.)
    pid_rec=PIDCalculator.calculate(K_p,T_p,d_p,pid_e or None,pid_type)

    st.markdown("#### Рекомендованные")
    Kc_rec = pid_rec["Gain (Kc)"]
    Ti_rec = pid_rec["Reset (Ti)"]
    Td_rec = pid_rec.get("Derivative (Td)")
    PB_rec = round(100.0 / Kc_rec, 4) if Kc_rec else None

    pc=st.columns(5 if Td_rec else 4)
    pc[0].metric("Gain Kc",  f"{Kc_rec:.4f}")
    pc[1].metric("PB = 100/Kc", f"{PB_rec:.2f}" if PB_rec else "—")
    pc[2].metric("Reset Ti", f"{Ti_rec:.4f}")
    if Td_rec is not None:
        pc[3].metric("Deriv Td", f"{Td_rec:.4f}")
        pc[4].metric("e (response time)", f"{pid_rec['e']:.4f}")
    else:
        pc[3].metric("e (response time)", f"{pid_rec['e']:.4f}")

    with st.expander("📐 Формулы (Desired Response)"):
        PB_formula = f"100 / {Kc_rec:.4f} = **{PB_rec:.2f}**" if PB_rec else "—"
        Td_val = f"**{Td_rec:.4f}**" if Td_rec else "—"
        st.markdown(f"""
| Параметр | Формула | Значение |
|---------|---------|---------|
| **Kc** — Коэффициент усиления | `(2T+d) / (K·(2e+d))` | **{Kc_rec:.4f}** |
| **PB** — Полоса пропорциональности | `100 / Kc` | **{PB_rec:.2f}** |
| **Ti** — Время интегрирования | `T + d/2` | **{Ti_rec:.4f}** |
| **Td** — Время дифференцирования | `T·d / (2T+d)` | {Td_val} |

Параметры процесса: K={K_p:.4f} · T={T_p:.4f} · d={d_p:.4f} · e={pid_rec['e']:.4f}
Использовано ступенек: {n_good} из {n_steps}
""")

    Kc_u=kc_ov if kc_ov!=0 else pid_rec["Gain (Kc)"]
    Ti_u=ti_ov if ti_ov!=0 else pid_rec["Reset (Ti)"]
    Td_u=td_ov if td_ov!=0 else (pid_rec.get("Derivative (Td)") or 0.)

    if any([kc_ov!=0,ti_ov!=0,td_ov!=0]):
        st.markdown("#### Применённые (скорректированные)")
        PB_u = round(100.0 / Kc_u, 4) if Kc_u else None
        ac=st.columns(4)
        ac[0].metric("Kc",  f"{Kc_u:.4f}", delta=f"{Kc_u-pid_rec['Gain (Kc)']:+.4f}")
        ac[1].metric("PB = 100/Kc", f"{PB_u:.2f}" if PB_u else "—")
        ac[2].metric("Ti",  f"{Ti_u:.4f}", delta=f"{Ti_u-pid_rec['Reset (Ti)']:+.4f}")
        if pid_type=="PID":
            ac[3].metric("Td", f"{Td_u:.4f}",
                         delta=f"{Td_u-(pid_rec.get('Derivative (Td)') or 0):+.4f}")

    pv_s=full_sim.get(best_key,pv_pct)
    mv0_v=float(np.mean(mv_pct[:max(5,steps[0].step_idx)]))
    mv_rec,_=PIDCalculator.simulate_pid(ta,pv_s,pid_sp,
        pid_rec["Gain (Kc)"],pid_rec["Reset (Ti)"],pid_rec.get("Derivative (Td)") or 0.,mv_init=mv0_v)
    mv_usr,_=PIDCalculator.simulate_pid(ta,pv_s,pid_sp,Kc_u,Ti_u,Td_u,mv_init=mv0_v)

    fp=make_subplots(rows=3,cols=1,shared_xaxes=True,row_heights=[.40,.35,.25],
        subplot_titles=["PV и уставка, %","MV выход регулятора, %","Ошибка e(t), %"],
        vertical_spacing=0.09)
    fp.add_trace(go.Scatter(x=ta,y=pv_pct,name="PV данные",
        line=dict(color="#94a3b8",width=1,dash="dot")),row=1,col=1)
    fp.add_hline(y=pid_sp,line_dash="dash",line_color="#dc2626",line_width=1.5,
        annotation_text=f"SP={pid_sp:.1f}%",annotation_font_color="#dc2626")
    fp.add_trace(go.Scatter(x=ta,y=mv_rec,
        name=f"MV рекоменд. Kc={pid_rec['Gain (Kc)']:.3f}",
        line=dict(color="#d97706",width=2)),row=2,col=1)
    if any([kc_ov!=0,ti_ov!=0,td_ov!=0]):
        fp.add_trace(go.Scatter(x=ta,y=mv_usr,
            name=f"MV скоррект. Kc={Kc_u:.3f}",
            line=dict(color="#16a34a",width=2,dash="dash")),row=2,col=1)
    fp.add_trace(go.Scatter(x=ta,y=pid_sp-pv_pct,name="e(t)",
        fill="tozeroy",fillcolor="rgba(220,38,38,0.06)",
        line=dict(color="#dc2626",width=1.5)),row=3,col=1)
    fp.add_hline(y=0,line_color="#cbd5e1",line_width=0.8,row=3)
    fp.update_layout(height=650,showlegend=True,**PLOTLY_BASE)
    apply_theme(fp,rows=3)
    for ann in fp.layout.annotations: ann.font.color="#475569"
    st.plotly_chart(fp,use_container_width=True)

st.divider()
st.caption(f"Process Model Analyzer v4.0 · ARX МНК · {n_good}/{n_steps} ступенек · Desired Response PID")
