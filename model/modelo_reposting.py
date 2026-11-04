"""Modelo de reposting por tabla y día de negocio: volumen (TRX) y velocidad (TRX por minuto); el tiempo = TRX / velocidad.

Métodos que compiten en backtest_reposting.py (se elige por tabla y por variable el de menor error a 1-90 días):
  - estacional: promedio de las últimas 4 ocurrencias del mismo día de la semana, con tendencia robusta (Theil-Sen de
    los últimos 180 días hábiles, acotada a ±40 %) y factores de calendario (inicio, 2.º día y cierre de mes, víspera
    de feriado, Nochebuena, 31-dic). Es el método que usaba el proyecto Reposting.
  - media60: media de los últimos 60 días hábiles (piso de referencia: ningún método debería ser peor).
  - lgbm: LightGBM sobre el log con variables de calendario (las mismas ideas que TPH y colas).
  - mezcla: media geométrica de lgbm y estacional (como en colas).
Cambios de nivel conocidos (ESCALONES): el histórico anterior se lleva al nivel vigente antes de entrenar, como en colas.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modelo_tph import PARAMS

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "data/processed"
TABLAS = ["IMG1", "IMG2", "IMG3", "IMG4", "IMG5", "IMG6", "IMG7", "IMG8"]
ETIQUETAS = {"IMG1": "ATM", "IMG2": "CNB – ATM", "IMG3": "Transf. Cuenta ↔ BGL", "IMG4": "Transacciones varias",
             "IMG5": "Volumen residual", "IMG6": "Registro de errores", "IMG7": "Transferencias directas BP-BP", "IMG8": "No financieras"}
# cambio de nivel del 7-sep-2026: IMG4 sube ~93 % y IMG8 baja ~78 % (volumen y velocidad); causa sin confirmar
ESCALONES = {"IMG4": ["2026-09-07"], "IMG8": ["2026-09-07"]}
METODOS = ["estacional", "media60", "lgbm", "mezcla"]
VARIABLES = ["trx", "ratio"]


def cargar(hasta=None):
    """Serie por día de negocio (solo hábiles) y tabla, más el calendario completo."""
    from limpieza_reposting import limpiar, FIN_DIA_NEGOCIO
    if hasta is None:
        l = pd.read_csv(SALIDA / "reposting_limpio.csv", parse_dates=["dia_negocio"])
    else:
        l, _ = limpiar()
        l = l[l.dia_negocio <= hasta]
    cal = pd.read_csv(SALIDA / "calendario_laboral.csv", parse_dates=["fecha"]).set_index("fecha")
    return l, cal


def activas(l, ventana=60):
    """Tablas con volumen sostenido en los últimos `ventana` días hábiles (IMG3 dejó de operar; IMG5 e IMG6 nunca operaron)."""
    p = l.pivot_table(index="dia_negocio", columns="tabla", values="trx").reindex(columns=TABLAS)
    u = p.tail(ventana)
    return [t for t in TABLAS if (u[t] > 0).mean() >= 0.5 and u[t].mean() >= 200]


# ---------------- calendario ----------------
def calendario_dias(cal, dias):
    """Variables de calendario para días de negocio (hábiles)."""
    lab = cal.laborable
    f = pd.DataFrame(index=pd.DatetimeIndex(dias))
    x = f.index
    f["dow"], f["dia_mes"], f["mes"] = x.dayofweek, x.day, x.month
    hab = lab[lab].index
    mes = pd.Series(hab, index=hab).groupby(hab.to_period("M"))
    pri = set(mes.min()); ult = set(mes.max())
    seg = set(pd.Series(hab, index=hab).groupby(hab.to_period("M")).apply(lambda s: s.iloc[1] if len(s) > 1 else s.iloc[0]))
    f["inicio_mes"] = x.isin(pri).astype(int)
    f["inicio_mes2"] = x.isin(seg).astype(int)
    f["cierre_mes"] = x.isin(ult).astype(int)
    sig = [x0 + pd.Timedelta(days=1) for x0 in x]
    f["pre_feriado"] = [int((not lab.get(s, True)) and s.dayofweek < 5) for s in sig]
    f["post_feriado"] = [int(cal.tipo_op.get(d, "") == "PRIMER DIA LABORAL" and d.dayofweek != 0) for d in x]
    f["primer_dia"] = [int(cal.tipo_op.get(d, "") == "PRIMER DIA LABORAL") for d in x]
    f["primer_batch"] = [int(cal.tipo_op.get(d, "") == "PRIMER BATCH") for d in x]
    f["quincena"] = 0
    for p, s in pd.Series(hab, index=hab).groupby(hab.to_period("M")):
        q = s[s.dt.day <= 15]
        if len(q):
            f.loc[f.index == q.iloc[-1], "quincena"] = 1
    f["nochebuena"] = ((x.month == 12) & (x.day == 24)).astype(int)
    f["fin_anio"] = ((x.month == 12) & (x.day == 31)).astype(int)
    f["t_dias"] = (x - pd.Timestamp("2024-01-01")).days
    return f


FEATURES = ["dow", "dia_mes", "mes", "inicio_mes", "inicio_mes2", "cierre_mes", "pre_feriado", "post_feriado", "primer_dia",
            "primer_batch", "quincena", "nochebuena", "fin_anio", "t_dias"]
ESPECIALES = ["inicio_mes", "inicio_mes2", "cierre_mes", "pre_feriado", "nochebuena", "fin_anio"]


# ---------------- nivelación por escalones ----------------
def nivelar(s, tabla, origen, n=10, minimo=3):
    """Lleva el histórico anterior a cada escalón (conocido antes de `origen`) al nivel posterior: razón de medianas
    de los `n` días hábiles a cada lado. Devuelve la serie nivelada."""
    s = s.copy()
    for e in [pd.Timestamp(x) for x in ESCALONES.get(tabla, []) if pd.Timestamp(x) < origen]:
        antes, desp = s[s.index < e].dropna().tail(n), s[s.index >= e].dropna().head(n)
        if len(antes) >= minimo and len(desp) >= minimo and antes.median() > 0:
            s[s.index < e] *= desp.median() / antes.median()
    return s


# ---------------- métodos ----------------
def _tendencia(s, ventana=180, tope=0.40):
    """Pendiente robusta (Theil-Sen) del log en los últimos `ventana` días hábiles, por día hábil."""
    from scipy.stats import theilslopes
    y = np.log(s.dropna().clip(lower=1)).tail(ventana)
    if len(y) < 40:
        return 0.0
    b = theilslopes(y.values, np.arange(len(y)))[0]
    return float(b), tope


def factores_calendario(s, F, k=3.0, minimo=3):
    """Factor por tipo de día especial: mediana de (valor / promedio de los 4 mismos días de la semana previos), relativo
    a los días normales, contraído hacia 1 con n/(n+k)."""
    df = pd.DataFrame({"y": s}).join(F)
    ref = df.groupby("dow").y.transform(lambda z: z.shift(1).rolling(4, min_periods=2).mean())
    r = (df.y / ref).replace([np.inf, -np.inf], np.nan)
    normal = r[(df[ESPECIALES].sum(axis=1) == 0)].median()
    out = {}
    for c in ESPECIALES:
        x = r[df[c] == 1].dropna()
        if len(x) >= (2 if c in ("nochebuena", "fin_anio") else minimo) and normal > 0:
            out[c] = float(np.exp(len(x) / (len(x) + k) * np.log(x.median() / normal)))
    return out


def estacional(s, F_hist, F_fut, con_calendario=True):
    s = s.dropna()
    if s.empty:
        return pd.Series(0.0, index=F_fut.index)
    base = {d: s[s.index.dayofweek == d].tail(4).mean() for d in range(5)}
    nivel = np.array([base.get(d, s.tail(20).mean()) for d in F_fut.index.dayofweek])
    tend = _tendencia(s)
    if isinstance(tend, tuple):
        b, tope = tend
        k = np.arange(1, len(F_fut) + 1)
        nivel = nivel * np.clip(np.exp(b * k), 1 - tope, 1 + tope)
    if con_calendario:
        fac = factores_calendario(s, F_hist)
        for c, v in fac.items():
            nivel = np.where(F_fut[c].values == 1, nivel * v, nivel)
    return pd.Series(np.maximum(nivel, 0), index=F_fut.index)


def media60(s, F_fut):
    return pd.Series(float(s.dropna().tail(60).mean()) if s.notna().any() else 0.0, index=F_fut.index)


def lgbm(s, F_hist, F_fut):
    y = s.dropna()
    X = F_hist.loc[y.index, FEATURES]
    m = lgb.LGBMRegressor(**{**PARAMS, "min_child_samples": 10}).fit(X, np.log1p(y.clip(lower=0)))
    return pd.Series(np.maximum(0, np.expm1(m.predict(F_fut[FEATURES]))), index=F_fut.index)


def predecir(s, tabla, origen, F_hist, F_fut, metodo):
    """Pronóstico de una serie (trx o ratio) de una tabla, entrenando con lo anterior a `origen`."""
    s = nivelar(s[s.index < origen], tabla, origen)
    if metodo == "media60":
        return media60(s, F_fut)
    if metodo == "estacional":
        return estacional(s, F_hist, F_fut)
    lg = lgbm(s, F_hist, F_fut)
    if metodo == "lgbm":
        return lg
    es = estacional(s, F_hist, F_fut)
    return np.sqrt((lg + 1) * (es + 1)) - 1
