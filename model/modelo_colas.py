"""Modelo de TPH por cola: LightGBM sobre el log del conteo con variables de calendario, promediado con el
perfil de las últimas 8 semanas.

El reparto entre colas cambió varias veces por decisiones de enrutamiento (ESCALONES). El histórico anterior a
cada escalón se lleva al nivel vigente antes de entrenar: supone que el enrutamiento actual se mantiene.
En backtest, nivelar, dar el escalón como variable o entrenar solo con lo reciente dan el mismo error (8,7-8,8 %).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features_tph import construir, FEATURES, SALIDA
from modelo_tph import PARAMS

# Primer día con el nivel nuevo. Detectados en los datos (salto de la mediana de 5 días laborables); la causa
# no está confirmada por el equipo de plataforma. Un cambio futuro conocido se agrega aquí con su fecha.
ESCALONES = {
    "OMN": ["2025-11-19", "2025-11-25", "2025-12-03", "2026-01-21"],
    "CI3": ["2025-11-19", "2025-11-25", "2025-12-03", "2026-01-21"],
    "OM2": ["2025-11-25", "2025-12-03", "2026-01-21", "2026-02-10"],
    "OM1": ["2026-03-04", "2026-03-10"],
    "CI1": ["2026-03-04", "2026-03-10"],
}
PERFIL_SIMPLE = ["FN1", "DP4", "OLR", "VST"]   # < 0,02 % del total: solo perfil de 8 semanas
FAMILIAS = {"OM": ["OM1", "OM2", "OMN"], "CI": ["CI1", "CI2", "CI3", "CIF"], "DEP / DP": ["DEP", "DP1", "DP2", "DP3", "DP4"], "GEN": ["GEN"]}
COLS = [c for c in FEATURES if c != "regimen"] + ["escalon"]
SEMANAS_PERFIL = 8
METODO = "mezcla"                             # elegido en backtest_colas.py


def cargar():
    """g: horas con las colas (NaN fuera de `ok`), estado y calendario. X: variables. ent: horas-cola usables
    para entrenar (ok y no excluidas por limpieza_colas.py)."""
    g = pd.read_csv(SALIDA / "colas_limpio.csv", parse_dates=["hora"]).set_index("hora")
    colas = [c for c in g.columns if c != "estado"]
    X = construir(g.index)
    g["es_laborable"], g["es_feriado"] = X.es_laborable.values == 1, X.es_feriado.values == 1
    ent = g[colas].notna() & (g.estado == "ok").values[:, None]
    ex = pd.read_csv(SALIDA / "colas_excluidas.csv", parse_dates=["hora"])
    for q, x in ex.groupby("cola"):
        ent.loc[x.hora.values, q] = False
    return g, X, ent, colas


def escalon(cola, fechas):
    """Número de cambios de enrutamiento de la cola ocurridos hasta cada fecha (el futuro queda en el último)."""
    f = pd.DatetimeIndex(fechas).normalize()
    return sum((f >= pd.Timestamp(e)).astype(int) for e in ESCALONES.get(cola, [])) + np.zeros(len(f), int)


def ultimo_escalon(cola, origen):
    e = [pd.Timestamp(x) for x in ESCALONES.get(cola, []) if pd.Timestamp(x) < origen]
    return max(e) if e else pd.Timestamp("2000-01-01")


def perfil(g, ent, cola, origen, idx):
    """Mediana por (día de la semana, hora) de las últimas 8 semanas, sin cruzar el último escalón."""
    desde = max(origen - pd.Timedelta(weeks=SEMANAS_PERFIL), ultimo_escalon(cola, origen))
    m = ent[cola] & (g.index < origen) & (g.index >= desde)
    y = g.loc[m, cola]
    med = y.groupby([y.index.dayofweek, y.index.hour]).median()
    p = pd.Series(pd.MultiIndex.from_arrays([idx.dayofweek, idx.hour]).map(med).values, index=idx, dtype=float)
    alt = pd.Series(idx.hour.map(y.groupby(y.index.hour).median()).values, index=idx, dtype=float)
    return p.fillna(alt).fillna(0.0)


def factores_escalon(g, ent, cola, origen, n=10, minimo=3):
    """Salto de nivel de cada escalón anterior a `origen`: mediana del total diario laborable de los `n` días
    posteriores / la de los `n` anteriores, sin cruzar otro escalón. Devuelve {fecha: razón}."""
    es = [pd.Timestamp(e) for e in ESCALONES.get(cola, []) if pd.Timestamp(e) < origen]
    if not es:
        return {}
    m = ent[cola] & (g.index < origen) & g.es_laborable & ~g.es_feriado
    tot = g.loc[m, cola].groupby(g.index[m].normalize()).agg(["sum", "size"])
    tot = tot.loc[tot["size"] >= 15, "sum"]
    lim = [pd.Timestamp("2000-01-01")] + es + [origen]
    f = {}
    for i, e in enumerate(es):
        antes, desp = tot[(tot.index >= lim[i]) & (tot.index < e)][-n:], tot[(tot.index >= e) & (tot.index < lim[i + 2])][:n]
        f[e] = float(desp.median() / antes.median()) if len(antes) >= minimo and len(desp) >= minimo and antes.median() > 0 else 1.0
    return f


def lgbm(g, X, ent, cola, origen, Xfut, cols=None, modo="nivelado"):
    """modo: 'nivelado' lleva el histórico anterior a cada escalón al nivel vigente (así la estacionalidad de 2025
    se aprende sin arrastrar el enrutamiento viejo); 'variable' da el escalón como variable; 'reciente' entrena
    solo desde el último escalón, sin variables anuales."""
    cols = list(cols or COLS)
    tr = (ent[cola] & (g.index < origen)).values
    Xtr, ytr = X[tr].assign(escalon=escalon(cola, X.index[tr])), g.loc[tr, cola].astype(float)
    if modo == "nivelado":
        adj = np.ones(len(ytr))
        for e, r in factores_escalon(g, ent, cola, origen).items():
            adj = adj * np.where(Xtr.index < e, r, 1.0)
        ytr, cols = ytr * adj, [c for c in cols if c != "escalon"]
    elif modo == "reciente" and cola in ESCALONES:
        rec = Xtr.index >= ultimo_escalon(cola, origen)
        if rec.sum() >= 24 * 60:
            Xtr, ytr, cols = Xtr[rec], ytr[rec], [c for c in cols if c not in ("escalon", "doy", "mes", "t_dias")]
    m = lgb.LGBMRegressor(**PARAMS).fit(Xtr[cols], np.log1p(ytr))
    Xf = Xfut.assign(escalon=escalon(cola, Xfut.index))
    return pd.Series(np.maximum(0, np.expm1(m.predict(Xf[cols]))), index=Xfut.index)


def predecir(g, X, ent, cola, origen, Xfut, metodo=None):
    """Pronóstico horario de una cola para las horas de Xfut, entrenando con lo anterior a `origen`."""
    metodo = "perfil" if cola in PERFIL_SIMPLE else (metodo or METODO)
    if metodo == "perfil":
        return perfil(g, ent, cola, origen, Xfut.index)
    modo = metodo.split("_")[1] if "_" in metodo else "nivelado"
    base = lgbm(g, X, ent, cola, origen, Xfut, modo=modo)
    if metodo.startswith("lgbm"):
        return base
    return np.sqrt((base + 1) * (perfil(g, ent, cola, origen, Xfut.index) + 1)) - 1   # media geométrica


def cuadrar(P, total, por="dia"):
    """Escala las colas para que sumen `total` (pronóstico de Financiero + No Financiero): un factor por día
    (conserva la forma horaria de cada cola) o por hora. Las horas vacías de P no cuentan."""
    s, t = P.sum(axis=1, min_count=1), total.reindex(P.index)
    t = t.where(s.notna())
    if por == "dia":
        d = P.index.normalize()
        s, t = s.groupby(d).transform("sum"), t.groupby(d).transform("sum")
    return P.mul((t / s.where(s > 0)).fillna(1.0), axis=0)
