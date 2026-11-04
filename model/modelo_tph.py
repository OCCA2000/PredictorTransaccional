"""Modelo de TPH: LightGBM sobre el log del conteo, con tendencia de nivel opcional y amortiguada."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import theilslopes

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features_tph import construir, FEATURES, SALIDA

PARAMS = dict(n_estimators=500, learning_rate=0.05, num_leaves=31, min_child_samples=30,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.9, reg_lambda=1.0, verbose=-1)
VENTANA_PENDIENTE_DIAS = 365
# En las noches de batch estas horas casi nunca traen dato (23h: 21 % de las noches, 00h: 1 %, 01h: 34 %):
# el pronóstico las deja vacías para que los totales diarios sean comparables con los reales.
HORAS_SIN_LINEA = [23, 0, 1]
COLAS_DESVIO, RAZON_DESVIO = ["OMN", "CI3"], 1.4


def sin_linea(X):
    """Horas de la ventana del batch sin transacciones en línea (X: variables de `construir`)."""
    return ((X.noche_batch == 1) & X.h.isin(HORAS_SIN_LINEA)).values


def dias_desvio():
    """Días con desvío de tráfico hacia OMN/CI3 (colas_excluidas.csv, de limpieza_colas.py). Esos días el total
    de No Financiero sale inflado (mediana +9 %, hasta +48 %): son reales pero no se usan para entrenarlo."""
    f = SALIDA / "colas_excluidas.csv"
    if not f.exists():
        return pd.DatetimeIndex([])
    ex = pd.read_csv(f, parse_dates=["hora"])
    ex = ex[(ex.motivo == "desvio") & ex.cola.isin(COLAS_DESVIO) & (ex.ratio > RAZON_DESVIO)]
    return pd.DatetimeIndex(sorted(set(ex.hora.dt.normalize())))


def cargar():
    g = pd.read_csv(SALIDA / "tph_limpio.csv", parse_dates=["hora"]).set_index("hora")
    g["excluir_nofin"] = g.index.normalize().isin(dias_desvio())
    return g, construir(g.index)


def pendiente_log_diaria(g, serie, hasta):
    """Pendiente robusta (Theil-Sen) del log del total diario de días laborables completos."""
    d = g[(g.index < hasta) & (g.index >= hasta - pd.Timedelta(days=VENTANA_PENDIENTE_DIAS))]
    d = d[(d.estado == "ok") & d.es_laborable & ~d.es_feriado]
    tot = d.groupby(d.index.normalize())[serie].agg(["sum", "size"])
    tot = tot[tot["size"] >= 15]
    t = (tot.index - tot.index[0]).days.values
    return float(theilslopes(np.log(tot["sum"].values), t)[0])


def factor_nivel(b, k, phi):
    """exp(b * suma amortiguada) para k días después del último día de entrenamiento."""
    k = np.asarray(k, float)
    acum = k if phi >= 1 else (1 - phi ** k) / (1 - phi)
    return np.exp(b * acum)


def entrenar_predecir(g, X, serie, origen, dias_futuros, usar_tendencia=False, phi=0.995, cols=None, Xfut=None):
    """Entrena con las horas 'ok' anteriores a `origen` y predice `Xfut` (o las horas reales del
    horizonte si Xfut=None). Devuelve (pred, b)."""
    cols = list(cols or FEATURES)
    tr = (g.index < origen) & (g.estado == "ok") & g[serie].notna()
    if "excluir_" + serie in g:
        tr &= ~g["excluir_" + serie]
    Xtr, ytr = X[tr], g.loc[tr, serie]
    b = 0.0
    k_tr = (origen - Xtr.index.normalize()).days.values  # días antes del origen
    if usar_tendencia:
        b = pendiente_log_diaria(g, serie, origen)
        ytr = ytr * np.exp(b * k_tr)  # nivela todo al nivel del origen (sube lo antiguo; antes tenía el signo invertido)
        cols = [c for c in cols if c != "t_dias"]
    m = lgb.LGBMRegressor(**PARAMS).fit(Xtr[cols], np.log1p(ytr))
    if Xfut is None:
        te = (g.index >= origen) & (g.index < origen + pd.Timedelta(days=dias_futuros))
        Xfut = X[te]
    p = np.expm1(m.predict(Xfut[cols]))
    if usar_tendencia:
        k = (Xfut.index.normalize() - origen).days.values + 1
        p = p * factor_nivel(b, k, phi)
    return pd.Series(p, index=Xfut.index), b
