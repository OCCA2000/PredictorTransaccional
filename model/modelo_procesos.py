"""Duración de los procesos críticos por subproceso: baselines de nivel reciente vs LightGBM."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "data/processed"
QUIEBRE_CFPD = pd.Timestamp("2026-06-02")   # primer día de negocio tras la última corrida acumulada (1-jun-2026)
# El orden fija los índices que usa el dashboard: los subprocesos nuevos (oct-2026) van al final.
SUBS = ["cfpd0111cm", "depd0800cm_Lote1", "depd0800cm_Lote2", "depd0800cm_Lote3", "depd0800cm_Lote4", "lond0501cm", "bnccrn2dcm", "bncd0801"]
ATIPICO_X_MEDIANA, ATIPICO_MIN = 10, 60   # duración > 10 veces la mediana del subproceso y > 60 min: registro erróneo


def cargar():
    o = pd.read_csv(SALIDA / "procesos_limpio.csv", parse_dates=["dia_negocio", "inicio", "fin"])
    # Tras el quiebre, los días en que solo corre cfpd0111c2 (lunes / primer día) cuentan como cfpd0111cm.
    esc2 = o.suproceso.str.startswith("cfpd0111c2") & (o.dia_negocio >= QUIEBRE_CFPD)
    sin_cm = ~o.dia_negocio.isin(o[o.suproceso == "cfpd0111cm"].dia_negocio)
    o.loc[esc2 & sin_cm, "suproceso"] = "cfpd0111cm"
    o = o[o.suproceso.isin(SUBS) & o.dur_calc_s.notna() & (o.tipo_dia != "no laborable")]
    o = o[o.ejec_n == o.ejec_total].copy()          # última ejecución de cada día de negocio
    o["dur"] = o.dur_calc_s / 60
    # p. ej. una ejecución de bnccrn2dcm de 23 h cuando dura 20 s
    med = o.groupby("suproceso").dur.transform("median")
    o = o[~((o.dur > ATIPICO_X_MEDIANA * med) & (o.dur > ATIPICO_MIN))].copy()
    t = pd.read_csv(SALIDA / "calidad_dias_tph.csv", parse_dates=["fecha"]).set_index("fecha")
    o["vol"] = o.dia_negocio.map((t.fin + t.nofin) / 1e6)
    return o


def X_dia(f):
    """Variables de calendario por día de negocio (f: DataFrame con dia_negocio y tipo_dia)."""
    x = pd.DataFrame(index=f.index)
    x["primer_dia"] = (f.tipo_dia == "primer dia").astype(int)
    x["dsem"] = f.dia_negocio.dt.dayofweek
    x["dia_mes"] = f.dia_negocio.dt.day
    x["fin_mes"] = (f.dia_negocio.dt.days_in_month - f.dia_negocio.dt.day <= 2).astype(int)
    return x


def pred_mediana_global(tr, te):
    m = tr.groupby("primer_dia").dur.median()
    return te.primer_dia.map(m).fillna(tr.dur.median())


def pred_nivel_reciente(tr, te, semanas=8):
    """Nivel = mediana de las últimas `semanas` semanas por (primer_dia); si faltan datos, histórico."""
    corte = tr.dia_negocio.max() - pd.Timedelta(weeks=semanas)
    rec = tr[tr.dia_negocio > corte]
    m = rec.groupby("primer_dia").dur.median()
    mg = tr.groupby("primer_dia").dur.median()
    return te.primer_dia.map(m).fillna(te.primer_dia.map(mg)).fillna(tr.dur.median())


def pred_nivel_x_mult(tr, te, semanas=8):
    """Nivel reciente de días normales x multiplicador histórico de 'primer día' (más robusto con pocos primer día recientes)."""
    corte = tr.dia_negocio.max() - pd.Timedelta(weeks=semanas)
    normal = tr[(tr.primer_dia == 0) & (tr.dia_negocio > corte)].dur.median()
    h = tr.groupby("primer_dia").dur.median()
    mult = (h.get(1, np.nan) / h.get(0, np.nan)) if 1 in h.index and 0 in h.index else 1.0
    mult = 1.0 if not np.isfinite(mult) else mult
    return pd.Series(np.where(te.primer_dia == 1, normal * mult, normal), index=te.index)


def pred_lgbm(tr, te, usar_vol=False):
    cols = ["primer_dia", "dsem", "dia_mes", "fin_mes"] + (["vol"] if usar_vol else [])
    mod = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.05, num_leaves=6, min_child_samples=15,
                            subsample=0.8, subsample_freq=1, reg_lambda=5.0, verbose=-1)
    mod.fit(tr[cols], np.log(tr.dur))
    return pd.Series(np.exp(mod.predict(te[cols])), index=te.index)
