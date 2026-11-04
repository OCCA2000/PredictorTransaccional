"""Backtest de origen móvil para TPH horarias: baseline de perfil vs LightGBM."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features_tph import construir, FEATURES, SALIDA

PARAMS = dict(n_estimators=500, learning_rate=0.05, num_leaves=31, min_child_samples=30,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.9, reg_lambda=1.0, verbose=-1)
ORIGENES = ["2025-09-30", "2026-03-31", "2026-06-25"]   # primer día a pronosticar
HORIZONTE = 90


def cargar():
    g = pd.read_csv(SALIDA / "tph_limpio.csv", parse_dates=["hora"]).set_index("hora")
    X = construir(g.index)
    return g, X


def baseline(Xtr, ytr, Xte):
    """Mediana de las últimas 8 semanas por (tipo de día, hora): feriado/laborable/sábado/domingo."""
    def tipo(X): return np.where(X.es_feriado == 1, 3, np.where(X.dow == 6, 2, np.where(X.dow == 5, 1, 0)))
    tr = pd.DataFrame({"tipo": tipo(Xtr), "h": Xtr.h.values, "y": ytr.values}, index=Xtr.index)
    tr = tr[tr.index >= tr.index.max() - pd.Timedelta(weeks=8)]
    med = tr.groupby(["tipo", "h"]).y.median()
    k = pd.MultiIndex.from_arrays([tipo(Xte), Xte.h.values])
    p = pd.Series(med.reindex(k).values, index=Xte.index)
    return p.fillna(tr.y.median())


def lgbm(Xtr, ytr, Xte, cols):
    m = lgb.LGBMRegressor(**PARAMS).fit(Xtr[cols], np.log1p(ytr))
    return pd.Series(np.expm1(m.predict(Xte[cols])), index=Xte.index)


def metricas(y, p):
    e = (p - y)
    horario = e.abs().sum() / y.sum()
    d = pd.DataFrame({"y": y, "p": p}).groupby(y.index.normalize()).sum()
    diario = (d.p - d.y).abs().sum() / d.y.sum()
    return dict(wape_hora=horario, wape_dia=diario, sesgo=e.sum() / y.sum())


def main():
    g, X = cargar()
    cols_reg = FEATURES
    cols_sin = [c for c in FEATURES if c != "regimen"]
    filas = []
    for serie in ("fin", "nofin"):
        for o in ORIGENES:
            o = pd.Timestamp(o)
            tr = (g.index < o) & (g.estado == "ok") & g[serie].notna()
            te = (g.index >= o) & (g.index < o + pd.Timedelta(days=HORIZONTE)) & (g.estado == "ok") & g[serie].notna()
            if te.sum() == 0: continue
            Xtr, ytr, Xte, yte = X[tr], g.loc[tr, serie], X[te], g.loc[te, serie]
            for nombre, p in (("perfil_8sem", baseline(Xtr, ytr, Xte)),
                              ("lgbm_sin_regimen", lgbm(Xtr, ytr, Xte, cols_sin)),
                              ("lgbm_con_regimen", lgbm(Xtr, ytr, Xte, cols_reg))):
                filas.append(dict(serie=serie, origen=o.date(), modelo=nombre, **metricas(yte, p)))
    r = pd.DataFrame(filas)
    r.to_csv(SALIDA / "backtest_tph_resumen.csv", index=False)
    return r


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    r = main()
    print(r.round(3).to_string(index=False))
    print(r.groupby(["serie", "modelo"])[["wape_hora", "wape_dia", "sesgo"]].mean().round(3).to_string())
