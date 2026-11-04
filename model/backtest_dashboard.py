"""Predicciones de backtest para el dashboard: para cada día pasado, lo que el modelo habría
pronosticado con 1, 7, 30 y 90 días de anticipación (origen semanal, el más cercano al horizonte).

Salidas:
  data/processed/backtest_tph_dashboard.csv      hora, serie, h, pred, origen
  data/processed/backtest_procesos_dashboard.csv dia_negocio, suproceso, h, dur_pred, ini_pred, origen
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import modelo_tph as M
import modelo_procesos as P
import pronostico_procesos as PP

SALIDA = M.SALIDA
HORIZONTES = [1, 7, 30, 90]
TENDENCIA = {"fin": True, "nofin": False}


def tph():
    g, X = M.cargar()
    ultimo = g.index.max().normalize()
    origenes = pd.date_range("2025-07-07", ultimo, freq="7D")
    pred = {}   # (serie, origen) -> Series de 90 días
    for s, tend in TENDENCIA.items():
        for o in origenes:
            h = min(90, (ultimo - o).days + 1)
            p, _ = M.entrenar_predecir(g, X, s, o, h, usar_tendencia=tend, phi=1.0)
            pred[(s, o)] = p
        print("tph", s, "listo", flush=True)
    filas = []
    dias = pd.date_range(origenes[0], ultimo)
    for s in TENDENCIA:
        for d in dias:
            for h in HORIZONTES:
                # origen semanal cuyo horizonte k=(d-o)+1 queda más cerca de h
                cand = [(abs((d - o).days + 1 - h), o) for o in origenes if 1 <= (d - o).days + 1 <= 90]
                if not cand: continue
                err, o = min(cand)
                if err > 3: continue
                p = pred[(s, o)]
                x = p[(p.index >= d) & (p.index < d + pd.Timedelta(days=1))]
                filas.append(pd.DataFrame({"hora": x.index, "serie": s, "h": h, "pred": x.round(0).values, "origen": o}))
    out = pd.concat(filas)
    out.to_csv(SALIDA / "backtest_tph_dashboard.csv", index=False)
    print("tph filas", len(out))


def procesos():
    o = P.cargar()
    W = PP.pivot(o)
    pdia = o.groupby("dia_negocio").apply(lambda x: int((x.tipo_dia == "primer dia").any()))
    ultimo = W.index.max()
    origenes = pd.date_range("2025-11-24", ultimo, freq="7D")
    pred = {og: PP.predecir(W, pdia, og, W.index[(W.index >= og) & (W.index < og + pd.Timedelta(days=90))]) for og in origenes}
    filas = []
    for d in W.index:
        for h in HORIZONTES:
            cand = [(abs((d - og).days + 1 - h), og) for og in origenes if 1 <= (d - og).days + 1 <= 90 and d in pred[og].index]
            if not cand: continue
            err, og = min(cand)
            if err > 3: continue
            p = pred[og].loc[d]
            for s in PP.CADENA:
                filas.append(dict(dia_negocio=d, suproceso=s, h=h, dur_pred=round(p[("dur", s)], 1), ini_pred=round(p[("ini", s)], 1), origen=og))
    out = pd.DataFrame(filas)
    out.to_csv(SALIDA / "backtest_procesos_dashboard.csv", index=False)
    print("procesos filas", len(out))


if __name__ == "__main__":
    procesos()
    tph()
