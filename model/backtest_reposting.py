"""Backtest de origen móvil del modelo de reposting: compara los métodos de modelo_reposting.py por tabla y variable.
Orígenes: fin de cada mes de sep-2025 a jul-2026, horizonte hasta 180 días (sin pasar del último dato real).
Los días posteriores a un cambio de nivel que el origen no conocía no se evalúan en esa tabla (ningún método puede verlo).
Error = WAPE diario por tramo (1-30, 31-90, 91-180 días). Ganador por tabla y variable: menor error a 1-90 días.
Salidas: data/processed/backtest_reposting.csv (predicciones), backtest_reposting_resumen.csv, reposting_ganadores.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import modelo_reposting as M
from limpieza_reposting import FIN_DIA_NEGOCIO

ORIGENES = pd.date_range("2025-09-30", "2026-07-31", freq="ME")
HORIZONTE = 180


def tramo(k):
    return "1-30" if k <= 30 else ("31-90" if k <= 90 else "91-180")


def correr():
    l, cal = M.cargar()
    act = M.activas(l)
    dias = sorted(l.dia_negocio.unique())
    F = M.calendario_dias(cal, pd.date_range(dias[0], FIN_DIA_NEGOCIO)[[cal.laborable.get(x, False) for x in pd.date_range(dias[0], FIN_DIA_NEGOCIO)]])
    filas = []
    for o in ORIGENES:
        fut = F[(F.index > o) & (F.index <= min(o + pd.Timedelta(days=HORIZONTE), FIN_DIA_NEGOCIO))]
        hist = F[F.index <= o]
        for t in act:
            for v in M.VARIABLES:
                s = l[l.tabla == t].set_index("dia_negocio")[v].reindex(F.index)
                real = s.reindex(fut.index)
                for m in M.METODOS:
                    p = M.predecir(s, t, o + pd.Timedelta(days=1), hist, fut, m)
                    filas.append(pd.DataFrame({"origen": o, "tabla": t, "variable": v, "metodo": m, "fecha": fut.index,
                                               "k": (fut.index - o).days, "real": real.values, "pred": p.values}))
    r = pd.concat(filas)
    # días posteriores a un escalón que el origen no conocía: no se evalúan
    for t, es in M.ESCALONES.items():
        for e in map(pd.Timestamp, es):
            r = r[~((r.tabla == t) & (r.origen < e) & (r.fecha >= e))]
    return r.dropna(subset=["real"])


def resumir(r):
    r = r.assign(tramo=r.k.map(tramo), ae=(r.pred - r.real).abs())
    g = r.groupby(["tabla", "variable", "metodo", "tramo"]).apply(
        lambda z: pd.Series({"wape": z.ae.sum() / z.real.abs().sum(), "sesgo": z.pred.sum() / z.real.sum() - 1, "n": len(z)})).reset_index()
    a90 = r[r.k <= 90].groupby(["tabla", "variable", "metodo"]).apply(lambda z: z.ae.sum() / z.real.abs().sum()).rename("wape_1_90").reset_index()
    gan = a90.loc[a90.groupby(["tabla", "variable"]).wape_1_90.idxmin()].reset_index(drop=True)
    return g, a90, gan


def main():
    r = correr()
    r.to_csv(M.SALIDA / "backtest_reposting.csv", index=False)
    g, a90, gan = resumir(r)
    g.to_csv(M.SALIDA / "backtest_reposting_resumen.csv", index=False)
    gan.to_csv(M.SALIDA / "reposting_ganadores.csv", index=False)
    pd.set_option("display.width", 200)
    print(a90.pivot_table(index=["variable", "tabla"], columns="metodo", values="wape_1_90").round(3).to_string())
    print("\nGanadores:\n", gan.round(3).to_string())
    # promedio ponderado por volumen (todas las tablas) por método
    rr = r[r.k <= 90].assign(ae=lambda z: (z.pred - z.real).abs())
    print("\nError 1-90 días, todas las tablas juntas:")
    print(rr.groupby(["variable", "metodo"]).apply(lambda z: z.ae.sum() / z.real.sum()).round(3).unstack().to_string())


if __name__ == "__main__":
    main()
