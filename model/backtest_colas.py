"""Backtest de origen móvil del modelo por cola: compara métodos, con y sin cuadre contra el total de TPH.

Orígenes posteriores al último cambio de enrutamiento (2026-03-10), horizonte de hasta 90 días. Se evalúa contra
el dato real de todas las horas `ok`, incluidos los días de desvío (que no se usan para entrenar).
Salida: data/processed/backtest_colas_resumen.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import modelo_colas as C
import modelo_tph as M

ORIGENES = ["2026-03-31", "2026-04-15", "2026-05-15", "2026-06-25", "2026-07-25", "2026-08-25"]
METODOS = ["perfil", "lgbm_nivelado", "mezcla_variable", "mezcla_nivelado", "mezcla_reciente"]
H = 90


def metricas(y, p):
    m = y.notna() & p.notna()
    y, p = y[m], p[m]
    if y.sum() <= 0:
        return np.nan, np.nan, np.nan
    yd, pd_ = y.groupby(y.index.normalize()).sum(), p.groupby(p.index.normalize()).sum()
    return float((y - p).abs().sum() / y.sum()), float((yd - pd_).abs().sum() / yd.sum()), float((p.sum() - y.sum()) / y.sum())


def main():
    g, X, ent, colas = C.cargar()
    gt, Xt = M.cargar()
    filas = []
    for o in map(pd.Timestamp, ORIGENES):
        te = (g.index >= o) & (g.index < o + pd.Timedelta(days=H)) & (g.estado == "ok")
        Xf = X[te]
        tot = sum(M.entrenar_predecir(gt, Xt, s, o, H, usar_tendencia=t, phi=1.0)[0] for s, t in (("fin", True), ("nofin", False))).reindex(Xf.index)
        for met in METODOS:
            P = pd.DataFrame({q: C.predecir(g, X, ent, q, o, Xf, met) for q in colas})
            for nom, PP in ((met, P), (met + "+cuadre_hora", C.cuadrar(P, tot, "hora")), (met + "+cuadre_dia", C.cuadrar(P, tot, "dia"))):
                k = (PP.index.normalize() - o).days + 1
                for tramo, mk in (("1-30", k <= 30), ("31-90", k > 30), ("1-90", k > 0)):
                    if not mk.any():
                        continue
                    for q in colas:
                        wh, wd, sg = metricas(g.loc[PP.index[mk], q], PP.loc[mk, q])
                        filas.append(dict(origen=o.date(), modelo=nom, tramo=tramo, cola=q, wape_hora=wh, wape_dia=wd, sesgo=sg, volumen=g.loc[PP.index[mk], q].sum()))
                    wh, wd, sg = metricas(g.loc[PP.index[mk], colas].sum(axis=1), PP[mk].sum(axis=1))
                    filas.append(dict(origen=o.date(), modelo=nom, tramo=tramo, cola="_SUMA", wape_hora=wh, wape_dia=wd, sesgo=sg, volumen=np.nan))
            print(o.date(), met, "listo", flush=True)
    r = pd.DataFrame(filas)
    r.to_csv(C.SALIDA / "backtest_colas_resumen.csv", index=False)
    return r


def pond(z):
    z = z.dropna(subset=["wape_dia"])
    return pd.Series({"wape_dia": np.average(z.wape_dia, weights=z.volumen), "wape_hora": np.average(z.wape_hora, weights=z.volumen),
                      "sesgo": np.average(z.sesgo, weights=z.volumen), "peor_cola_dia": z.wape_dia.max()})


if __name__ == "__main__":
    pd.set_option("display.width", 220, "display.max_rows", 200)
    r = main()
    c = r[(r.cola != "_SUMA") & ~r.cola.isin(C.PERFIL_SIMPLE)]
    print("\nPonderado por volumen, todas las colas con modelo, por tramo (media de orígenes):")
    print(c.groupby(["tramo", "modelo", "origen"]).apply(pond).groupby(["tramo", "modelo"]).mean().round(3).to_string())
    print("\nWAPE diario 1-90 por origen:")
    print(c[c.tramo == "1-90"].groupby(["modelo", "origen"]).apply(pond).wape_dia.unstack().round(3).to_string())
    print("\nWAPE diario 1-90 por cola (media de orígenes), con cuadre por día:")
    print(r[(r.tramo == "1-90") & r.modelo.str.endswith("cuadre_dia")].groupby(["cola", "modelo"]).wape_dia.mean().unstack().round(3).to_string())
