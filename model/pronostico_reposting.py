"""Pronóstico de reposting a 365 días (días de negocio hábiles) con el método ganador de backtest_reposting.py.
Bandas 10-90 %: cuantiles del error (log) del método ganador en el backtest, por tabla, variable y tramo de horizonte
(181-365 usa la de 91-180). El tiempo se calcula como TRX / velocidad; su banda sale del error del tiempo así derivado.
Tablas con un cambio de nivel en los últimos 60 días hábiles: banda ×1,5 (el backtest no puede validar el nivel nuevo).
Tablas inactivas: pronóstico 0.
La noche entera (TRX total de las tablas activas y duración = la de la tabla más lenta) lleva su propia banda del backtest.
Salidas: data/processed/pronostico_reposting.csv, pronostico_reposting_noche.csv, bandas_reposting.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import modelo_reposting as M
from limpieza_reposting import FIN_DIA_NEGOCIO

HORIZONTE_DIAS = 365
FACTOR_QUIEBRE = 1.5


def bandas(gan):
    r = pd.read_csv(M.SALIDA / "backtest_reposting.csv", parse_dates=["origen", "fecha"])
    r["tramo"] = np.where(r.k <= 30, "1-30", np.where(r.k <= 90, "31-90", "91-180"))
    sel = r.merge(gan[["tabla", "variable", "metodo"]], on=["tabla", "variable", "metodo"])
    w = sel.pivot_table(index=["origen", "tabla", "fecha", "tramo"], columns="variable", values=["pred", "real"])
    t = pd.DataFrame({"variable": "tmin", "pred": w[("pred", "trx")] / w[("pred", "ratio")], "real": w[("real", "trx")] / w[("real", "ratio")]}).reset_index()
    e = pd.concat([sel[["tabla", "variable", "tramo", "pred", "real"]], t[["tabla", "variable", "tramo", "pred", "real"]]])
    # la noche entera: volumen total de las tablas activas y duración de la noche (la de la tabla más lenta)
    wt = w.copy()
    wt.columns = [f"{x}_{y}" for x, y in wt.columns]
    wt = wt.dropna().reset_index()
    wt = wt[wt.groupby(["origen", "fecha"]).tabla.transform("size") == wt.tabla.nunique()]
    wt["pt"], wt["rt"] = wt.pred_trx / wt.pred_ratio, wt.real_trx / wt.real_ratio
    g = wt.groupby(["origen", "fecha", "tramo"])
    noche = pd.concat([pd.DataFrame({"variable": "trx", "pred": g.pred_trx.sum(), "real": g.real_trx.sum()}),
                       pd.DataFrame({"variable": "tmin", "pred": g.pt.max(), "real": g.rt.max()})]).reset_index()
    noche["tabla"] = "NOCHE"
    e = pd.concat([e, noche[["tabla", "variable", "tramo", "pred", "real"]]])
    e = e[(e.pred > 0) & (e.real > 0)].replace([np.inf, -np.inf], np.nan).dropna()
    e["err"] = np.log(e.real / e.pred)
    q = e.groupby(["tabla", "variable", "tramo"]).err.quantile([0.1, 0.9]).unstack()
    q.columns = ["q_lo", "q_hi"]
    q = q.reset_index()
    extra = q[q.tramo == "91-180"].assign(tramo="181-365")
    return pd.concat([q, extra])


def noche(p, q, act):
    """Por día: TRX total de las tablas activas y duración de la noche (tabla más lenta), con su banda del backtest."""
    a = p[p.tabla.isin(act)].reset_index(drop=True)
    idx = a.groupby("fecha").tmin_p50.idxmax()
    n = pd.DataFrame({"fecha": a.fecha.loc[idx].values, "tramo": a.tramo.loc[idx].values, "tabla_lenta": a.tabla.loc[idx].values,
                      "trx_p50": a.groupby("fecha").trx_p50.sum().values, "tmin_p50": a.tmin_p50.loc[idx].values})
    for v in ("trx", "tmin"):
        b = q[(q.tabla == "NOCHE") & (q.variable == v)].set_index("tramo")
        n[f"{v}_p10"] = n[f"{v}_p50"] * np.exp(np.minimum(n.tramo.map(b.q_lo), 0))
        n[f"{v}_p90"] = n[f"{v}_p50"] * np.exp(np.maximum(n.tramo.map(b.q_hi), 0))
    return n


def main():
    l, cal = M.cargar()
    act = M.activas(l)
    gan = pd.read_csv(M.SALIDA / "reposting_ganadores.csv")
    q = bandas(gan)
    q.to_csv(M.SALIDA / "bandas_reposting.csv", index=False)
    todos = pd.date_range(l.dia_negocio.min(), FIN_DIA_NEGOCIO + pd.Timedelta(days=HORIZONTE_DIAS))
    habiles = todos[[bool(cal.laborable.get(x, False)) for x in todos]]
    F = M.calendario_dias(cal, habiles)
    hist, fut = F[F.index <= FIN_DIA_NEGOCIO], F[F.index > FIN_DIA_NEGOCIO]
    k = (fut.index - FIN_DIA_NEGOCIO).days
    tramo = np.where(k <= 30, "1-30", np.where(k <= 90, "31-90", np.where(k <= 180, "91-180", "181-365")))
    origen = FIN_DIA_NEGOCIO + pd.Timedelta(days=1)
    filas = []
    for t in M.TABLAS:
        out = pd.DataFrame({"fecha": fut.index, "tabla": t, "dow": fut.index.dayofweek, "tramo": tramo})
        if t not in act:
            for v in ("trx", "ratio", "tmin"):
                out[[f"{v}_p50", f"{v}_p10", f"{v}_p90"]] = 0.0
            out["metodo_trx"] = out["metodo_ratio"] = "inactiva"
            filas.append(out)
            continue
        quiebre = any(pd.Timestamp(e) > FIN_DIA_NEGOCIO - pd.Timedelta(days=90) for e in M.ESCALONES.get(t, []))
        for v in ("trx", "ratio"):
            m = gan[(gan.tabla == t) & (gan.variable == v)].metodo.iloc[0]
            s = l[l.tabla == t].set_index("dia_negocio")[v].reindex(F.index)
            out[f"{v}_p50"] = M.predecir(s, t, origen, hist, fut, m).values
            out[f"metodo_{v}"] = m
        out["tmin_p50"] = np.where(out.ratio_p50 > 0, out.trx_p50 / out.ratio_p50, np.nan)
        for v in ("trx", "ratio", "tmin"):
            b = q[(q.tabla == t) & (q.variable == v)].set_index("tramo")
            lo, hi = out.tramo.map(b.q_lo).values, out.tramo.map(b.q_hi).values
            if quiebre:
                lo, hi = lo * FACTOR_QUIEBRE, hi * FACTOR_QUIEBRE
            out[f"{v}_p10"] = out[f"{v}_p50"] * np.exp(np.minimum(lo, 0))
            out[f"{v}_p90"] = out[f"{v}_p50"] * np.exp(np.maximum(hi, 0))
        filas.append(out)
    p = pd.concat(filas).sort_values(["fecha", "tabla"])
    p.to_csv(M.SALIDA / "pronostico_reposting.csv", index=False)
    noche(p, q, act).to_csv(M.SALIDA / "pronostico_reposting_noche.csv", index=False)
    print(f"pronóstico {p.fecha.min().date()} -> {p.fecha.max().date()} | {p.fecha.nunique()} días hábiles | activas: {act}")
    print(p[p.tabla.isin(act)].groupby("tabla")[["trx_p50", "ratio_p50", "tmin_p50"]].mean().round(1).to_string())


if __name__ == "__main__":
    main()
