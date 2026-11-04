"""Pronóstico horario de TPH (Financiero y No Financiero) para los 365 días siguientes al último dato.

 - Financiero: LightGBM + tendencia de nivel (Theil-Sen, sin amortiguar; el backtest a 360 días la prefirió).
 - No Financiero: LightGBM sin tendencia (la pendiente queda contaminada por la caída de marzo de 2026).
 - Bandas 10-90%: cuantiles de los errores de backtest (origen móvil) por serie, tramo de horizonte y franja horaria.
 - Las horas 23:00-01:59 de las noches de batch quedan vacías (sin transacciones en línea) y no suman al total diario.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import modelo_tph as M
from features_tph import construir, SALIDA

TENDENCIA = {"fin": True, "nofin": False}
ORIGENES_BANDAS = ["2025-07-01", "2025-09-01", "2025-11-01", "2026-01-01", "2026-03-31", "2026-06-25"]
TRAMOS = [(1, 30), (31, 90), (91, 180), (181, 365)]
DIAS = 365


def franja(h):
    return np.where(pd.Index(h).isin([23, 0, 1, 2, 3, 4]), "noche", "dia")


def residuos_backtest(g, X):
    """log((real+1)/(pred+1)) por hora ok, con horizonte en días, serie y franja; y errores diarios.
    Además de cada serie, la suma de ambas («tot»): su banda sale de sus propios errores, no de sumar bandas."""
    hor, dia = [], []
    horas_ok = g.groupby(g.index.normalize()).estado.apply(lambda s: (s == "ok").sum())
    for o in ORIGENES_BANDAS:
        o = pd.Timestamp(o)
        h = min(DIAS, (g.index.max().normalize() - o).days + 1)
        P = {s: M.entrenar_predecir(g, X, s, o, h, usar_tendencia=t, phi=1.0)[0] for s, t in TENDENCIA.items()}
        P["tot"] = P["fin"] + P["nofin"]
        idx = P["fin"].index
        Y = {s: g.loc[idx, s] for s in TENDENCIA}
        Y["tot"] = Y["fin"] + Y["nofin"]
        for serie in list(TENDENCIA) + ["tot"]:
            y, p = Y[serie], P[serie]
            ok = (g.loc[idx, "estado"] == "ok") & y.notna()
            y, p = y[ok], p[ok]
            k = (y.index.normalize() - o).days + 1
            hor.append(pd.DataFrame({"serie": serie, "k": k, "franja": franja(y.index.hour), "r": np.log((y + 1) / (p + 1)).values}))
            dd = pd.DataFrame({"y": y, "p": p}).groupby(y.index.normalize()).sum()
            dd = dd[horas_ok.reindex(dd.index) >= 17]
            dia.append(pd.DataFrame({"serie": serie, "k": (dd.index - o).days + 1, "r": np.log(dd.y / dd.p).values}))
    return pd.concat(hor), pd.concat(dia)


def tramo(k):
    return np.select([(k >= a) & (k <= b) for a, b in TRAMOS], [f"{a}-{b}" for a, b in TRAMOS], default="181-365")


def cuantiles(res, por, lo=0.10, hi=0.90):
    res = res.assign(tramo=tramo(res.k.values))
    q = res.groupby(por + ["tramo"]).r.quantile([lo, hi]).unstack().rename(columns={lo: "q_lo", hi: "q_hi"})
    # las bandas no pueden achicarse al crecer el horizonte (los tramos largos tienen pocos orígenes)
    orden = [f"{a}-{b}" for a, b in TRAMOS]
    q = q.loc[sorted(q.index, key=lambda i: (i[:-1], orden.index(i[-1])))]
    q["q_lo"] = q.groupby(level=por).q_lo.cummin()
    q["q_hi"] = q.groupby(level=por).q_hi.cummax()
    return q


def main():
    g, X = M.cargar()
    ultimo = g.index.max().normalize()
    origen = ultimo + pd.Timedelta(days=1)
    horas = pd.date_range(origen, periods=DIAS * 24, freq="h")
    Xf = construir(horas)

    rh, rd = residuos_backtest(g, X)
    qh = cuantiles(rh, ["serie", "franja"])
    qd = cuantiles(rd, ["serie"])
    qh.to_csv(SALIDA / "bandas_tph_horarias.csv"); qd.to_csv(SALIDA / "bandas_tph_diarias.csv")

    out = pd.DataFrame(index=horas); out.index.name = "hora"
    k = (Xf.index.normalize() - origen).days.values + 1
    tr = tramo(k); fr = franja(Xf.h.values)
    for serie, tend in TENDENCIA.items():
        p, b = M.entrenar_predecir(g, X, serie, origen, DIAS, usar_tendencia=tend, phi=1.0, Xfut=Xf)
        p = p.mask(M.sin_linea(Xf))
        q = qh.loc[serie]
        lo = np.array([q.loc[(f, t), "q_lo"] for f, t in zip(fr, tr)]); hi = np.array([q.loc[(f, t), "q_hi"] for f, t in zip(fr, tr)])
        out[f"{serie}_p50"] = p.round(0)
        out[f"{serie}_p10"] = np.maximum(0, (p + 1) * np.exp(lo) - 1).round(0)
        out[f"{serie}_p90"] = ((p + 1) * np.exp(hi) - 1).round(0)
        print(f"{serie}: pendiente b={b:.5f}/día -> {100*(np.exp(365*b)-1):.1f}% anual" if tend else f"{serie}: sin tendencia")
    # total (Financiero + No Financiero): mediana = suma de medianas; banda con los errores de la suma
    p = out.fin_p50 + out.nofin_p50
    q = qh.loc["tot"]
    lo = np.array([q.loc[(f, t), "q_lo"] for f, t in zip(fr, tr)]); hi = np.array([q.loc[(f, t), "q_hi"] for f, t in zip(fr, tr)])
    out["tot_p50"] = p
    out["tot_p10"] = np.maximum(0, (p + 1) * np.exp(lo) - 1).round(0)
    out["tot_p90"] = ((p + 1) * np.exp(hi) - 1).round(0)
    out["es_feriado"] = Xf.es_feriado.values; out["es_laborable"] = Xf.es_laborable.values
    out["noche_batch"] = Xf.noche_batch.values
    out.to_csv(SALIDA / "pronostico_tph_horario.csv")

    # resumen diario
    f = out.groupby(out.index.normalize())
    dia = pd.DataFrame({"es_laborable": f.es_laborable.first(), "es_feriado": f.es_feriado.first()})
    kd = (dia.index - origen).days.values + 1; trd = tramo(kd)
    for serie in list(TENDENCIA) + ["tot"]:
        tot = f[f"{serie}_p50"].sum()
        dia[f"{serie}_total_p50"] = tot
        dia[f"{serie}_total_p10"] = (tot * np.exp([qd.loc[(serie, t), "q_lo"] for t in trd])).round(0)
        dia[f"{serie}_total_p90"] = (tot * np.exp([qd.loc[(serie, t), "q_hi"] for t in trd])).round(0)
        dia[f"{serie}_tph_pico"] = f[f"{serie}_p50"].max()
        dia[f"{serie}_hora_pico"] = out.groupby(out.index.normalize())[f"{serie}_p50"].apply(lambda s: s.idxmax().hour)
    dia.to_csv(SALIDA / "pronostico_tph_diario.csv", index_label="fecha")
    print(f"pronóstico {horas[0]} -> {horas[-1]} ({len(horas)} horas, {len(dia)} días)")
    return out, dia, qh, qd


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    out, dia, qh, qd = main()
    print("bandas horarias (factor sobre p50, 10%-90%):")
    print(np.exp(qh).round(2).to_string())
    print("bandas diarias:"); print(np.exp(qd).round(2).to_string())
