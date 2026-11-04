"""Pronóstico horario de TPH por cola para los 365 días siguientes al último dato.

 - Método: el de modelo_colas.METODO (media geométrica de LightGBM y perfil de 8 semanas); las colas residuales
   usan solo el perfil. Supone que el enrutamiento vigente se mantiene.
 - Cuadre: cada día se escala para que la suma de colas sea el pronóstico de Financiero + No Financiero de
   pronostico_tph.py (correr antes). En backtest el cuadre deja el error igual (8,8 % diario) y baja el sesgo
   de -5,5 % a -1,1 %; a cambio, las colas heredan los errores del total (origen 2026-07-25: 14,8 % vs 8,7 %).
 - Bandas 10-90 %: cuantiles de los errores de backtest con orígenes posteriores al último cambio de
   enrutamiento (2026-03-10). Solo hay horizontes de hasta 180 días: el tramo 181-365 usa la banda de 91-180
   ensanchada con lo que se ensancha la de No Financiero en pronostico_tph.py.
 - Las horas 23:00-01:59 de las noches de batch quedan vacías, igual que en TPH.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import modelo_colas as C
import modelo_tph as M
from modelo_tph import sin_linea
from features_tph import construir, SALIDA
from pronostico_tph import franja, tramo, cuantiles, TRAMOS, DIAS, TENDENCIA

ORIGENES_BANDAS = ["2026-03-31", "2026-04-15", "2026-05-15", "2026-06-25", "2026-07-25", "2026-08-25"]
ULTIMO_TRAMO, PENULTIMO = "181-365", "91-180"


def residuos_backtest(g, X, ent, colas):
    """Errores log por hora y por día de cada cola, con su horizonte k; y métricas por cola y tramo."""
    hor, dia = [], []
    gt, Xt = M.cargar()
    horas_ok = (g.estado == "ok").groupby(g.index.normalize()).sum()
    for o in map(pd.Timestamp, ORIGENES_BANDAS):
        te = (g.index >= o) & (g.estado == "ok")
        Xf = X[te]
        k = (Xf.index.normalize() - o).days.values + 1
        tot = sum(M.entrenar_predecir(gt, Xt, s, o, int(k.max()), usar_tendencia=t, phi=1.0)[0] for s, t in TENDENCIA.items())
        P = C.cuadrar(pd.DataFrame({q: C.predecir(g, X, ent, q, o, Xf) for q in colas}), tot)
        for q in colas:
            p, y = P[q], g.loc[Xf.index, q]
            hor.append(pd.DataFrame({"cola": q, "origen": o, "k": k, "franja": franja(Xf.index.hour), "y": y.values, "p": p.values}))
            dd = pd.DataFrame({"y": y, "p": p}).groupby(Xf.index.normalize()).sum()
            dd = dd[horas_ok.reindex(dd.index) >= 17]
            dia.append(pd.DataFrame({"cola": q, "origen": o, "k": (dd.index - o).days + 1, "y": dd.y.values, "p": dd.p.values}))
        print("backtest", o.date(), "listo", flush=True)
    hor, dia = pd.concat(hor), pd.concat(dia)
    hor["r"], dia["r"] = np.log((hor.y + 1) / (hor.p + 1)), np.log((dia.y + 1) / (dia.p + 1))
    # en las colas mínimas los días casi sin pronóstico (fines de semana en cero) dan razones sin sentido
    dia["r"] = dia.r.where(dia.p >= 0.05 * dia.groupby("cola").p.transform("median"))
    return hor, dia


def metricas(hor, dia):
    f = []
    for (q, t), x in hor[hor.k <= 90].assign(tramo=lambda z: tramo(z.k.values)).groupby(["cola", "tramo"]):
        d = dia[(dia.cola == q) & (tramo(dia.k.values) == t)]
        f.append(dict(cola=q, tramo=t, wape_hora=(x.y - x.p).abs().sum() / max(x.y.sum(), 1), wape_dia=(d.y - d.p).abs().sum() / max(d.y.sum(), 1),
                      sesgo=(x.p.sum() - x.y.sum()) / max(x.y.sum(), 1)))
    return pd.DataFrame(f)


def completar_ultimo_tramo(q, extra):
    """Agrega el tramo 181-365 (sin backtests en el régimen actual) a partir del de 91-180."""
    f = []
    for idx, r in q.iterrows():
        if idx[-1] == PENULTIMO:
            f.append((idx[:-1] + (ULTIMO_TRAMO,), r.q_lo + extra[0], r.q_hi + extra[1]))
    add = pd.DataFrame([x[1:] for x in f], index=pd.MultiIndex.from_tuples([x[0] for x in f], names=q.index.names), columns=["q_lo", "q_hi"])
    return pd.concat([q[q.index.get_level_values("tramo") != ULTIMO_TRAMO], add]).sort_index()


def main():
    g, X, ent, colas = C.cargar()
    origen = g.index.max().normalize() + pd.Timedelta(days=1)
    horas = pd.date_range(origen, periods=DIAS * 24, freq="h")
    Xf = construir(horas)

    rh, rd = residuos_backtest(g, X, ent, colas)
    metricas(rh, rd).round(4).to_csv(SALIDA / "metricas_colas.csv", index=False)
    bt_h = pd.read_csv(SALIDA / "bandas_tph_horarias.csv").set_index(["serie", "franja", "tramo"]).loc["nofin"]
    bt_d = pd.read_csv(SALIDA / "bandas_tph_diarias.csv").set_index(["serie", "tramo"]).loc["nofin"]
    extra_d = (min(0, bt_d.at[ULTIMO_TRAMO, "q_lo"] - bt_d.at[PENULTIMO, "q_lo"]), max(0, bt_d.at[ULTIMO_TRAMO, "q_hi"] - bt_d.at[PENULTIMO, "q_hi"]))
    extra_h = (min(0, bt_h.at[("dia", ULTIMO_TRAMO), "q_lo"] - bt_h.at[("dia", PENULTIMO), "q_lo"]), max(0, bt_h.at[("dia", ULTIMO_TRAMO), "q_hi"] - bt_h.at[("dia", PENULTIMO), "q_hi"]))
    qh = completar_ultimo_tramo(cuantiles(rh, ["cola", "franja"]), extra_h)
    qd = completar_ultimo_tramo(cuantiles(rd, ["cola"]), extra_d)
    qh.to_csv(SALIDA / "bandas_colas_horarias.csv"); qd.to_csv(SALIDA / "bandas_colas_diarias.csv")

    k = (Xf.index.normalize() - origen).days.values + 1
    tr, fr, vacia = tramo(k), franja(Xf.h.values), sin_linea(Xf)
    tph = pd.read_csv(SALIDA / "pronostico_tph_horario.csv", parse_dates=["hora"]).set_index("hora")
    P = pd.DataFrame({q: C.predecir(g, X, ent, q, origen, Xf).mask(vacia) for q in colas})
    sin_cuadrar = P.sum(axis=1).groupby(P.index.normalize()).sum()
    P = C.cuadrar(P, tph.fin_p50 + tph.nofin_p50)
    factor = P.sum(axis=1).groupby(P.index.normalize()).sum() / sin_cuadrar
    out = {}
    for q in colas:
        p = P[q]
        lo = np.array([qh.at[(q, f, t), "q_lo"] for f, t in zip(fr, tr)]); hi = np.array([qh.at[(q, f, t), "q_hi"] for f, t in zip(fr, tr)])
        out[f"{q}_p50"] = p.round(0)
        out[f"{q}_p10"] = np.maximum(0, (p + 1) * np.exp(lo) - 1).round(0)
        out[f"{q}_p90"] = ((p + 1) * np.exp(hi) - 1).round(0)
    out = pd.DataFrame(out, index=horas); out.index.name = "hora"
    out["es_feriado"], out["es_laborable"], out["noche_batch"] = Xf.es_feriado.values, Xf.es_laborable.values, Xf.noche_batch.values
    out.to_csv(SALIDA / "pronostico_colas_horario.csv")

    # resumen diario (formato largo)
    fecha = out.index.normalize()
    kd = (fecha.unique() - origen).days.values + 1; trd = tramo(kd)
    filas = []
    for q in colas:
        f = out.groupby(fecha)
        tot = f[f"{q}_p50"].sum()
        filas.append(pd.DataFrame({"fecha": tot.index, "cola": q, "total_p50": tot.values,
                                   "total_p10": (tot.values * np.exp([qd.at[(q, t), "q_lo"] for t in trd])).round(0),
                                   "total_p90": (tot.values * np.exp([qd.at[(q, t), "q_hi"] for t in trd])).round(0),
                                   "tph_pico": f[f"{q}_p50"].max().values, "tph_pico_p90": f[f"{q}_p90"].max().values,
                                   "hora_pico": f[f"{q}_p50"].apply(lambda s: s.idxmax().hour).values}))
    dia = pd.concat(filas)
    dia.to_csv(SALIDA / "pronostico_colas_diario.csv", index=False)

    # la suma diaria de colas debe coincidir con la pestaña de TPH (salvo redondeo)
    tphd = pd.read_csv(SALIDA / "pronostico_tph_diario.csv", parse_dates=["fecha"]).set_index("fecha")
    razon = dia.groupby("fecha").total_p50.sum() / (tphd.fin_total_p50 + tphd.nofin_total_p50)
    assert (razon - 1).abs().max() < 1e-4, f"la suma de colas no cuadra con TPH (razón {razon.min():.4f}-{razon.max():.4f})"
    print(f"factor de cuadre aplicado (mediana por tramo): {factor.groupby(tramo(kd)).median().reindex([f'{a}-{b}' for a, b in TRAMOS]).round(3).to_dict()}")
    print(f"pronóstico {horas[0]} -> {horas[-1]} | {len(colas)} colas")
    return out, dia, qh, qd


if __name__ == "__main__":
    pd.set_option("display.width", 220, "display.max_rows", 200)
    out, dia, qh, qd = main()
    print("bandas diarias (factor sobre p50, 10 %-90 %):")
    print(np.exp(qd).round(2).unstack("tramo").to_string())
    print(pd.read_csv(SALIDA / "metricas_colas.csv").pivot(index="cola", columns="tramo").round(3).to_string())
