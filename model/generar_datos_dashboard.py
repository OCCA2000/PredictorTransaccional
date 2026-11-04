"""Empaqueta histórico, pronóstico, bandas y backtest en data/processed/dashboard_data.json."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import modelo_procesos as P

SALIDA = P.SALIDA
ESTADO = {"ok": 0, "batch": 1, "sin_dato": 2, "caida": 3}
HORIZ = [1, 7, 30, 90]
FC_HORAS_DIAS = 90   # detalle horario pronosticado de la pestaña de colas (el real va completo)


def lista(a, nd=0):
    a = np.asarray(a, float)
    return [None if not np.isfinite(x) else (int(round(x)) if nd == 0 else round(float(x), nd)) for x in a]


def tph():
    g = pd.read_csv(SALIDA / "tph_limpio.csv", parse_dates=["hora"]).set_index("hora")
    fc = pd.read_csv(SALIDA / "pronostico_tph_horario.csv", parse_dates=["hora"]).set_index("hora")
    cal = pd.read_csv(SALIDA / "calendario_laboral.csv", parse_dates=["fecha"]).set_index("fecha")
    dias = pd.date_range(g.index.min().normalize(), fc.index.max().normalize())
    lab = "".join("1" if cal.at[d, "laborable"] else "0" for d in dias)
    fer = "".join("1" if (not cal.at[d, "laborable"] and d.dayofweek < 5) else "0" for d in dias)
    # tipo de día del calendario operativo, una letra por día (ver limpieza_procesos.tipo_operativo)
    COD = {"DIA NORMAL": "N", "PRIMER DIA LABORAL": "P", "PRIMER BATCH": "B", "FDS NORMAL": "S", "FERIADO": "F"}
    op = "".join(COD[cal.at[d, "tipo_op"]] for d in dias)
    bd = pd.read_csv(SALIDA / "bandas_tph_diarias.csv")
    bandas = {s: {r.tramo: [round(float(np.exp(r.q_lo)), 3), round(float(np.exp(r.q_hi)), 3)] for r in x.itertuples()} for s, x in bd.groupby("serie")}
    out = {"t0": str(g.index.min()), "fc_t0": str(fc.index.min()), "last_real": str(g.index.max()),
           "cal_d0": str(dias[0].date()), "lab": lab, "fer": fer, "op": op, "dia_banda": bandas, "series": {}}
    bt = pd.read_csv(SALIDA / "backtest_tph_dashboard.csv", parse_dates=["hora", "origen"])
    bt0 = bt.hora.min().normalize(); bt1 = g.index.max()
    idx = pd.date_range(bt0, bt1, freq="h")
    out["back_t0"] = str(bt0)
    met = {}
    g["tot"] = g.fin + g.nofin                     # total: vacío si falta cualquiera de las dos
    serie_bt = lambda s, h: bt[(bt.serie == s) & (bt.h == h)].drop_duplicates("hora").set_index("hora").pred.reindex(idx)
    for s in ("fin", "nofin", "tot"):
        d = {"hist": lista(g[s]), "estado": [ESTADO[e] for e in g.estado], "fc50": lista(fc[f"{s}_p50"]),
             "fc10": lista(fc[f"{s}_p10"]), "fc90": lista(fc[f"{s}_p90"]), "back": {}}
        for h in HORIZ:
            x = serie_bt("fin", h) + serie_bt("nofin", h) if s == "tot" else serie_bt(s, h)
            if s != "tot":                         # el total solo necesita sus métricas (ahorra ~0,5 MB)
                d["back"][str(h)] = lista(x)
            y = g[s].reindex(idx); ok = x.notna() & y.notna()
            hor = float((x[ok] - y[ok]).abs().sum() / y[ok].sum())
            dd = pd.DataFrame({"y": y[ok], "p": x[ok]}).groupby(lambda t: t.normalize()).sum()
            met.setdefault(s, {})[str(h)] = {"wape_hora": round(hor, 4), "wape_dia": round(float((dd.p - dd.y).abs().sum() / dd.y.sum()), 4), "sesgo": round(float((x[ok] - y[ok]).sum() / y[ok].sum()), 4)}
        out["series"][s] = d
    out["metricas"] = met
    return out


def procesos():
    o = P.cargar()
    subs = P.SUBS
    tipos = sorted(o.tipo_dia.unique().tolist())
    h = o.assign(ini=(o.inicio - o.dia_negocio).dt.total_seconds() / 60)
    hist = [[str(r.dia_negocio.date()), subs.index(r.suproceso), round(r.ini, 1), round(r.dur, 1), tipos.index(r.tipo_dia)] for r in h.itertuples()]
    fc = pd.read_csv(SALIDA / "pronostico_procesos.csv", parse_dates=["fecha_negocio", "inicio_p50", "inicio_p10", "inicio_p90", "fin_p50"])
    allsubs = subs + ["depd0800cm_total"]
    fil = []
    for r in fc.itertuples():
        m = lambda t: None if pd.isna(t) else round((t - r.fecha_negocio).total_seconds() / 60, 1)
        fil.append([str(r.fecha_negocio.date()), allsubs.index(r.suproceso), r.dur_min_p50, r.dur_min_p10, r.dur_min_p90, m(r.inicio_p50), m(r.inicio_p10), m(r.inicio_p90), tipos.index(r.tipo_dia) if r.tipo_dia in tipos else -1, r.tipo_dia])
    bt = pd.read_csv(SALIDA / "backtest_procesos_dashboard.csv", parse_dates=["dia_negocio"])
    back = [[str(r.dia_negocio.date()), subs.index(r.suproceso), int(r.h), r.dur_pred, r.ini_pred] for r in bt.itertuples()]
    # métricas de backtest de duración e inicio por subproceso y horizonte
    real = o.assign(ini=(o.inicio - o.dia_negocio).dt.total_seconds() / 60).set_index(["dia_negocio", "suproceso"])[["dur", "ini"]]
    met = {}
    for (s, hh), x in bt.groupby(["suproceso", "h"]):
        j = x.set_index(["dia_negocio", "suproceso"]).join(real, how="inner")
        met.setdefault(s, {})[str(hh)] = {"wape_dur": round(float((j.dur_pred - j.dur).abs().sum() / j.dur.sum()), 4),
                                          "mae_ini_min": round(float((j.ini_pred - j.ini).abs().mean()), 1), "n": int(len(j))}
    rc = pd.read_csv(SALIDA / "registros_contables.csv")
    reg = [[r.dia_negocio, None if pd.isna(r.real) else round(r.real / 1e6, 3), None if pd.isna(r.p50) else round(r.p50 / 1e6, 3),
            None if pd.isna(r.p10) else round(r.p10 / 1e6, 3), None if pd.isna(r.p90) else round(r.p90 / 1e6, 3)] for r in rc.itertuples()]
    return {"reg": reg, "subs": subs, "allsubs": allsubs, "tipos": tipos, "hist": hist, "fc": fil, "back": back, "metricas": met,
            "ultimo": str(o.dia_negocio.max().date())}


def colas():
    """Por cola: totales y picos diarios (todo el histórico y los 365 días de pronóstico) y detalle horario de todo
    el histórico real (el simulador de hilos puede usar cualquier día real) y de los primeros 90 días pronosticados."""
    import modelo_colas as C
    g = pd.read_csv(SALIDA / "colas_limpio.csv", parse_dates=["hora"]).set_index("hora")
    qs = [c for c in g.columns if c != "estado"]
    fh = pd.read_csv(SALIDA / "pronostico_colas_horario.csv", parse_dates=["hora"]).set_index("hora")
    fd = pd.read_csv(SALIDA / "pronostico_colas_diario.csv", parse_dates=["fecha"])
    bd = pd.read_csv(SALIDA / "bandas_colas_diarias.csv").set_index(["cola", "tramo"])
    bh = pd.read_csv(SALIDA / "bandas_colas_horarias.csv").set_index(["cola", "franja", "tramo"])
    met = pd.read_csv(SALIDA / "metricas_colas.csv").set_index(["cola", "tramo"])
    ex = pd.read_csv(SALIDA / "colas_excluidas.csv", parse_dates=["hora"])
    fecha = g.index.normalize()
    malo = g.estado.isin(["sin_dato", "caida"]).groupby(fecha).any()
    f0 = fh.index.min()
    h0 = g.index.min()
    gh, fhh = g[g.index >= h0], fh[fh.index < f0 + pd.Timedelta(days=FC_HORAS_DIAS)]
    ult = g[g.index >= g.index.max() - pd.Timedelta(days=365)]
    e = lambda x: round(float(np.exp(x)), 3)
    series = {}
    for q in qs:
        gq = g[q].groupby(fecha)
        tot = gq.sum(min_count=1).mask(malo)
        fq = fd[fd.cola == q]
        tmax = ult[q].idxmax()
        series[q] = {
            "hd": lista(tot), "hp": lista(gq.max().mask(malo)), "hh": lista(gq.apply(lambda s: s.idxmax().hour if s.notna().any() else np.nan).mask(malo)),
            "fd": lista(fq.total_p50), "fp": lista(fq.tph_pico), "fp90": lista(fq.tph_pico_p90), "fh": lista(fq.hora_pico),
            "h": lista(gh[q]), "f": lista(fhh[f"{q}_p50"]),
            "banda_d": {t: [e(r.q_lo), e(r.q_hi)] for t, r in bd.loc[q].iterrows()},
            "banda_h": {fr: {t: [e(r.q_lo), e(r.q_hi)] for t, r in bh.loc[(q, fr)].iterrows()} for fr in ("dia", "noche")},
            "desvios": sorted({str(d.date()) for d in ex[(ex.cola == q) & (ex.motivo == "desvio")].hora.dt.normalize()}),
            "max_hist": [int(ult.at[tmax, q]), str(tmax)[:16]],
            "met": {t: {k: round(float(v), 4) for k, v in r.items()} for t, r in met.loc[q].iterrows()},
        }
    return {"colas": qs, "familias": C.FAMILIAS, "escalones": C.ESCALONES, "perfil_simple": C.PERFIL_SIMPLE,
            "h_t0": str(h0), "fc_horas": len(fhh), "series": series, "hilos": hilos()}


def hilos():
    """Hilos activos por cola, pico real del régimen vigente y razones por hora de la muestra medida (DEP)."""
    t = pd.read_csv(SALIDA / "hilos_colas.csv", parse_dates=["pico_cuando"])
    r = pd.read_csv(SALIDA / "hilos_referencia_dep.csv")
    c = pd.read_csv(SALIDA / "hilos_calibracion.csv").iloc[0]
    return {"max": int(t.hilos_max.iloc[0]), "medida": t.cola[t.medida].iloc[0], "fecha_muestra": str(r.fecha_muestra.iloc[0]),
            "fecha_tiempos": str(c.fecha_tiempos), "cal": {"z_p90": float(c.z_p90), "z_max": float(c.z_max), "c_w": float(c.c_w)},
            "w": {x.cola: [float(x.w_ms), float(x.w_lo), float(x.w_hi), int(x.max_txn_seg)] for x in t.itertuples()},
            "activos": {x.cola: int(x.hilos_activos) for x in t.itertuples()},
            "pico": {x.cola: [round(float(x.tph_pico)), str(x.pico_cuando)[:16]] for x in t.itertuples()},
            "ref": [[float(x.rafaga), float(x.w90), float(x.f_p90), float(x.f_max), int(bool(x.coincide)), float(x.k_p90), float(x.k_max)] for x in r.itertuples()]}


def reposting():
    """Pestaña Reposting: por tabla activa, histórico por día de negocio (TRX, minutos, TRX/min) y pronóstico con banda
    10-90 %; la noche completa (TRX de las tablas activas y duración = la de la tabla más lenta). Series alineadas a
    `hd` (días con corrida) y `fd` (días hábiles pronosticados)."""
    import modelo_reposting as MR
    from limpieza_reposting import FIN_DIA_NEGOCIO
    l = pd.read_csv(SALIDA / "reposting_limpio.csv", parse_dates=["dia_negocio"])
    p = pd.read_csv(SALIDA / "pronostico_reposting.csv", parse_dates=["fecha"])
    n = pd.read_csv(SALIDA / "pronostico_reposting_noche.csv", parse_dates=["fecha"])
    g = pd.read_csv(SALIDA / "reposting_ganadores.csv")
    act = [t for t in l[l.tabla.isin(g.tabla)].groupby("tabla").trx.sum().sort_values(ascending=False).index]
    hd, fd = sorted(l.dia_negocio.unique()), sorted(n.fecha)
    l["tmin"] = l.tmin.where((l.tmin > 0) & (l.tmin < 600))
    l["ratio"] = l.trx / l.tmin
    h, f = {}, {}
    for t in act:
        z = l[l.tabla == t].set_index("dia_negocio").reindex(hd)
        h[t] = [lista(z.trx), lista(z.tmin, 1), lista(z.ratio, 1)]
        z = p[p.tabla == t].set_index("fecha").reindex(fd)
        f[t] = [lista(z.trx_p50), lista(z.trx_p10), lista(z.trx_p90), lista(z.tmin_p50, 1), lista(z.tmin_p10, 1), lista(z.tmin_p90, 1), lista(z.ratio_p50, 1), lista(z.ratio_p10, 1), lista(z.ratio_p90, 1)]
    la = l[l.tabla.isin(act)]
    completo = la.dropna(subset=["trx"]).groupby("dia_negocio").tabla.nunique().reindex(hd) == len(act)
    tot = la.groupby("dia_negocio").trx.sum(min_count=1).reindex(hd).where(completo)
    idx = la.dropna(subset=["tmin"]).groupby("dia_negocio").tmin.idxmax()
    lenta = la.loc[idx].set_index("dia_negocio").reindex(hd)
    hn = [lista(tot), lista(lenta.tmin, 1), [None if pd.isna(x) else x for x in lenta.tabla]]
    fn = [lista(n.trx_p50), lista(n.trx_p10), lista(n.trx_p90), lista(n.tmin_p50, 1), lista(n.tmin_p10, 1), lista(n.tmin_p90, 1), n.tabla_lenta.tolist()]
    inactivas = {}
    for t in [x for x in MR.TABLAS if x not in act]:
        z = l[(l.tabla == t) & (l.trx > 0)]
        inactivas[t] = f"sin actividad desde el {z.dia_negocio.max():%d-%m-%Y}" if len(z) and z.trx.sum() > 1000 else "nunca tuvo volumen"
    return {"tablas": act, "inactivas": inactivas, "etiquetas": MR.ETIQUETAS, "escalones": MR.ESCALONES,
            "metodos": {t: z.set_index("variable").metodo.to_dict() for t, z in g.groupby("tabla")},
            "ultimo": str(FIN_DIA_NEGOCIO.date()), "hd": [str(x.date()) for x in hd], "fd": [str(x.date()) for x in fd],
            "h": h, "f": f, "hn": hn, "fn": fn}

def main():
    import picos_historicos as PH
    d = {"generado": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), "tph": tph(), "proc": procesos(), "colas": colas(), "picos": PH.calcular(), "reposting": reposting()}
    (SALIDA / "dashboard_data.json").write_text(json.dumps(d, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    kb = (SALIDA / "dashboard_data.json").stat().st_size / 1024
    print(f"dashboard_data.json {kb:.0f} KB")
    print(json.dumps(d["tph"]["metricas"], indent=1)); print(json.dumps(d["proc"]["metricas"]["depd0800cm_Lote1"]))


if __name__ == "__main__":
    main()
