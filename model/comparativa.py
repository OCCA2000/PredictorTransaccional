"""Comparativa: pronóstico vigente frente a los datos reales posteriores al entrenamiento (data/comparacion/).

Los modelos se entrenaron con datos hasta el 25-sep-2026 (transacciones, colas y reposting) y el día de negocio
28-sep-2026 (procesos), así que todo el período comparado es fuera de muestra.

Fuentes reales (data/comparacion/, unificadas a partir de los extractos del banco; para agregar días nuevos se
añaden filas a estos CSV):
  - transacciones_hora.csv: TRX por hora, Financiero y No Financiero (vacío = hora sin dato en el extracto).
  - transacciones_cola_dia.csv: TRX por día y cola. Suma exactamente el total de transacciones reales; las pocas filas
    sin código de cola del extracto (16 TRX en 25 días) se descartaron al unificar.
  - procesos_criticos.csv: ejecuciones del batch (versión DATA_CORREGIDA del banco, sin DESCRIPCIÓN ni RESPONSABLE).
  - reposting_tabla_dia.csv: TRX y minutos por día de negocio y tabla después del 25-sep-2026, sacados del registro de
    reposting del banco con la misma limpieza del entrenamiento (limpieza_reposting.limpiar: día de negocio, sin
    depuraciones ni copias). Solo entran las tablas activas.

Métricas: error = suma |pronóstico - real| / suma real (WAPE); confiabilidad = 1 - error (si el error pasa de 100 %, 0);
diferencia (sesgo) = suma pronóstico / suma real - 1 (negativo = el pronóstico quedó por debajo); en rango = el real cae
dentro de la banda P10-P90 del pronóstico.

Correcciones de los datos reales (todas listadas en la hoja «Correcciones»):
  - Procesos: el archivo no viene en orden y algunas filas traen la fecha del día siguiente o del anterior. Cada
    subproceso corre una vez por noche: sus ejecuciones se ordenan por fecha y hora y la n-ésima va a la n-ésima noche;
    las noches se asignan, en orden, a los días laborables del calendario. Se verifica que todos los subprocesos tengan
    el mismo número de noches.
  - Colas: las 19 colas son comparables (NO_COMPARABLES queda vacío; si un extracto futuro viniera incompleto, las
    colas afectadas se agregan ahí y quedan fuera del cálculo general).
Salida: data/processed/comparativa_pronostico_real.xlsx
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exportar_predicciones_excel import escribir, Font, FONT, Alignment, DIAS

RAIZ = Path(__file__).resolve().parents[1]
CMP, S = RAIZ / "data/comparacion", RAIZ / "data/processed"
OUT = S / "comparativa_pronostico_real.xlsx"
NO_COMPARABLES = set()
CADENA = ["depd0800cm_Lote1", "depd0800cm_Lote2", "depd0800cm_Lote3", "depd0800cm_Lote4", "lond0501cm", "bnccrn2dcm", "cfpd0111cm", "bncd0801"]
CORTE_SUB = {"bncd0801": 14}
wape = lambda r, p: float(np.abs(p - r).sum() / r.sum())
sesgo = lambda r, p: float(p.sum() / r.sum() - 1)
confiabilidad = lambda r, p: max(0.0, 1 - wape(r, p))


def transacciones():
    fh = pd.read_csv(S / "pronostico_tph_horario.csv", parse_dates=["hora"]).set_index("hora")
    fd = pd.read_csv(S / "pronostico_tph_diario.csv", parse_dates=["fecha"]).set_index("fecha")
    t = pd.read_csv(CMP / "transacciones_hora.csv", parse_dates=["hora"]).set_index("hora")
    real = {"fin": t.financiero, "nofin": t.no_financiero}
    dias = sorted(set(real["fin"].index.normalize()))
    d = pd.DataFrame(index=pd.DatetimeIndex(dias, name="fecha"))
    hor = []
    for s, r in real.items():
        d[f"real_{s}"] = r.groupby(r.index.normalize()).sum()
        d[f"pron_{s}"] = fd[f"{s}_total_p50"].reindex(d.index)
        d[f"p10_{s}"], d[f"p90_{s}"] = fd[f"{s}_total_p10"].reindex(d.index), fd[f"{s}_total_p90"].reindex(d.index)
        p = fh[f"{s}_p50"].reindex(r.index)
        hor.append(pd.DataFrame({f"real_{s}": r, f"pron_{s}": p, f"p10_{s}": fh[f"{s}_p10"].reindex(r.index), f"p90_{s}": fh[f"{s}_p90"].reindex(r.index)}))
    d["real_tot"], d["pron_tot"] = d.real_fin + d.real_nofin, d.pron_fin + d.pron_nofin
    d["p10_tot"], d["p90_tot"] = fd["tot_total_p10"].reindex(d.index), fd["tot_total_p90"].reindex(d.index)
    return d, pd.concat(hor, axis=1)


def colas():
    c = pd.read_csv(CMP / "transacciones_cola_dia.csv", parse_dates=["fecha"])
    r = c.pivot_table(index="fecha", columns="cola", values="trx", aggfunc="sum")
    p = pd.read_csv(S / "pronostico_colas_diario.csv", parse_dates=["fecha"])
    largo = r.stack().rename("real").reset_index()
    largo = largo.merge(p[["fecha", "cola", "total_p50", "total_p10", "total_p90", "tph_pico", "tph_pico_p90"]], on=["fecha", "cola"], how="left")
    largo["comparable"] = ~largo.cola.isin(NO_COMPARABLES)
    return largo


def procesos():
    x = pd.read_csv(CMP / "procesos_criticos.csv")
    x["sub"] = x.SUPROCESO.replace({"cfpd0111c2 (Lunes)": "cfpd0111cm", "cfpd0111c2": "cfpd0111cm"})
    t = lambda v: pd.to_timedelta(str(v)[-8:])
    x["ti"], x["tf"] = x["HORA INICIO"].map(t), x["HORA FIN"].map(t)
    x["dur"] = ((x.tf - x.ti).dt.total_seconds() % 86400) / 60
    # cada subproceso corre una vez por noche: su n-ésima ejecución (por fecha y hora del archivo) es la n-ésima noche
    x["orden"] = pd.to_datetime(x.FECHA) + x.ti
    x["noche"] = x.groupby("sub").orden.rank(method="first").astype(int)
    cuenta = x.groupby("sub").size()
    assert cuenta.nunique() == 1, f"los subprocesos no tienen el mismo número de noches: {cuenta.to_dict()}"
    cal = pd.read_csv(S / "calendario_laboral.csv", parse_dates=["fecha"]).set_index("fecha")
    l1 = x[x["sub"] == "depd0800cm_Lote1"].sort_values("orden").iloc[0]
    f0 = pd.Timestamp(l1.FECHA).normalize() - pd.Timedelta(days=int(l1.ti < pd.Timedelta(hours=12)))
    lab = cal.index[(cal.index >= f0) & cal.laborable]
    n = int(x.noche.max())
    dias = lab[:n]
    x["dia_negocio"] = x.noche.map(dict(zip(range(1, n + 1), dias)))
    x["fecha_archivo"] = pd.to_datetime(x.FECHA).dt.normalize()
    corte = x["sub"].map(CORTE_SUB).fillna(12)
    x["inicio"] = x.dia_negocio + x.ti + pd.to_timedelta((x.ti.dt.total_seconds() / 3600 < corte).astype(int), unit="D")
    x["fecha_esperada"] = x.inicio.dt.normalize()
    x["corregida"] = x.fecha_archivo != x.fecha_esperada
    x["ini_min"] = (x.inicio - x.dia_negocio).dt.total_seconds() / 60
    x = x.sort_values(["dia_negocio", "sub", "inicio"]).drop_duplicates(["dia_negocio", "sub"], keep="last")
    fc = pd.read_csv(S / "pronostico_procesos.csv", parse_dates=["fecha_negocio", "inicio_p50", "inicio_p10", "inicio_p90"])
    fc["ini_p50"] = (fc.inicio_p50 - fc.fecha_negocio).dt.total_seconds() / 60
    fc["ini_p10"] = (fc.inicio_p10 - fc.fecha_negocio).dt.total_seconds() / 60
    fc["ini_p90"] = (fc.inicio_p90 - fc.fecha_negocio).dt.total_seconds() / 60
    m = x.merge(fc.rename(columns={"fecha_negocio": "dia_negocio", "suproceso": "sub"}), on=["dia_negocio", "sub"], how="inner")
    # total de los 4 lotes
    lot = x[x["sub"].str.startswith("depd0800cm_Lote")].groupby("dia_negocio").agg(dur=("dur", "sum"), n=("dur", "size"), ini_min=("ini_min", "min"))
    lot = lot[lot.n == 4].reset_index().assign(sub="depd0800cm_total")
    mt = lot.merge(fc.rename(columns={"fecha_negocio": "dia_negocio", "suproceso": "sub"}), on=["dia_negocio", "sub"], how="inner")
    det = pd.concat([m, mt], ignore_index=True)
    # registros contables (cfpd0111) y fin de la cadena
    reg = x[(x["sub"] == "cfpd0111cm") & x.REGISTROS.notna()][["dia_negocio", "REGISTROS"]]
    rc = pd.read_csv(S / "registros_contables.csv", parse_dates=["dia_negocio"])
    reg = reg.merge(rc[["dia_negocio", "p50", "p10", "p90"]], on="dia_negocio", how="inner").dropna(subset=["p50"])   # solo noches pronosticadas
    return x, det, reg, dias


def reposting():
    real = pd.read_csv(CMP / "reposting_tabla_dia.csv", parse_dates=["dia_negocio"])
    p = pd.read_csv(S / "pronostico_reposting.csv", parse_dates=["fecha"])
    p = p[p.metodo_trx != "inactiva"]
    m = real.merge(p, left_on=["dia_negocio", "tabla"], right_on=["fecha", "tabla"])
    return m.dropna(subset=["trx", "tmin"]).sort_values(["dia_negocio", "tabla"])


def hhmm(m):
    if pd.isna(m):
        return None
    m = int(round(m)) % 1440
    return f"{m // 60:02d}:{m % 60:02d}"


def main():
    d, h = transacciones()
    cq = colas()
    praw, det, reg, noches = procesos()
    rp = reposting()

    # ---------------- hojas ----------------
    td = pd.DataFrame({"Fecha": d.index, "Día": [DIAS[i] for i in d.index.dayofweek]})
    for s, nm in (("fin", "Financieras"), ("nofin", "No Financieras"), ("tot", "Total")):
        td[f"{nm} real"] = d[f"real_{s}"].values; td[f"{nm} pronóstico"] = d[f"pron_{s}"].round(0).values
        td[f"{nm} P10"] = d[f"p10_{s}"].round(0).values; td[f"{nm} P90"] = d[f"p90_{s}"].round(0).values
        td[f"{nm} diferencia"] = (d[f"pron_{s}"] / d[f"real_{s}"] - 1).values
        td[f"{nm} en rango"] = np.where((d[f"real_{s}"] >= d[f"p10_{s}"]) & (d[f"real_{s}"] <= d[f"p90_{s}"]), "Sí", "No")
    th = pd.DataFrame({"Hora": h.index})
    for s, nm in (("fin", "Financieras"), ("nofin", "No Financieras")):
        th[f"{nm} real"] = h[f"real_{s}"].values; th[f"{nm} pronóstico"] = h[f"pron_{s}"].round(0).values
        th[f"{nm} diferencia"] = (h[f"pron_{s}"] / h[f"real_{s}"] - 1).values
        th[f"{nm} en rango"] = np.where(h[f"real_{s}"].isna() | h[f"pron_{s}"].isna(), "", np.where((h[f"real_{s}"] >= h[f"p10_{s}"]) & (h[f"real_{s}"] <= h[f"p90_{s}"]), "Sí", "No"))

    cr = cq.groupby("cola").apply(lambda z: pd.Series({"Real": z.real.sum(), "Pronóstico": z.total_p50.sum(), "Diferencia": z.total_p50.sum() / z.real.sum() - 1,
                                                      "Confiabilidad diaria": confiabilidad(z.real, z.total_p50), "Días en rango": int(((z.real >= z.total_p10) & (z.real <= z.total_p90)).sum()),
                                                      "Días": len(z), "Comparable": "Sí" if z.comparable.iloc[0] else "No (extracto incompleto)"})).reset_index().rename(columns={"cola": "Cola"})
    cr = cr.sort_values("Real", ascending=False)
    cdia = cq.assign(dif=cq.total_p50 / cq.real - 1, rango=np.where((cq.real >= cq.total_p10) & (cq.real <= cq.total_p90), "Sí", "No"))
    cdia = pd.DataFrame({"Fecha": cdia.fecha, "Cola": cdia.cola, "Real": cdia.real, "Pronóstico": cdia.total_p50, "P10": cdia.total_p10, "P90": cdia.total_p90,
                         "Diferencia": cdia.dif, "En rango": cdia.rango, "Comparable": np.where(cdia.comparable, "Sí", "No")}).sort_values(["Fecha", "Cola"])

    det = det.copy()
    det["orden"] = det["sub"].map({s: i for i, s in enumerate(CADENA + ["depd0800cm_total"])})
    det = det.sort_values(["dia_negocio", "orden"])
    pdet = pd.DataFrame({"Día de negocio": det.dia_negocio, "Subproceso": det["sub"],
                         "Inicio real": det.ini_min.map(hhmm), "Inicio pronóstico": det.ini_p50.map(hhmm),
                         "Diferencia inicio (min)": (det.ini_p50 - det.ini_min).round(0),
                         "Inicio en rango": np.where(det.ini_p10.isna(), "", np.where((det.ini_min >= det.ini_p10) & (det.ini_min <= det.ini_p90), "Sí", "No")),
                         "Duración real (min)": det.dur.round(1), "Duración pronóstico (min)": det.dur_min_p50,
                         "Duración P10": det.dur_min_p10, "Duración P90": det.dur_min_p90,
                         "Diferencia duración": det.dur_min_p50 / det.dur - 1,
                         "Duración en rango": np.where((det.dur >= det.dur_min_p10) & (det.dur <= det.dur_min_p90), "Sí", "No")})
    filas = []
    for s in ["depd0800cm_total"] + CADENA:
        z = det[det["sub"] == s]
        if z.empty:
            continue
        zi = z[z.ini_p10.notna()]
        filas.append({"Subproceso": s, "Noches": len(z), "Duración real (prom. min)": z.dur.mean(), "Duración pronóstico (prom. min)": z.dur_min_p50.mean(),
                      "Confiabilidad de duración": confiabilidad(z.dur, z.dur_min_p50), "Diferencia duración": sesgo(z.dur, z.dur_min_p50),
                      "Duración en rango": f"{int(((z.dur >= z.dur_min_p10) & (z.dur <= z.dur_min_p90)).sum())}/{len(z)}",
                      "Error medio del inicio (min)": (z.ini_p50 - z.ini_min).abs().mean(), "Inicio en rango": (f"{int(((zi.ini_min >= zi.ini_p10) & (zi.ini_min <= zi.ini_p90)).sum())}/{len(zi)}" if len(zi) else "")})
    pres = pd.DataFrame(filas)
    cf = det[det["sub"] == "cfpd0111cm"]
    fin_err = ((cf.ini_p50 + cf.dur_min_p50) - (cf.ini_min + cf.dur)).abs().mean()
    treg = pd.DataFrame({"Día de negocio": reg.dia_negocio, "Registros reales": reg.REGISTROS, "Pronóstico": reg.p50.round(0), "P10": reg.p10.round(0), "P90": reg.p90.round(0),
                         "Diferencia": reg.p50 / reg.REGISTROS - 1, "En rango": np.where((reg.REGISTROS >= reg.p10) & (reg.REGISTROS <= reg.p90), "Sí", "No")})
    corr = praw[praw.corregida][["fecha_archivo", "fecha_esperada", "dia_negocio", "SUPROCESO", "HORA INICIO", "HORA FIN"]].rename(columns={
        "fecha_archivo": "FECHA en el archivo", "fecha_esperada": "Fecha según la secuencia", "dia_negocio": "Día de negocio asignado", "SUPROCESO": "Subproceso"})
    corr["HORA INICIO"], corr["HORA FIN"] = corr["HORA INICIO"].astype(str).str[-8:], corr["HORA FIN"].astype(str).str[-8:]

    # resumen
    cc = cq[cq.comparable]
    R = []
    def fila(ind, per, n, real, pron, err, rango, nota=""):
        R.append({"Indicador": ind, "Período": per, "Datos": n, "Real": real, "Pronóstico": pron, "Diferencia": (pron / real - 1) if real else None, "Confiabilidad": max(0.0, 1 - err), "En rango": rango, "Nota": nota})
    per = f"{d.index.min():%d-%m-%Y} a {d.index.max():%d-%m-%Y}"
    for s, nm in (("fin", "TXN Financieras · por día"), ("nofin", "TXN No Financieras · por día"), ("tot", "TXN total · por día")):
        fila(nm, per, f"{len(d)} días", d[f"real_{s}"].sum(), d[f"pron_{s}"].sum(), wape(d[f"real_{s}"], d[f"pron_{s}"]),
             f"{int(((d[f'real_{s}'] >= d[f'p10_{s}']) & (d[f'real_{s}'] <= d[f'p90_{s}'])).sum())}/{len(d)}")
    for s, nm in (("fin", "TXN Financieras · por hora"), ("nofin", "TXN No Financieras · por hora")):
        ok = h[f"real_{s}"].notna() & h[f"pron_{s}"].notna(); z = h[ok]
        fila(nm, per, f"{int(ok.sum())} horas", z[f"real_{s}"].sum(), z[f"pron_{s}"].sum(), wape(z[f"real_{s}"], z[f"pron_{s}"]),
             f"{int(((z[f'real_{s}'] >= z[f'p10_{s}']) & (z[f'real_{s}'] <= z[f'p90_{s}'])).sum())}/{int(ok.sum())}", "solo horas con dato real y pronóstico")
    fila(f"Colas · {cc.cola.nunique()} comparables, por día y cola", per, f"{len(cc)} cola-días", cc.real.sum(), cc.total_p50.sum(), wape(cc.real, cc.total_p50),
         f"{int(((cc.real >= cc.total_p10) & (cc.real <= cc.total_p90)).sum())}/{len(cc)}", ("sin " + ", ".join(sorted(NO_COMPARABLES))) if NO_COMPARABLES else "todas las colas")
    nper = f"{det.dia_negocio.min():%d-%m-%Y} a {det.dia_negocio.max():%d-%m-%Y}"
    for s in ["depd0800cm_total", "lond0501cm", "bnccrn2dcm", "cfpd0111cm", "bncd0801"]:
        z = det[det["sub"] == s]
        fila(f"Duración {s} (min)", nper, f"{len(z)} noches", z.dur.sum(), z.dur_min_p50.sum(), wape(z.dur, z.dur_min_p50),
             f"{int(((z.dur >= z.dur_min_p10) & (z.dur <= z.dur_min_p90)).sum())}/{len(z)}", f"error medio del inicio: {(z.ini_p50 - z.ini_min).abs().mean():.0f} min")
    fila("Registros contables", nper, f"{len(reg)} noches", reg.REGISTROS.sum(), reg.p50.sum(), wape(reg.REGISTROS, reg.p50),
         f"{int(((reg.REGISTROS >= reg.p10) & (reg.REGISTROS <= reg.p90)).sum())}/{len(reg)}")
    R.append({"Indicador": "Fin de la cadena (fin de cfpd0111cm)", "Período": nper, "Datos": f"{len(cf)} noches", "Nota": f"error medio de la hora de fin: {fin_err:.0f} min"})
    rper = f"{rp.dia_negocio.min():%d-%m-%Y} a {rp.dia_negocio.max():%d-%m-%Y}"
    fila(f"Reposting TRX · {rp.tabla.nunique()} tablas activas, por día y tabla", rper, f"{len(rp)} tabla-días", rp.trx.sum(), rp.trx_p50.sum(), wape(rp.trx, rp.trx_p50),
         f"{int(((rp.trx >= rp.trx_p10) & (rp.trx <= rp.trx_p90)).sum())}/{len(rp)}", "días de negocio con corrida registrada")
    fila("Reposting tiempo (min) · por día y tabla", rper, f"{len(rp)} tabla-días", rp.tmin.sum(), rp.tmin_p50.sum(), wape(rp.tmin, rp.tmin_p50),
         f"{int(((rp.tmin >= rp.tmin_p10) & (rp.tmin <= rp.tmin_p90)).sum())}/{len(rp)}")
    res = pd.DataFrame(R)
    trp = pd.DataFrame({"Día de negocio": rp.dia_negocio, "Tabla": rp.tabla, "TRX real": rp.trx, "TRX pronóstico": rp.trx_p50.round(0), "TRX P10": rp.trx_p10.round(0),
                        "TRX P90": rp.trx_p90.round(0), "Diferencia TRX": rp.trx_p50 / rp.trx - 1, "TRX en rango": np.where((rp.trx >= rp.trx_p10) & (rp.trx <= rp.trx_p90), "Sí", "No"),
                        "Tiempo real (min)": rp.tmin.round(1), "Tiempo pronóstico (min)": rp.tmin_p50.round(1), "Diferencia tiempo": rp.tmin_p50 / rp.tmin - 1,
                        "Tiempo en rango": np.where((rp.tmin >= rp.tmin_p10) & (rp.tmin <= rp.tmin_p90), "Sí", "No")})

    # ---------------- Excel ----------------
    wb = openpyxl.Workbook(); ws = wb.active; ws.title = "Léeme"
    lineas = [("Comparativa: pronóstico vigente frente a datos reales", True),
              (f"Transacciones y colas: {d.index.min():%d-%m-%Y} a {d.index.max():%d-%m-%Y} ({len(d)} días). Procesos: noches del {det.dia_negocio.min():%d-%m-%Y} al {det.dia_negocio.max():%d-%m-%Y} ({det.dia_negocio.nunique()} días de negocio).", False),
              ("Los modelos se entrenaron con datos hasta el 25-sep-2026 (transacciones, colas y reposting) y el día de negocio 28-sep-2026 (procesos): todo el período es fuera de muestra.", False),
              (f"Reposting: días de negocio del {rp.dia_negocio.min():%d-%m-%Y} al {rp.dia_negocio.max():%d-%m-%Y} con corrida registrada (reposting_tabla_dia.csv; el reposting del día D corre en la madrugada de D+1; la depuración del 17-oct no entra).", False), ("", False),
              ("Cómo leer las métricas", True),
              ("Diferencia = pronóstico / real - 1 (negativo: el pronóstico quedó por debajo).", False),
              ("Error = suma de |pronóstico - real| / suma de lo real (desvío promedio sin signo, ponderado por volumen). Confiabilidad = 100 % - error: un error de 10 % es una confiabilidad de 90 % (si el error pasa de 100 %, la confiabilidad es 0).", False),
              ("En rango = el valor real cae dentro de la banda P10-P90 del pronóstico (se espera ~80 %).", False), ("", False),
              ("Correcciones a los datos reales (hoja «Correcciones»)", True),
              ("Procesos: el archivo (versión DATA_CORREGIDA del banco) no viene en orden y algunas filas traen la fecha del día siguiente o del anterior. Cada subproceso corre una vez por noche: sus ejecuciones se ordenan por fecha y hora, la n-ésima va a la n-ésima noche y las noches se asignan en orden a los días laborables.", False),
              ("Colas: transacciones_cola_dia.csv suma exactamente el total real de transacciones; las 19 colas son comparables. Al unificar los extractos se descartaron 4 filas sin código de cola (16 TRX en total).", False),
              ("Transacciones: el total diario real incluye 1-2 horas de madrugada que el pronóstico deja vacías (ventana del batch); su efecto en la confiabilidad es menor a 0,1 puntos.", False)]
    for i, (t, b) in enumerate(lineas, 1):
        c = ws.cell(i, 1, t); c.font = Font(name=FONT, size=14 if i == 1 else 10, bold=b); c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.column_dimensions["A"].width = 140
    pct = "0.0%"
    hojas = [("Resumen", res, {"Real": "#,##0", "Pronóstico": "#,##0", "Diferencia": pct, "Confiabilidad": pct}, {"Indicador": 40, "Período": 24, "Nota": 40}),
             ("TXN_diario", td, {"Fecha": "yyyy-mm-dd", **{k: "#,##0" for k in td.columns if any(w in k for w in ("real", "pronóstico", "P10", "P90"))}, **{k: pct for k in td.columns if "diferencia" in k}}, None),
             ("TXN_horario", th, {"Hora": "yyyy-mm-dd hh:mm", **{k: "#,##0" for k in th.columns if "real" in k or "pronóstico" in k}, **{k: pct for k in th.columns if "diferencia" in k}}, {"Hora": 18}),
             ("Colas_resumen", cr, {"Real": "#,##0", "Pronóstico": "#,##0", "Diferencia": pct, "Confiabilidad diaria": pct}, {"Comparable": 24}),
             ("Colas_diario", cdia, {"Fecha": "yyyy-mm-dd", "Real": "#,##0", "Pronóstico": "#,##0", "P10": "#,##0", "P90": "#,##0", "Diferencia": pct}, None),
             ("Procesos_resumen", pres, {"Duración real (prom. min)": "0.0", "Duración pronóstico (prom. min)": "0.0", "Confiabilidad de duración": pct, "Diferencia duración": pct, "Error medio del inicio (min)": "0"}, {"Subproceso": 20}),
             ("Procesos_detalle", pdet, {"Día de negocio": "yyyy-mm-dd", "Diferencia duración": pct, "Duración real (min)": "0.0", "Duración pronóstico (min)": "0.0", "Duración P10": "0.0", "Duración P90": "0.0"}, {"Subproceso": 20}),
             ("Registros_contables", treg, {"Día de negocio": "yyyy-mm-dd", "Registros reales": "#,##0", "Pronóstico": "#,##0", "P10": "#,##0", "P90": "#,##0", "Diferencia": pct}, None),
             ("Reposting", trp, {"Día de negocio": "yyyy-mm-dd", **{k: "#,##0" for k in trp.columns if k.startswith("TRX ") and "rango" not in k}, "Diferencia TRX": pct, "Diferencia tiempo": pct}, None),
             ("Correcciones", corr, {"FECHA en el archivo": "yyyy-mm-dd", "Fecha según la secuencia": "yyyy-mm-dd", "Día de negocio asignado": "yyyy-mm-dd"}, {"Subproceso": 20})]
    for nom, df, fmt, wid in hojas:
        escribir(wb.create_sheet(nom), df.reset_index(drop=True), fmt=fmt, widths=wid)
    wb.save(OUT)
    print("guardado", OUT)
    pd.set_option("display.width", 250, "display.max_columns", 20)
    print(res.drop(columns=["Período"]).to_string(index=False))
    print(cr.to_string(index=False)); print(pres.round(3).to_string(index=False)); print(treg.to_string(index=False))
    print("noches reales asignadas:", [f"{x:%d-%m}" for x in noches], "| filas con fecha corregida:", len(corr))


if __name__ == "__main__":
    main()
