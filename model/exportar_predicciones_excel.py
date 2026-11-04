"""Exporta todos los pronósticos a un solo Excel, data/processed/ForecastBaNCS.xlsx (mismas cifras que el dashboard):
transacciones, colas, procesos críticos y reposting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

RAIZ = Path(__file__).resolve().parents[1]
S = RAIZ / "data/processed"
OUT = S / "ForecastBaNCS.xlsx"
FONT = "Arial"
HDR_FILL = PatternFill("solid", fgColor="1C3F6E")
HDR_FONT = Font(name=FONT, bold=True, color="FFFFFF", size=10)
BASE = Font(name=FONT, size=10)
BOLD = Font(name=FONT, size=10, bold=True)
ZFILL = {"Confiable (1-90 d)": "E3F2E8", "Intermedia (91-180 d)": "FFF3D6", "Indicativa (181-365 d)": "ECECEC"}
DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
ORDEN = ["depd0800cm_Lote1", "depd0800cm_Lote2", "depd0800cm_Lote3", "depd0800cm_Lote4", "depd0800cm_total", "lond0501cm", "bnccrn2dcm", "cfpd0111cm", "bncd0801"]


def zona(k):
    return "Confiable (1-90 d)" if k <= 90 else ("Intermedia (91-180 d)" if k <= 180 else "Indicativa (181-365 d)")


def tramo(k):
    return "1-30" if k <= 30 else ("31-90" if k <= 90 else ("91-180" if k <= 180 else "181-365"))


def cabecera(ws, fila, textos, col0=1):
    for j, t in enumerate(textos, col0):
        c = ws.cell(fila, j, t)
        c.font, c.fill = HDR_FONT, HDR_FILL
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)


def escribir(ws, df, fmt=None, widths=None, zona_col=None):
    """Escribe df con encabezado, formatos, filtro y panel congelado."""
    fmt = fmt or {}
    cabecera(ws, 1, list(df.columns))
    ws.row_dimensions[1].height = 34
    for row in df.itertuples(index=False):
        ws.append([None if (isinstance(v, float) and np.isnan(v)) else (v.to_pydatetime() if isinstance(v, pd.Timestamp) else v) for v in row])
    for j, col in enumerate(df.columns, 1):
        L = get_column_letter(j)
        for cell in ws[L][1:]:
            cell.font = BASE
            if col in fmt:
                cell.number_format = fmt[col]
        ws.column_dimensions[L].width = (widths or {}).get(col, max(12, min(26, len(str(col)) + 3)))
    ws.freeze_panes = "B2"
    ws.auto_filter.ref = ws.dimensions
    if zona_col:
        j = list(df.columns).index(zona_col) + 1
        for r in range(2, ws.max_row + 1):
            c = ws.cell(r, j)
            c.fill = PatternFill("solid", fgColor=ZFILL.get(c.value, "FFFFFF"))


def columna_formula(ws, titulo, formula, fmt, ancho):
    j = ws.max_column + 1
    cabecera(ws, 1, [titulo], j)
    for r in range(2, ws.max_row + 1):
        c = ws.cell(r, j, formula.format(r=r))
        c.font, c.number_format = BASE, fmt
    ws.column_dimensions[get_column_letter(j)].width = ancho


def main():
    h = pd.read_csv(S / "pronostico_tph_horario.csv", parse_dates=["hora"]).set_index("hora")
    f0 = h.index.min().normalize()
    h["fecha"] = h.index.normalize()
    h["k"] = (h.fecha - f0).dt.days + 1
    h["batch"] = h.noche_batch.eq(1) & h.index.hour.isin([23, 0, 1])
    cols = ["fin_p10", "fin_p50", "fin_p90", "nofin_p10", "nofin_p50", "nofin_p90"]
    hv = h.copy()
    hv.loc[hv.batch, cols] = np.nan          # sin transacciones en línea en la ventana del batch
    ban = pd.read_csv(S / "bandas_tph_diarias.csv")
    ban["lo"], ban["hi"] = np.exp(ban.q_lo), np.exp(ban.q_hi)
    ban = ban.set_index(["serie", "tramo"])

    # tipo de día del calendario operativo del banco (limpieza_procesos.tipo_operativo)
    cal = pd.read_csv(S / "calendario_laboral.csv", parse_dates=["fecha"]).set_index("fecha")
    TOP = {"DIA NORMAL": "Día normal", "PRIMER DIA LABORAL": "Primer día laboral", "PRIMER BATCH": "Primer batch",
           "FDS NORMAL": "Fin de semana", "FERIADO": "Feriado"}
    tipo_de = lambda fechas: pd.Index(fechas).normalize().map(cal.tipo_op).map(TOP)
    tipo = tipo_de(hv.index)
    th = pd.DataFrame({"Hora": hv.index, "Zona": hv.k.map(zona).values, "Tipo de día": tipo,
                       "Ventana batch (sin dato en línea)": np.where(hv.batch, "Sí", "No"),
                       "Financiero P10": hv.fin_p10.values, "Financiero P50": hv.fin_p50.values, "Financiero P90": hv.fin_p90.values,
                       "No Financiero P10": hv.nofin_p10.values, "No Financiero P50": hv.nofin_p50.values, "No Financiero P90": hv.nofin_p90.values})
    filas = []
    for d, g in hv.groupby("fecha"):
        k = int(g.k.iloc[0])
        t = tramo(k)
        r = {"Fecha": d, "Día": DIAS[d.dayofweek], "Tipo de día": TOP[cal.at[d, "tipo_op"]],
             "Zona": zona(k), "Días desde el último dato": k}
        for s, nm in (("fin", "Financiero"), ("nofin", "No Financiero")):
            tot = g[f"{s}_p50"].sum(skipna=True)
            r[f"{nm} total P50"] = round(tot)
            r[f"{nm} total P10"] = round(tot * ban.loc[(s, t), "lo"])
            r[f"{nm} total P90"] = round(tot * ban.loc[(s, t), "hi"])
            r[f"{nm} TPH pico"] = g[f"{s}_p50"].max()
            r[f"{nm} hora pico"] = int(g[f"{s}_p50"].idxmax().hour)
        filas.append(r)
    td = pd.DataFrame(filas)

    pr = pd.read_csv(S / "pronostico_procesos.csv", parse_dates=["fecha_negocio", "inicio_p50", "inicio_p10", "inicio_p90", "fin_p50"])
    pr["k"] = (pr.fecha_negocio - f0).dt.days + 1
    pr["tipo_dia"] = tipo_de(pr.fecha_negocio)
    det = pd.DataFrame({"Día de negocio": pr.fecha_negocio, "Día": pr.fecha_negocio.dt.dayofweek.map(lambda i: DIAS[i]), "Tipo de día": pr.tipo_dia,
                        "Zona": pr.k.map(zona), "Subproceso": pr.suproceso, "Duración P10 (min)": pr.dur_min_p10, "Duración P50 (min)": pr.dur_min_p50,
                        "Duración P90 (min)": pr.dur_min_p90, "Inicio P10": pr.inicio_p10, "Inicio P50": pr.inicio_p50, "Inicio P90": pr.inicio_p90, "Fin P50": pr.fin_p50})
    det["o"] = det.Subproceso.map(ORDEN.index)
    det = det.sort_values(["Día de negocio", "o"]).drop(columns="o")
    w = pr.pivot_table(index="fecha_negocio", columns="suproceso", values=["dur_min_p50", "dur_min_p10", "dur_min_p90", "inicio_p50"], aggfunc="first")
    kk = (w.index - f0).days + 1
    res = pd.DataFrame({"Día de negocio": w.index, "Día": w.index.dayofweek.map(lambda i: DIAS[i]),
                        "Tipo de día": pr.groupby("fecha_negocio").tipo_dia.first().values, "Zona": [zona(k) for k in kk],
                        "Inicio Lote1 P50": w[("inicio_p50", "depd0800cm_Lote1")].values,
                        "depd0800cm total P10 (min)": w[("dur_min_p10", "depd0800cm_total")].values,
                        "depd0800cm total P50 (min)": w[("dur_min_p50", "depd0800cm_total")].values,
                        "depd0800cm total P90 (min)": w[("dur_min_p90", "depd0800cm_total")].values,
                        "lond0501cm P50 (min)": w[("dur_min_p50", "lond0501cm")].values,
                        "bnccrn2dcm P50 (s)": (w[("dur_min_p50", "bnccrn2dcm")] * 60).round(0).values,
                        "Inicio cfpd0111cm P50": w[("inicio_p50", "cfpd0111cm")].values,
                        "cfpd0111cm P10 (min)": w[("dur_min_p10", "cfpd0111cm")].values,
                        "cfpd0111cm P50 (min)": w[("dur_min_p50", "cfpd0111cm")].values,
                        "cfpd0111cm P90 (min)": w[("dur_min_p90", "cfpd0111cm")].values,
                        "Inicio bncd0801 P50": w[("inicio_p50", "bncd0801")].values,
                        "bncd0801 P50 (min)": w[("dur_min_p50", "bncd0801")].values})
    rc = pd.read_csv(S / "registros_contables.csv", parse_dates=["dia_negocio"]).set_index("dia_negocio")
    for q in ("p10", "p50", "p90"):
        res[f"Registros contables {q.upper()} (M)"] = (res["Día de negocio"].map(rc[q]) / 1e6).values

    # colas: diario en formato largo (365 días) y P50 horario de los primeros 90 días
    cdia = pd.read_csv(S / "pronostico_colas_diario.csv", parse_dates=["fecha"])
    kc = (cdia.fecha - f0).dt.days + 1
    col_d = pd.DataFrame({"Fecha": cdia.fecha, "Día": cdia.fecha.dt.dayofweek.map(lambda i: DIAS[i]), "Tipo de día": tipo_de(cdia.fecha), "Zona": kc.map(zona), "Cola": cdia.cola,
                       "Total P10": cdia.total_p10, "Total P50": cdia.total_p50, "Total P90": cdia.total_p90,
                       "TPH pico P50": cdia.tph_pico, "TPH pico P90": cdia.tph_pico_p90, "Hora pico": cdia.hora_pico})
    chor = pd.read_csv(S / "pronostico_colas_horario.csv", parse_dates=["hora"]).set_index("hora")
    chor = chor[chor.index < f0 + pd.Timedelta(days=90)]
    colas = [c[:-4] for c in chor.columns if c.endswith("_p50")]
    col_h = pd.DataFrame({"Hora": chor.index, **{q: chor[f"{q}_p50"].values for q in colas}})

    wb = Workbook()
    ws = wb.active
    ws.title = "Léeme"
    lineas = [("BaNCS Forecast — transacciones, colas, procesos críticos y reposting", True),
              (f"Generado {pd.Timestamp.now():%Y-%m-%d %H:%M}. Pronóstico del {f0:%Y-%m-%d} al {h.index.max():%Y-%m-%d} (365 días).", False), ("", False),
              ("Hojas", True),
              ("TPH_diario: total, banda 10-90%, TPH pico y hora pico por día y serie. Los totales NO suman las horas de la ventana del batch.", False),
              ("TPH_horario: P10 / P50 (mediana) / P90 por hora. Las horas de la ventana del batch (23:00-01:59 en noches laborables) van vacías: no hay transacciones en línea.", False),
              ("Procesos_resumen: una fila por día de negocio con la cadena del batch. 'Fin de la cadena' es fórmula: inicio de cfpd0111cm + su duración. bnccrn2dcm (segundos) corre tras lond0501cm; bncd0801 corre después y no depende de la cadena.", False),
              ("Procesos_detalle: una fila por día y subproceso (duración e inicio con banda, fin P50). Las horas posteriores a las 24:00 caen en la madrugada del día siguiente.", False),
              ("Colas_diario: una fila por día y cola (19 colas) con total, banda 10-90%, TPH pico y hora pico. La suma de colas de cada día es Financiero + No Financiero.", False),
              ("Colas_horario_90d: P50 por hora y cola de los primeros 90 días. Supone que el enrutamiento actual entre colas se mantiene.", False),
              ("Hilos_colas: hilos activos por cola (máximo 99) y hilos requeridos en la hora más exigente de los primeros 90 días, con el techo del pronóstico. P90 = cubre el 90% de los instantes; máximo = cubre el instante más cargado. DEP está medido (un día segundo a segundo); las demás se estiman con su tiempo promedio por transacción de 30 días laborables, del 1-sep al 13-oct-2026 (redondeado a 10 ms: se usa el extremo alto) y el salto por ráfagas medido en DEP.", False),
              ("Picos_historicos: picos reales (sin los datos del 26-sep al 13-oct usados en la comparativa). Transacciones: hora con más transacciones (TPH; TPS = TPH / 3600) y día con más transacciones, en todo el historial y desde el cambio de nivel de No Financieras (12-mar-2026). Colas: igual, sin días de desvío ni caídas, y desde el último cambio de enrutamiento de cada cola. Contabilidad: desde la contabilidad diaria (2-jun-2026), noche con más registros contables y mayor ritmo (registros por minuto de cfpd0111cm). Reposting: noche con más transacciones y noche más larga, de la noche completa y de cada tabla (IMG4 e IMG8 también desde su cambio de nivel del 7-sep-2026).", False),
              ("", False),
              ("Zonas de confianza (días desde el último dato real)", True),
              ("Confiable: días 1-90 (hasta 2026-12-24). Intermedia: 91-180. Indicativa: 181-365, asume que el nivel actual se mantiene.", False), ("", False),
              ("Limitaciones", True),
              ("• No Financiero cayó ~20% entre feb y mar de 2026 (causa sin confirmar); una caída así no se puede anticipar. Se usa una variable de régimen.", False),
              ("• Con 21 meses de historia (TPH) y 12 meses (procesos) la estacionalidad anual se aprende de un solo año.", False),
              ("• Financiero incluye una tendencia de crecimiento (+21% anual) y su sesgo histórico es de -4% a -9%.", False),
              ("• La hora de inicio de los procesos tiene un margen de decenas de minutos (error medio de ~16 min en el Lote1 a ~35 min en cfpd0111cm).", False),
              ("• P10 y P90 salen de errores reales de backtest, no de una distribución teórica.", False), ("", False)]
    rp = pd.read_csv(S / "pronostico_reposting.csv", parse_dates=["fecha"])
    rg = pd.read_csv(S / "reposting_ganadores.csv")
    import modelo_reposting as MR
    lineas += [("Reposting (hojas Reposting_diario y Reposting_mensual)", True),
               ("Reposting_diario: una fila por día de negocio y tabla (IMG1 a IMG8): TRX, velocidad (TRX por minuto) y tiempo con banda 10-90%. El reposting del día de negocio D corre en la madrugada de D+1; solo hay filas para días hábiles.", False),
               ("Reposting_mensual: TRX y horas totales P50 por mes y tabla.", False),
               ("Método por tabla y variable: el de menor error a 1-90 días en el backtest (11 orígenes, oct-2025 a jul-2026). " +
                "; ".join(f"{t}: TRX {g.set_index('variable').metodo['trx']}, velocidad {g.set_index('variable').metodo['ratio']}" for t, g in rg.groupby("tabla")) + ".", False),
               ("El tiempo es TRX / velocidad. IMG3, IMG5 e IMG6 están inactivas: su pronóstico es 0.", False),
               ("IMG4 e IMG8 cambiaron de nivel el 7-sep-2026 (causa sin confirmar): el histórico anterior se llevó al nivel nuevo y su banda se amplió 1,5 veces.", False)]
    for i, (t, b) in enumerate(lineas, 1):
        c = ws.cell(i, 1, t)
        c.font = Font(name=FONT, size=14 if i == 1 else 10, bold=b)
        c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.column_dimensions["A"].width = 130

    ws = wb.create_sheet("TPH_diario")
    fmt = {"Fecha": "yyyy-mm-dd"}
    fmt.update({c: "#,##0" for c in td.columns if "total" in c or "TPH pico" in c})
    escribir(ws, td, fmt=fmt, widths={"Zona": 22, "Día": 11, "Tipo de día": 14}, zona_col="Zona")
    cf = get_column_letter(list(td.columns).index("Financiero total P50") + 1)
    cn = get_column_letter(list(td.columns).index("No Financiero total P50") + 1)
    columna_formula(ws, "Total P50 (Fin + No Fin)", "=" + cf + "{r}+" + cn + "{r}", "#,##0", 18)

    ws = wb.create_sheet("TPH_horario")
    fmt = {"Hora": "yyyy-mm-dd hh:mm"}
    fmt.update({c: "#,##0" for c in th.columns if c.endswith(("P10", "P50", "P90"))})
    escribir(ws, th, fmt=fmt, widths={"Hora": 18, "Zona": 22, "Ventana batch (sin dato en línea)": 16}, zona_col="Zona")

    ws = wb.create_sheet("Procesos_resumen")
    fmt = {"Día de negocio": "yyyy-mm-dd", "Inicio Lote1 P50": "yyyy-mm-dd hh:mm", "Inicio cfpd0111cm P50": "yyyy-mm-dd hh:mm", "Inicio bncd0801 P50": "yyyy-mm-dd hh:mm", "bnccrn2dcm P50 (s)": "0"}
    fmt.update({c: "0.0" for c in res.columns if "min" in c or "(M)" in c})
    escribir(ws, res, fmt=fmt, widths={"Zona": 22, "Inicio Lote1 P50": 18, "Inicio cfpd0111cm P50": 20}, zona_col="Zona")
    ci = get_column_letter(list(res.columns).index("Inicio cfpd0111cm P50") + 1)
    cd = get_column_letter(list(res.columns).index("cfpd0111cm P50 (min)") + 1)
    columna_formula(ws, "Fin de la cadena P50", "=" + ci + "{r}+" + cd + "{r}/1440", "yyyy-mm-dd hh:mm", 20)

    ws = wb.create_sheet("Procesos_detalle")
    fmt = {"Día de negocio": "yyyy-mm-dd", "Inicio P10": "yyyy-mm-dd hh:mm", "Inicio P50": "yyyy-mm-dd hh:mm", "Inicio P90": "yyyy-mm-dd hh:mm", "Fin P50": "yyyy-mm-dd hh:mm"}
    fmt.update({c: "0.0" for c in det.columns if "min" in c})
    escribir(ws, det, fmt=fmt, widths={"Zona": 22, "Subproceso": 22, "Inicio P10": 18, "Inicio P50": 18, "Inicio P90": 18, "Fin P50": 18}, zona_col="Zona")

    ws = wb.create_sheet("Colas_diario")
    fmt = {"Fecha": "yyyy-mm-dd"}
    fmt.update({c: "#,##0" for c in col_d.columns if "Total" in c or "TPH" in c})
    escribir(ws, col_d, fmt=fmt, widths={"Zona": 22, "Día": 11}, zona_col="Zona")
    cp = get_column_letter(list(col_d.columns).index("TPH pico P90") + 1)
    columna_formula(ws, "TPS pico P90", "=" + cp + "{r}/3600", "#,##0.0", 14)

    ws = wb.create_sheet("Colas_horario_90d")
    fmt = {"Hora": "yyyy-mm-dd hh:mm"}
    fmt.update({q: "#,##0" for q in colas})
    escribir(ws, col_h, fmt=fmt, widths={"Hora": 18})

    # hilos por cola: mismas cifras que la tabla del dashboard para los primeros 90 días (techo del pronóstico)
    import hilos_colas as HC
    hc = pd.read_csv(S / "hilos_colas.csv", parse_dates=["pico_cuando"])
    hr = pd.read_csv(S / "hilos_referencia_dep.csv").set_index("hora")
    calib = pd.read_csv(S / "hilos_calibracion.csv").iloc[0].to_dict()
    techo = lambda v: int(np.ceil(v - 0.005))
    filas = []
    for x in hc.itertuples():
        p90h = chor[f"{x.cola}_p90"].dropna()
        if x.medida:
            req, rmx = techo((p90h * p90h.index.hour.map(hr.k_p90)).max()), techo((p90h * p90h.index.hour.map(hr.k_max)).max())
            bajo = req
        else:
            req, rmx = techo(HC.hilos_estimados(p90h.max(), x.w_hi, calib, "p90")), techo(HC.hilos_estimados(p90h.max(), x.w_hi, calib, "max"))
            bajo = max(1, techo(HC.hilos_estimados(p90h.max(), x.w_lo, calib, "p90")))
        filas.append({"Cola": x.cola, "Hilos activos": x.hilos_activos, "Máximo": x.hilos_max, "Margen": x.hilos_max - x.hilos_activos,
                      "T. servicio promedio (ms)": x.w_ms, "T. servicio usado (ms)": x.w_ms if x.medida else x.w_hi, "Pico en un segundo (trx)": x.max_txn_seg,
                      "TPS pico real": x.tps_pico, "Fecha del pico real": x.pico_cuando, "Pico pronosticado P90, 90 días (TPS)": p90h.max() / 3600,
                      "Pico pronosticado / pico real": p90h.max() / x.tph_pico, "Hilos req. P90 (mínimo por redondeo)": bajo, "Hilos req. P90": req,
                      "Uso P90": req / x.hilos_activos, "Hilos req. máximo": rmx, "Uso máximo": rmx / x.hilos_activos,
                      "Supera 99": "Sí" if rmx > x.hilos_max else "No", "Base": "medido" if x.medida else "estimado"})
    th = pd.DataFrame(filas).sort_values("Uso máximo", ascending=False)
    ws = wb.create_sheet("Hilos_colas")
    escribir(ws, th, fmt={"TPS pico real": "#,##0.0", "Fecha del pico real": "yyyy-mm-dd hh:mm", "Pico pronosticado P90, 90 días (TPS)": "#,##0.0",
                          "Pico pronosticado / pico real": "0.0%", "Uso P90": "0.0%", "Uso máximo": "0.0%", "T. servicio promedio (ms)": "0", "T. servicio usado (ms)": "0"},
             widths={"Fecha del pico real": 18})

    import picos_historicos as PH
    tp = PH.tabla(PH.calcular())
    ws = wb.create_sheet("Picos_historicos")
    escribir(ws, tp, fmt={"TPH pico": "#,##0", "TPS pico": "#,##0.0", "Transacciones ese día": "#,##0", "Ritmo pico (registros/min)": "#,##0",
                          "Duración cfpd0111cm (min)": "0.0", "Duración de la noche (min)": "0.0"}, widths={"Ámbito": 48, "Ventana": 52, "Hora del pico": 18, "Día con más transacciones": 16})

    rp["k"] = (rp.fecha - f0).dt.days + 1
    rd = pd.DataFrame({"Día de negocio": rp.fecha, "Día": rp.fecha.dt.dayofweek.map(lambda i: DIAS[i]), "Tipo de día": tipo_de(rp.fecha), "Zona": rp.k.map(zona),
                       "Tabla": rp.tabla, "Descripción": rp.tabla.map(MR.ETIQUETAS), "Método TRX": rp.metodo_trx, "Método velocidad": rp.metodo_ratio,
                       "TRX P10": rp.trx_p10, "TRX P50": rp.trx_p50, "TRX P90": rp.trx_p90, "Velocidad P50 (TRX/min)": rp.ratio_p50,
                       "Tiempo P10 (min)": rp.tmin_p10, "Tiempo P50 (min)": rp.tmin_p50, "Tiempo P90 (min)": rp.tmin_p90})
    ws = wb.create_sheet("Reposting_diario")
    fmt = {"Día de negocio": "yyyy-mm-dd", "Velocidad P50 (TRX/min)": "#,##0.0"}
    fmt.update({c: "#,##0" for c in rd.columns if c.startswith("TRX")})
    fmt.update({c: "0.0" for c in rd.columns if c.startswith("Tiempo")})
    escribir(ws, rd, fmt=fmt, widths={"Zona": 22, "Día": 11, "Tipo de día": 16, "Descripción": 26}, zona_col="Zona")
    rp["Mes"] = rp.fecha.dt.strftime("%Y-%m")
    m = rp.groupby(["Mes", "tabla"]).agg(trx=("trx_p50", "sum"), h=("tmin_p50", "sum"), dias=("fecha", "nunique")).reset_index()
    rm = m.pivot(index="Mes", columns="tabla", values="trx").add_prefix("TRX ")
    rm = rm.join(m.pivot(index="Mes", columns="tabla", values="h").div(60).add_prefix("Horas "))
    rm.insert(0, "Días hábiles", m.groupby("Mes").dias.first())
    ws = wb.create_sheet("Reposting_mensual")
    escribir(ws, rm.reset_index(), fmt={**{c: "#,##0" for c in rm.columns if c.startswith("TRX")}, **{c: "0.0" for c in rm.columns if c.startswith("Horas")}})

    wb.save(OUT)
    print("guardado", OUT, f"{OUT.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
