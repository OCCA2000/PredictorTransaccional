"""Limpieza de PROCESOS CRITICOS ACUMULADOS.xlsx -> data/processed/procesos_limpio.csv

Las columnas DESCRIPCIÓN y RESPONSABLE nunca se leen (usecols). REGISTROS se descarta
(casi vacía). Cada ejecución se asigna a su DÍA DE NEGOCIO:
  - inicio >= 12:00  -> el batch es de ese mismo día
  - inicio <  12:00  -> madrugada: es el batch del día calendario anterior
  (bncd0801 empieza a veces hasta las 12:30 y pertenece igual al batch de la noche anterior: su corte es 14:00)
El batch no corre fines de semana ni feriados de Ecuador; si el día de negocio resultante
no es laborable se marca (`dia_negocio_no_laborable`) para revisión, no se mueve.
Solo se usan días de negocio hasta FIN_DIA_NEGOCIO, el mismo corte del entrenamiento anterior.
"""
from pathlib import Path
import datetime as dt
import pandas as pd
import holidays

RAIZ = Path(__file__).resolve().parents[1]
# Versión de oct-2026: mismas filas que la anterior (data/raw/PROCESOS_CRITICOS_ACUMULADOS.xlsx) más bnccrn2dcm y bncd0801
XLSX = RAIZ / "data/raw/REPORTES TRANSACCIONALES 2610 1210/PROCESOS CRITICOS/PROCESOS CRITICOS ACUMULADOS.xlsx"
SALIDA = RAIZ / "data/processed"
CORTE_HORA = 12
CORTE_SUB = {"bncd0801": 14}
FIN_DIA_NEGOCIO = pd.Timestamp("2026-09-28")
TOL_DUP_MIN = 10
PROHIBIDAS = {"DESCRIPCIÓN", "DESCRIPCION", "RESPONSABLE", "REGISTROS"}
DIAS = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]
# Ajustes al calendario de la librería `holidays`, contrastados con el volumen de TPH:
#  - puentes de Año Nuevo decretados: esos días el volumen fue el de un fin de semana (53-57 % del normal)
FERIADOS_EXTRA = {dt.date(2025, 1, 2), dt.date(2025, 1, 3), dt.date(2026, 1, 2),
#  - suspensiones de jornada decretadas (sector público y privado), que el calendario operativo del banco también marca:
#    jueves 30-abr-2026 (Decreto 354; ese día no hubo batch y el volumen bajó) y viernes 20-nov-2026 (Decreto 507)
                  dt.date(2026, 4, 30), dt.date(2026, 11, 20)}
#  - feriados trasladados (Ley de feriados: martes -> lunes; miércoles o jueves -> viernes; sábado -> viernes;
#    domingo -> lunes): la librería deja como feriado la fecha original además de la trasladada («observado»). Se quita
#    la original en `calendario()` (p. ej. 1-may-2024 -> vie 3; 9-oct-2024 -> vie 11; 1-may-2025 -> vie 2). Año Nuevo
#    y Navidad no se trasladan, y Difuntos se deja como en el calendario operativo del banco.
#  - excepción: el jueves 9-oct-2025 (trasladado al viernes 10) el banco no corrió el batch ni el reposting y el volumen
#    bajó: se mantiene como no laborable para el banco.
TRASLADO_NO_APLICADO = {dt.date(2025, 10, 9)}
NO_FERIADOS = set()


def a_time(v):
    if isinstance(v, str):  # hay celdas de hora guardadas como texto
        return dt.time.fromisoformat(v.strip())
    return v.time() if isinstance(v, (dt.datetime, pd.Timestamp)) else v


def corregir_fechas(d):
    """Aplica data/processed/correcciones_fecha.csv (sub, p_ini, delta_dias): corrige la FECHA
    de ejecuciones cuya fecha en PROCESOS no coincide con la del scheduler (RITM1037408).
    Modifica `d` en el lugar."""
    c = pd.read_csv(SALIDA / "correcciones_fecha.csv", parse_dates=["p_ini"])
    m = d[["SUPROCESO", "inicio"]].merge(c, left_on=["SUPROCESO", "inicio"], right_on=["sub", "p_ini"], how="left")
    delta = pd.to_timedelta(m["delta_dias"].fillna(0).to_numpy(), unit="D")
    d["fecha_corregida"] = m["delta_dias"].notna().to_numpy()
    d["FECHA"] = d["FECHA"] + delta
    d["inicio"] = d["inicio"] + delta
    print(f"fechas corregidas: {int(d.fecha_corregida.sum())} filas de {len(c)} correcciones")


def aplicar_parches_ritm(d):
    """Aplica data/processed/parches_ritm.csv. 'reemplazar': pone en la fila los tiempos del RITM
    (si el fin del RITM es igual al inicio, duración 0, se conserva el fin de PROCESOS).
    'agregar': ejecuciones que están en el RITM y no en PROCESOS (si la duración del RITM es 0
    se agrega sin fin, para que no entre como duración 0 al modelo). Devuelve `d` ampliado."""
    c = pd.read_csv(SALIDA / "parches_ritm.csv", parse_dates=["p_ini", "r_ini", "r_fin"])
    d["ritm_parche"] = ""
    proc_de = d.drop_duplicates("SUPROCESO").set_index("SUPROCESO")["PROCESO"]
    nuevas = []
    for _, x in c.iterrows():
        if x.tipo == "reemplazar":
            m = (d["SUPROCESO"] == x["sub"]) & (d["inicio"] == x.p_ini)
            assert m.sum() == 1, (x["sub"], x.p_ini, int(m.sum()))
            i = d.index[m][0]
            fecha_antes = d.at[i, "FECHA"]
            if x.usar_fin_ritm:
                d.at[i, "fin"] = x.r_fin
            d.at[i, "inicio"] = x.r_ini
            d.at[i, "FECHA"] = x.r_ini.normalize()
            d.at[i, "ini_t"] = x.r_ini.time()
            d.at[i, "dur_calc_s"] = (d.at[i, "fin"] - x.r_ini).total_seconds()
            d.at[i, "dur_difiere"] = False
            d.at[i, "ritm_parche"] = "reemplazada"
            d.at[i, "fecha_corregida"] = bool(d.at[i, "fecha_corregida"]) or fecha_antes != x.r_ini.normalize()
        else:
            nuevas.append({"FECHA": x.r_ini.normalize(), "PROCESO": proc_de[x["sub"]], "SUPROCESO": x["sub"],
                           "ini_t": x.r_ini.time(), "inicio": x.r_ini,
                           "fin": x.r_fin if x.usar_fin_ritm else pd.NaT,
                           "dur_calc_s": x.dur_ritm_s if x.usar_fin_ritm else float("nan"),
                           "dur_difiere": False, "fecha_corregida": False,
                           "ritm_parche": "agregada" if x.usar_fin_ritm else "agregada_sin_fin",
                           "fecha_original": pd.NaT})
    print(f"parches RITM: {int((d.ritm_parche == 'reemplazada').sum())} filas reemplazadas, {len(nuevas)} agregadas")
    return pd.concat([d, pd.DataFrame(nuevas)], ignore_index=True)


# Calendario operativo del banco (oct-2026). Sus tipos se recalculan con las mismas reglas sobre nuestro calendario
# laboral (verificado con los datos), así que coincide con el archivo salvo en las fechas donde los datos lo
# contradicen (ver `comparar_operativo`). La DEPURACION del archivo no se usa: aplica solo al proceso de reposting
# (página futura), no a TPH, colas ni procesos críticos; esos fines de semana se tratan como cualquier otro.
CAL_OPERATIVO = RAIZ / "data/raw/Calendario_Operativo_Final_2024_2027.xlsx"
TIPOS_OP = ["DIA NORMAL", "PRIMER DIA LABORAL", "PRIMER BATCH", "FDS NORMAL", "FERIADO"]


def leer_operativo():
    c = pd.read_excel(CAL_OPERATIVO, sheet_name="Calendario", usecols=["Fecha", "Tipo Dia"])
    return c.set_index("Fecha")["Tipo Dia"].str.strip()


def tipo_operativo(lab):
    """Tipo de día con las definiciones del calendario operativo:
    PRIMER DIA LABORAL = hábil tras un día no hábil; PRIMER BATCH = hábil siguiente al primer día laboral;
    FERIADO = día hábil de semana que no se trabaja, y el sábado/domingo pegado a él (bloque feriado); FDS NORMAL = resto."""
    out = {}
    for i, d in enumerate(lab.index):
        prev = lab.iloc[i - 1] if i > 0 else True
        prev2 = lab.iloc[i - 2] if i > 1 else True
        if lab.iloc[i]:
            out[d] = "PRIMER DIA LABORAL" if not prev else ("PRIMER BATCH" if not prev2 else "DIA NORMAL")
        elif d.weekday() < 5:
            out[d] = "FERIADO"
        else:
            a = i
            while a > 0 and not lab.iloc[a - 1]:
                a -= 1
            b = i
            while b < len(lab) - 1 and not lab.iloc[b + 1]:
                b += 1
            out[d] = "FERIADO" if any(x.weekday() < 5 for x in lab.index[a:b + 1]) else "FDS NORMAL"
    return pd.Series(out)


def comparar_operativo(tipo_op):
    """Fechas en que nuestro tipo de día difiere del archivo del banco (se revisaron una por una contra TPH y batch)."""
    op = leer_operativo()
    op = op[op != "DEPURACION"]          # sin depuración: no aplica a estos datos
    j = pd.DataFrame({"archivo": op, "nuestro": tipo_op.reindex(op.index)})
    return j[j.archivo != j.nuestro]


def calendario(desde, hasta):
    h = holidays.country_holidays("EC", years=range(desde.year, hasta.year + 1))
    trasladados = set()
    for f, nombre in h.items():                      # fecha original de un feriado que tiene fecha «observado»
        if "observado" in nombre.lower() and not any(k in nombre.lower() for k in ("año nuevo", "navidad", "difuntos")):
            base = nombre.lower().replace("(observado)", "").strip()
            trasladados |= {g for g, n in h.items() if n.lower().strip() == base and g.year == f.year and g != f and g.weekday() < 5}
    feri = (set(h) - (trasladados - TRASLADO_NO_APLICADO) | FERIADOS_EXTRA) - NO_FERIADOS
    dias = pd.date_range(desde, hasta, freq="D")
    lab = pd.Series([d.weekday() < 5 and d.date() not in feri for d in dias], index=dias)
    prev_lab = lab.shift(1, fill_value=True)
    # primer día = laborable cuyo día calendario anterior no lo era
    primer = lab & ~prev_lab
    # primer batch = el siguiente día laborable al primer día no es necesario: el batch del
    # primer día corre en la madrugada del día calendario siguiente (ver `tipo_dia_batch`)
    tipo = pd.Series(
        [("no laborable" if not l else "primer dia" if p else DIAS[d.weekday()])
         for d, l, p in zip(dias, lab, primer)], index=dias)
    return pd.DataFrame({"laborable": lab, "tipo_dia": tipo, "tipo_op": tipo_operativo(lab)})


def main():
    d = pd.read_excel(XLSX, sheet_name=0, usecols=lambda c: c.strip().upper() not in PROHIBIDAS)
    d.columns = [c.strip() for c in d.columns]
    n0 = len(d)
    d["ini_t"] = d["HORA INICIO"].map(a_time)
    d["fin_t"] = d["HORA FIN"].map(a_time)
    sec = lambda t: t.hour * 3600 + t.minute * 60 + t.second
    d["ini_s"] = d["ini_t"].map(sec)
    d["fin_s"] = d["fin_t"].map(sec)
    d["fecha_original"] = d["FECHA"]
    d["inicio"] = d["FECHA"] + pd.to_timedelta(d["ini_s"], unit="s")
    corregir_fechas(d)
    # duración desde inicio/fin (módulo 24 h); TIEMPO_SEGUNDOS trae valores de 24-25 h
    d["dur_calc_s"] = (d["fin_s"] - d["ini_s"]) % 86400
    d["dur_difiere"] = (d["dur_calc_s"] - d["TIEMPO_SEGUNDOS"]).abs() > 120
    d["fin"] = d["inicio"] + pd.to_timedelta(d["dur_calc_s"], unit="s")
    d = aplicar_parches_ritm(d)

    # día de negocio
    madrugada = d["ini_t"].map(lambda t: t.hour) < d["SUPROCESO"].map(CORTE_SUB).fillna(CORTE_HORA)
    d["dia_negocio"] = d["FECHA"] - pd.to_timedelta(madrugada.astype(int), unit="D")
    n_corte = int((d["dia_negocio"] > FIN_DIA_NEGOCIO).sum())
    d = d[d["dia_negocio"] <= FIN_DIA_NEGOCIO].copy()
    print(f"filas posteriores al día de negocio {FIN_DIA_NEGOCIO.date()} descartadas: {n_corte}")
    cal = calendario(pd.Timestamp("2024-01-01"), pd.Timestamp("2027-12-31"))  # cubre también las TPH (desde 2024-12)
    dif = comparar_operativo(cal["tipo_op"])
    print(f"calendario operativo del banco: {len(dif)} fechas con tipo distinto al nuestro (revisadas contra los datos):")
    print(dif.to_string())
    d["tipo_dia"] = d["dia_negocio"].map(cal["tipo_dia"])
    d["dia_negocio_no_laborable"] = ~d["dia_negocio"].map(cal["laborable"])
    d["dow"] = d["dia_negocio"].dt.weekday.map(lambda i: DIAS[i])
    d = d.rename(columns={"TIPO_DIA": "tipo_dia_original", "DIA SEMANA": "dow_original"})

    # duplicados: misma ejecución registrada dos veces (inicio a <= 10 min) -> la última
    k = ["dia_negocio", "PROCESO", "SUPROCESO"]
    d = d.sort_values(k + ["inicio"]).reset_index(drop=True)
    # las filas con fecha corregida (RITM) son ejecuciones reales distintas de las que ya había
    # en ese día de negocio, aunque difieran por minutos: no participan en la deduplicación
    sin_corr = d[~d["fecha_corregida"] & (d["ritm_parche"] == "")]
    gap = sin_corr.groupby(k)["inicio"].diff()
    es_dup = gap.notna() & (gap <= pd.Timedelta(minutes=TOL_DUP_MIN))
    # conservar la última de cada racha: se elimina la fila ANTERIOR a cada duplicado
    elim_sc = es_dup.groupby([sin_corr[c] for c in k]).shift(-1, fill_value=False)
    elim = pd.Series(False, index=d.index)
    elim.loc[elim_sc.index] = elim_sc
    n_dup = int(elim.sum())
    d = d[~elim].copy()
    d["ejec_n"] = d.groupby(k).cumcount() + 1
    d["ejec_total"] = d.groupby(k)["inicio"].transform("size")

    cols = ["dia_negocio", "PROCESO", "SUPROCESO", "ejec_n", "ejec_total", "inicio", "fin",
            "dur_calc_s", "TIEMPO_SEGUNDOS", "dur_difiere", "tipo_dia", "dow",
            "dia_negocio_no_laborable", "tipo_dia_original", "dow_original", "fecha_original", "fecha_corregida", "ritm_parche"]
    out = d[cols].rename(columns={"PROCESO": "proceso", "SUPROCESO": "suproceso",
                                  "TIEMPO_SEGUNDOS": "dur_original_s"})
    SALIDA.mkdir(parents=True, exist_ok=True)
    out.to_csv(SALIDA / "procesos_limpio.csv", index=False)
    cal.to_csv(SALIDA / "calendario_laboral.csv", index_label="fecha")

    print(f"filas entrada {n0} | duplicados eliminados {n_dup} | salida {len(out)}")
    print("ejecuciones distintas el mismo día de negocio (ejec_total>1):", int((out.ejec_total > 1).sum()))
    print("duración calculada != original (>2 min):", int(out.dur_difiere.sum()))
    print("día de negocio NO laborable:", int(out.dia_negocio_no_laborable.sum()))
    print(out.tipo_dia.value_counts().to_string())


if __name__ == "__main__":
    main()
