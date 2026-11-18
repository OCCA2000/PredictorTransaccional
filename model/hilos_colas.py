"""Hilos por cola: referencia medida de DEP (AnalisisColas.xlsx) y estimación de las demás colas con tiempo
de servicio por hora (data/processed/w_por_cola_hora.csv, generado por w_por_cola_hora.py).

1) DEP, medida. El Excel trae un día (2026-09-15) segundo a segundo de la cola «IN», que es DEP en el CSV de
   colas (se verifica aquí: mismas transacciones por hora de 06:00 a 19:00). Su fórmula, por hora:
       hilos requeridos = TECHO( MAX( concurrencia P90, TPS P90 x W P90 / 1000 ) x (1 + margen) )
   y la demanda se escala en proporción (W no cambia). De esa muestra se guardan, por hora del día:
       rafaga = TPS P90 / TPS promedio      w90 = tiempo de servicio P90 (ms)
       f_p90  = MAX(concurrencia P90, TPS P90 x W P90 / 1000) / (TPS P90 x W P90 / 1000)
       f_max  = concurrencia máxima / (TPS P90 x W P90 / 1000)
       k      = hilos por transacción/hora = rafaga x w90 / 1000 x f / 3600   ->  hilos(TPH) = TPH x k
   En el día de muestra reproduce el Excel. Las horas 00-04 no están completas: usan las razones de las 05:00.

2) Demás colas, estimadas. Se leen tiempos por cola y hora de w_por_cola_hora.csv (que procesa el histograma
   TXN_GRUPO_X_TIEMPO_RESPUESTA_CONSOLIDADO.csv con la mezcla real de transacciones de cada hora). El salto
   de la carga media al P90 y al máximo se calibra con DEP:
       L = TPS x w_hora / 1000          hilos(L) = L + z x RAIZ(L)
   z resulta estable entre horas con cargas muy distintas (P90: 2,2-3,0; máximo: 6,6-12,9), a diferencia de un
   factor proporcional; se usa el mayor de las horas diurnas. Se valida que, aplicado a todas las colas con su
   tiempo por hora, da cobertura adecuada (nunca por debajo de lo requerido).

Entrada: w_por_cola_hora.csv debe existir (ejecutar w_por_cola_hora.py primero).
Salidas: data/processed/hilos_referencia_dep.csv, hilos_colas.csv, hilos_calibracion.csv
"""
import math
from pathlib import Path
import numpy as np
import pandas as pd
import openpyxl

RAIZ = Path(__file__).resolve().parents[1]
XLSX = RAIZ / "data/raw/AnalisisColas.xlsx"
SALIDA = RAIZ / "data/processed"
COLA_EXCEL, COLA_MEDIDA = "IN", "DEP"
HILOS_MAX = 99
# Hilos activos de forma permanente por cola (consola de BANCS, octubre de 2026). ELM y VIS (1 hilo cada una)
# no aparecen en el CSV de transacciones; SDV (2) no tiene actividad.
HILOS_ACTIVOS = {"DEP": 90, "DP1": 90, "DP2": 50, "DP3": 60, "DP4": 10, "CIF": 90, "CI1": 90, "CI2": 60, "CI3": 90,
                 "LON": 20, "OMN": 90, "OM1": 90, "OM2": 80, "PFM": 1, "GEN": 46, "CAS": 20, "VST": 3, "FN1": 5, "OLR": 6}
# Códigos de dos letras del resumen de tiempos -> cola del CSV (deducido por volumen y participación del día).
# Se ignoran la fila sin código, los códigos 00-06 y TF (no está entre las colas con hilos de la consola).
CODIGOS = {"BR": "LON", "C1": "CI1", "C2": "CI2", "C3": "CI3", "CA": "CAS", "CI": "CIF", "D1": "DP1", "D2": "DP2", "D3": "DP3",
           "D4": "DP4", "GA": "FN1", "GL": "GEN", "IN": "DEP", "O1": "OM1", "O2": "OM2", "OL": "OLR", "OM": "OMN", "SP": "PFM", "VS": "VST"}
REDONDEO_MS = 5                   # el tiempo viene con 2 decimales de segundo: ±5 ms
REGIMEN_DESDE = "2026-03-10"      # último cambio de enrutamiento (modelo_colas.ESCALONES)
PRIMERA_HORA_COMPLETA = 5
HORAS_DIURNAS = list(range(6, 20))
T, W, C, CMAX, BUSY = "TRANSACCIONES_INICIADAS", "PROMEDIO_TXN_TIME_MSS", "CONCURRENCIA_MEDIA_P90_CALENDARIO_10MS", "CONCURRENCIA_MEDIA_MAXIMA_10MS", "BUSY_MS_SEGUNDO"


def leer_muestra():
    ws = openpyxl.load_workbook(XLSX, data_only=True, read_only=True)["DatosentradaSeg"]
    filas = list(ws.iter_rows(values_only=True))
    d = pd.DataFrame(filas[1:], columns=filas[0])
    assert d.COLA.nunique() == 1 and d.COLA.iloc[0] == COLA_EXCEL and d.FECHA.nunique() == 1
    d["h"] = d.HORA.astype(str).str[:2].astype(int)
    return d, pd.Timestamp(str(d.FECHA.iloc[0]))


def referencia(d):
    p90 = lambda s: float(np.percentile(s, 90))
    r = d.groupby("h").apply(lambda x: pd.Series({"trx": x[T].sum(), "segundos": len(x), "tps90": p90(x[T]), "w90": p90(x[W]),
                                                  "w_prom": x[BUSY].sum() / x[T].sum(), "c90": p90(x[C]), "cmax": x[CMAX].max()}))
    r["tps_prom"] = r.trx / 3600
    r["carga"] = r.tps_prom * r.w_prom / 1000          # hilos ocupados en promedio en la hora
    r["little"] = r.tps90 * r.w90 / 1000
    r["base"] = np.maximum(r.c90, r.little)
    r["rafaga"], r["f_p90"], r["f_max"] = r.tps90 / r.tps_prom, r.base / r.little, r.cmax / r.little
    r["completa"] = r.segundos == 3600
    r = r.reindex(range(24))
    rz = ["rafaga", "w90", "f_p90", "f_max"]
    r.loc[r.index < PRIMERA_HORA_COMPLETA, rz] = r.loc[PRIMERA_HORA_COMPLETA, rz].values
    r["completa"] = r.completa.fillna(False) & (r.index >= PRIMERA_HORA_COMPLETA)
    return r


def hilos(tph, r, h, criterio="f_p90"):
    """Hilos (sin redondear) que pide un volumen horario `tph` a la hora `h` con las razones de la muestra."""
    return tph / 3600 * r.at[h, "rafaga"] * r.at[h, "w90"] / 1000 * r.at[h, criterio]


def calibrar(r):
    """Salto de la carga media al P90 y al máximo en las horas diurnas de DEP (se toma el mayor de cada uno)."""
    x = r.loc[HORAS_DIURNAS]
    return {"z_p90": float(((x.base - x.carga) / np.sqrt(np.maximum(x.carga, 0.1))).max()),
            "z_max": float(((x.cmax - x.carga) / np.sqrt(np.maximum(x.carga, 0.1))).max())}


def hilos_estimados(tph, w_hora, z, criterio="p90"):
    """Hilos (sin redondear) para un volumen horario con tiempo de servicio de la hora `w_hora`."""
    L = np.asarray(tph, float) / 3600 * np.asarray(w_hora, float) / 1000
    return L + z[f"z_{criterio}"] * np.sqrt(np.maximum(L, 0.1))


def leer_w_por_hora():
    """Lee tiempos por cola y hora de w_por_cola_hora.csv (generado por w_por_cola_hora.py).
    Retorna tabla con w_ms diario (promediado) por cola."""
    if not (SALIDA / "w_por_cola_hora.csv").exists():
        raise SystemExit(f"falta w_por_cola_hora.csv: ejecutar antes model/w_por_cola_hora.py")
    w = pd.read_csv(SALIDA / "w_por_cola_hora.csv", sep=";")
    # Tiempo promedio por cola (ponderado por transacciones)
    g = w.groupby("COLA")
    w_ms = (g.TRX * g.W_MIX).sum() / g.TRX.sum()
    r = pd.DataFrame({"codigo": g["COLA"].first(), "dias_tempo": w.FECHA_HORA.str[:10].nunique(), "w_ms": w_ms})
    r = r.set_index(r.codigo)
    return r


def main():
    d, fecha = leer_muestra()
    r = referencia(d)
    g = pd.read_csv(SALIDA / "colas_limpio.csv", parse_dates=["hora"]).set_index("hora")

    # la cola del Excel es DEP: mismas transacciones por hora de 06:00 a 19:00 (diferencias de 0 a 11). De noche
    # y de madrugada los conteos difieren, así que sus razones se aplican igual pero quedan marcadas como no coincidentes.
    real = g.loc[(g.index >= fecha) & (g.index < fecha + pd.Timedelta(days=1)), COLA_MEDIDA]
    real.index = real.index.hour
    comp = r.index[r.completa]
    r["coincide"] = ((real.reindex(r.index) / r.trx - 1).abs() < 1e-3).fillna(False)
    assert r.loc[HORAS_DIURNAS, "coincide"].all(), "la cola del Excel no coincide con DEP en el CSV de colas"

    # réplica de los números del Excel
    p90 = lambda s: float(np.percentile(s, 90))
    base = max(p90(d[C]), p90(d[T]) * p90(d[W]) / 1000)
    assert math.ceil(base) == 14 and math.ceil(base * 1.9) == 26, "no se reproduce «hilos requeridos» del Excel"
    por_hora = pd.Series({h: math.ceil(round(hilos(r.at[h, "trx"] * 1.9, r, h), 9)) for h in comp})
    assert por_hora.max() == 33 and por_hora.idxmax() == 18, "no se reproduce la hora más exigente del simulador (+90 %)"

    r["k_p90"] = r.rafaga * r.w90 / 1000 * r.f_p90 / 3600
    r["k_max"] = r.rafaga * r.w90 / 1000 * r.f_max / 3600
    ref = r[["trx", "tps_prom", "tps90", "w90", "w_prom", "carga", "c90", "cmax", "base", "rafaga", "f_p90", "f_max", "k_p90", "k_max", "completa", "coincide"]].copy()
    ref.insert(0, "fecha_muestra", fecha.date())
    ref.to_csv(SALIDA / "hilos_referencia_dep.csv", index_label="hora", float_format="%.12g")

    # calibración del salto por ráfagas con DEP
    cal = calibrar(r)
    # tiempos por cola (de w_por_cola_hora.csv)
    tm = leer_w_por_hora()
    # guardar calibración
    meses = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
    fx = lambda x: f"{x.day}-{meses[x.month - 1]}"
    pd.DataFrame([dict(cal, fecha_muestra=fecha.date())]).to_csv(SALIDA / "hilos_calibracion.csv", index=False, float_format="%.6g")
    
    # validación: el método estimado, aplicado a DEP con el tiempo del Excel, no queda por debajo de lo medido
    w_dep = d[BUSY].sum() / d[T].sum()
    x = r.loc[HORAS_DIURNAS]
    e90 = hilos_estimados(x.trx, w_dep, cal, "p90")
    emx = hilos_estimados(x.trx, w_dep, cal, "max")
    assert (e90 >= x.base - 1e-9).all() and (emx >= x.cmax - 1e-9).all(), "el método estimado subestima a DEP"

    # indicadores de todas las colas con el pico real del régimen vigente
    u = g[g.index >= REGIMEN_DESDE]
    filas = []
    for q, n in HILOS_ACTIVOS.items():
        mx, cuando = float(u[q].max()), u[q].idxmax()
        w_ms_q = tm.loc[q, "w_ms"] if q in tm.index else w_dep
        filas.append(dict(cola=q, hilos_activos=n, hilos_max=HILOS_MAX, tph_pico=mx, tps_pico=mx / 3600, pico_cuando=cuando,
                          tps_por_hilo=mx / 3600 / n, medida=q == COLA_MEDIDA, w_ms=w_ms_q))
    t = pd.DataFrame(filas)
    w_hi = t.w_ms + REDONDEO_MS
    t["carga_pico"] = t.tps_pico * w_hi / 1000
    t["hilos_p90_pico"] = np.ceil(hilos_estimados(t.tph_pico, w_hi, cal, "p90") - 0.005)
    t["hilos_max_pico"] = np.ceil(hilos_estimados(t.tph_pico, w_hi, cal, "max") - 0.005)
    t.to_csv(SALIDA / "hilos_colas.csv", index=False, float_format="%.4f")

    print(f"muestra: cola {COLA_EXCEL} = {COLA_MEDIDA}, {fecha.date()}, {len(d)} segundos | réplica del Excel: 14 hilos base, 26 con +90 %, 33 a las 18:00")
    print(f"DEP día de muestra: hilos requeridos (P90) máx {math.ceil(r.base.max())} | W P90 {r.w90[comp].min():.0f}-{r.w90[comp].max():.0f} ms")
    print(f"tiempos: de w_por_cola_hora.csv (ejecutar w_por_cola_hora.py primero)")
    print("calibración con DEP (horas diurnas):", {k: round(v, 2) for k, v in cal.items()})
    print("validación en DEP, estimado / medido (P90):", f"{(e90 / x.base).min():.2f}-{(e90 / x.base).max():.2f}", " | máximo:", f"{(emx / x.cmax).min():.2f}-{(emx / x.cmax).max():.2f}")
    print(t.sort_values("hilos_max_pico", ascending=False)[["cola", "hilos_activos", "w_ms", "tps_pico", "carga_pico", "hilos_p90_pico", "hilos_max_pico"]].round(1).to_string(index=False))
    return ref, t


if __name__ == "__main__":
    main()
