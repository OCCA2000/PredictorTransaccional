"""Tiempo de servicio por cola, día y hora, con el catálogo Cat_TXN_Cola_ACT.xlsx (transacción -> cola).

Qué hay en los archivos de entrada (data/Archivos de entrada - Corte 2026-10-27):
  - TXN_X_HORA_GRUPO_X_CANAL_TIEMPO_PROMEDIO.csv NO trae un tiempo por hora: repite el mismo valor por transacción en todas las
    horas, y está contaminado (1045, la mitad de DEP, sale con -4,37 ms). De ese archivo solo sirve el conteo por hora.
  - TXN_GRUPO_X_TIEMPO_RESPUESTA_CONSOLIDADO.csv trae la distribución diaria del tiempo por transacción (TXN_TIME_MSS -> TRX_COUNT),
    con valores imposibles (negativos: transacciones que cruzan la medianoche, hasta -86 399 890 ms; y de horas). Se limpia.

Método:
  1. w(txn, día) = promedio de la distribución limpia (0 <= tiempo < TIEMPO_MAX_MS).
  2. W(cola, día) = ponderado por transacciones; percentiles P50/P90/P99 de la distribución de la cola.
  3. W(cola, hora) = SUMA(TRX(txn, hora) x w(txn, día)) / SUMA(TRX(txn, hora)): la mezcla de transacciones de cada hora
     (conteo real por hora) con el tiempo propio de cada transacción ese día. Si una transacción no tiene tiempo ese día se
     usa su mediana histórica.
  4. Ruteo real: si existe ruteo_txn.csv (rafaga_colas.py, del día con detalle por transacción), los códigos que corren en parte
     en otra cola se reparten con esas fracciones (p. ej. la 21031, de GEN en el catálogo, corre 75 % en TF). Así TRX y W_MIX son
     los de la cola donde de verdad corren; TRX_COLA_CSV sigue siendo el conteo por catálogo (el que se pronostica).

Salidas (data/processed, separador «;»):
  w_por_cola_dia.csv     FECHA, COLA, TRX, W_MEDIO, W_P50, W_P90, W_P99, PCT_DESCARTADO   (por catálogo)
  w_por_cola_hora.csv    FECHA_HORA, COLA, TRX, W_MIX, TXN_TOP, PCT_PESO_TOP, TRX_COLA_CSV   (por cola real)
  w_validacion_cruce.csv transacciones del cruce por catálogo vs. RESUMEN_CONTEO_POR_HORA_COLA.csv, por cola
  w_txn_sin_cola.csv     códigos sin cola en el catálogo
"""
from pathlib import Path
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
ENTRADA = RAIZ / "data/Archivos de entrada - Corte 2026-10-27"
SALIDA = RAIZ / "data/processed"
CHUNK = 5_000_000
TIEMPO_MAX_MS = 300_000       # 5 min; por encima son cruces de hora/día, no tiempo de servicio
RUTEO = SALIDA / "ruteo_txn.csv"


def leer_catalogo():
    c = pd.read_excel(ENTRADA / "Cat_TXN_Cola_ACT.xlsx").rename(columns={"TXN": "txn", "Cola": "cola"})
    return c.set_index("txn").cola


def distribucion(cat):
    """Distribución diaria por (día, cola, transacción, tiempo), con el tiempo ya validado."""
    partes = []
    for b in pd.read_csv(ENTRADA / "TXN_GRUPO_X_TIEMPO_RESPUESTA_CONSOLIDADO.csv", sep=";", dtype={"TRAN_CODE": str}, chunksize=CHUNK):
        b["txn"] = b.TRAN_CODE.astype(int)
        b["valido"] = (b.TXN_TIME_MSS >= 0) & (b.TXN_TIME_MSS < TIEMPO_MAX_MS)
        partes.append(b.groupby(["FECHA", "txn", "TXN_TIME_MSS", "valido"], as_index=False).TRX_COUNT.sum())
    d = pd.concat(partes).groupby(["FECHA", "txn", "TXN_TIME_MSS", "valido"], as_index=False).TRX_COUNT.sum()
    d["cola"] = d.txn.map(cat)
    return d


def percentil(h, q):
    """h: histograma (FECHA, cola, TXN_TIME_MSS, TRX_COUNT) ya ordenado por tiempo dentro de cada grupo."""
    c = h.groupby(["FECHA", "cola"]).TRX_COUNT.cumsum()
    tot = h.groupby(["FECHA", "cola"]).TRX_COUNT.transform("sum")
    return h[c >= q * tot].groupby(["FECHA", "cola"]).TXN_TIME_MSS.first()


def tiempos_dia(d):
    v = d[d.valido].assign(peso=lambda x: x.TXN_TIME_MSS * x.TRX_COUNT)
    wt = v.groupby(["FECHA", "txn"]).agg(trx_v=("TRX_COUNT", "sum"), peso=("peso", "sum"))
    wt["w"] = wt.peso / wt.trx_v
    wt = wt.reset_index()

    dc = d.dropna(subset=["cola"])
    tot = dc.groupby(["FECHA", "cola"]).TRX_COUNT.sum().rename("TRX")
    vc = dc[dc.valido]
    h = vc.groupby(["FECHA", "cola", "TXN_TIME_MSS"], as_index=False).TRX_COUNT.sum().sort_values(["FECHA", "cola", "TXN_TIME_MSS"])
    g = vc.groupby(["FECHA", "cola"]).TRX_COUNT.sum().rename("trx_v").to_frame()
    g["W_MEDIO"] = vc.assign(p=vc.TXN_TIME_MSS * vc.TRX_COUNT).groupby(["FECHA", "cola"]).p.sum() / g.trx_v
    for q, n in ((.5, "W_P50"), (.9, "W_P90"), (.99, "W_P99")):
        g[n] = percentil(h, q)
    g = g.join(tot)
    g["PCT_DESCARTADO"] = 1 - g.trx_v / g.TRX
    g = g.reset_index().rename(columns={"cola": "COLA"})
    return wt, g[["FECHA", "COLA", "TRX", "W_MEDIO", "W_P50", "W_P90", "W_P99", "PCT_DESCARTADO"]]


def mezcla_horaria(cat, wt):
    partes = []
    for b in pd.read_csv(ENTRADA / "TXN_X_HORA_GRUPO_X_CANAL_TIEMPO_PROMEDIO.csv", sep=";", dtype={"TRAN_CODE": str},
                         usecols=["FECHA_HORA", "TRAN_CODE", "TRX_COUNT"], chunksize=CHUNK):
        b["txn"] = b.TRAN_CODE.astype(int)
        partes.append(b.groupby(["FECHA_HORA", "txn"], as_index=False).TRX_COUNT.sum())
    a = pd.concat(partes).groupby(["FECHA_HORA", "txn"], as_index=False).TRX_COUNT.sum()
    a["FECHA"] = a.FECHA_HORA.str[:10]
    a["cola"] = a.txn.map(cat)
    sin = a[a.cola.isna()].groupby("txn").TRX_COUNT.sum().sort_values(ascending=False)
    sin.rename("trx").to_csv(SALIDA / "w_txn_sin_cola.csv", index_label="txn")
    print(f"transacciones sin cola en el catálogo: {len(sin)} códigos, {a.loc[a.cola.isna(), 'TRX_COUNT'].sum() / a.TRX_COUNT.sum():.4%} del volumen")
    a = a.dropna(subset=["cola"]).merge(wt[["FECHA", "txn", "w"]], on=["FECHA", "txn"], how="left")
    mediana = wt.groupby("txn").w.median()
    a["w"] = a.w.fillna(a.txn.map(mediana))
    print(f"transacciones sin tiempo ni ese día ni en el histórico: {a.loc[a.w.isna(), 'TRX_COUNT'].sum() / a.TRX_COUNT.sum():.4%} del volumen")
    a["peso"] = a.TRX_COUNT * a.w
    cruce = a.groupby(["FECHA_HORA", "cola"]).TRX_COUNT.sum().rename("TRX").reset_index().rename(columns={"cola": "COLA"})

    # ruteo real (rafaga_colas.py): cada código se reparte entre las colas donde de verdad corre, según el día con detalle
    if RUTEO.exists():
        ru = pd.read_csv(RUTEO)[["txn", "cola_real", "fraccion"]]
        a = a.merge(ru, on="txn", how="left")
        a["cola"] = a.cola_real.fillna(a.cola)
        f = a.fraccion.fillna(1.0)
        a["TRX_COUNT"], a["peso"] = a.TRX_COUNT * f, a.peso * f
        print(f"ruteo real aplicado: {ru.txn.nunique()} códigos repartidos entre colas")
    else:
        print(f"sin {RUTEO.name}: las colas quedan según el catálogo (correr rafaga_colas.py)")

    p = a.dropna(subset=["w"])
    g = p.groupby(["FECHA_HORA", "cola"]).agg(trx_w=("TRX_COUNT", "sum"), peso=("peso", "sum")).reset_index()
    g["W_MIX"] = g.peso / g.trx_w
    top = p.groupby(["FECHA_HORA", "cola", "txn"], as_index=False).peso.sum().sort_values("peso", ascending=False).drop_duplicates(["FECHA_HORA", "cola"])
    g = g.merge(top.rename(columns={"txn": "TXN_TOP", "peso": "peso_top"}), on=["FECHA_HORA", "cola"])
    g["PCT_PESO_TOP"] = g.peso_top / g.peso
    tot = a.groupby(["FECHA_HORA", "cola"]).TRX_COUNT.sum().rename("TRX").reset_index()
    g = g.merge(tot, on=["FECHA_HORA", "cola"])
    return g.rename(columns={"cola": "COLA"}), cruce


def main():
    cat = leer_catalogo()
    wt, dia = tiempos_dia(distribucion(cat))
    dia.to_csv(SALIDA / "w_por_cola_dia.csv", index=False, sep=";", float_format="%.4f")
    print(f"descartado de la distribución por tiempo inválido: {(dia.PCT_DESCARTADO * dia.TRX).sum() / dia.TRX.sum():.4%} de las transacciones")

    g, cruce = mezcla_horaria(cat, wt)
    r = pd.read_csv(ENTRADA / "RESUMEN_CONTEO_POR_HORA_COLA.csv", sep=";").melt("FECHA_HORA", var_name="COLA", value_name="TRX_COLA_CSV")
    g = g.merge(r, on=["FECHA_HORA", "COLA"], how="left")
    g[["FECHA_HORA", "COLA", "TRX", "W_MIX", "TXN_TOP", "PCT_PESO_TOP", "TRX_COLA_CSV"]].to_csv(SALIDA / "w_por_cola_hora.csv", index=False, sep=";", float_format="%.4f")

    # validación del catálogo (antes del ruteo) contra el conteo por cola que usa el modelo
    c = cruce.merge(r, on=["FECHA_HORA", "COLA"], how="left")
    h = c.dropna(subset=["TRX_COLA_CSV"])
    v = c.groupby("COLA").agg(trx_cruce=("TRX", "sum"), trx_csv=("TRX_COLA_CSV", "sum"))
    v["ratio"] = v.trx_cruce / v.trx_csv
    v["horas"] = h.groupby("COLA").size()
    v["horas_dif_mayor_1pct"] = h[(h.TRX / h.TRX_COLA_CSV - 1).abs() > 0.01].groupby("COLA").size()
    v.fillna(0).to_csv(SALIDA / "w_validacion_cruce.csv", float_format="%.4f")
    print(v.fillna(0).round(3).to_string())


if __name__ == "__main__":
    main()
