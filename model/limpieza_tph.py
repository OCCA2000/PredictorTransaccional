"""Limpieza de RESUMEN_CONTEO_POR_HORA_TIPO_HISTORICO_2024_2026.csv -> data/processed/tph_limpio.csv

Series: Financiero y No Financiero, transacciones por hora (TPH). Reglas:
  - Se usa desde 2024-12-19: antes solo hay un día aislado (2024-11-06) y 42 días sin datos.
  - Rejilla horaria completa. Cada hora recibe un `estado`:
      ok          dato presente y plausible
      batch       hora ausente entre 23:00 y 04:59 en un día con datos: ventana del batch nocturno
                  (las noches laborables tienen 2-4 horas ausentes, las no laborables casi nunca)
      sin_dato    hora ausente por falla de extracción (día completo ausente u horas diurnas ausentes)
      caida       hora diurna presente con un valor implausiblemente bajo (<25% de lo esperado en
                  alguna serie y <45% en las dos) en un día NO feriado: se pone NaN, original en *_orig
  - Los feriados de Ecuador y los fines de semana NO se limpian: su caída de volumen es real.
  - Las horas 21-23 con volumen bajo cuando arranca el batch son comportamiento real: se conservan.
"""
from pathlib import Path
import pandas as pd
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
CSV = RAIZ / "data/raw/RESUMEN_CONTEO_POR_HORA_TIPO_HISTORICO_2024_2026.csv"
SALIDA = RAIZ / "data/processed"
INICIO = "2024-12-19"
HORAS_BATCH = {23, 0, 1, 2, 3, 4}
UMBRAL_UNA, UMBRAL_AMBAS = 0.25, 0.45
H_DIURNAS = range(7, 21)  # 07:00-20:59, donde una caída no puede ser el arranque del batch


def main():
    t = pd.read_csv(CSV, sep=";", encoding="utf-8-sig")
    # Soportar tanto "HORA" como "FECHA_HORA"
    col_hora = "FECHA_HORA" if "FECHA_HORA" in t.columns else "HORA"
    t["hora"] = pd.to_datetime(t[col_hora])
    t = t.drop(columns=col_hora)
    # Normalizar nombres de columnas (puede ser "Financiero"/"No Financiero" o "FINANCIERO"/"NO FINANCIERO")
    rename_map = {}
    for old, new in [("Financiero", "fin"), ("FINANCIERO", "fin"), ("No Financiero", "nofin"), ("NO FINANCIERO", "nofin")]:
        if old in t.columns:
            rename_map[old] = new
    t = t.rename(columns=rename_map).set_index("hora").sort_index()
    n_raw = len(t)
    assert not t.index.duplicated().any()
    t = t[t.index >= INICIO]

    g = pd.DataFrame(index=pd.date_range(t.index.min().normalize(), t.index.max().normalize() + pd.Timedelta(hours=23), freq="h"))
    g = g.join(t)
    g.index.name = "hora"
    g["fin_orig"], g["nofin_orig"] = g["fin"], g["nofin"]
    g["fecha"] = g.index.normalize()
    g["h"] = g.index.hour
    g["dow"] = g.index.dayofweek

    # calendario laboral (feriados de Ecuador) generado por limpieza_procesos.py
    cal = pd.read_csv(SALIDA / "calendario_laboral.csv", parse_dates=["fecha"]).set_index("fecha")
    g["es_feriado"] = g["fecha"].map(lambda d: d in cal.index and (not cal.at[d, "laborable"]) and d.dayofweek < 5)
    g["es_laborable"] = g["fecha"].map(lambda d: bool(cal.at[d, "laborable"]) if d in cal.index else d.dayofweek < 5)

    # estado de las horas ausentes
    dias_con_datos = set(t.index.normalize())
    falta = g["fin"].isna() & g["nofin"].isna()
    g["estado"] = "ok"
    en_ventana = g["h"].isin(HORAS_BATCH) & g["fecha"].isin(dias_con_datos)
    g.loc[falta & en_ventana, "estado"] = "batch"
    g.loc[falta & ~en_ventana, "estado"] = "sin_dato"

    # caídas: comparar con la mediana de la misma (día de semana, hora) en días laborables normales
    ref = g[(g.estado == "ok") & g["es_laborable"] & ~g["es_feriado"]]
    for c in ("fin", "nofin"):
        med = ref.groupby(["dow", "h"])[c].median()
        g["ratio_" + c] = g[c] / pd.MultiIndex.from_frame(g[["dow", "h"]]).map(med)
    cond = (g.estado == "ok") & g["h"].isin(H_DIURNAS) & ~g["es_feriado"] & g["es_laborable"] \
        & (((g.ratio_fin < UMBRAL_UNA) | (g.ratio_nofin < UMBRAL_UNA)) & (g.ratio_fin < UMBRAL_AMBAS) & (g.ratio_nofin < UMBRAL_AMBAS))
    g.loc[cond, "estado"] = "caida"
    g.loc[g.estado == "caida", ["fin", "nofin"]] = np.nan

    cols = ["fin", "nofin", "fin_orig", "nofin_orig", "estado", "es_feriado", "es_laborable", "ratio_fin", "ratio_nofin"]
    out = g[cols].copy()
    out[["ratio_fin", "ratio_nofin"]] = out[["ratio_fin", "ratio_nofin"]].round(3)
    out.to_csv(SALIDA / "tph_limpio.csv")

    # calidad por día
    d = g.groupby("fecha").agg(horas_ok=("estado", lambda s: int((s == "ok").sum())),
                               horas_batch=("estado", lambda s: int((s == "batch").sum())),
                               horas_sin_dato=("estado", lambda s: int((s == "sin_dato").sum())),
                               horas_caida=("estado", lambda s: int((s == "caida").sum())),
                               fin=("fin", "sum"), nofin=("nofin", "sum"))
    d["dow"] = d.index.dayofweek
    d["es_feriado"] = g.groupby("fecha")["es_feriado"].first()
    d["es_laborable"] = g.groupby("fecha")["es_laborable"].first()
    d["dia_completo"] = (d.horas_sin_dato == 0) & (d.horas_caida == 0)
    d.to_csv(SALIDA / "calidad_dias_tph.csv")

    print(f"filas crudas {n_raw} | desde {INICIO}: {len(t)} | rejilla {len(g)} horas, {g['fecha'].nunique()} días")
    print(g.estado.value_counts().to_string())
    print(f"días completos (sin sin_dato ni caida): {int(d.dia_completo.sum())} de {len(d)}")
    print("días con alguna hora sin_dato:", int((d.horas_sin_dato > 0).sum()), "| días con caída:", int((d.horas_caida > 0).sum()))
    return out, d


if __name__ == "__main__":
    main()
