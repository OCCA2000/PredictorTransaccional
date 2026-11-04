"""Limpieza de RESUMEN_CONTEO_POR_HORA_COLA_HISTORICO_2024_2026.csv -> data/processed/colas_limpio.csv

Las mismas transacciones de TPH, repartidas por cola. Es la misma fuente que el CSV por tipo: la suma de las
colas es igual a Financiero + No Financiero en todas las horas (se verifica aquí). Reglas:
  - Misma rejilla y mismo `estado` por hora que tph_limpio.csv (ok / batch / sin_dato / caida): correr antes
    limpieza_tph.py. El valor de la cola solo se conserva en las horas `ok`.
  - SDV no tiene actividad (máximo 10 transacciones en una hora): se descarta.
  - colas_excluidas.csv lista las horas que son reales pero NO se usan para entrenar (el dato se conserva):
      desvio      día en que el nivel de la cola se aparta más de ×1,42 (o ÷1,42) de la mediana de los días
                  de su mismo tipo anteriores Y siguientes (5 laborables, o 3 sábados / 3 domingos), hacia el
                  mismo lado. Son desvíos temporales de tráfico entre colas (p. ej. 2026-07-27 y 28: OM1 a
                  40 %, OMN y CI3 a ×3; domingo 2026-09-20: OM2 a 16 %, OMN a ×7). En los últimos días de la
                  serie, sin «después», basta un salto de ×1,65 contra lo anterior.
      caida_cola  hora diurna (07-20) de día laborable con la cola por debajo del 30 % de la mediana de la misma
                  hora en los 10 días laborables vecinos, mientras el total sigue por encima del 70 %.
    Los cambios de nivel permanentes (escalones) no se excluyen: van como variable en modelo_colas.py.
"""
from pathlib import Path
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
CSV = RAIZ / "data/raw/RESUMEN_CONTEO_POR_HORA_COLA_HISTORICO_2024_2026.csv"
SALIDA = RAIZ / "data/processed"
SIN_ACTIVIDAD = ["SDV"]
RESIDUALES = ["FN1", "DP4", "OLR", "VST"]   # < 0,02 % del total cada una: sin reglas de exclusión
UMBRAL_DESVIO = 0.35                        # |log| contra ambos lados
UMBRAL_DESVIO_FINAL = 0.50                  # |log| contra lo anterior, para los últimos días de la serie
UMBRAL_CAIDA, UMBRAL_TOTAL = 0.30, 0.70
H_DIURNAS = range(7, 21)
MIN_HORAS_DIA = 18


def main():
    c = pd.read_csv(CSV, sep=";", encoding="utf-8-sig")
    c["hora"] = pd.to_datetime(c["HORA"])
    c = c.drop(columns="HORA").set_index("hora").sort_index()
    assert not c.index.duplicated().any()
    g = pd.read_csv(SALIDA / "tph_limpio.csv", parse_dates=["hora"]).set_index("hora")

    # misma fuente que el CSV por tipo: la suma de colas debe ser Financiero + No Financiero
    j = g[["fin_orig", "nofin_orig"]].join(c, how="inner").dropna(subset=["fin_orig"])
    dif = (j[c.columns].sum(axis=1) - j.fin_orig - j.nofin_orig).abs()
    assert len(j) == g.fin_orig.notna().sum() and dif.max() == 0, f"las colas no cuadran con TPH (dif. máx. {dif.max()})"

    colas = [q for q in c.columns if q not in SIN_ACTIVIDAD]
    orden = c.loc[c.index >= g.index.min(), colas].sum().sort_values(ascending=False).index.tolist()
    out = g[["estado", "es_feriado", "es_laborable"]].join(c[orden])
    out.loc[out.estado != "ok", orden] = np.nan
    out[orden + ["estado"]].to_csv(SALIDA / "colas_limpio.csv")

    ok = out[out.estado == "ok"]
    excl = []
    evaluar = [q for q in orden if q not in RESIDUALES]

    # desvíos: días completos con el nivel de la cola fuera de lugar contra ambos lados. Cada día se compara con
    # días de su mismo tipo: laborables entre sí (5 por lado), sábados con sábados y domingos con domingos (3).
    n = ok.groupby(ok.index.normalize()).size()
    d = ok.groupby(ok.index.normalize())[evaluar].sum()[n >= MIN_HORAS_DIA]
    lab = ok.groupby(ok.index.normalize()).es_laborable.first().reindex(d.index).values
    grupos = [(d[lab], 5), (d[~lab & (d.index.dayofweek == 5)], 3), (d[~lab & (d.index.dayofweek == 6)], 3)]
    for dg, v in grupos:
        dl = np.log(dg.clip(lower=1))
        antes, desp = dl.shift(1).rolling(v).median(), dl.shift(-v).rolling(v).median()
        desvio = ((dl - antes).abs() > UMBRAL_DESVIO) & ((dl - desp).abs() > UMBRAL_DESVIO) & (np.sign(dl - antes) == np.sign(dl - desp))
        # últimos días de la serie: aún no hay «después» con qué comparar; se exige un salto mayor contra lo anterior
        desvio |= desp.isna() & ((dl - antes).abs() > UMBRAL_DESVIO_FINAL)
        for q in evaluar:
            for dia in desvio.index[desvio[q]]:
                h = ok.index[ok.index.normalize() == dia]
                excl.append(pd.DataFrame({"hora": h, "cola": q, "motivo": "desvio", "ratio": round(float(np.exp(dl.at[dia, q] - antes.at[dia, q])), 2)}))

    # caídas propias de una cola en horas diurnas de días laborables
    hd = ok[ok.es_laborable & ~ok.es_feriado & ok.index.hour.isin(H_DIURNAS)]
    tot = hd[orden].sum(axis=1)
    for _, x in hd.groupby(hd.index.hour):
        rt = tot.loc[x.index] / tot.loc[x.index].rolling(11, center=True, min_periods=6).median()
        for q in evaluar:
            r = x[q] / x[q].rolling(11, center=True, min_periods=6).median()
            m = (r < UMBRAL_CAIDA) & (rt > UMBRAL_TOTAL)
            if m.any():
                excl.append(pd.DataFrame({"hora": r.index[m], "cola": q, "motivo": "caida_cola", "ratio": r[m].round(2).values}))
    excl = pd.concat(excl).drop_duplicates(["hora", "cola"]).sort_values(["hora", "cola"])
    excl.to_csv(SALIDA / "colas_excluidas.csv", index=False)

    print(f"colas: {len(orden)} ({', '.join(orden)}) | descartadas: {SIN_ACTIVIDAD}")
    print(f"horas ok: {len(ok)} | suma de colas = Financiero + No Financiero en las {len(j)} horas con dato")
    dias = excl[excl.motivo == "desvio"].assign(dia=lambda e: e.hora.dt.normalize()).groupby("cola").dia.nunique()
    print("días de desvío por cola:", dias.to_dict())
    print("horas con caída propia por cola:", excl[excl.motivo == "caida_cola"].groupby("cola").size().to_dict())
    print(f"horas-cola excluidas del entrenamiento: {len(excl)} de {len(ok) * len(orden)} ({100 * len(excl) / (len(ok) * len(orden)):.2f} %)")
    return out, excl


if __name__ == "__main__":
    main()
