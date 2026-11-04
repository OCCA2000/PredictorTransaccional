"""Registros contables = columna REGISTROS de PROCESOS_CRITICOS_ACUMULADOS (solo cfpd0111, el proceso contable).

REGISTROS ≈ 3,1 × transacciones financieras del día de negocio. Pronóstico = razón reciente x financieras pronosticadas.
No se usa como variable del modelo de duración (no mejora el error de cfpd tras el quiebre de junio de 2026).
Nunca se leen DESCRIPCIÓN ni RESPONSABLE.
Salida: data/processed/registros_contables.csv (dia_negocio, tipo, real, p50, p10, p90).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import modelo_procesos as P

RAIZ, SALIDA = P.RAIZ, P.SALIDA
PROHIBIDAS = {"DESCRIPCIÓN", "DESCRIPCION", "RESPONSABLE"}
MIN_VALIDO = 5e6          # por debajo: captura dañada (15-19 jun 2026, 12 dic 2025)


def reales():
    from limpieza_procesos import XLSX               # mismo archivo de procesos que la limpieza
    r = pd.read_excel(XLSX, usecols=lambda c: c.strip().upper() not in PROHIBIDAS)
    r.columns = [c.strip() for c in r.columns]
    r = r[r.REGISTROS.notna()].copy()
    r["k"] = r.FECHA.dt.strftime("%Y-%m-%d") + " " + r.SUPROCESO.astype(str)
    l = pd.read_csv(SALIDA / "procesos_limpio.csv", parse_dates=["dia_negocio"])
    l = l[l.suproceso.str.startswith("cfpd0111")].copy()
    l["k"] = l.fecha_original.astype(str) + " " + l.suproceso
    l = l.sort_values("ejec_n").drop_duplicates("k", keep="last")
    m = r.merge(l[["k", "dia_negocio", "tipo_dia"]], on="k")
    m = m[m.REGISTROS >= MIN_VALIDO]
    m["cm"] = m.SUPROCESO.eq("cfpd0111cm")
    m = m.sort_values(["dia_negocio", "cm"]).drop_duplicates("dia_negocio", keep="last")
    return m[["dia_negocio", "tipo_dia", "REGISTROS"]].rename(columns={"REGISTROS": "real"})


def fin_dia_negocio():
    t = pd.read_csv(SALIDA / "tph_limpio.csv", parse_dates=["hora"])
    dn = np.where(t.hora.dt.hour < 12, (t.hora - pd.Timedelta(days=1)).dt.normalize(), t.hora.dt.normalize())
    return t.groupby(dn).agg(fin=("fin", "sum"), n=("fin", lambda s: s.notna().sum()))


def main():
    re = reales()
    f = fin_dia_negocio()
    j = re.join(f, on="dia_negocio")
    ok = j[(j.dia_negocio >= P.QUIEBRE_CFPD) & (j.n >= 20)]
    ra = ok.real / ok.fin; ra = ra[ra < 4]                      # descarta días con TPH incompleto
    rm, r10, r90 = ra.median(), ra.quantile(.10), ra.quantile(.90)
    # financieras pronosticadas por día de negocio
    h = pd.read_csv(SALIDA / "pronostico_tph_horario.csv", parse_dates=["hora"])
    dn = np.where(h.hora.dt.hour < 12, (h.hora - pd.Timedelta(days=1)).dt.normalize(), h.hora.dt.normalize())
    fh = h.groupby(dn).fin_p50.sum()
    d = pd.read_csv(SALIDA / "pronostico_tph_diario.csv", parse_dates=["fecha"]).set_index("fecha")
    pr = pd.read_csv(SALIDA / "pronostico_procesos.csv", parse_dates=["fecha_negocio"])
    dias = pd.Index(sorted(pr.fecha_negocio.unique()))
    tipo = pr.drop_duplicates("fecha_negocio").set_index("fecha_negocio").tipo_dia
    lo = (d.fin_total_p10 / d.fin_total_p50).reindex(dias).fillna(.9); hi = (d.fin_total_p90 / d.fin_total_p50).reindex(dias).fillna(1.1)
    p50 = fh.reindex(dias) * rm
    fc = pd.DataFrame({"dia_negocio": dias, "tipo_dia": tipo.reindex(dias).values, "p50": p50.values,
                       "p10": (p50 * lo * (r10 / rm)).values, "p90": (p50 * hi * (r90 / rm)).values}).dropna(subset=["p50"])
    ult = re.dia_negocio.max()
    fc = fc[fc.dia_negocio > ult]
    out = pd.concat([re, fc], ignore_index=True)
    out.insert(1, "fuente", np.where(out.real.notna(), "real", "pronóstico"))
    out.to_csv(SALIDA / "registros_contables.csv", index=False)
    print(f"razón registros/financieras (desde {P.QUIEBRE_CFPD.date()}): mediana {rm:.3f}  p10 {r10:.3f}  p90 {r90:.3f}  n={len(ra)}")
    print(f"real {len(re)} días ({re.dia_negocio.min().date()} -> {ult.date()}) | pronóstico {len(fc)} días hasta {fc.dia_negocio.max().date()}")
    print(fc.head(6).round(0).to_string())
    return out, dict(mediana=rm, p10=r10, p90=r90)


if __name__ == "__main__":
    main()
