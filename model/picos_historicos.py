"""Picos históricos (solo datos reales de entrenamiento; no entran los reales de data/comparacion/).

- Transacciones (Financieras, No Financieras, total): hora con más transacciones (TPH; TPS = TPH / 3600, promedio
  dentro de esa hora) y día con más transacciones. Dos ventanas: todo el historial y desde el cambio de nivel de
  No Financieras (feb-mar 2026, ver CAMBIO_NOFIN). Solo horas «ok» y días completos (sin horas vacías ni caídas).
- Colas: lo mismo por cola, sin días de desvío de tráfico ni horas con caída. Dos ventanas: todo el historial y desde
  el último cambio de enrutamiento de la cola (modelo_colas.ESCALONES); sin cambios, las dos coinciden.
- Contabilidad (cfpd0111cm): solo desde la contabilidad diaria (QUIEBRE_CFPD, 2-jun-2026). Noche con más registros
  contables y mayor ritmo (registros por minuto = registros de la noche / duración de cfpd0111cm).
- Reposting (por día de negocio, tablas activas): día con más TRX y noche más larga, del total (la duración de la noche
  es la de la tabla más lenta) y de cada tabla. Las tablas con un cambio de nivel (modelo_reposting.ESCALONES) tienen
  además la ventana desde ese cambio.
Salida: data/processed/picos_historicos.csv (una fila por pico) y el dict de `calcular()` para el dashboard.
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import modelo_procesos as P
import modelo_colas as MC

S = P.SALIDA
CAMBIO_NOFIN = pd.Timestamp("2026-03-12")   # fin de la caída de nivel de No Financieras (mismo tramo que marca el dashboard)


def _pico(hor, completo):
    """hor: serie horaria válida; completo: serie booleana por día. Devuelve (tph, hora, total_dia, dia)."""
    h = hor.dropna()
    d = hor.groupby(hor.index.normalize()).sum(min_count=1)
    d = d[completo.reindex(d.index, fill_value=False)].dropna()
    if h.empty or d.empty:
        return None
    return {"tph": float(h.max()), "hora": str(h.idxmax())[:16], "dia_total": float(d.max()), "dia": str(d.idxmax().date())}


def transacciones():
    g = pd.read_csv(S / "tph_limpio.csv", parse_dates=["hora"]).set_index("hora")
    g["tot"] = g.fin + g.nofin
    malo = g.estado.isin(["sin_dato", "caida"])
    completo = ~malo.groupby(g.index.normalize()).any() & (g.groupby(g.index.normalize()).size() == 24)
    out = {}
    for s in ("fin", "nofin", "tot"):
        x = g[s].where(~malo)
        out[s] = {"todo": _pico(x, completo), "desde": _pico(x[x.index >= CAMBIO_NOFIN], completo)}
    return {"cambio": str(CAMBIO_NOFIN.date()), "series": out}


def colas():
    g = pd.read_csv(S / "colas_limpio.csv", parse_dates=["hora"]).set_index("hora")
    ex = pd.read_csv(S / "colas_excluidas.csv", parse_dates=["hora"])
    malo = g.estado.isin(["sin_dato", "caida"])
    fecha = g.index.normalize()
    completo = ~malo.groupby(fecha).any() & (g.groupby(fecha).size() == 24)
    out = {}
    for q in [c for c in g.columns if c != "estado"]:
        e = ex[ex.cola == q]
        desv = set(e[e.motivo == "desvio"].hora.dt.normalize())
        caida = set(e[e.motivo == "caida_cola"].hora)
        x = g[q].where(~malo & ~g.index.isin(caida) & ~fecha.isin(desv))
        comp = completo & ~completo.index.isin(desv) & ~completo.index.isin({t.normalize() for t in caida})
        esc = MC.ESCALONES.get(q)
        desde = pd.Timestamp(esc[-1]) if esc else None
        out[q] = {"todo": _pico(x, comp), "desde": _pico(x[x.index >= desde], comp) if desde is not None else None,
                  "cambio": str(desde.date()) if desde is not None else None}
    return out


def contabilidad():
    """Récords de la contabilidad en dos ventanas: todo el historial (hasta el quiebre las noches de primer día acumulaban
    el fin de semana: ~2,7 veces los registros) y desde la contabilidad diaria (QUIEBRE_CFPD). Ritmo = registros / duración
    de cfpd0111cm. Una noche cuyo ritmo es más de 2 veces el siguiente mayor se descarta como dato atípico (duración
    incompleta); queda listada en `atipicos`."""
    o = P.cargar()
    cf = o[o.suproceso == "cfpd0111cm"].set_index("dia_negocio")
    r = pd.read_csv(S / "registros_contables.csv", parse_dates=["dia_negocio"])
    r = r[(r.fuente == "real") & r.real.notna()].set_index("dia_negocio").real
    tipo = o.drop_duplicates("dia_negocio").set_index("dia_negocio").tipo_dia

    def ventana(r):
        rit = (r / cf.dur.reindex(r.index)).dropna().sort_values(ascending=False)
        atip = []
        while len(rit) > 1 and rit.iloc[0] > 2 * rit.iloc[1]:
            atip.append({"dia": str(rit.index[0].date()), "valor": float(rit.iloc[0]), "registros": float(r[rit.index[0]]), "dur": float(cf.dur[rit.index[0]])})
            rit = rit.iloc[1:]
        d_r, d_t = r.idxmax(), rit.index[0]
        return {"desde": str(r.index.min().date()), "hasta": str(r.index.max().date()), "noches": int(len(r)),
                "registros": {"valor": float(r.max()), "dia": str(d_r.date()), "tipo": str(tipo.get(d_r, "")), "dur": float(cf.dur.get(d_r, float("nan")))},
                "ritmo": {"valor": float(rit.iloc[0]), "dia": str(d_t.date()), "registros": float(r[d_t]), "dur": float(cf.dur[d_t])},
                "atipicos": atip}

    out = ventana(r[r.index >= P.QUIEBRE_CFPD])
    out["todo"] = ventana(r)
    return out


def reposting():
    import modelo_reposting as MR
    l = pd.read_csv(S / "reposting_limpio.csv", parse_dates=["dia_negocio"])
    act = pd.read_csv(S / "reposting_ganadores.csv").tabla.unique().tolist()
    l = l[l.tabla.isin(act)]
    dur = l[(l.tmin > 0) & (l.tmin < 600)]

    def pico(x, d):
        if x.dropna().empty or d.dropna().empty:
            return None
        return {"trx": float(x.max()), "dia": str(x.idxmax().date()), "tmin": float(d.max()), "dia_tmin": str(d.idxmax().date())}

    completo = l.dropna(subset=["trx"]).groupby("dia_negocio").tabla.nunique() == len(act)
    tot = l.groupby("dia_negocio").trx.sum(min_count=1)[completo[completo].index]
    out = {"total": {"todo": pico(tot, dur.groupby("dia_negocio").tmin.max())}, "tablas": {}}
    for t in act:
        z = l[l.tabla == t].set_index("dia_negocio")
        d = dur[dur.tabla == t].set_index("dia_negocio").tmin
        e = MR.ESCALONES.get(t)
        desde = pd.Timestamp(e[-1]) if e else None
        out["tablas"][t] = {"todo": pico(z.trx, d), "cambio": str(desde.date()) if desde is not None else None,
                            "desde": pico(z.trx[z.index >= desde], d[d.index >= desde]) if desde is not None else None}
    return out


def calcular():
    return {"tph": transacciones(), "colas": colas(), "conta": contabilidad(), "rep": reposting()}


def tabla(p):
    """Una fila por pico, para el CSV y el Excel."""
    filas = []
    nom = {"fin": "TXN Financieras", "nofin": "TXN No Financieras", "tot": "TXN total"}
    for s, v in p["tph"]["series"].items():
        for k, et in (("todo", "todo el historial"), ("desde", f"desde {p['tph']['cambio']} (cambio de nivel de No Financieras)")):
            if s == "fin" and k == "desde":   # el cambio de nivel fue solo de No Financieras
                continue
            x = v[k]
            filas.append({"Ámbito": nom[s], "Ventana": et, "TPH pico": x["tph"], "TPS pico": x["tph"] / 3600, "Hora del pico": x["hora"],
                          "Día con más transacciones": x["dia"], "Transacciones ese día": x["dia_total"]})
    for q, v in p["colas"].items():
        for k, et in (("todo", "todo el historial"), ("desde", f"desde {v['cambio']} (último cambio de enrutamiento)")):
            x = v[k]
            if x is None:
                continue
            filas.append({"Ámbito": f"Cola {q}", "Ventana": et, "TPH pico": x["tph"], "TPS pico": x["tph"] / 3600, "Hora del pico": x["hora"],
                          "Día con más transacciones": x["dia"], "Transacciones ese día": x["dia_total"]})
    c = p["conta"]
    for v, et in ((c["todo"], f"todo el historial (desde {c['todo']['desde']}; hasta el 1-jun-2026 el primer día acumulaba el fin de semana)"),
                  (c, f"desde {c['desde']} (contabilidad diaria)")):
        filas.append({"Ámbito": "Contabilidad · registros contables por noche", "Ventana": et, "Día con más transacciones": v["registros"]["dia"],
                      "Transacciones ese día": v["registros"]["valor"]})
        filas.append({"Ámbito": "Contabilidad · ritmo de cfpd0111cm (registros por minuto)", "Ventana": et, "Ritmo pico (registros/min)": v["ritmo"]["valor"],
                      "Día con más transacciones": v["ritmo"]["dia"], "Transacciones ese día": v["ritmo"]["registros"], "Duración cfpd0111cm (min)": v["ritmo"]["dur"]})
    r = p["rep"]
    for nombre, v in [("Reposting · noche completa (tablas activas)", r["total"])] + [(f"Reposting · {t}", v) for t, v in r["tablas"].items()]:
        for k, et in (("todo", "todo el historial"), ("desde", f"desde {v.get('cambio')} (cambio de nivel)")):
            x = v.get(k)
            if x is None:
                continue
            filas.append({"Ámbito": nombre, "Ventana": et, "Día con más transacciones": x["dia"], "Transacciones ese día": x["trx"],
                          "Noche más larga": x["dia_tmin"], "Duración de la noche (min)": x["tmin"]})
    return pd.DataFrame(filas)


if __name__ == "__main__":
    p = calcular()
    t = tabla(p)
    t.to_csv(S / "picos_historicos.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 20)
    print(t.round(1).to_string())
