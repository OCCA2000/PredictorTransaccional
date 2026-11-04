"""Verifica PROCESOS_CRITICOS_ACUMULADOS.xlsx contra RITM1037408.xlsx (bitácora del scheduler).

RITM: solo se leen Job Name, Start Time, End Time (Application, Sub Application Name y
Folder nunca se leen). PROCESOS: DESCRIPCIÓN y RESPONSABLE nunca se leen.
La correspondencia subproceso -> job se dedujo por coincidencia de horario (ver salida).
"""
from pathlib import Path
import datetime as dt
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RAW = RAIZ / "data/raw"
MAPA = {  # subproceso (PROCESOS) -> job (RITM)
    "cfpd0111cm": "cfpd0111cm",
    "cfpd0111c2 (Lunes)": "BSE_CFPD0111C2",
    "depd0800cm_Lote1": "beod_1-depd0800ms_001_035_controlM",
    "depd0800cm_Lote2": "beod_1-depd0800ms_036_070_controlM",
    "depd0800cm_Lote3": "beod_1-depd0800ms_071_105_controlM",
    "depd0800cm_Lote4": "beod_1-depd0800ms_106_140_controlM",
    "lond0501cm": "beod_1-lond0501ms.controlM",
}
DIAS = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]


def a_time(v):
    if isinstance(v, str):
        return dt.time.fromisoformat(v.strip())
    return v.time() if isinstance(v, (dt.datetime, pd.Timestamp)) else v


def cargar():
    r = pd.read_excel(RAW / "RITM1037408.xlsx",
                      usecols=lambda c: str(c).strip().lower() in {"job name", "start time", "end time"})
    r.columns = ["job", "ini", "fin"]
    for c in ("ini", "fin"):
        r[c] = pd.to_datetime(r[c], format="%B %d, %Y %I:%M:%S %p")
    r["job"] = r.job.str.strip()
    p = pd.read_excel(RAW / "PROCESOS_CRITICOS_ACUMULADOS.xlsx", sheet_name=0,
                      usecols=lambda c: c.strip().upper() in {"FECHA", "PROCESO", "SUPROCESO", "HORA INICIO", "HORA FIN", "DIA SEMANA"})
    p.columns = [c.strip() for c in p.columns]
    t = lambda v: pd.Timedelta(hours=a_time(v).hour, minutes=a_time(v).minute, seconds=a_time(v).second)
    p["ini"] = p["FECHA"] + p["HORA INICIO"].map(t)
    dur = (p["HORA FIN"].map(t) - p["HORA INICIO"].map(t)) % pd.Timedelta(days=1)
    p["fin"] = p["ini"] + dur
    return r, p


def main():
    r, p = cargar()
    lo, hi = r.ini.min(), r.ini.max()
    p = p[p.SUPROCESO.isin(MAPA)].copy()
    p["en_cobertura"] = (p.ini >= lo) & (p.ini <= hi)
    print(f"cobertura RITM: {lo} -> {hi}")
    print(f"filas PROCESOS (subprocesos mapeados): {len(p)} | dentro de cobertura: {int(p.en_cobertura.sum())} | fuera (no verificables): {int((~p.en_cobertura).sum())}")

    DIA = pd.Timedelta(days=1)
    filas = []
    for sub, job in MAPA.items():
        a = p[(p.SUPROCESO == sub) & p.en_cobertura]
        b = r[r.job == job]
        ui, uj, pares = set(), set(), {}
        # emparejar solo por la menor diferencia de hora del día (±45 min); el desfase de
        # fecha (|k| <= 1) no se penaliza: es parte de lo que se mide
        cand = []
        for i, x in a.iterrows():
            for j, y in b.iterrows():
                d = x.ini - y.ini
                k = round(d / DIA)
                if abs(k) <= 1 and abs(d - k * DIA) <= pd.Timedelta(minutes=45):
                    cand.append((abs(d - k * DIA).total_seconds(), abs(k), i, j))
        cand.sort()
        for _, _, i, j in cand:
            if i not in ui and j not in uj:
                ui.add(i); uj.add(j); pares[i] = (j, "par")
        for i, x in a.iterrows():
            if i in pares:
                y = b.loc[pares[i][0]]
                filas.append(dict(sub=sub, job=job, p_ini=x.ini, p_fin=x.fin, p_dow=x["DIA SEMANA"],
                                  r_ini=y.ini, r_fin=y.fin, estado=pares[i][1]))
            else:
                filas.append(dict(sub=sub, job=job, p_ini=x.ini, p_fin=x.fin, p_dow=x["DIA SEMANA"], estado="solo_PROCESOS"))
        for j, y in b.iterrows():
            if j not in uj:
                filas.append(dict(sub=sub, job=job, r_ini=y.ini, r_fin=y.fin, estado="solo_RITM"))
    v = pd.DataFrame(filas)
    par = v.estado.isin(["par", "par_desfase_mayor"])
    dd = (v.p_ini - v.r_ini)
    v.loc[par, "desfase_dias"] = (dd / pd.Timedelta(days=1)).round()
    v.loc[par, "d_ini_s"] = (dd - v.desfase_dias * pd.Timedelta(days=1)).dt.total_seconds()
    v.loc[par, "d_fin_s"] = ((v.p_fin - v.r_fin) - v.desfase_dias * pd.Timedelta(days=1)).dt.total_seconds()
    v.loc[par, "fecha_ok"] = v.p_ini.dt.date == v.r_ini.dt.date
    v.loc[par, "dow_ok"] = v.p_dow == v.r_ini.dt.weekday.map(lambda i: DIAS[int(i)] if pd.notna(i) else None)
    v.loc[par, "hora_ini_ok"] = v.d_ini_s == 0
    v.loc[par, "hora_fin_ok"] = v.d_fin_s == 0
    # PROCESOS suele guardar minutos: ¿coincide al truncar los segundos del RITM?
    v.loc[par, "hora_ini_ok_min"] = v.p_ini == v.r_ini.dt.floor("min")
    v.loc[par, "hora_fin_ok_min"] = v.p_fin == v.r_fin.dt.floor("min")
    v.to_csv(RAIZ / "data/processed/verificacion_ritm.csv", index=False)
    return v


if __name__ == "__main__":
    main()


def clasificar_fechas():
    """Por fila de PROCESOS: ¿la fecha coincide con la del RITM, está corrida o no se puede saber?

    Se compara el mejor candidato del mismo día (d0) con el mejor de un día antes/después (d1),
    por diferencia de hora del día. Solo se afirma algo si uno de los dos es claramente mejor:
      - confirmada : d0 <= 120 s y d1 > 300 s
      - desfase    : d1 <= 120 s y d0 > 300 s   (el RITM tiene otra fecha)
      - ambigua    : ambos <= 300 s (el proceso corre casi a la misma hora todos los días)
      - sin_pareja : ninguno <= 120 s
    """
    r, p = cargar()
    lo, hi = r.ini.min(), r.ini.max()
    p = p[p.SUPROCESO.isin(MAPA) & (p.ini >= lo) & (p.ini <= hi)]
    S = lambda t: t.total_seconds()
    filas = []
    for i, x in p.iterrows():
        b = r[r.job == MAPA[x.SUPROCESO]]
        best = {0: (1e9, None), 1: (1e9, None), -1: (1e9, None)}
        for _, y in b.iterrows():
            d = x.ini - y.ini
            k = round(d / pd.Timedelta(days=1))
            if abs(k) <= 1:
                e = abs(S(d - k * pd.Timedelta(days=1)))
                if e < best[k][0]:
                    best[k] = (e, y)
        d0, y0 = best[0]
        d1, y1 = min(best[1], best[-1], key=lambda t: t[0])
        if d0 <= 120 and d1 > 300: c, y = "confirmada", y0
        elif d1 <= 120 and d0 > 300: c, y = "desfase", y1
        elif d0 <= 300 and d1 <= 300: c, y = "ambigua", (y0 if d0 <= d1 else y1)
        elif min(d0, d1) <= 120: c, y = ("confirmada", y0) if d0 <= d1 else ("ambigua", y1)
        else: c, y = "sin_pareja", None
        filas.append(dict(sub=x.SUPROCESO, p_ini=x.ini, p_dow=x["DIA SEMANA"], clase=c, d_mismo_dia_s=d0, d_otro_dia_s=d1,
                          r_ini=None if y is None else y.ini))
    out = pd.DataFrame(filas)
    out["dia_correcto"] = out.r_ini.dt.normalize()
    out.to_csv(RAIZ / "data/processed/verificacion_ritm_fechas.csv", index=False)
    return out


def verificar_limpio():
    """Compara procesos_limpio.csv (ya con fechas corregidas) contra el RITM, solo dentro de la
    ventana en que existe cada job. Asignación uno a uno por subproceso (inicio + 0.5*fin);
    desfase de fecha permitido ±1 día solo para poder detectarlo. Escribe verificacion_limpio.csv."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment as lsa
    r, _ = cargar()
    p = pd.read_csv(RAIZ / "data/processed/procesos_limpio.csv", parse_dates=["inicio", "fin"])
    D = 86400.0
    filas = []
    for sub, job in MAPA.items():
        b = r[r.job == job].reset_index(drop=True)
        a = p[(p.suproceso == sub) & (p.inicio >= b.ini.min() - pd.Timedelta(minutes=5)) & (p.inicio <= b.ini.max() + pd.Timedelta(minutes=5))].reset_index(drop=True)
        C = np.full((len(a), len(b)), 1e6); K = np.zeros_like(C)
        for i, x in a.iterrows():
            for j, y in b.iterrows():
                d = (x.inicio - y.ini).total_seconds(); k = round(d / D)
                if abs(k) <= 1:
                    fin_cmp = pd.notna(x.fin) and y.fin != y.ini  # sin fin, o RITM con duración 0: solo inicio
                    C[i, j] = abs(d - k * D) + (0.5 * abs((x.fin - y.fin).total_seconds() - k * D) if fin_cmp else 0) + 5 * abs(k); K[i, j] = k  # +5 s/día: desempata a favor de k=0
        ri, cj = lsa(np.where(C < 1800, C, 1e6))
        usados_b = set()
        par = {i: j for i, j in zip(ri, cj) if C[i, j] < 1800}
        for i, x in a.iterrows():
            if i in par:
                y = b.loc[par[i]]; usados_b.add(par[i])
                filas.append(dict(sub=sub, dia_negocio=x.dia_negocio, p_ini=x.inicio, p_fin=x.fin, r_ini=y.ini, r_fin=y.fin,
                                  k=int(K[i, par[i]]), fecha_corregida=x.fecha_corregida, estado="par"))
            else:
                filas.append(dict(sub=sub, dia_negocio=x.dia_negocio, p_ini=x.inicio, p_fin=x.fin, fecha_corregida=x.fecha_corregida, estado="solo_PROCESOS"))
        for j, y in b.iterrows():
            if j not in usados_b:
                filas.append(dict(sub=sub, r_ini=y.ini, r_fin=y.fin, estado="solo_RITM"))
    v = pd.DataFrame(filas)
    m = v.estado == "par"
    v.loc[m, "d_ini_s"] = ((v.p_ini - v.r_ini) - v.k * pd.Timedelta(days=1)).dt.total_seconds()
    v.loc[m, "d_fin_s"] = ((v.p_fin - v.r_fin) - v.k * pd.Timedelta(days=1)).dt.total_seconds()
    v.loc[m & (v.r_fin == v.r_ini), "d_fin_s"] = float("nan")  # RITM con duración 0: fin no comparable
    v.to_csv(RAIZ / "data/processed/verificacion_limpio.csv", index=False)
    return v
