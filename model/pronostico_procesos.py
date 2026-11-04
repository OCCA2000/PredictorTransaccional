"""Pronóstico de duración e inicio de los procesos críticos por día de negocio (365 días).

Duración por subproceso = nivel reciente (mediana de 8 semanas, días normales) x multiplicador de
'primer día' x efecto del día de la semana x efecto del primer/último día hábil del mes x volumen
relativo de transacciones financieras del día (elasticidad estimada entre 0 y 1; no aplica a lond0501cm). Un baseline simple superó a LightGBM en el backtest (WAPE 12-13% vs 14-15%).
bncd0801 tiene además la «corrida del 28» (ver dias_28): minutos extra de duración y de inicio estimados con las
corridas del 28 del histórico, que se excluyen de sus efectos de calendario normales.
Inicio: Lote1 y bncd0801 = mediana reciente por tipo de día; los demás = inicio de su predecesor (PRED) +
duración predicha + pausa mediana reciente (encadenado, equivalente en error al directo).
Bandas 10-90% = errores de backtests de origen móvil por tramo de horizonte (duración: simétricas, percentil 80 del
error absoluto; inicio: cuantiles 10 y 90). En los backtests el volumen de los días futuros es el pronosticado, no el real.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import modelo_procesos as P
import modelo_tph as M
from features_tph import construir

SALIDA = P.SALIDA
CADENA = ["depd0800cm_Lote1", "depd0800cm_Lote2", "depd0800cm_Lote3", "depd0800cm_Lote4", "lond0501cm", "bnccrn2dcm", "cfpd0111cm", "bncd0801"]
# De qué subproceso depende el inicio de cada uno (None = inicio propio). Medido en los datos (oct-2025 a sep-2026):
#  - bnccrn2dcm arranca ~78 min después de que termina lond0501cm y nunca antes (hay pasos intermedios fuera de los datos).
#  - cfpd0111cm no depende de bnccrn2dcm (27 % de las noches arranca antes de que termine): sigue encadenado a lond0501cm.
#  - bncd0801 no sigue a la cadena (corr. 0,29 con el fin de cfpd0111cm): su inicio se pronostica por su cuenta.
PRED = {"depd0800cm_Lote1": None, "depd0800cm_Lote2": "depd0800cm_Lote1", "depd0800cm_Lote3": "depd0800cm_Lote2",
        "depd0800cm_Lote4": "depd0800cm_Lote3", "lond0501cm": "depd0800cm_Lote4", "bnccrn2dcm": "lond0501cm",
        "cfpd0111cm": "lond0501cm", "bncd0801": None}
ORIGENES_BANDAS = pd.date_range("2025-12-01", "2026-08-01", freq="MS")
ORIGENES_CFPD = pd.date_range("2026-07-20", "2026-09-14", freq="7D")   # solo regímenes posteriores al quiebre
# semanas para el nivel reciente (por defecto 8): con 4 baja el error de los lotes y de lond0501cm a 7 y 30 días; en cfpd0111cm no mejora a 30 días
SEMANAS_NIVEL = {s: 4 for s in CADENA[:5]}
# Corrida del 28 de bncd0801 (DJ - Campañas): la que se ejecuta el día 28 dura ~7 veces lo normal (+75 min) y arranca
# más tarde. Si el 28 no tiene corrida (domingo o lunes) se adelanta a la última corrida anterior (dic-2025 y sep-2026;
# nunca se vio pasar a la siguiente). Registros dudosos de esas corridas, fuera de la estimación del efecto:
#  - 27-oct-2025: mismo inicio y duración que el sábado 25 (copia); 26-jun-2026: sábado de 54 segundos.
#  - 26-mar-2026: la corrida de 106 min fechada el viernes 27-mar a las 09:37 es la del sábado 28 (falta ese sábado y
#    la otra corrida del 27 es copia de la del 26): en el entrenamiento cuenta como corrida del 28.
DIA28_DUDOSOS = pd.to_datetime(["2025-10-27", "2026-06-26"])
DIA28_MAL_FECHADOS = pd.to_datetime(["2026-03-26"])
TRAMOS = [(1, 30), (31, 90), (91, 180), (181, 365)]
FIN_PRON = pd.Timestamp("2027-09-24")


def tramo(k):
    return np.select([(k >= a) & (k <= b) for a, b in TRAMOS], [f"{a}-{b}" for a, b in TRAMOS], default="181-365")


def pivot(o):
    o = o.copy()
    o["ini"] = (o.inicio - o.dia_negocio).dt.total_seconds() / 60
    o["fn"] = (o.fin - o.dia_negocio).dt.total_seconds() / 60
    # duración 0: inicio y fin registrados en el mismo minuto (bnccrn2dcm dura ~20 s). Sirve para el inicio,
    # pero no para el modelo de duración, que trabaja en escala logarítmica.
    o["dur"] = o.dur.where(o.dur > 0)
    return o.pivot_table(index="dia_negocio", columns="suproceso", values=["ini", "fn", "dur"], aggfunc="last", dropna=False)


_LAB = None
def zona_mes(ix):
    """Primer día hábil del mes, último día hábil del mes o resto (ahí se concentra el efecto de cierre/apertura)."""
    global _LAB
    if _LAB is None:
        cal = pd.read_csv(SALIDA / "calendario_laboral.csv", parse_dates=["fecha"]).set_index("fecha")
        lab = pd.Series(cal.index[cal.laborable]); g = lab.groupby([lab.dt.year, lab.dt.month])
        _LAB = (set(g.min()), set(g.max()))
    ix = pd.DatetimeIndex(ix)
    return np.select([ix.isin(_LAB[0]), ix.isin(_LAB[1])], ["ini", "fin"], "resto")


_EJEC28 = None
def dias_28(ix):
    """Días de negocio cuya corrida de bncd0801 es «la del 28»: la última que se ejecuta hasta el 28 del mes.

    bncd0801 se ejecuta la madrugada siguiente al día de negocio (lunes a viernes -> martes a sábado).
    """
    global _EJEC28
    if _EJEC28 is None:
        cal = pd.read_csv(SALIDA / "calendario_laboral.csv", parse_dates=["fecha"]).set_index("fecha")
        lab = pd.Series(cal.index[cal.laborable]); ej = lab + pd.Timedelta(days=1)
        ok = ej.dt.day <= 28
        _EJEC28 = set(lab[ok].groupby([ej[ok].dt.year, ej[ok].dt.month]).max())
    return pd.DatetimeIndex(ix).isin(_EJEC28)


def efecto_28(tr, ef, extra):
    """Minutos extra de duración e inicio de la corrida del 28 de bncd0801 frente a lo normal de ese día.

    Duración: real - lo esperado sin el efecto (mediana móvil de días normales x calendario); inicio: real - mediana
    del mismo día de la semana. Mediana de las corridas del 28 del histórico (0 si hay menos de 3).
    """
    s = "bncd0801"
    f28 = (dias_28(tr.index) | tr.index.isin(DIA28_MAL_FECHADOS)) & ~tr.index.isin(DIA28_DUDOSOS)
    d, ini = tr[("dur", s)], tr[("ini", s)]
    norm = ~(dias_28(tr.index) | tr.index.isin(DIA28_MAL_FECHADOS))
    base = np.log(d.where(norm)).rolling(40, min_periods=10, center=True).median()
    dd = (d - np.exp(base + extra(tr.index)))[f28].dropna()
    ib = ini[norm].groupby(tr.index[norm].dayofweek).median()
    di = (ini - pd.Series(tr.index.dayofweek, index=tr.index).map(ib))[f28].dropna()
    return (float(dd.median()) if len(dd) >= 3 else 0.0), (float(di.median()) if len(di) >= 3 else 0.0)


def efectos_calendario(d, k=4):
    """Efecto (log) del día de la semana y de la zona del mes sobre la duración, respecto de su nivel local.

    Mediana por grupo del residuo frente a la mediana móvil, encogida hacia 0 con n/(n+k) para no sobreajustar.
    """
    base = d.rolling(40, min_periods=10, center=True).median()
    r = np.log(d / base).dropna()
    if len(r) < 20: return {"dow": {}, "zona": {}}, r
    r = r.clip(r.quantile(.02), r.quantile(.98))
    out = {}
    for nom, g in (("dow", r.index.dayofweek), ("zona", zona_mes(r.index))):
        s = pd.Series(r.values, index=g); m = s.groupby(level=0).median(); n = s.groupby(level=0).size()
        out[nom] = (m * n / (n + k)).to_dict()
    return out, r


def efecto_zona_min(x, pd_, k=4):
    """Minutos extra del primer/último día hábil del mes sobre una serie de horas o pausas (x), frente a su nivel local.

    Se mide en días normales (no 'primer día') y se encoge con n/(n+k).
    """
    x = x.dropna()
    r = (x - x.rolling(40, min_periods=10, center=True).median()).dropna()
    r = r[pd_.reindex(r.index).fillna(0) == 0]
    if len(r) < 20: return {}
    s = pd.Series(r.values, index=zona_mes(r.index)); m = s.groupby(level=0).median(); n = s.groupby(level=0).size()
    return {z: float(m[z] * n[z] / (n[z] + k)) for z in ("ini", "fin") if z in m.index}


def ajuste(ef, ix):
    return np.array([ef["dow"].get(x, 0.0) for x in ix.dayofweek]) + np.array([ef["zona"].get(x, 0.0) for x in zona_mes(ix)])


_VOL, _TPH = {}, None
def volumen_relativo(origen):
    """log(financieras del día de negocio / su mediana móvil de ~6 semanas): real antes de `origen`, pronóstico después.

    Al ser relativo a su nivel local no arrastra la tendencia de las financieras: solo mueve la duración
    de cfpd0111cm dentro del mes (el proceso contable procesa ≈ 3 registros por transacción financiera).
    El pronóstico es el vigente (pronostico_tph_horario.csv) si `origen` es posterior al último dato de TPH;
    en un backtest se rehace solo con datos anteriores a `origen`, para no usar volumen que no se conocía.
    """
    global _TPH
    if _TPH is None:
        g, X = M.cargar()
        pron = pd.read_csv(SALIDA / "pronostico_tph_horario.csv", parse_dates=["hora"]).set_index("hora").fin_p50
        cal = pd.read_csv(SALIDA / "calendario_laboral.csv", parse_dates=["fecha"]).set_index("fecha")
        _TPH = (g, X, pron, cal.index[cal.laborable])
    g, X, pron, laborables = _TPH
    corte = min(pd.Timestamp(origen).normalize(), pron.index.min())
    if corte not in _VOL:
        if corte == pron.index.min():
            fut = pron
        else:
            Xf = construir(pd.date_range(corte, corte + pd.Timedelta(days=366), freq="h", inclusive="left"))
            fut, _ = M.entrenar_predecir(g, X, "fin", corte, 0, usar_tendencia=True, phi=1.0, Xfut=Xf)   # igual que pronostico_tph
            fut = fut.mask(M.sin_linea(Xf))
        v = pd.concat([g.loc[g.index < corte, "fin"], fut])
        dn = np.where(v.index.hour < 12, (v.index - pd.Timedelta(days=1)).normalize(), v.index.normalize())
        v = v.groupby(dn).agg(["sum", "count"])
        v = v.loc[v["count"] >= 20, "sum"]
        v = np.log(v[v.index.isin(laborables)])
        _VOL[corte] = v - v.rolling(30, min_periods=8, center=True).median()
    return _VOL[corte]


def predecir(W, pdia, origen, dias):
    """Predice duración e inicio (min desde 00:00 del día de negocio) para `dias` (índice de fechas)."""
    tr0 = W[W.index < origen]
    pt = pdia.reindex(dias).fillna(0).astype(int)
    out = {}
    ini_prev = None
    for i, s in enumerate(CADENA):
        # cfpd0111cm: tras el quiebre de junio-2026 solo aplican los datos posteriores (el régimen anterior ya no vale)
        tr = tr0[tr0.index >= P.QUIEBRE_CFPD] if (s == "cfpd0111cm" and origen > P.QUIEBRE_CFPD + pd.Timedelta(weeks=6)) else tr0
        ptr = pdia.reindex(tr.index)
        rec = tr[tr.index > origen - pd.Timedelta(weeks=SEMANAS_NIVEL.get(s, 8))]; pr_ = pdia.reindex(rec.index)
        # duración: nivel reciente (días normales) x multiplicador histórico de primer día
        d_tr = tr[("dur", s)].dropna()
        if s == "bncd0801":   # las corridas del 28 tienen su propio efecto (efecto_28)
            d_tr = d_tr[~(dias_28(d_tr.index) | d_tr.index.isin(DIA28_MAL_FECHADOS))]
        ef, _ = efectos_calendario(d_tr)
        vol, e = None, 0.0
        if s != "lond0501cm" and not (s == "cfpd0111cm" and origen <= P.QUIEBRE_CFPD + pd.Timedelta(weeks=6)):
            vol = volumen_relativo(origen)
            y = np.log(d_tr) - np.log(d_tr).rolling(40, min_periods=10, center=True).median() - ajuste(ef, d_tr.index)
            j = pd.DataFrame({"y": y, "x": vol.reindex(d_tr.index)}).dropna()
            if len(j) >= 25: e = float(np.clip(np.polyfit(j.x, j.y.clip(j.y.quantile(.02), j.y.quantile(.98)), 1)[0], 0, 1))
        extra = lambda ix: ajuste(ef, ix) + (e * vol.reindex(ix).fillna(0).values if e else 0.0)
        # nivel reciente "neutro": días normales de las últimas 8 semanas, sin su efecto de calendario/volumen
        rn = rec[("dur", s)][pr_ == 0].dropna()
        rn = rn[rn.index.isin(d_tr.index)]
        normal = np.exp(np.median(np.log(rn.values) - extra(rn.index))) if len(rn) else np.nan
        hn = np.log(d_tr) - extra(d_tr.index)
        h = hn.groupby(ptr.reindex(d_tr.index)).median()
        mult = np.exp(h.get(1, np.nan) - h.get(0, np.nan)) if (1 in h.index and 0 in h.index) else 1.0
        mult = 1.0 if not np.isfinite(mult) else mult
        dur = pd.Series(np.where(pt == 1, normal * mult, normal) * np.exp(extra(dias)), index=dias)
        e28 = efecto_28(tr, ef, extra) if s == "bncd0801" else (0.0, 0.0)
        dur = dur + np.where(dias_28(dias), e28[0], 0.0)
        out[("dur", s)] = dur
        # inicio: propio (mediana reciente por tipo de día) o encadenado al fin de su predecesor
        a = PRED[s]
        if a is None:
            m = rec[("ini", s)].groupby(pr_).median(); mg = tr[("ini", s)].groupby(ptr).median()
            ini = pt.map(m).fillna(pt.map(mg))
            ez = efecto_zona_min(tr[("ini", s)], ptr)
            ini = ini + np.array([ez.get(z, 0.0) for z in zona_mes(dias)]) + np.where(dias_28(dias), e28[1], 0.0)
        else:
            gap = (rec[("ini", s)] - rec[("fn", a)]).groupby(pr_).median(); gapg = (tr[("ini", s)] - tr[("fn", a)]).groupby(ptr).median()
            g = pt.map(gap).fillna(pt.map(gapg)).fillna(0)
            ez = efecto_zona_min(tr[("ini", s)] - tr[("fn", a)], ptr)
            g = g + np.array([ez.get(z, 0.0) for z in zona_mes(dias)])
            ini = out[("ini", a)] + out[("dur", a)] + g
        out[("ini", s)] = pd.Series(ini.values, index=dias)
    return pd.DataFrame(out)


def residuos(W, pdia):
    filas = []
    for og in ORIGENES_BANDAS.union(ORIGENES_CFPD):
        te = W[(W.index >= og) & (W.index < og + pd.Timedelta(days=365))]
        if len(te) < 5: continue
        p = predecir(W, pdia, og, te.index)
        k = (te.index - og).days + 1
        for s in CADENA:
            if (s == "cfpd0111cm") != (og in ORIGENES_CFPD): continue   # cfpd: solo orígenes posteriores al quiebre
            ok = te[("dur", s)].notna()
            filas.append(pd.DataFrame({"sub": s, "var": "dur", "tramo": tramo(k[ok]), "r": np.log(te[("dur", s)][ok] / p[("dur", s)][ok]).values}))
            ok = te[("ini", s)].notna()
            filas.append(pd.DataFrame({"sub": s, "var": "ini", "tramo": tramo(k[ok]), "r": (te[("ini", s)][ok] - p[("ini", s)][ok]).values}))
        if og in ORIGENES_CFPD: continue
        lotes = CADENA[:4]
        okd = te[[("dur", s) for s in lotes]].notna().all(axis=1)
        tot_r = te[[("dur", s) for s in lotes]].sum(axis=1)[okd]; tot_p = p[[("dur", s) for s in lotes]].sum(axis=1)[okd]
        filas.append(pd.DataFrame({"sub": "depd0800cm_total", "var": "dur", "tramo": tramo(k[okd.values]), "r": np.log(tot_r / tot_p).values}))
    return pd.concat(filas)


def main():
    o = P.cargar()
    W = pivot(o)
    pdia = o.groupby("dia_negocio").apply(lambda x: int((x.tipo_dia == "primer dia").any()))
    res = residuos(W, pdia)
    q = res.groupby(["sub", "var", "tramo"]).r.quantile([0.10, 0.90]).unstack().rename(columns={0.10: "lo", 0.90: "hi"})
    q["n"] = res.groupby(["sub", "var", "tramo"]).size()
    # duración: banda simétrica (en log) = percentil 80 del error absoluto. Los cuantiles 10/90 quedaban torcidos por
    # los escalones de nivel (los lotes bajaron ~20 % en julio de 2026), y un escalón futuro puede ir en cualquier dirección
    w = res[res["var"] == "dur"].assign(r=lambda x: x.r.abs()).groupby(["sub", "var", "tramo"]).r.quantile(0.80)
    q.loc[w.index, "lo"], q.loc[w.index, "hi"] = -w, w
    # las bandas no pueden achicarse al crecer el horizonte (los tramos largos tienen pocos orígenes)
    orden = [f"{a}-{b}" for a, b in TRAMOS]
    for (sub, var), g in q.groupby(level=[0, 1]):
        lo_min, hi_max = np.inf, -np.inf
        for t in orden:
            if (sub, var, t) not in q.index: continue
            lo_min = min(lo_min, q.at[(sub, var, t), "lo"]); hi_max = max(hi_max, q.at[(sub, var, t), "hi"])
            q.at[(sub, var, t), "lo"], q.at[(sub, var, t), "hi"] = lo_min, hi_max
    # cfpd0111cm solo tiene backtests posteriores al quiebre (horizontes cortos): en los tramos largos sin datos (n=0)
    # usan la más ancha entre su última banda medida y la banda mediana de los otros subprocesos en ese tramo
    for (sub, var), g in list(q.groupby(level=[0, 1])):
        ref = None
        for tr_nom in orden:
            if (sub, var, tr_nom) in q.index: ref = tr_nom; continue
            if ref is None: continue
            otros = [s for s in CADENA if (s, var, tr_nom) in q.index and q.at[(s, var, tr_nom), "n"] > 0]
            lo = min(q.at[(sub, var, ref), "lo"], np.median([q.at[(s, var, tr_nom), "lo"] for s in otros]))
            hi = max(q.at[(sub, var, ref), "hi"], np.median([q.at[(s, var, tr_nom), "hi"] for s in otros]))
            q.loc[(sub, var, tr_nom), :] = [lo, hi, 0]
    q = q.sort_index()
    q.to_csv(SALIDA / "bandas_procesos.csv")

    ultimo = W.index.max()
    cal = pd.read_csv(SALIDA / "calendario_laboral.csv", parse_dates=["fecha"]).set_index("fecha")
    dias = cal.index[(cal.index > ultimo) & (cal.index <= FIN_PRON) & cal.laborable]
    pdia_f = pd.Series((cal.loc[dias, "tipo_dia"] == "primer dia").astype(int).values, index=dias)
    pdia_all = pd.concat([pdia, pdia_f])
    origen = ultimo + pd.Timedelta(days=1)
    p = predecir(W, pdia_all, origen, dias)
    k = (dias - origen).days + 1
    tr_ = tramo(k)

    filas = []
    for s in CADENA + ["depd0800cm_total"]:
        if s == "depd0800cm_total":
            d50 = p[[("dur", x) for x in CADENA[:4]]].sum(axis=1)
            i50 = p[("ini", CADENA[0])]; f50 = i50 + d50
        else:
            d50 = p[("dur", s)]; i50 = p[("ini", s)]; f50 = i50 + d50
        qd = q.loc[(s, "dur")]
        lo = np.array([qd.loc[t, "lo"] for t in tr_]); hi = np.array([qd.loc[t, "hi"] for t in tr_])
        f = pd.DataFrame({"fecha_negocio": dias, "suproceso": s, "tipo_dia": cal.loc[dias, "tipo_dia"].values,
                          "dur_min_p50": d50.values.round(1), "dur_min_p10": (d50.values * np.exp(lo)).round(1),
                          "dur_min_p90": (d50.values * np.exp(hi)).round(1)})
        if s != "depd0800cm_total":
            qi = q.loc[(s, "ini")]
            ilo = np.array([qi.loc[t, "lo"] for t in tr_]); ihi = np.array([qi.loc[t, "hi"] for t in tr_])
            to_dt = lambda m: (dias + pd.to_timedelta(m, unit="m")).round("min")
            f["inicio_p50"] = to_dt(i50.values); f["inicio_p10"] = to_dt(i50.values + ilo); f["inicio_p90"] = to_dt(i50.values + ihi)
            f["fin_p50"] = to_dt(f50.values)
        else:
            f["inicio_p50"] = (dias + pd.to_timedelta(i50.values, unit="m")).round("min"); f["fin_p50"] = (dias + pd.to_timedelta(f50.values, unit="m")).round("min")
        filas.append(f)
    out = pd.concat(filas).sort_values(["fecha_negocio", "suproceso"])
    out.to_csv(SALIDA / "pronostico_procesos.csv", index=False)
    print(f"pronóstico {dias.min().date()} -> {dias.max().date()} | {len(dias)} días de negocio | {out.suproceso.nunique()} series")
    return out, q


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    out, q = main()
    print(np.exp(q.loc[(slice(None), "dur"), ["lo", "hi"]]).round(2).join(q.n).to_string())
    print(q.loc[(slice(None), "ini"), ["lo", "hi"]].round(0).to_string())
