"""Variables de calendario y régimen para el modelo de TPH (una fila por hora)."""
from pathlib import Path
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "data/processed"
T0_REGIMEN = pd.Timestamp("2026-02-17")  # centro de la rampa ajustada a No Financiero
W_REGIMEN = 10.0                         # ancho (días): 10% -> 90% en ~44 días
INICIO = pd.Timestamp("2024-12-19")
# La caída de No Financiero fue en realidad cuatro escalones (2025-12-03, 2026-01-21, 2026-03-04 y 2026-03-10:
# -4,7 %, -6,2 %, -10,7 % y -11,1 %), visibles en los cambios de enrutamiento entre colas. Se probó reemplazar
# la rampa por esos escalones (como variable y nivelando el histórico) en 14 orígenes: no mejora, se descartó.

FEATURES = ["h", "dow", "dia_mes", "mes", "doy", "es_feriado", "es_laborable", "es_finde",
            "dias_a_feriado", "dias_desde_feriado", "es_primer_dia", "es_ultimo_laborable_mes",
            "noche_batch", "regimen", "t_dias"]


def regimen(fechas):
    """0 antes de la caída de No Financiero, 1 después (rampa logística). Futuro = 1."""
    t = (pd.DatetimeIndex(fechas) - T0_REGIMEN).days.values.astype(float)
    return 1.0 / (1.0 + np.exp(-t / W_REGIMEN))


def construir(horas):
    """DataFrame de variables para un índice horario cualquiera (pasado o futuro)."""
    cal = pd.read_csv(SALIDA / "calendario_laboral.csv", parse_dates=["fecha"]).set_index("fecha")
    f = pd.DataFrame(index=pd.DatetimeIndex(horas))
    f.index.name = "hora"
    fecha = f.index.normalize()
    f["fecha"] = fecha
    f["h"] = f.index.hour
    f["dow"] = f.index.dayofweek
    f["dia_mes"] = f.index.day
    f["mes"] = f.index.month
    f["doy"] = f.index.dayofyear
    lab = fecha.map(lambda d: bool(cal.at[d, "laborable"]))
    f["es_laborable"] = lab.astype(int)
    f["es_finde"] = (f["dow"] >= 5).astype(int)
    f["es_feriado"] = ((~lab) & (f["dow"] < 5)).astype(int)
    f["es_primer_dia"] = fecha.map(lambda d: cal.at[d, "tipo_dia"] == "primer dia").astype(int)
    # días hasta el próximo / desde el último feriado entre semana (acotado a ±5)
    fer = cal.index[(~cal.laborable) & (cal.index.dayofweek < 5)]
    fer_arr = fer.values
    def dist(d, sig):
        d64 = np.datetime64(d)
        if sig > 0:
            x = fer_arr[fer_arr >= d64]
            return min(int((x[0] - d64) / np.timedelta64(1, "D")), 5) if len(x) else 5
        x = fer_arr[fer_arr <= d64]
        return min(int((d64 - x[-1]) / np.timedelta64(1, "D")), 5) if len(x) else 5
    u = fecha.unique()
    a = {d: dist(d, 1) for d in u}; b = {d: dist(d, -1) for d in u}
    f["dias_a_feriado"] = fecha.map(a).astype(int)
    f["dias_desde_feriado"] = fecha.map(b).astype(int)
    # último día laborable del mes
    ult = cal[cal.laborable].groupby(cal[cal.laborable].index.to_period("M")).apply(lambda s: s.index.max())
    f["es_ultimo_laborable_mes"] = fecha.map(lambda d: int(ult.get(d.to_period("M"), pd.NaT) == d)).astype(int)
    # ventana del batch: noche (23-04) de un día laborable (el día de negocio es el día calendario
    # para h=23 y el anterior para h<12)
    dn = fecha.where(f["h"] >= 12, fecha - pd.Timedelta(days=1))
    labn = dn.map(lambda d: bool(cal.at[d, "laborable"]) if d in cal.index else d.dayofweek < 5)
    f["noche_batch"] = (f["h"].isin([23, 0, 1, 2, 3, 4]) & labn).astype(int)
    f["regimen"] = regimen(fecha)
    f["t_dias"] = (fecha - INICIO).days
    return f
