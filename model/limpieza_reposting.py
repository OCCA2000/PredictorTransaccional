"""Limpieza de data/raw/Reposting.xlsx -> data/processed/reposting_limpio.csv y reposting_excluidas.csv

El archivo trae una hoja por mes con bloques de 8 filas (IMG1..IMG8) por fecha: hora de inicio y fin, tiempo total y TRX.
Reglas (revisadas con el usuario el 2026-10-22):
  - Día de negocio. El reposting de un día hábil corre la madrugada siguiente. Hasta el lunes 21-jul-2025 el archivo
    anotaba la fecha de esa madrugada (la hoja «Instrucciones» lo llama «día de reposting»): el día de negocio es el día
    hábil anterior (sábado 01:00 -> viernes; martes 02:00 -> lunes). Desde la segunda corrida del 21-jul-2025 (anotada
    «Se ejecutó el 22/07/2025») la fecha ya es el día de negocio («fecha batch»). La primera corrida del 21-jul (01:54)
    es la del viernes 18 (anotado «Damo indica que esta información es del viernes que se registra con fecha de lunes»).
  - Depuración y pruebas: los bloques anotados «Depuracion»/«PRUEBAS REPOSTING» en el archivo, y los fines de semana de
    depuración del calendario operativo del banco, quedan fuera (la depuración se verá aparte).
  - Duplicados: un bloque idéntico en dos hojas se cuenta una vez; un bloque vacío se descarta si hay otro con datos (la
    hoja resumen «Sem #» solo se usa si el mes no trae el bloque); un bloque idéntico a otro de una fecha anterior es una
    copia (17-21 jun-2025 copiados de mayo) y se descarta.
  - Dos corridas del mismo día de negocio: antes del cambio, si dos bloques traen la misma fecha y el día siguiente queda
    sin corrida, el mayor es el del día de la fecha (13-sep y 18-oct-2024: el viernes pesa más). En el resto se usa la
    primera corrida y las demás quedan fuera hasta que el dueño de los datos confirme (29/30-jun-2024, 16-jun-2025,
    27-abr-2026, 8-may-2026).
  - Fin de semana después del cambio sin depuración: si el viernes anterior no tiene corrida es la de ese viernes
    (30-may-2026); si no, queda fuera (30-ago-2025).
  - TRX idéntico al día anterior en 2 o más tablas: error de captura, queda como hueco (no se inventa).
Se guarda todo hasta el último dato; el entrenamiento usa hasta FIN_DIA_NEGOCIO (mismo corte que TPH y colas).
"""
import re
from datetime import datetime, date
from pathlib import Path
import numpy as np
import pandas as pd
import openpyxl

RAIZ = Path(__file__).resolve().parents[1]
XLSX = RAIZ / "data/raw/Reposting.xlsx"
SALIDA = RAIZ / "data/processed"
TABLAS = ["IMG1", "IMG2", "IMG3", "IMG4", "IMG5", "IMG6", "IMG7", "IMG8"]
FIN_DIA_NEGOCIO = pd.Timestamp("2026-09-25")
CAMBIO_FECHA = pd.Timestamp("2025-07-21")         # desde la 2.ª corrida de este día la fecha es el día de negocio
NOTA = re.compile(r"depur|prueba", re.I)
# notas de depuración escritas fuera de la celda de la fecha: «Diciembre - Depuracion» (debajo de la 2.ª corrida del
# 23-dic-2024) y «Depuracion» (encima del 26-may-2025)
DEPURACION_NOTA_SUELTA = [("2024-12-23", "05:51"), ("2025-05-26", None)]
MESES = {"ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6, "jul": 7, "ago": 8, "sep": 9, "oct": 10, "nov": 11, "dic": 12}


def _pista_mes(hoja):
    s = hoja.lower()
    mes = next((n for k, n in MESES.items() if k in s), None)
    a = re.search(r"20\d{2}", s)
    return mes, int(a.group()) if a else None


def _fecha(v, hoja):
    """Fecha de la celda (objeto fecha o texto dd/mm/aaaa con notas). El nombre de la hoja desambigua día/mes."""
    mes, anio = _pista_mes(hoja)
    if isinstance(v, (datetime, date)):
        y, m, d = v.year, v.month, v.day
        if mes and m != mes and d == mes:
            m, d = d, m
    else:
        x = re.search(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})", str(v))
        if not x:
            return None
        n1, n2, y = int(x.group(1)), int(x.group(2)), int(x.group(3))
        y = y + 2000 if y < 100 else y
        d, m = (n2, n1) if (mes and n1 == mes and n2 != mes) else (n1, n2)
    if anio and y != anio and abs(y - anio) == 1 and m in (1, 12):
        pass                                          # hojas que cruzan el año (31-dic en la hoja de enero)
    try:
        return pd.Timestamp(y, m, d)
    except ValueError:
        return None


def _min(v):
    """Duración 'HH:MM:SS' en minutos (errores de Excel y restas negativas -> NaN)."""
    if v is None:
        return np.nan
    if hasattr(v, "hour") and not isinstance(v, datetime):
        return v.hour * 60 + v.minute + v.second / 60
    s = str(v).strip()
    m = re.match(r"^(\d{1,3}):(\d{2}):(\d{2})$", s)
    if m:
        h, mi, se = map(int, m.groups())
        return h * 60 + mi + se / 60
    return np.nan


def _hora(v):
    if v is None or str(v).strip() in ("", "-", "None"):
        return ""
    if hasattr(v, "hour"):
        return f"{v.hour:02d}:{v.minute:02d}:{v.second:02d}"
    return str(v).strip()[:8]


def _trx(v):
    s = re.sub(r"[^\d\-]", "", str(v)) if v is not None else ""
    return float(s) if s not in ("", "-") else np.nan


def extraer(xlsx=XLSX):
    """Una fila por tabla y bloque (corrida), con la fecha del archivo y la nota de depuración/prueba si la tiene."""
    wb = openpyxl.load_workbook(xlsx, data_only=True)
    filas, bloque = [], 0
    for ws in wb.worksheets:
        if ws.title.lower().startswith("instruc"):
            continue
        fecha, nota, col = None, "", None
        for row in ws.iter_rows(values_only=True):
            low = [str(c).strip().lower() if c is not None else "" for c in row]
            if any(x in ("fecha", "fechas") for x in low) and "tabla" in low:
                # columnas según el encabezado (desde oct-2025 las hojas empiezan una columna más a la derecha)
                col = {"fecha": next(i for i, x in enumerate(low) if x in ("fecha", "fechas")), "tabla": low.index("tabla")}
                for i, x in enumerate(low):
                    for k in ("inicio", "fin", "total", "trx"):
                        if k in x and k not in col:
                            col[k] = i
                continue
            if col is None:
                continue
            g = lambda k: row[col[k]] if k in col and len(row) > col[k] else None
            f0 = g("fecha")
            if f0 is not None and (isinstance(f0, (datetime, date)) or re.search(r"\d{1,2}/\d{1,2}/\d{2,4}", str(f0))):
                fecha = _fecha(f0, ws.title)
                nota = str(f0) if NOTA.search(str(f0)) else ""          # nota escrita en la propia celda de la fecha
                bloque += 1
            tabla = str(g("tabla")).strip() if g("tabla") is not None else ""
            if tabla in TABLAS and fecha is not None and any(f["bloque"] == bloque and f["tabla"] == tabla for f in filas[-8:]):
                bloque += 1                                 # segunda corrida bajo la misma fecha (sin repetir la celda)
            if tabla in TABLAS and fecha is not None:
                filas.append({"hoja": ws.title, "bloque": bloque, "fecha_archivo": fecha, "tabla": tabla, "inicio": _hora(g("inicio")),
                              "fin": _hora(g("fin")), "tmin": _min(g("total")), "trx": _trx(g("trx")), "nota": nota.replace("\n", " ")})
    return pd.DataFrame(filas)


def _calendario():
    cal = pd.read_csv(SALIDA / "calendario_laboral.csv", parse_dates=["fecha"]).set_index("fecha")
    op = pd.read_excel(RAIZ / "data/raw/Calendario_Operativo_Final_2024_2027.xlsx", sheet_name="Calendario", usecols=["Fecha", "Tipo Dia"])
    depur = set(op.loc[op["Tipo Dia"].str.strip() == "DEPURACION", "Fecha"])
    return cal, depur


def limpiar(xlsx=XLSX):
    """Devuelve (limpio, excluidas). limpio: una fila por día de negocio y tabla."""
    cal, depur = _calendario()
    lab = lambda x: bool(cal.laborable.get(x, False))

    def hab_anterior(x):
        y = x - pd.Timedelta(days=1)
        while not lab(y):
            y -= pd.Timedelta(days=1)
        return y

    d = extraer(xlsx)
    # un bloque = una corrida; su firma permite detectar duplicados y copias
    b = d.groupby("bloque").agg(hoja=("hoja", "first"), fecha_archivo=("fecha_archivo", "first"), nota=("nota", "first"),
                                inicio=("inicio", lambda s: s[d.loc[s.index, "tabla"] == "IMG7"].iloc[0] if (d.loc[s.index, "tabla"] == "IMG7").any() else ""),
                                trx_total=("trx", "sum"), con_dato=("trx", lambda s: s.notna().sum()))
    firma = d.sort_values("tabla").groupby("bloque").apply(lambda z: tuple(zip(z.tabla, z.inicio, z.fin, z.trx.fillna(-1))))
    b["firma"] = firma
    exc = []
    def fuera(ids, motivo):
        for i in ids:
            exc.append({"bloque": i, "fecha_archivo": b.at[i, "fecha_archivo"], "inicio": b.at[i, "inicio"], "hoja": b.at[i, "hoja"],
                        "trx_total": b.at[i, "trx_total"], "motivo": motivo})
        b.drop(index=[i for i in ids if i in b.index], inplace=True)

    # 1) idénticos en dos hojas (mismo día): se cuenta uno
    fuera(b[b.duplicated(["fecha_archivo", "firma"])].index.tolist(), "duplicado idéntico en otra hoja")
    # 2) bloques vacíos si hay otro con datos el mismo día; la hoja resumen solo si el mes no trae el bloque
    vacio = b[(b.con_dato == 0) & b.fecha_archivo.duplicated(keep=False)].index.tolist()
    fuera(vacio, "bloque vacío (hay otro con datos)")
    sem = b[b.hoja.str.strip().str.lower().str.startswith("sem")]
    fuera([i for i in sem.index if (b.fecha_archivo == b.at[i, "fecha_archivo"]).sum() > 1], "hoja resumen «Sem #» (el mes ya trae el bloque)")
    # 3) copias de otra fecha: firma igual a la de un bloque de fecha anterior
    vista, copias = {}, []
    for i in b.sort_values("fecha_archivo").index:
        f = b.at[i, "firma"]
        if b.at[i, "con_dato"] >= 4 and f in vista and vista[f] < b.at[i, "fecha_archivo"]:
            copias.append(i)
        vista.setdefault(f, b.at[i, "fecha_archivo"])
    for i in copias:
        exc.append({"bloque": i, "fecha_archivo": b.at[i, "fecha_archivo"], "inicio": b.at[i, "inicio"], "hoja": b.at[i, "hoja"],
                    "trx_total": b.at[i, "trx_total"], "motivo": f"copia de la corrida del {vista[b.at[i, 'firma']].date()}"})
    b.drop(index=copias, inplace=True)
    # 4) depuración y pruebas: nota en la celda de la fecha, o nota suelta junto al bloque (revisadas una por una)
    for i in b.index:
        if any(str(b.at[i, "fecha_archivo"].date()) == f and (ini is None or str(b.at[i, "inicio"]).startswith(ini)) for f, ini in DEPURACION_NOTA_SUELTA):
            b.at[i, "nota"] = "Depuracion (nota junto al bloque)"
    nota = b.nota.fillna("")
    fuera(b[nota.str.contains("prueba", case=False)].index.tolist(), "prueba (anotada en el archivo)")
    fuera(b[nota.str.contains("depur", case=False)].index.tolist(), "depuración (anotada en el archivo)")
    # 5) día de negocio
    h = pd.to_numeric(b.inicio.str[:2], errors="coerce")
    antigua = (b.fecha_archivo < CAMBIO_FECHA) | ((b.fecha_archivo == CAMBIO_FECHA) & (h < 5))
    b["regla"] = np.where(antigua, "fecha de ejecución -> día hábil anterior", "fecha = día de negocio")
    b["dia_negocio"] = [hab_anterior(f) if a else f for f, a in zip(b.fecha_archivo, antigua)]
    # fin de semana o feriado con la fecha nueva: depuración del banco, viernes faltante o fuera
    for i in b[~antigua & ~b.dia_negocio.map(lab)].index:
        f = b.at[i, "dia_negocio"]
        if f in depur:
            fuera([i], "depuración (calendario operativo del banco)")
        elif hab_anterior(f) not in set(b.dia_negocio):
            b.at[i, "dia_negocio"] = hab_anterior(f)
            b.at[i, "regla"] = "fin de semana sin el viernes anterior -> viernes"
        else:
            fuera([i], "corrida en fin de semana sin depuración ni anotación (por confirmar)")
    # 6) dos corridas el mismo día de negocio
    for f, z in list(b.groupby("dia_negocio")):
        if len(z) < 2:
            continue
        mismo = z.fecha_archivo.nunique() == 1 and (z.regla.iloc[0].startswith("fecha de ejecución"))
        sig = z.fecha_archivo.iloc[0]
        if mismo and lab(sig) and sig not in set(b.dia_negocio):
            mayor = z.trx_total.idxmax()
            b.at[mayor, "dia_negocio"] = sig
            b.at[mayor, "regla"] = "dos corridas con la misma fecha: la mayor es la del día de la fecha"
            continue
        orden = z.sort_values(["fecha_archivo", "inicio"]).index
        fuera(list(orden[1:]), "segunda corrida del mismo día de negocio (por confirmar)")

    # filas por tabla de las corridas que quedan
    out = d[d.bloque.isin(b.index)].merge(b[["dia_negocio", "regla"]], left_on="bloque", right_index=True)
    out = out[out.dia_negocio.notna()].copy()
    out["ratio"] = np.where((out.tmin > 0) & (out.trx > 0), out.trx / out.tmin, np.nan)
    # 7) TRX idéntico al día anterior en 2+ tablas: error de captura
    out = out.sort_values(["tabla", "dia_negocio"])
    rep = (out.trx == out.groupby("tabla").trx.shift(1)) & (out.trx > 1000)
    malos = rep.groupby(out.dia_negocio).transform("sum") >= 2
    for f in sorted(out.loc[malos & rep, "dia_negocio"].unique()):
        exc.append({"bloque": None, "fecha_archivo": f, "dia_negocio": f, "inicio": "", "hoja": "", "trx_total": np.nan,
                    "motivo": "TRX idéntico al día anterior en 2 o más tablas (queda como hueco)"})
    out.loc[malos & rep, ["trx", "tmin", "ratio"]] = np.nan
    out["tipo_op"] = out.dia_negocio.map(cal.tipo_op)
    cols = ["dia_negocio", "tabla", "trx", "tmin", "ratio", "inicio", "fin", "fecha_archivo", "regla", "tipo_op"]
    limpio = out[cols].sort_values(["dia_negocio", "tabla"]).reset_index(drop=True)
    excl = pd.DataFrame(exc).sort_values("fecha_archivo", na_position="last")
    return limpio, excl


def main():
    limpio, excl = limpiar()
    entreno = limpio[limpio.dia_negocio <= FIN_DIA_NEGOCIO]
    entreno.to_csv(SALIDA / "reposting_limpio.csv", index=False)
    excl.to_csv(SALIDA / "reposting_excluidas.csv", index=False)
    cal, _ = _calendario()
    habiles = [x for x in pd.date_range(entreno.dia_negocio.min(), FIN_DIA_NEGOCIO) if cal.laborable.get(x, False)]
    sin = [x for x in habiles if x not in set(entreno.dia_negocio)]
    print(f"días de negocio con corrida: {entreno.dia_negocio.nunique()} (hasta {FIN_DIA_NEGOCIO.date()}) | hábiles sin corrida: {len(sin)}")
    print("  ", [str(x.date()) for x in sin])
    print("corridas excluidas por motivo:\n", excl.motivo.value_counts().to_string())
    print("asignación del día de negocio:\n", entreno.drop_duplicates(["dia_negocio"]).regla.value_counts().to_string())


if __name__ == "__main__":
    main()
