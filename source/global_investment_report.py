from __future__ import annotations

from pathlib import Path
from datetime import datetime
import re

import numpy as np
import pandas as pd


# ======================================================================================
# MARKET AI - GLOBAL INVESTMENT REPORT V3.1
# ======================================================================================
#
# OBJETIVO
# -------
# Convertir los resultados de los 9 mercados en un informe de inversion explicable.
#
# PRINCIPIO CENTRAL
# -----------------
#
#   OPORTUNIDAD IA
#        |
#        v
#   REGIMEN / DECISION ENGINE
#        |
#        v
#   VALIDACION WALK-FORWARD
#        |
#        v
#   RIESGO
#        |
#        v
#   MOMENTO TECNICO
#        |
#        v
#   AUTORIZACION
#
# Un score alto NO significa automaticamente comprar.
#
# ======================================================================================


BASE_DIR = Path(__file__).resolve().parent.parent
REPORTES_DIR = BASE_DIR / "reportes"
GLOBAL_DIR = REPORTES_DIR / "global"

GLOBAL_REPORT = REPORTES_DIR / "GLOBAL_INVESTMENT_REPORT.txt"

MERCADOS = [
    "merval",
    "crypto",
    "nasdaq",
    "cedears",
    "etf",
    "forex",
    "commodities",
    "bonos",
    "indices",
]

CAPITAL_BASE = 100_000.0
TOP_GLOBAL = 15
TOP_MERCADO = 5


# ======================================================================================
# UTILIDADES
# ======================================================================================


def linea(char="=", n=112):
    return char * n


def safe_float(value, default=np.nan):
    try:
        if value is None:
            return default

        if isinstance(value, str):
            value = value.strip()
            value = value.replace("%", "")
            value = value.replace(",", ".")

        return float(value)

    except (TypeError, ValueError):
        return default


def fmt_num(value, dec=2, default="-"):
    x = safe_float(value)

    if pd.isna(x):
        return default

    return f"{x:.{dec}f}"


def fmt_pct(value, dec=2, default="-"):
    x = safe_float(value)

    if pd.isna(x):
        return default

    return f"{x:.{dec}f}%"


def fmt_return_pct(value, dec=2, default="-"):
    """Formatea retornos almacenados como decimal o porcentaje."""
    x = safe_float(value)

    if pd.isna(x):
        return default

    if abs(x) <= 1.0:
        x *= 100.0

    return f"{x:.{dec}f}%"


def texto_seguro(value, default="-"):
    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except Exception:
        pass

    txt = str(value).strip()

    return txt if txt else default


def leer_texto(path):
    if not path.exists():
        return ""

    for encoding in (
        "utf-8",
        "utf-8-sig",
        "cp1252",
        "latin-1",
    ):
        try:
            return path.read_text(encoding=encoding)
        except Exception:
            continue

    return ""


def limpiar_texto(texto):
    if not texto:
        return ""

    texto = texto.replace("\ufeff", "")
    texto = texto.replace("\r\n", "\n")
    texto = texto.replace("\r", "\n")

    return "\n".join(
        ln.rstrip()
        for ln in texto.splitlines()
        if ln.strip()
    )


def extraer_valor(texto, patrones):
    if not texto:
        return None

    for patron in patrones:
        m = re.search(
            patron,
            texto,
            flags=re.IGNORECASE | re.MULTILINE,
        )

        if m:
            return m.group(1).strip()

    return None


def normalizar_columna(df, candidatos):
    mapa = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for candidato in candidatos:
        key = candidato.strip().lower()

        if key in mapa:
            return mapa[key]

    return None


def obtener_col(row, candidatos, default="-"):
    for nombre in candidatos:
        if nombre in row.index:
            valor = row[nombre]

            try:
                if pd.isna(valor):
                    continue
            except Exception:
                pass

            return valor

    return default


# ======================================================================================
# DATOS DEL ACTIVO
# ======================================================================================


def obtener_ticker(row):
    return texto_seguro(
        obtener_col(
            row,
            [
                "Ticker",
                "ticker",
                "Activo",
                "Symbol",
            ],
        )
    )


def obtener_nombre(row):
    return texto_seguro(
        obtener_col(
            row,
            [
                "Nombre",
                "nombre",
                "Name",
            ],
            obtener_ticker(row),
        )
    )


def obtener_senal(row):
    return texto_seguro(
        obtener_col(
            row,
            [
                "Señal",
                "Senal",
                "Signal",
                "Fortaleza_Senal",
            ],
        )
    )


def obtener_score(row):
    return safe_float(
        obtener_col(
            row,
            [
                "Score_Final",
                "score_final",
                "Score",
                "Consensus_Score",
            ],
        )
    )


def obtener_probabilidad(row):
    return safe_float(
        obtener_col(
            row,
            [
                "Consensus_Probability",
                "Probabilidad",
                "Probability",
                "Prob",
            ],
        )
    )


def obtener_close(row):
    return safe_float(
        obtener_col(
            row,
            [
                "Close",
                "Adj Close",
                "Precio",
            ],
        )
    )


def obtener_rsi(row):
    return safe_float(
        obtener_col(
            row,
            [
                "RSI_14",
                "RSI",
            ],
        )
    )


def obtener_return_5d(row):
    return safe_float(
        obtener_col(
            row,
            [
                "Return_5d",
                "Retorno_5d",
            ],
        )
    )


def obtener_return_20d(row):
    return safe_float(
        obtener_col(
            row,
            [
                "Return_20d",
                "Retorno_20d",
            ],
        )
    )


def obtener_atr(row):
    return safe_float(
        obtener_col(
            row,
            [
                "ATR_14",
                "ATR",
            ],
        )
    )


def obtener_volatilidad(row):
    return safe_float(
        obtener_col(
            row,
            [
                "Volatility_20",
                "Volatilidad_20",
                "Volatility",
            ],
        )
    )


def obtener_agreement(row):
    return safe_float(
        obtener_col(
            row,
            [
                "Agreement_Score",
                "Agreement",
            ],
        )
    )


def obtener_dispersion(row):
    return safe_float(
        obtener_col(
            row,
            [
                "Model_Dispersion",
                "Dispersion",
            ],
        )
    )


def obtener_buy_votes(row):
    return safe_float(
        obtener_col(
            row,
            [
                "Buy_Votes",
                "buy_votes",
            ],
        ),
        0.0,
    )


def obtener_neutral_votes(row):
    return safe_float(
        obtener_col(
            row,
            [
                "Neutral_Votes",
                "neutral_votes",
            ],
        ),
        0.0,
    )


def obtener_sell_votes(row):
    return safe_float(
        obtener_col(
            row,
            [
                "Sell_Votes",
                "sell_votes",
            ],
        ),
        0.0,
    )


def obtener_models_used(row):
    return texto_seguro(
        obtener_col(
            row,
            [
                "Models_Used",
                "Modelos_Usados",
            ],
        )
    )


# ======================================================================================
# CARGA DE RANKINGS
# ======================================================================================


def carpeta_mercado(mercado):
    return REPORTES_DIR / mercado


def buscar_ranking(mercado):
    carpeta = carpeta_mercado(mercado)

    candidatos = [
        carpeta / "ranking_diario.csv",
        carpeta / f"ranking_diario_{mercado}.csv",
    ]

    for path in candidatos:
        if path.exists():
            return path

    encontrados = sorted(
        carpeta.glob("*ranking*diario*.csv")
    )

    return encontrados[0] if encontrados else None


def cargar_ranking(mercado):
    path = buscar_ranking(mercado)

    if path is None:
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)

    except UnicodeDecodeError:
        df = pd.read_csv(
            path,
            encoding="latin-1",
        )

    except Exception as exc:
        print(
            f"AVISO {mercado.upper()}: "
            f"no se pudo leer ranking: {exc}"
        )
        return pd.DataFrame()

    if df.empty:
        return df

    df = df.copy()

    df["Mercado_Global"] = mercado.upper()

    col_score = normalizar_columna(
        df,
        [
            "Score_Final",
            "Score",
            "Consensus_Score",
        ],
    )

    if col_score:
        df["_score_global"] = pd.to_numeric(
            df[col_score],
            errors="coerce",
        )
    else:
        df["_score_global"] = np.nan

    col_prob = normalizar_columna(
        df,
        [
            "Consensus_Probability",
            "Probability",
            "Probabilidad",
        ],
    )

    if col_prob:
        df["_prob_global"] = pd.to_numeric(
            df[col_prob],
            errors="coerce",
        )
    else:
        df["_prob_global"] = np.nan

    return df


# ======================================================================================
# ARCHIVOS COMPLEMENTARIOS
# ======================================================================================


def cargar_contexto_txt(mercado):
    carpeta = carpeta_mercado(mercado)

    paths = {
        "regime":
            carpeta /
            f"market_regime_{mercado}_v2.txt",

        "decision":
            carpeta /
            f"decision_engine_{mercado}.txt",

        "portfolio":
            carpeta /
            f"portfolio_optimizer_{mercado}.txt",

        "walk":
            carpeta /
            f"walk_forward_resumen_{mercado}.txt",

        "daily":
            carpeta /
            f"market_ai_daily_report_{mercado}.txt",
    }

    salida = {}

    for key, path in paths.items():
        salida[key] = limpiar_texto(
            leer_texto(path)
        )

    return salida


# ======================================================================================
# INTERPRETACION DECISION ENGINE
# ======================================================================================


def interpretar_contexto(contexto):

    combinado = "\n".join(
        txt
        for txt in contexto.values()
        if txt
    )

    regimen = extraer_valor(
        combinado,
        [
            r"R[eé]gimen\s*[:=]\s*([^\n]+)",
            r"Market Regime\s*[:=]\s*([^\n]+)",
            r"Regime\s*[:=]\s*([^\n]+)",
        ],
    )

    decision = extraer_valor(
        combinado,
        [
            r"Decisi[oó]n(?:\s+IA)?\s*[:=]\s*([^\n]+)",
            r"Recomendaci[oó]n\s*[:=]\s*([^\n]+)",
            r"Acci[oó]n\s*[:=]\s*([^\n]+)",
        ],
    )

    riesgo = extraer_valor(
        combinado,
        [
            r"Riesgo\s*[:=]\s*([^\n]+)",
            r"Nivel de riesgo\s*[:=]\s*([^\n]+)",
        ],
    )

    market_index = extraer_valor(
        combinado,
        [
            r"Market AI Index\s*[:=]\s*([0-9.,]+)",
            r"AI Index\s*[:=]\s*([0-9.,]+)",
        ],
    )

    exposicion = extraer_valor(
        combinado,
        [
            r"Exposici[oó]n\s*[:=]\s*([0-9.,]+\s*%)",
            r"Acciones\s*[:=]\s*([0-9.,]+\s*%)",
            r"Activos\s*[:=]\s*([0-9.,]+\s*%)",
        ],
    )

    cash = extraer_valor(
        combinado,
        [
            r"Cash\s*[:=]\s*([0-9.,]+\s*%)",
            r"Liquidez\s*[:=]\s*([0-9.,]+\s*%)",
        ],
    )

    cash_origen = "MOTOR" if cash else "-"

    if not cash and exposicion:
        exp_num = safe_float(exposicion)

        if not pd.isna(exp_num) and 0 <= exp_num <= 100:
            cash = f"{100.0 - exp_num:.2f}%"
            cash_origen = "DERIVADO DE EXPOSICION"

    return {
        "regimen":
            regimen or "NO IDENTIFICADO",

        "decision":
            decision or "NO IDENTIFICADA",

        "riesgo":
            riesgo or "-",

        "market_ai_index":
            market_index or "-",

        "exposicion":
            exposicion or "-",

        "cash":
            cash or "-",

        "cash_origen":
            cash_origen,
    }


# ======================================================================================
# WALK FORWARD
# ======================================================================================


def parsear_walk_forward(texto):

    if not texto:
        return {
            "disponible": False,
        }

    def num(patrones):
        valor = extraer_valor(
            texto,
            patrones,
        )

        return safe_float(valor)

    return {
        "disponible": True,

        "capital_inicial":
            num([
                r"capital_inicial\s*:\s*([0-9.,]+)",
            ]),

        "capital_final":
            num([
                r"capital_final\s*:\s*([0-9.,]+)",
            ]),

        "retorno":
            num([
                r"retorno_total_%\s*:\s*(-?[0-9.,]+)",
            ]),

        "operaciones":
            num([
                r"operaciones\s*:\s*([0-9.,]+)",
            ]),

        "win_rate":
            num([
                r"win_rate_%\s*:\s*(-?[0-9.,]+)",
            ]),

        "profit_factor":
            num([
                r"profit_factor\s*:\s*(-?[0-9.,]+)",
            ]),

        "max_drawdown":
            num([
                r"max_drawdown_%\s*:\s*(-?[0-9.,]+)",
            ]),

        "score_usado":
            extraer_valor(
                texto,
                [
                    r"score_usado\s*:\s*([^\n]+)",
                ],
            ) or "-",
    }


def evaluar_walk_forward(wf):

    if not wf.get("disponible"):
        return {
            "estado": "SIN DATOS",
            "puntos": 0,
            "razones": [
                "No existe validacion walk-forward disponible."
            ],
        }

    pf = wf.get("profit_factor", np.nan)
    retorno = wf.get("retorno", np.nan)
    dd = wf.get("max_drawdown", np.nan)
    win = wf.get("win_rate", np.nan)

    puntos = 0
    razones = []

    if not pd.isna(pf):
        if pf >= 1.30:
            puntos += 3
            razones.append(f"Profit Factor solido: {pf:.3f}.")
        elif pf >= 1.10:
            puntos += 2
            razones.append(f"Profit Factor positivo: {pf:.3f}.")
        elif pf >= 1.00:
            razones.append(f"Profit Factor apenas suficiente: {pf:.3f}.")
        else:
            puntos -= 5
            razones.append(f"Profit Factor insuficiente: {pf:.3f}.")

    if not pd.isna(retorno):
        if retorno > 0:
            puntos += 1
            razones.append(
                f"Retorno walk-forward positivo: {retorno:.2f}%."
            )
        else:
            puntos -= 5
            razones.append(
                f"Retorno walk-forward negativo: {retorno:.2f}%."
            )

    if not pd.isna(dd):
        abs_dd = abs(dd)

        if abs_dd <= 20:
            puntos += 3
            razones.append(
                f"Drawdown historico controlado: {dd:.2f}%."
            )
        elif abs_dd <= 35:
            puntos += 1
            razones.append(
                f"Drawdown historico moderado: {dd:.2f}%."
            )
        elif abs_dd <= 50:
            puntos -= 2
            razones.append(
                f"Drawdown historico elevado: {dd:.2f}%."
            )
        elif abs_dd <= 70:
            puntos -= 5
            razones.append(
                f"Drawdown historico muy elevado: {dd:.2f}%."
            )
        else:
            puntos -= 8
            razones.append(
                f"Drawdown historico extremo: {dd:.2f}%."
            )

    if not pd.isna(win):
        if win >= 55:
            puntos += 2
            razones.append(f"Win Rate robusto: {win:.2f}%.")
        elif win >= 48:
            razones.append(f"Win Rate intermedio: {win:.2f}%.")
        elif win < 40:
            puntos -= 2
            razones.append(f"Win Rate debil: {win:.2f}%.")

    # Gate prudencial: el retorno no puede compensar un DD extremo.
    if not pd.isna(dd) and abs(dd) > 70:
        estado = "RIESGO EXTREMO"
    elif (
        (not pd.isna(retorno) and retorno < 0)
        or (not pd.isna(pf) and pf < 1.0)
    ):
        estado = "DESFAVORABLE"
    elif not pd.isna(dd) and abs(dd) > 50:
        estado = "RIESGO ALTO"
    elif puntos >= 6:
        estado = "FAVORABLE"
    elif puntos >= 2:
        estado = "ACEPTABLE CON RIESGO"
    elif puntos >= -2:
        estado = "DEBIL"
    else:
        estado = "DESFAVORABLE"

    return {
        "estado": estado,
        "puntos": puntos,
        "razones": razones,
    }


# ======================================================================================
# CALIDAD IA
# ======================================================================================


def calidad_score(score):

    if pd.isna(score):
        return "SIN DATOS"

    if score >= 80:
        return "EXCEPCIONAL"

    if score >= 72:
        return "MUY ALTA"

    if score >= 65:
        return "ALTA"

    if score >= 58:
        return "MEDIA"

    if score >= 50:
        return "BAJA"

    return "MUY BAJA"


# ======================================================================================
# ANALISIS TECNICO
# ======================================================================================


def evaluar_momento_tecnico(row):

    rsi = obtener_rsi(row)
    ret5 = obtener_return_5d(row)
    ret20 = obtener_return_20d(row)

    puntos = 0
    favor = []
    contra = []

    if not pd.isna(rsi):

        if rsi >= 80:
            puntos -= 3

            contra.append(
                f"RSI {rsi:.2f}: sobrecompra extrema; "
                f"riesgo elevado de correccion."
            )

        elif rsi >= 70:
            puntos -= 1

            contra.append(
                f"RSI {rsi:.2f}: activo en zona de "
                f"sobrecompra."
            )

        elif 50 <= rsi < 70:
            puntos += 1

            favor.append(
                f"RSI {rsi:.2f}: momentum positivo "
                f"sin sobrecompra extrema."
            )

        elif 30 <= rsi < 50:
            favor.append(
                f"RSI {rsi:.2f}: zona neutral."
            )

        else:
            contra.append(
                f"RSI {rsi:.2f}: debilidad tecnica / "
                f"sobreventa."
            )

    if not pd.isna(ret5):

        if ret5 > 0:
            favor.append(
                f"Momentum de 5 dias positivo: "
                f"{fmt_return_pct(ret5)}."
            )

        elif ret5 < 0:
            contra.append(
                f"Momentum de 5 dias negativo: "
                f"{fmt_return_pct(ret5)}."
            )

    if not pd.isna(ret20):

        if ret20 > 0:
            favor.append(
                f"Tendencia de 20 dias positiva: "
                f"{fmt_return_pct(ret20)}."
            )

        elif ret20 < 0:
            contra.append(
                f"Tendencia de 20 dias negativa: "
                f"{fmt_return_pct(ret20)}."
            )

    if puntos >= 1:
        estado = "FAVORABLE"

    elif puntos <= -2:
        estado = "DESFAVORABLE"

    else:
        estado = "NEUTRAL"

    return {
        "estado": estado,
        "puntos": puntos,
        "favor": favor,
        "contra": contra,
    }


# ======================================================================================
# MOTOR DE AUTORIZACION
# ======================================================================================


def autorizar_inversion(
    row,
    mercado,
    info,
    wf_eval,
):

    score = obtener_score(row)
    prob = obtener_probabilidad(row)
    senal = obtener_senal(row).upper()

    buy_votes = obtener_buy_votes(row)
    neutral_votes = obtener_neutral_votes(row)
    sell_votes = obtener_sell_votes(row)

    tecnico = evaluar_momento_tecnico(row)

    favor = []
    contra = []
    bloqueos = []

    puntos = 0

    # ------------------------------------------------------------------
    # SCORE
    # ------------------------------------------------------------------

    if pd.isna(score):
        bloqueos.append(
            "No existe Score Final valido."
        )

    elif score >= 80:
        puntos += 4
        favor.append(
            f"Score IA excepcional: {score:.2f}."
        )

    elif score >= 72:
        puntos += 3
        favor.append(
            f"Score IA muy alto: {score:.2f}."
        )

    elif score >= 65:
        puntos += 2
        favor.append(
            f"Score IA alto: {score:.2f}."
        )

    elif score >= 58:
        puntos += 0
        contra.append(
            f"Score IA solamente medio: {score:.2f}."
        )

    else:
        puntos -= 3
        contra.append(
            f"Score IA insuficiente: {score:.2f}."
        )

    # ------------------------------------------------------------------
    # SENAL
    # ------------------------------------------------------------------

    if "COMPRA FUERTE" in senal:
        puntos += 3

        favor.append(
            "El ranking clasifica al activo como "
            "COMPRA FUERTE."
        )

    elif "COMPRA" in senal:
        puntos += 2

        favor.append(
            "El ranking clasifica al activo como COMPRA."
        )

    elif "VIGILAR" in senal:
        puntos -= 1

        contra.append(
            "La señal actual es VIGILAR."
        )

    elif "EVITAR" in senal:
        puntos -= 5

        bloqueos.append(
            "El ranking actual indica EVITAR."
        )

    # ------------------------------------------------------------------
    # PROBABILIDAD
    # ------------------------------------------------------------------

    if not pd.isna(prob):

        prob_pct = (
            prob * 100
            if prob <= 1
            else prob
        )

        favor.append(
            f"Probabilidad de consenso IA: "
            f"{prob_pct:.2f}%."
        )

        if prob_pct < 10:
            contra.append(
                "La probabilidad absoluta de consenso "
                "es reducida."
            )

    # ------------------------------------------------------------------
    # VOTOS
    # ------------------------------------------------------------------

    total_votes = (
        buy_votes +
        neutral_votes +
        sell_votes
    )

    if total_votes > 0:

        if buy_votes > sell_votes:
            favor.append(
                f"Votos de modelos: "
                f"{int(buy_votes)} compra, "
                f"{int(neutral_votes)} neutral, "
                f"{int(sell_votes)} venta."
            )

        elif sell_votes > buy_votes:
            contra.append(
                f"Los votos no respaldan claramente "
                f"la compra: "
                f"{int(buy_votes)} compra, "
                f"{int(neutral_votes)} neutral, "
                f"{int(sell_votes)} venta."
            )

    # ------------------------------------------------------------------
    # REGIMEN
    # ------------------------------------------------------------------

    regimen = (
        texto_seguro(info.get("regimen"))
        .upper()
    )

    if any(
        x in regimen
        for x in [
            "BULL",
            "ALCISTA",
            "POSITIVE",
        ]
    ):
        puntos += 2

        favor.append(
            f"Regimen de mercado favorable: "
            f"{info['regimen']}."
        )

    elif "NEUTRAL" in regimen:
        puntos -= 1

        contra.append(
            "El regimen general del mercado es NEUTRAL."
        )

    elif any(
        x in regimen
        for x in [
            "BEAR",
            "BAJISTA",
            "EXTREME",
        ]
    ):
        puntos -= 4

        bloqueos.append(
            f"Regimen de mercado desfavorable: "
            f"{info['regimen']}."
        )

    # ------------------------------------------------------------------
    # DECISION ENGINE
    # ------------------------------------------------------------------

    decision = (
        texto_seguro(info.get("decision"))
        .upper()
    )

    if any(
        x in decision
        for x in [
            "COMPRAR",
            "COMPRA",
            "INVERTIR",
        ]
    ):
        puntos += 3

        favor.append(
            f"Decision Engine favorable: "
            f"{info['decision']}."
        )

    elif "VIGILAR" in decision:
        puntos -= 2

        contra.append(
            "Decision Engine global: VIGILAR."
        )

    elif any(
        x in decision
        for x in [
            "NO INVERTIR",
            "EVITAR",
            "SALIR",
        ]
    ):
        puntos -= 6

        bloqueos.append(
            f"Decision Engine desfavorable: "
            f"{info['decision']}."
        )

    # ------------------------------------------------------------------
    # WALK FORWARD
    # ------------------------------------------------------------------

    if wf_eval["estado"] == "FAVORABLE":
        puntos += 3
        favor.append(
            "La validacion walk-forward del mercado es favorable."
        )

    elif wf_eval["estado"] == "ACEPTABLE CON RIESGO":
        contra.append(
            "La validacion walk-forward es aceptable, pero "
            "mantiene riesgo economico relevante."
        )

    elif wf_eval["estado"] == "DEBIL":
        puntos -= 3
        contra.append(
            "La validacion walk-forward es debil."
        )

    elif wf_eval["estado"] == "RIESGO ALTO":
        puntos -= 4
        contra.append(
            "El mercado presenta RIESGO ALTO en el walk-forward "
            "(Max Drawdown historico superior al 50%). "
            "Es un filtro de riesgo del mercado, no una invalidacion "
            "historica especifica de este activo."
        )

    elif wf_eval["estado"] == "RIESGO EXTREMO":
        puntos -= 6
        contra.append(
            "El mercado presenta RIESGO EXTREMO en el walk-forward "
            "(Max Drawdown historico superior al 70%). "
            "Se exige confirmacion reforzada y menor exposicion, "
            "pero no se atribuye ese drawdown al activo individual."
        )

    elif wf_eval["estado"] == "DESFAVORABLE":
        puntos -= 6
        bloqueos.append(
            "La validacion walk-forward del mercado es desfavorable."
        )

    # ------------------------------------------------------------------
    # MOMENTO TECNICO
    # ------------------------------------------------------------------

    puntos += tecnico["puntos"]

    favor.extend(
        tecnico["favor"]
    )

    contra.extend(
        tecnico["contra"]
    )

    # ------------------------------------------------------------------
    # DECISION FINAL
    # ------------------------------------------------------------------

    if bloqueos:
        estado = "BLOQUEADA"

    elif tecnico["estado"] == "DESFAVORABLE":
        estado = "ESPERAR"

    elif "VIGILAR" in decision:
        estado = "ESPERAR"

    elif puntos >= 9:
        estado = "AUTORIZADA"

    elif puntos >= 5:
        estado = "ESPERAR CONFIRMACION"

    else:
        estado = "NO AUTORIZADA"

    # ------------------------------------------------------------------
    # CONFIANZA
    # ------------------------------------------------------------------

    if puntos >= 10:
        confianza = "ALTA"

    elif puntos >= 5:
        confianza = "MEDIA"

    else:
        confianza = "BAJA"

    return {
        "estado": estado,
        "confianza": confianza,
        "puntos": puntos,
        "favor": favor,
        "contra": contra,
        "bloqueos": bloqueos,
        "tecnico": tecnico,
    }


# ======================================================================================
# NIVELES DE RIESGO
# ======================================================================================


def niveles_referencia(row):

    close = obtener_close(row)
    atr = obtener_atr(row)

    if (
        pd.isna(close)
        or pd.isna(atr)
        or close <= 0
        or atr <= 0
    ):
        return None

    atr_pct = (atr / close) * 100.0

    if atr_pct <= 1.5:
        multiplicador = 1.5
        vol = "BAJA"
    elif atr_pct <= 3.5:
        multiplicador = 2.0
        vol = "MEDIA"
    else:
        multiplicador = 2.5
        vol = "ALTA"

    distancia = atr * multiplicador
    stop = max(0.0, close - distancia)
    tp1 = close + distancia * 1.5
    tp2 = close + distancia * 2.5
    riesgo_pct = (distancia / close) * 100.0

    return {
        "entrada": close,
        "stop": stop,
        "tp1": tp1,
        "tp2": tp2,
        "riesgo_pct": riesgo_pct,
        "atr_pct": atr_pct,
        "multiplicador": multiplicador,
        "volatilidad_atr": vol,
    }


# ======================================================================================
# EXPLICACION DE UN ACTIVO
# ======================================================================================


def generar_ficha_activo(
    row,
    mercado,
    info,
    wf,
    wf_eval,
):

    auth = autorizar_inversion(
        row,
        mercado,
        info,
        wf_eval,
    )

    ticker = obtener_ticker(row)
    nombre = obtener_nombre(row)
    score = obtener_score(row)
    prob = obtener_probabilidad(row)

    prob_pct = np.nan

    if not pd.isna(prob):
        prob_pct = (
            prob * 100
            if prob <= 1
            else prob
        )

    out = []

    out.append(
        f"{ticker} - {nombre}"
    )

    out.append("-" * 76)

    out.append(
        f"Oportunidad IA       : "
        f"{obtener_senal(row)}"
    )

    out.append(
        f"Score Final          : "
        f"{fmt_num(score)}"
    )

    out.append(
        f"Calidad IA           : "
        f"{calidad_score(score)}"
    )

    out.append(
        f"Probabilidad IA      : "
        f"{fmt_pct(prob_pct)}"
    )

    out.append(
        f"Autorizacion         : "
        f"{auth['estado']}"
    )

    out.append(
        f"Confianza            : "
        f"{auth['confianza']}"
    )

    out.append(
        f"Puntuacion decision  : "
        f"{auth['puntos']}"
    )

    out.append("")

    out.append("RAZONES A FAVOR")

    if auth["favor"]:
        for razon in auth["favor"]:
            out.append(
                f"  + {razon}"
            )
    else:
        out.append(
            "  + No se identificaron factores "
            "positivos suficientes."
        )

    out.append("")
    out.append("RAZONES EN CONTRA")

    if auth["contra"]:
        for razon in auth["contra"]:
            out.append(
                f"  - {razon}"
            )
    else:
        out.append(
            "  - No se identificaron objeciones "
            "relevantes."
        )

    if auth["bloqueos"]:

        out.append("")
        out.append("BLOQUEOS")

        for razon in auth["bloqueos"]:
            out.append(
                f"  ! {razon}"
            )

    out.append("")
    out.append("VALIDACION WALK-FORWARD")

    out.append(
        f"  Estado              : "
        f"{wf_eval['estado']}"
    )
    out.append(
        "  Alcance             : FILTRO HISTORICO DEL MERCADO; "
        "no valida directamente el Score Final V2.1 del activo"
    )

    out.append(
        f"  Puntuacion economica: "
        f"{wf_eval['puntos']}"
    )

    for razon_wf in wf_eval.get("razones", []):
        out.append(
            f"  - {razon_wf}"
        )

    if wf.get("disponible"):

        out.append(
            f"  Retorno historico   : "
            f"{fmt_pct(wf.get('retorno'))}"
        )

        out.append(
            f"  Win Rate            : "
            f"{fmt_pct(wf.get('win_rate'))}"
        )

        out.append(
            f"  Profit Factor       : "
            f"{fmt_num(wf.get('profit_factor'), 3)}"
        )

        out.append(
            f"  Max Drawdown        : "
            f"{fmt_pct(wf.get('max_drawdown'))}"
        )

        out.append(
            f"  Score historico     : "
            f"{texto_seguro(wf.get('score_usado'))}"
        )

    out.append("")
    out.append("MOMENTO TECNICO")

    out.append(
        f"  RSI 14              : "
        f"{fmt_num(obtener_rsi(row))}"
    )

    out.append(
        f"  Retorno 5d          : "
        f"{fmt_return_pct(obtener_return_5d(row))}"
    )

    out.append(
        f"  Retorno 20d         : "
        f"{fmt_return_pct(obtener_return_20d(row))}"
    )

    out.append(
        f"  Estado tecnico      : "
        f"{auth['tecnico']['estado']}"
    )

    niveles = niveles_referencia(row)

    out.append("")

    if auth["estado"] == "AUTORIZADA":

        out.append("PLAN OPERATIVO DE REFERENCIA")

        if niveles:

            out.append(
                f"  Entrada referencia  : "
                f"{niveles['entrada']:.4f}"
            )

            out.append(
                f"  Volatilidad ATR     : "
                f"{niveles['volatilidad_atr']} "
                f"({niveles['atr_pct']:.2f}% del precio)"
            )

            out.append(
                f"  Stop ATR x{niveles['multiplicador']:.1f}     : "
                f"{niveles['stop']:.4f}"
            )

            out.append(
                f"  TP1                 : "
                f"{niveles['tp1']:.4f}"
            )

            out.append(
                f"  TP2                 : "
                f"{niveles['tp2']:.4f}"
            )

            out.append(
                f"  Riesgo hasta stop   : "
                f"{niveles['riesgo_pct']:.2f}%"
            )

        else:
            out.append(
                "  No hay ATR/precio suficiente para "
                "calcular niveles operativos."
            )

    else:

        out.append("ACCION")

        if auth["estado"] == "ESPERAR":

            out.append(
                "  Mantener en vigilancia. Existe una "
                "oportunidad, pero el momento actual "
                "no justifica una entrada."
            )

        elif auth["estado"] == "ESPERAR CONFIRMACION":

            out.append(
                "  Esperar confirmacion adicional antes "
                "de comprometer capital."
            )

        elif auth["estado"] == "BLOQUEADA":

            out.append(
                "  NO INVERTIR actualmente. Uno o mas "
                "filtros de riesgo bloquean la operacion."
            )

        else:

            out.append(
                "  No existe autorizacion suficiente "
                "para invertir."
            )

    out.append("")
    out.append("QUE DEBERIA CAMBIAR PARA MEJORAR LA AUTORIZACION")

    if "VIGILAR" in info["decision"].upper():
        out.append(
            "  - El Decision Engine deberia abandonar "
            "el estado VIGILAR."
        )

    if wf_eval["estado"] in (
        "DEBIL",
        "DESFAVORABLE",
        "ACEPTABLE CON RIESGO",
    ):
        out.append(
            "  - Mejorar la evidencia historica walk-forward del mercado, "
            "especialmente el equilibrio entre retorno, Profit Factor "
            "y Max Drawdown."
        )

    elif wf_eval["estado"] in (
        "RIESGO ALTO",
        "RIESGO EXTREMO",
    ):
        out.append(
            "  - Mantener confirmacion reforzada mientras el filtro "
            "historico del mercado conserve un Max Drawdown elevado."
        )

    if auth["tecnico"]["estado"] == "DESFAVORABLE":
        out.append(
            "  - Normalizar el momento tecnico antes "
            "de considerar una entrada."
        )

    if obtener_score(row) < 65:
        out.append(
            "  - Elevar el Score Final por encima "
            "de la zona de alta calidad."
        )

    out.append("")

    return "\n".join(out)


# ======================================================================================
# INFORME POR MERCADO
# ======================================================================================


def generar_informe_mercado(
    mercado,
    df,
    contexto,
):

    info = interpretar_contexto(
        contexto
    )

    wf = parsear_walk_forward(
        contexto.get("walk", "")
    )

    wf_eval = evaluar_walk_forward(
        wf
    )

    out = []

    out.append(linea())
    out.append(
        f"MARKET AI - INFORME DE INVERSION | "
        f"{mercado.upper()}"
    )
    out.append(linea())
    out.append("")

    out.append(
        f"Regimen             : "
        f"{info['regimen']}"
    )

    out.append(
        f"Decision Engine     : "
        f"{info['decision']}"
    )

    out.append(
        f"Riesgo              : "
        f"{info['riesgo']}"
    )

    out.append(
        f"Market AI Index     : "
        f"{info['market_ai_index']}"
    )

    out.append(
        f"Exposicion motor    : "
        f"{info['exposicion']}"
    )

    out.append(
        f"Cash motor          : "
        f"{info['cash']}"
    )

    if info.get("cash_origen") == "DERIVADO DE EXPOSICION":
        out.append(
            "Cash origen         : DERIVADO (100% - exposicion motor)"
        )

    out.append(
        f"Walk-Forward        : "
        f"{wf_eval['estado']}"
    )

    if df.empty:
        out.append("")
        out.append(
            "No existe ranking disponible."
        )
        return "\n".join(out)

    df = df.sort_values(
        "_score_global",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # RESUMEN
    # ------------------------------------------------------------------

    out.append("")
    out.append(linea("-", 112))
    out.append("RESUMEN DE OPORTUNIDADES")
    out.append(linea("-", 112))

    encabezado = (
        f"{'#':>2} "
        f"{'TICKER':<14} "
        f"{'SENAL':<18} "
        f"{'SCORE':>7} "
        f"{'PROB':>8} "
        f"{'RSI':>7} "
        f"{'AUTORIZACION':<22}"
    )

    out.append(encabezado)
    out.append("-" * len(encabezado))

    top = df.head(
        TOP_MERCADO
    )

    autorizaciones = []

    for i, (_, row) in enumerate(
        top.iterrows(),
        start=1,
    ):

        auth = autorizar_inversion(
            row,
            mercado,
            info,
            wf_eval,
        )

        autorizaciones.append(
            auth
        )

        prob = obtener_probabilidad(row)

        if not pd.isna(prob):
            if prob <= 1:
                prob *= 100

        out.append(
            f"{i:>2} "
            f"{obtener_ticker(row):<14.14} "
            f"{obtener_senal(row):<18.18} "
            f"{fmt_num(obtener_score(row)):>7} "
            f"{fmt_pct(prob):>8} "
            f"{fmt_num(obtener_rsi(row)):>7} "
            f"{auth['estado']:<22.22}"
        )

    # ------------------------------------------------------------------
    # DECISION DEL MERCADO
    # ------------------------------------------------------------------

    out.append("")
    out.append(linea("-", 112))
    out.append("DECISION DE INVERSION DEL MERCADO")
    out.append(linea("-", 112))

    estados = [
        a["estado"]
        for a in autorizaciones
    ]

    autorizadas = estados.count(
        "AUTORIZADA"
    )

    bloqueadas = estados.count(
        "BLOQUEADA"
    )

    esperando = sum(
        1
        for x in estados
        if x.startswith("ESPERAR")
    )

    out.append(
        f"Activos autorizados : {autorizadas}"
    )

    out.append(
        f"Activos esperando   : {esperando}"
    )

    out.append(
        f"Activos bloqueados  : {bloqueadas}"
    )

    if autorizadas > 0:

        out.append(
            "Conclusion          : "
            "EXISTEN OPERACIONES AUTORIZADAS"
        )

    elif esperando > 0:

        out.append(
            "Conclusion          : "
            "VIGILAR - ESPERAR CONFIRMACION"
        )

    else:

        out.append(
            "Conclusion          : "
            "NO INVERTIR ACTUALMENTE"
        )

    # ------------------------------------------------------------------
    # EXPLICACION
    # ------------------------------------------------------------------

    out.append("")
    out.append(linea("-", 112))
    out.append("ANALISIS DETALLADO")
    out.append(linea("-", 112))
    out.append("")

    for _, row in top.iterrows():

        out.append(
            generar_ficha_activo(
                row,
                mercado,
                info,
                wf,
                wf_eval,
            )
        )

        out.append(
            linea(".", 90)
        )

    return "\n".join(out)


# ======================================================================================
# INFORME GLOBAL
# ======================================================================================


def generar_global(
    rankings,
    contextos,
):

    frames = []

    for mercado, df in rankings.items():

        if df.empty:
            continue

        temp = df.copy()

        temp["Mercado_Global"] = (
            mercado.upper()
        )

        frames.append(temp)

    if frames:

        global_df = pd.concat(
            frames,
            ignore_index=True,
            sort=False,
        )

        global_df = global_df.sort_values(
            "_score_global",
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)

    else:

        global_df = pd.DataFrame()

    out = []

    out.append(linea())
    out.append(
        "MARKET AI - INFORME GLOBAL DE INVERSION V3.1"
    )
    out.append(linea())
    out.append("")

    out.append(
        f"Generado             : "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    out.append(
        f"Mercados analizados  : "
        f"{sum(not x.empty for x in rankings.values())}"
    )

    out.append(
        f"Activos analizados   : "
        f"{sum(len(x) for x in rankings.values())}"
    )

    out.append(
        f"Capital referencia   : "
        f"{CAPITAL_BASE:,.2f}"
    )

    # ------------------------------------------------------------------
    # MERCADOS
    # ------------------------------------------------------------------

    out.append("")
    out.append(linea("-", 112))
    out.append("1. ESTADO DE LOS 9 MERCADOS")
    out.append(linea("-", 112))

    mercado_resultados = {}

    out.append(
        f"{'MERCADO':<14}"
        f"{'REGIMEN':<18}"
        f"{'DECISION':<20}"
        f"{'WF':<16}"
        f"{'AUTORIZADAS':>12}"
    )

    out.append("-" * 84)

    for mercado in MERCADOS:

        df = rankings.get(
            mercado,
            pd.DataFrame(),
        )

        contexto = contextos.get(
            mercado,
            {},
        )

        info = interpretar_contexto(
            contexto
        )

        wf = parsear_walk_forward(
            contexto.get("walk", "")
        )

        wf_eval = evaluar_walk_forward(
            wf
        )

        autorizadas = 0
        esperar = 0
        bloqueadas = 0

        if not df.empty:

            df_ord = df.sort_values(
                "_score_global",
                ascending=False,
                na_position="last",
            )

            for _, row in df_ord.head(
                TOP_MERCADO
            ).iterrows():

                auth = autorizar_inversion(
                    row,
                    mercado,
                    info,
                    wf_eval,
                )

                if auth["estado"] == "AUTORIZADA":
                    autorizadas += 1

                elif auth["estado"].startswith(
                    "ESPERAR"
                ):
                    esperar += 1

                elif auth["estado"] == "BLOQUEADA":
                    bloqueadas += 1

        mercado_resultados[mercado] = {
            "info": info,
            "wf": wf,
            "wf_eval": wf_eval,
            "autorizadas": autorizadas,
            "esperar": esperar,
            "bloqueadas": bloqueadas,
        }

        out.append(
            f"{mercado.upper():<14}"
            f"{info['regimen']:<18.18}"
            f"{info['decision']:<20.20}"
            f"{wf_eval['estado']:<16}"
            f"{autorizadas:>12}"
        )

    out.append("")
    out.append(
        "Nota V3.1: el Walk-Forward prioriza supervivencia economica "
        "y se interpreta como filtro historico del MERCADO."
    )
    out.append(
        "DD > 50% = RIESGO ALTO; DD > 70% = RIESGO EXTREMO. "
        "Penalizan la autorizacion, pero no bloquean por si solos "
        "a cada activo individual."
    )
    out.append(
        "DESFAVORABLE (retorno negativo o Profit Factor < 1) "
        "continua siendo un bloqueo duro."
    )

    # ------------------------------------------------------------------
    # TOP GLOBAL
    # ------------------------------------------------------------------

    out.append("")
    out.append(linea("-", 112))
    out.append("2. TOP OPORTUNIDADES GLOBALES")
    out.append(linea("-", 112))

    oportunidades = []

    if not global_df.empty:

        for _, row in global_df.iterrows():

            mercado = str(
                row["Mercado_Global"]
            ).lower()

            mr = mercado_resultados[
                mercado
            ]

            auth = autorizar_inversion(
                row,
                mercado,
                mr["info"],
                mr["wf_eval"],
            )

            oportunidades.append(
                (
                    row,
                    auth,
                    mercado,
                )
            )

    encabezado = (
        f"{'#':>2} "
        f"{'MERCADO':<12} "
        f"{'ACTIVO':<14} "
        f"{'SCORE':>7} "
        f"{'SENAL':<18} "
        f"{'AUTORIZACION':<22}"
    )

    out.append(encabezado)
    out.append("-" * len(encabezado))

    for i, (
        row,
        auth,
        mercado,
    ) in enumerate(
        oportunidades[:TOP_GLOBAL],
        start=1,
    ):

        out.append(
            f"{i:>2} "
            f"{mercado.upper():<12.12} "
            f"{obtener_ticker(row):<14.14} "
            f"{fmt_num(obtener_score(row)):>7} "
            f"{obtener_senal(row):<18.18} "
            f"{auth['estado']:<22.22}"
        )

    # ------------------------------------------------------------------
    # AUTORIZADAS / ESPERAR / BLOQUEADAS
    # ------------------------------------------------------------------

    autorizadas = [
        x
        for x in oportunidades
        if x[1]["estado"] == "AUTORIZADA"
    ]

    esperando = [
        x
        for x in oportunidades
        if x[1]["estado"].startswith(
            "ESPERAR"
        )
    ]

    bloqueadas = [
        x
        for x in oportunidades
        if x[1]["estado"] == "BLOQUEADA"
    ]

    out.append("")
    out.append(linea("-", 112))
    out.append("3. DECISION GLOBAL DE INVERSION")
    out.append(linea("-", 112))

    out.append(
        f"Operaciones autorizadas : "
        f"{len(autorizadas)}"
    )

    out.append(
        f"Oportunidades esperando : "
        f"{len(esperando)}"
    )

    out.append(
        f"Oportunidades bloqueadas: "
        f"{len(bloqueadas)}"
    )

    decisiones = [
        mercado_resultados[m]["info"][
            "decision"
        ].upper()
        for m in MERCADOS
    ]

    vigilando = sum(
        "VIGILAR" in d
        for d in decisiones
    )

    if len(autorizadas) == 0:

        estado_global = (
            "PRUDENCIA - NO HAY COMPRAS "
            "AUTORIZADAS"
        )

    elif vigilando >= 5:

        estado_global = (
            "PRUDENCIA - EXPOSICION REDUCIDA"
        )

    else:

        estado_global = (
            "EXISTEN OPORTUNIDADES AUTORIZADAS"
        )

    out.append("")
    out.append(
        f"Estado global           : "
        f"{estado_global}"
    )

    out.append(
        f"Mercados en VIGILAR     : "
        f"{vigilando}/{len(MERCADOS)}"
    )

    # ------------------------------------------------------------------
    # COMPRAR AHORA
    # ------------------------------------------------------------------

    out.append("")
    out.append(linea("-", 112))
    out.append("4. COMPRAR AHORA")
    out.append(linea("-", 112))

    if not autorizadas:

        out.append(
            "NINGUNA OPERACION AUTORIZADA."
        )

        out.append("")
        out.append(
            "Aunque existen activos con scores "
            "atractivos, los filtros superiores "
            "de regimen, Decision Engine, "
            "walk-forward o momento tecnico "
            "no permiten convertir esas señales "
            "en compras actualmente."
        )

    else:

        for row, auth, mercado in autorizadas:

            out.append(
                f"{mercado.upper():<12} "
                f"{obtener_ticker(row):<14} "
                f"Score {obtener_score(row):.2f} "
                f"| Confianza {auth['confianza']}"
            )

    # ------------------------------------------------------------------
    # ESPERAR
    # ------------------------------------------------------------------

    out.append("")
    out.append(linea("-", 112))
    out.append("5. MEJORES OPORTUNIDADES PARA VIGILAR")
    out.append(linea("-", 112))

    for row, auth, mercado in esperando[:10]:

        out.append(
            f"{mercado.upper():<12} "
            f"{obtener_ticker(row):<14} "
            f"Score {obtener_score(row):>6.2f} | "
            f"{auth['estado']}"
        )

        if auth["contra"]:
            out.append(
                f"   Motivo principal: "
                f"{auth['contra'][0]}"
            )

        elif auth["bloqueos"]:
            out.append(
                f"   Motivo principal: "
                f"{auth['bloqueos'][0]}"
            )

    # ------------------------------------------------------------------
    # BLOQUEADAS
    # ------------------------------------------------------------------

    out.append("")
    out.append(linea("-", 112))
    out.append("6. NO COMPRAR / BLOQUEADAS")
    out.append(linea("-", 112))

    if not bloqueadas:

        out.append(
            "No existen bloqueos absolutos "
            "entre las oportunidades analizadas."
        )

    else:

        for row, auth, mercado in bloqueadas[:10]:

            out.append(
                f"{mercado.upper():<12} "
                f"{obtener_ticker(row):<14} "
                f"Score {obtener_score(row):>6.2f}"
            )

            for razon in auth["bloqueos"][:2]:
                out.append(
                    f"   BLOQUEO: {razon}"
                )

    # ------------------------------------------------------------------
    # EXPLICACION DE LAS PRINCIPALES
    # ------------------------------------------------------------------

    out.append("")
    out.append(linea("-", 112))
    out.append("7. POR QUE MARKET AI TOMA ESTA DECISION")
    out.append(linea("-", 112))

    out.append("")
    out.append(
        "Market AI separa deliberadamente una "
        "OPORTUNIDAD de una AUTORIZACION DE INVERSION."
    )

    out.append("")
    out.append(
        "Un activo puede tener un Score Final elevado "
        "y aun asi quedar en espera o bloqueado."
    )

    out.append("")
    out.append(
        "La autorizacion considera conjuntamente:"
    )

    out.append(
        "  - calidad del Score Final;"
    )

    out.append(
        "  - señal del ranking;"
    )

    out.append(
        "  - consenso de modelos;"
    )

    out.append(
        "  - regimen general del mercado;"
    )

    out.append(
        "  - Decision Engine;"
    )

    out.append(
        "  - validacion walk-forward;"
    )

    out.append(
        "  - drawdown historico;"
    )

    out.append(
        "  - Profit Factor;"
    )

    out.append(
        "  - RSI y momento tecnico."
    )

    out.append("")
    out.append(
        "Por ello, COMPRA FUERTE no significa "
        "automaticamente COMPRAR AHORA."
    )

    # ------------------------------------------------------------------
    # DETALLE TOP 5 GLOBAL
    # ------------------------------------------------------------------

    out.append("")
    out.append(linea("-", 112))
    out.append("8. ANALISIS DE LAS 5 PRINCIPALES OPORTUNIDADES")
    out.append(linea("-", 112))
    out.append("")

    for row, auth, mercado in oportunidades[:5]:

        mr = mercado_resultados[
            mercado
        ]

        out.append(
            generar_ficha_activo(
                row,
                mercado,
                mr["info"],
                mr["wf"],
                mr["wf_eval"],
            )
        )

        out.append(
            linea(".", 100)
        )

    # ------------------------------------------------------------------
    # INFORMES INDIVIDUALES
    # ------------------------------------------------------------------

    out.append("")
    out.append(linea())
    out.append("INFORMES DETALLADOS DE LOS 9 MERCADOS")
    out.append(linea())

    for mercado in MERCADOS:

        out.append("")
        out.append(
            generar_informe_mercado(
                mercado,
                rankings.get(
                    mercado,
                    pd.DataFrame(),
                ),
                contextos.get(
                    mercado,
                    {},
                ),
            )
        )

    # ------------------------------------------------------------------
    # CONCLUSION FINAL
    # ------------------------------------------------------------------

    out.append("")
    out.append(linea())
    out.append("CONCLUSION FINAL MARKET AI")
    out.append(linea())
    out.append("")

    out.append(
        f"Estado general        : "
        f"{estado_global}"
    )

    out.append(
        f"Mercados en vigilancia: "
        f"{vigilando}/{len(MERCADOS)}"
    )

    out.append(
        f"Compras autorizadas   : "
        f"{len(autorizadas)}"
    )

    out.append(
        f"En espera             : "
        f"{len(esperando)}"
    )

    out.append(
        f"Bloqueadas            : "
        f"{len(bloqueadas)}"
    )

    out.append("")

    if not autorizadas:

        out.append(
            "DECISION: preservar capital y esperar."
        )

        out.append("")
        out.append(
            "El sistema detecta oportunidades "
            "estadisticas, pero actualmente no "
            "encuentra suficiente confirmacion "
            "conjunta para transformar esas "
            "oportunidades en operaciones."
        )

    else:

        out.append(
            "DECISION: existen operaciones "
            "autorizadas, manteniendo disciplina "
            "de riesgo y tamaño de posicion."
        )

    out.append("")
    out.append(
        "Este informe no interpreta el Score Final "
        "como probabilidad de rentabilidad."
    )

    out.append(
        "Score, probabilidad IA y autorizacion son "
        "conceptos diferentes."
    )

    out.append("")
    out.append(
        "La prioridad del sistema es preservar "
        "capital cuando la evidencia es contradictoria."
    )

    out.append(
        "V3.1 mantiene la regla de supervivencia, pero distingue "
        "riesgo historico del mercado de validacion individual del activo."
    )

    return "\n".join(out)


# ======================================================================================
# MAIN
# ======================================================================================


def main():

    print("")
    print(linea())
    print(
        "MARKET AI - GLOBAL INVESTMENT REPORT V3.1"
    )
    print(linea())
    print("")

    GLOBAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rankings = {}
    contextos = {}

    for mercado in MERCADOS:

        print(
            f"Leyendo {mercado.upper():12} ...",
            end=" ",
        )

        df = cargar_ranking(
            mercado
        )

        contexto = cargar_contexto_txt(
            mercado
        )

        rankings[mercado] = df
        contextos[mercado] = contexto

        if df.empty:

            print(
                "SIN RANKING"
            )

        else:

            print(
                f"OK | {len(df)} activos"
            )

        informe_individual = (
            generar_informe_mercado(
                mercado,
                df,
                contexto,
            )
        )

        path_individual = (
            GLOBAL_DIR /
            f"informe_{mercado}.txt"
        )

        path_individual.write_text(
            informe_individual,
            encoding="utf-8",
        )

    informe_global = generar_global(
        rankings,
        contextos,
    )

    GLOBAL_REPORT.write_text(
        informe_global,
        encoding="utf-8",
    )

    print("")
    print(linea("-"))
    print("ARCHIVOS GENERADOS")
    print(linea("-"))
    print("")

    print(
        f"GLOBAL : {GLOBAL_REPORT}"
    )

    print(
        f"DETALLE: {GLOBAL_DIR}"
    )

    print("")
    print(
        "OK - INFORME GLOBAL V3.1 GENERADO"
    )
    print(linea())


if __name__ == "__main__":
    main()
