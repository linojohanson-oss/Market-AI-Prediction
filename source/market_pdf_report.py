from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional
import sys
import unicodedata

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.market_scoring import generar_market_scoring


def _pdf_text(value: Any) -> str:
    """Convierte texto a una forma compatible con las fuentes base de FPDF."""
    text = str(value)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _normalizar(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    return "".join(char for char in text if not unicodedata.combining(char)).lower()


def _buscar_columna(df: pd.DataFrame, *candidatas: str) -> Optional[str]:
    disponibles = {_normalizar(columna): columna for columna in df.columns}
    for candidata in candidatas:
        encontrada = disponibles.get(_normalizar(candidata))
        if encontrada is not None:
            return encontrada
    return None


def _valor_dict(datos: dict, *claves: str, default: Any = "-") -> Any:
    normalizadas = {_normalizar(clave): valor for clave, valor in datos.items()}
    for clave in claves:
        if _normalizar(clave) in normalizadas:
            return normalizadas[_normalizar(clave)]
    return default


def _ranking_desde_dict(scoring: dict) -> list[dict]:
    ranking = _valor_dict(
        scoring,
        "ranking_top10",
        "ranking",
        "top10",
        default=[],
    )
    return ranking if isinstance(ranking, list) else []


def _ordenar_scoring(df: pd.DataFrame) -> tuple[pd.DataFrame, Optional[str]]:
    score_col = _buscar_columna(df, "Score_Final", "score_final", "score", "Score")
    if score_col is None:
        return df.tail(10).copy(), None

    trabajo = df.copy()
    trabajo[score_col] = pd.to_numeric(trabajo[score_col], errors="coerce")
    trabajo = trabajo.dropna(subset=[score_col])
    return trabajo.sort_values(score_col, ascending=False).head(10), score_col


def _buscar_historico(mercado: str) -> Optional[Path]:
    procesados = ROOT / "datos" / mercado / "procesados"
    candidatos = [
        procesados / f"{mercado.upper()}_INDICADORES.csv",
        procesados / f"{mercado}_indicadores.csv",
    ]

    for candidato in candidatos:
        if candidato.exists():
            return candidato

    if procesados.exists():
        encontrados = sorted(procesados.glob("*INDICADORES*.csv"))
        if encontrados:
            return encontrados[0]
    return None


def _agregar_linea(pdf, etiqueta: str, valor: Any, altura: int = 7) -> None:
    pdf.cell(0, altura, _pdf_text(f"{etiqueta}: {valor}"), ln=True)


def _agregar_ranking_dict(pdf, ranking: Iterable[dict]) -> None:
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, _pdf_text("Top oportunidades"), ln=True)
    pdf.set_font("Arial", "", 10)

    for posicion, item in enumerate(ranking, start=1):
        ticker = _valor_dict(item, "ticker", "Ticker", default="-")
        nombre = _valor_dict(item, "name", "nombre", "Nombre", default="")
        score = _valor_dict(item, "score", "Score_Final", default="-")
        signal = _valor_dict(item, "signal", "senal", "Señal", default="-")
        linea = f"{posicion:02}. {ticker} | {nombre} | Score {score} | {signal}"
        pdf.multi_cell(0, 6, _pdf_text(linea))


def _agregar_ranking_dataframe(pdf, top: pd.DataFrame, score_col: Optional[str]) -> None:
    ticker_col = _buscar_columna(top, "Ticker", "ticker", "Symbol", "simbolo")
    nombre_col = _buscar_columna(top, "Nombre", "name", "descripcion")
    senal_col = _buscar_columna(top, "Señal", "senal", "signal")

    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, _pdf_text("Top oportunidades"), ln=True)
    pdf.set_font("Arial", "", 10)

    for posicion, (_, fila) in enumerate(top.iterrows(), start=1):
        ticker = fila.get(ticker_col, "-") if ticker_col else "-"
        nombre = fila.get(nombre_col, "") if nombre_col else ""
        score = fila.get(score_col, "-") if score_col else "-"
        senal = fila.get(senal_col, "-") if senal_col else "-"
        if isinstance(score, (int, float)):
            score = f"{score:.2f}"
        linea = f"{posicion:02}. {ticker} | {nombre} | Score {score} | {senal}"
        pdf.multi_cell(0, 6, _pdf_text(linea))


def _crear_grafico_ranking(
    top: pd.DataFrame,
    score_col: Optional[str],
    salida: Path,
) -> bool:
    if top.empty or score_col is None:
        return False

    ticker_col = _buscar_columna(top, "Ticker", "ticker", "Symbol", "simbolo")
    if ticker_col is None:
        return False

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(top[ticker_col].astype(str), top[score_col], color="#2f80ed")
    ax.set_title("Ranking Top 10 - Score final")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(salida, dpi=140)
    plt.close(fig)
    return True


def _crear_grafico_historico(
    mercado: str,
    ticker: str,
    salida: Path,
) -> bool:
    historico_path = _buscar_historico(mercado)
    if historico_path is None:
        return False

    df = pd.read_csv(historico_path)
    ticker_col = _buscar_columna(df, "Ticker", "ticker", "Symbol", "simbolo")
    fecha_col = _buscar_columna(df, "Date", "fecha", "datetime")
    close_col = _buscar_columna(df, "Close", "cierre", "adj close", "Adj Close")

    if ticker_col is None or fecha_col is None or close_col is None:
        return False

    trabajo = df[df[ticker_col].astype(str).eq(str(ticker))].copy()
    if trabajo.empty:
        return False

    trabajo[fecha_col] = pd.to_datetime(trabajo[fecha_col], errors="coerce")
    trabajo[close_col] = pd.to_numeric(trabajo[close_col], errors="coerce")
    trabajo = trabajo.dropna(subset=[fecha_col, close_col]).sort_values(fecha_col).tail(60)
    if trabajo.empty:
        return False

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(trabajo[fecha_col], trabajo[close_col], color="#27ae60", linewidth=1.8)
    ax.set_title(f"{ticker} - ultimas 60 ruedas")
    ax.set_ylabel("Cierre")
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(salida, dpi=140)
    plt.close(fig)
    return True


def generar_reporte_pdf_market_ai(mercado: str, scoring: Any = None) -> Path:
    """Genera el reporte PDF sin recalcular el scoring si el pipeline lo aporta."""
    try:
        from fpdf import FPDF
    except ImportError as exc:
        raise RuntimeError(
            "No esta instalada la dependencia fpdf. "
            "Instale el paquete fpdf2 en el entorno de Market."
        ) from exc

    mercado = str(mercado).lower().strip()
    if scoring is None:
        scoring = generar_market_scoring(mercado)

    output_dir = ROOT / "reportes" / mercado
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"market_ai_daily_report_{mercado}.pdf"

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "MARKET AI DAILY REPORT", ln=True)
    pdf.set_font("Arial", "", 11)
    _agregar_linea(pdf, "Mercado", mercado.upper())
    _agregar_linea(pdf, "Generado", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    pdf.ln(5)

    top = pd.DataFrame()
    score_col = None
    mejor_ticker = None

    if isinstance(scoring, dict):
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 8, _pdf_text("Sintesis de mercado"), ln=True)
        pdf.set_font("Arial", "", 10)

        campos = [
            ("Regimen", ("regimen",)),
            ("Accion recomendada", ("accion", "accion_recomendada")),
            ("Decision IA", ("decision",)),
            ("Market AI Index", ("mai", "market_ai_index")),
            ("Semaforo", ("semaforo",)),
            ("Cash recomendado", ("cash",)),
            ("Exposicion", ("acciones", "exposicion")),
            ("Maximo de posiciones", ("max_pos", "maximo_posiciones")),
        ]
        for etiqueta, claves in campos:
            valor = _valor_dict(scoring, *claves, default=None)
            if valor is not None:
                _agregar_linea(pdf, etiqueta, valor)

        ranking = _ranking_desde_dict(scoring)
        if ranking:
            pdf.ln(4)
            _agregar_ranking_dict(pdf, ranking[:10])
            mejor_ticker = _valor_dict(ranking[0], "ticker", "Ticker", default=None)
    elif isinstance(scoring, pd.DataFrame):
        top, score_col = _ordenar_scoring(scoring)
        _agregar_ranking_dataframe(pdf, top, score_col)
        ticker_col = _buscar_columna(top, "Ticker", "ticker", "Symbol", "simbolo")
        if ticker_col and not top.empty:
            mejor_ticker = top.iloc[0][ticker_col]
    else:
        raise TypeError(
            "El scoring debe ser un diccionario o un DataFrame de pandas."
        )

    if not top.empty:
        grafico_ranking = output_dir / f"ranking_top10_{mercado}.png"
        try:
            if _crear_grafico_ranking(top, score_col, grafico_ranking):
                pdf.ln(5)
                pdf.image(str(grafico_ranking), w=180)
        except Exception as exc:
            print(f"AVISO PDF: no se pudo crear el grafico de ranking: {exc}")

    if mejor_ticker:
        grafico_historico = output_dir / f"historico_{mejor_ticker}.png"
        try:
            if _crear_grafico_historico(mercado, str(mejor_ticker), grafico_historico):
                pdf.ln(5)
                pdf.set_font("Arial", "B", 13)
                pdf.cell(0, 8, _pdf_text(f"Evolucion reciente - {mejor_ticker}"), ln=True)
                pdf.image(str(grafico_historico), w=180)
        except Exception as exc:
            print(f"AVISO PDF: no se pudo crear el grafico historico: {exc}")

    pdf.output(str(pdf_path))
    print(f"PDF generado: {pdf_path}")
    return pdf_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Genera el reporte PDF de Market AI")
    parser.add_argument("mercado", nargs="?", default="merval")
    argumentos = parser.parse_args()
    generar_reporte_pdf_market_ai(argumentos.mercado)
