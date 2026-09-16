from __future__ import annotations

import json
import os
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORTES_DIR = ROOT / "reportes"
GRAFICOS_DIR = REPORTES_DIR / "global" / "graficos"
JSON_GLOBAL = REPORTES_DIR / "resumen_global.json"
PDF_FINAL = REPORTES_DIR / "dashboard_global_graficos_PRO.pdf"
LOGO_PATH = ROOT / "logo.png"

MERCADOS_DEFAULT = [
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

CATEGORIAS = [
    "COMPRA_FUERTE",
    "COMPRA",
    "VIGILAR",
    "NEUTRAL",
    "EVITAR",
]


def _normalizar(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return text.strip().lower()


def _texto_pdf(value: Any) -> str:
    text = str(value)
    text = unicodedata.normalize("NFKD", text)
    return "".join(char for char in text if not unicodedata.combining(char))


def _buscar_columna(df: pd.DataFrame, *candidatas: str) -> Optional[str]:
    mapa = {_normalizar(columna): columna for columna in df.columns}
    for candidata in candidatas:
        encontrada = mapa.get(_normalizar(candidata))
        if encontrada is not None:
            return encontrada
    return None


def _leer_csv(path: Path) -> pd.DataFrame:
    ultimo_error = None
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            ultimo_error = exc
    if ultimo_error:
        raise ultimo_error
    return pd.read_csv(path)


def _mercados_activos() -> list[str]:
    try:
        from config import MARKET_DEFINITIONS, SUPPORTED_MARKETS

        activos = []
        for mercado in SUPPORTED_MARKETS:
            definicion = MARKET_DEFINITIONS.get(mercado, {})
            if definicion.get("symbols"):
                activos.append(str(mercado).lower())
        if activos:
            return activos
    except Exception as exc:
        print(f"AVISO PDF GLOBAL: no se pudo leer config.py: {exc}")
    return MERCADOS_DEFAULT.copy()


def _display_name(mercado: str) -> str:
    try:
        from config import MARKET_DEFINITIONS

        definicion = MARKET_DEFINITIONS.get(mercado, {})
        return str(definicion.get("display_name", mercado.upper()))
    except Exception:
        return mercado.upper()


def _buscar_ranking(mercado: str) -> Optional[Path]:
    carpeta = REPORTES_DIR / mercado
    candidatos = [
        carpeta / "ranking_diario.csv",
        carpeta / f"ranking_diario_{mercado}.csv",
    ]
    for candidato in candidatos:
        if candidato.exists():
            return candidato
    if carpeta.exists():
        encontrados = sorted(carpeta.glob("*ranking*diario*.csv"))
        if encontrados:
            return encontrados[0]
    return None


def _cargar_modelo_activo(mercado: str) -> dict:
    path = REPORTES_DIR / mercado / "modelo_activo.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        print(f"AVISO {mercado.upper()}: modelo_activo.json invalido: {exc}")
        return {}


def _categoria_senal(value: Any) -> str:
    key = _normalizar(value).upper().replace(" ", "_").replace("-", "_")
    equivalencias = {
        "COMPRA_FUERTE": "COMPRA_FUERTE",
        "COMPRAR_FUERTE": "COMPRA_FUERTE",
        "COMPRA": "COMPRA",
        "COMPRAR": "COMPRA",
        "VIGILAR": "VIGILAR",
        "OBSERVAR": "VIGILAR",
        "NEUTRAL": "NEUTRAL",
        "MANTENER": "NEUTRAL",
        "EVITAR": "EVITAR",
        "VENTA": "EVITAR",
        "VENDER": "EVITAR",
    }
    return equivalencias.get(key, "NEUTRAL")


def _preparar_ranking(df: pd.DataFrame, mercado: str) -> pd.DataFrame:
    ticker_col = _buscar_columna(df, "Ticker", "ticker", "Activo", "Symbol")
    score_col = _buscar_columna(
        df,
        "Score_Final",
        "score_final",
        "Score",
        "Consensus_Score",
    )
    senal_col = _buscar_columna(df, "Señal", "Senal", "Signal")

    if ticker_col is None or score_col is None:
        raise RuntimeError(
            f"El ranking de {mercado.upper()} no contiene Ticker y Score_Final."
        )

    trabajo = pd.DataFrame(
        {
            "mercado": _display_name(mercado),
            "ticker": df[ticker_col].astype(str).str.strip(),
            "score": pd.to_numeric(df[score_col], errors="coerce"),
            "senal": (
                df[senal_col].map(_categoria_senal)
                if senal_col is not None
                else "NEUTRAL"
            ),
        }
    )
    trabajo = trabajo.dropna(subset=["score"])
    trabajo = trabajo[trabajo["ticker"].ne("")]
    trabajo = trabajo.sort_values("score", ascending=False)
    trabajo = trabajo.drop_duplicates(subset=["ticker"], keep="first")
    return trabajo.reset_index(drop=True)


def construir_resumen() -> tuple[dict, dict[str, pd.DataFrame], list[str]]:
    resumen = {}
    rankings = {}
    faltantes = []

    for mercado in _mercados_activos():
        path = _buscar_ranking(mercado)
        if path is None:
            faltantes.append(mercado)
            print(f"AVISO {mercado.upper():15}: no existe ranking_diario.csv")
            continue

        try:
            ranking = _preparar_ranking(_leer_csv(path), mercado)
        except Exception as exc:
            faltantes.append(mercado)
            print(f"AVISO {mercado.upper():15}: ranking invalido: {exc}")
            continue

        if ranking.empty:
            faltantes.append(mercado)
            print(f"AVISO {mercado.upper():15}: ranking vacio")
            continue

        rankings[mercado] = ranking
        senales = {categoria: [] for categoria in CATEGORIAS}

        for _, fila in ranking.iterrows():
            senales[fila["senal"]].append(
                {
                    "ticker": fila["ticker"],
                    "score": round(float(fila["score"]), 6),
                }
            )

        resumen[mercado] = {
            "mercado": _display_name(mercado),
            "modelo_activo": _cargar_modelo_activo(mercado),
            "señales": senales,
            "fuente_ranking": str(path),
            "fecha_actualizacion": datetime.now().isoformat(),
        }

        print(
            f"OK {mercado.upper():15}: "
            f"{len(ranking):3} activos | {path.name}"
        )

    return resumen, rankings, faltantes


def guardar_resumen_global(resumen: dict) -> Path:
    REPORTES_DIR.mkdir(parents=True, exist_ok=True)
    JSON_GLOBAL.write_text(
        json.dumps(resumen, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    return JSON_GLOBAL


def generar_grafico_senales(rankings: dict[str, pd.DataFrame]) -> Path:
    mercados = []
    compras = []
    vigilar = []
    neutrales = []
    evitar = []

    for ranking in rankings.values():
        mercados.append(str(ranking.iloc[0]["mercado"]))
        conteos = ranking["senal"].value_counts()
        compras.append(int(conteos.get("COMPRA", 0) + conteos.get("COMPRA_FUERTE", 0)))
        vigilar.append(int(conteos.get("VIGILAR", 0)))
        neutrales.append(int(conteos.get("NEUTRAL", 0)))
        evitar.append(int(conteos.get("EVITAR", 0)))

    salida = GRAFICOS_DIR / "grafico_senales.png"
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(mercados, compras, color="#27ae60", label="Compra")
    ax.bar(mercados, vigilar, bottom=compras, color="#f2c94c", label="Vigilar")
    base_neutral = [a + b for a, b in zip(compras, vigilar)]
    ax.bar(mercados, neutrales, bottom=base_neutral, color="#95a5a6", label="Neutral")
    base_evitar = [a + b + c for a, b, c in zip(compras, vigilar, neutrales)]
    ax.bar(mercados, evitar, bottom=base_evitar, color="#eb5757", label="Evitar")
    ax.set_title("Señales actuales por mercado")
    ax.set_ylabel("Cantidad de activos")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    fig.savefig(salida, dpi=150)
    plt.close(fig)
    return salida


def generar_graficos_por_mercado(
    rankings: dict[str, pd.DataFrame],
) -> list[tuple[str, Path]]:
    rutas = []
    for mercado, ranking in rankings.items():
        top = ranking.head(12).sort_values("score", ascending=True)
        salida = GRAFICOS_DIR / f"alfa_{mercado}.png"
        colores = [
            {
                "COMPRA_FUERTE": "#1b8f4b",
                "COMPRA": "#27ae60",
                "VIGILAR": "#f2c94c",
                "NEUTRAL": "#95a5a6",
                "EVITAR": "#eb5757",
            }.get(senal, "#2f80ed")
            for senal in top["senal"]
        ]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top["ticker"], top["score"], color=colores)
        ax.set_title(f"Ranking de oportunidades - {_display_name(mercado)}")
        ax.set_xlabel("Score final")
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        fig.savefig(salida, dpi=150)
        plt.close(fig)
        rutas.append((_display_name(mercado), salida))
    return rutas


def generar_top_global(rankings: dict[str, pd.DataFrame]) -> Path:
    combinado = pd.concat(rankings.values(), ignore_index=True)
    top = combinado.sort_values("score", ascending=False).head(15).copy()
    top["etiqueta"] = top["ticker"] + " · " + top["mercado"]
    top = top.sort_values("score", ascending=True)

    salida = GRAFICOS_DIR / "top15_global.png"
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["etiqueta"], top["score"], color="#173f73")
    ax.set_title("Top 15 de oportunidades globales")
    ax.set_xlabel("Score final")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(salida, dpi=150)
    plt.close(fig)
    return salida


def _footer(pdf: canvas.Canvas) -> None:
    width, _ = A4
    pdf.setFont("Helvetica", 8)
    pdf.drawString(45, 22, f"Market AI Prediction - {datetime.now():%d/%m/%Y %H:%M}")
    pdf.drawRightString(width - 45, 22, f"Pagina {pdf.getPageNumber()}")


def _pagina_grafico(pdf: canvas.Canvas, titulo: str, imagen: Path) -> None:
    width, height = A4
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(45, height - 60, _texto_pdf(titulo))
    pdf.drawImage(
        str(imagen),
        45,
        90,
        width=width - 90,
        height=height - 180,
        preserveAspectRatio=True,
        anchor="c",
    )
    _footer(pdf)
    pdf.showPage()


def generar_pdf(
    resumen: dict,
    rankings: dict[str, pd.DataFrame],
    faltantes: list[str],
    grafico_senales: Path,
    graficos_mercado: list[tuple[str, Path]],
    grafico_global: Path,
) -> Path:
    REPORTES_DIR.mkdir(parents=True, exist_ok=True)
    temporal = PDF_FINAL.with_suffix(".tmp.pdf")
    pdf = canvas.Canvas(str(temporal), pagesize=A4)
    width, height = A4

    if LOGO_PATH.exists():
        pdf.drawImage(
            str(LOGO_PATH),
            45,
            height - 190,
            width=145,
            height=100,
            preserveAspectRatio=True,
        )

    pdf.setFont("Helvetica-Bold", 28)
    pdf.drawString(45, height - 235, "Reporte Global de Mercados")
    pdf.setFont("Helvetica", 16)
    pdf.drawString(45, height - 270, "Market AI Prediction - V2.1")
    pdf.setFont("Helvetica", 11)
    pdf.drawString(45, height - 305, f"Generado: {datetime.now():%d/%m/%Y %H:%M}")
    pdf.drawString(45, height - 325, f"Mercados incluidos: {len(rankings)}")
    if faltantes:
        pdf.drawString(
            45,
            height - 345,
            _texto_pdf("Sin ranking actual: " + ", ".join(m.upper() for m in faltantes)),
        )
    _footer(pdf)
    pdf.showPage()

    _pagina_grafico(pdf, "1. Señales por mercado", grafico_senales)

    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(45, height - 60, "2. Resumen ejecutivo")
    pdf.setFont("Helvetica-Bold", 9)
    y = height - 100
    pdf.drawString(45, y, "MERCADO")
    pdf.drawString(155, y, "TOP ACTIVO")
    pdf.drawString(280, y, "SCORE")
    pdf.drawString(345, y, "COMPRAS")
    pdf.drawString(420, y, "EVITAR")
    y -= 18
    pdf.setFont("Helvetica", 9)
    for mercado, ranking in rankings.items():
        top = ranking.iloc[0]
        conteos = ranking["senal"].value_counts()
        compras = int(conteos.get("COMPRA", 0) + conteos.get("COMPRA_FUERTE", 0))
        pdf.drawString(45, y, _texto_pdf(_display_name(mercado)))
        pdf.drawString(155, y, _texto_pdf(top["ticker"]))
        pdf.drawString(280, y, f"{float(top['score']):.2f}")
        pdf.drawString(345, y, str(compras))
        pdf.drawString(420, y, str(int(conteos.get("EVITAR", 0))))
        y -= 18
    _footer(pdf)
    pdf.showPage()

    for mercado, ruta in graficos_mercado:
        _pagina_grafico(pdf, f"3. Ranking - {mercado}", ruta)

    _pagina_grafico(pdf, "4. Top 15 global", grafico_global)
    pdf.save()
    os.replace(temporal, PDF_FINAL)
    return PDF_FINAL


def generar_dashboard_global_pdf() -> Path:
    print("=" * 88)
    print("MARKET AI - PDF GLOBAL CON GRAFICOS V2.1")
    print("=" * 88)

    GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)
    resumen, rankings, faltantes = construir_resumen()

    if not rankings:
        raise RuntimeError(
            "No se encontro ningun ranking_diario.csv actual para generar el PDF global."
        )

    json_path = guardar_resumen_global(resumen)
    grafico_senales = generar_grafico_senales(rankings)
    graficos_mercado = generar_graficos_por_mercado(rankings)
    grafico_global = generar_top_global(rankings)
    pdf_path = generar_pdf(
        resumen,
        rankings,
        faltantes,
        grafico_senales,
        graficos_mercado,
        grafico_global,
    )

    print("-" * 88)
    print(f"Mercados incluidos : {len(rankings)}")
    print(f"Mercados faltantes : {len(faltantes)}")
    print(f"Resumen actualizado: {json_path}")
    print(f"Graficos           : {GRAFICOS_DIR}")
    print(f"PDF global         : {pdf_path}")
    print("=" * 88)
    print("OK PDF GLOBAL V2.1 GENERADO")
    return pdf_path


def main() -> None:
    generar_dashboard_global_pdf()


if __name__ == "__main__":
    main()
