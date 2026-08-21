import os
import json
import sys
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from datetime import datetime
from pathlib import Path

BASE_PATH = str(Path(__file__).resolve().parent / "reportes")
JSON_GLOBAL = os.path.join(BASE_PATH, "pipeline_maestro.json")
LOGO_PATH = str(Path(__file__).resolve().parent / "logo.png")


def _normalizar_senales(senales):
    base = {
        "COMPRA_FUERTE": [],
        "COMPRA": [],
        "VIGILAR": [],
        "NEUTRAL": [],
        "EVITAR": []
    }
    if isinstance(senales, dict):
        for k in base:
            v = senales.get(k, [])
            base[k] = v if isinstance(v, list) else []
    return base


def normalizar_resumen(data):
    """Convierte el JSON global a un dict homogéneo apto para gráficos/PDF."""
    if isinstance(data, dict):
        items = list(data.items())
        normalizado = {}
        for key, info in items:
            if not isinstance(info, dict):
                continue
            nombre_mercado = str(info.get("mercado", key)).upper()
            normalizado[str(key)] = {
                "mercado": nombre_mercado,
                "señales": _normalizar_senales(info.get("señales", {}))
            }
        return normalizado

    if isinstance(data, list):
        normalizado = {}
        for i, info in enumerate(data):
            if not isinstance(info, dict):
                continue
            raw_market = str(info.get("mercado", f"mercado_{i + 1}")).strip()
            key = raw_market.lower() or f"mercado_{i + 1}"
            normalizado[key] = {
                "mercado": raw_market.upper(),
                "señales": _normalizar_senales(info.get("señales", {}))
            }
        return normalizado

    return {}


# ============================================================
# CARGAR RESUMEN
# ============================================================
def cargar_resumen():
    if not os.path.exists(JSON_GLOBAL):
        print(f"❌ ERROR: No se encontró {JSON_GLOBAL}")
        return None

    with open(JSON_GLOBAL, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception as e:
            print("❌ ERROR: El JSON está corrupto:", e)
            return None

    data = normalizar_resumen(data)

    if not data:
        print("❌ ERROR: El JSON está vacío")
        return None

    return data


# ============================================================
# GRÁFICO — SEÑALES POR MERCADO (PRIMERO)
# ============================================================
def generar_grafico_senales(resumen):
    mercados = []
    compras = []
    evitar = []

    for mercado, info in resumen.items():
        mercados.append(info.get("mercado", str(mercado).upper()))
        compras.append(
            len(info.get("señales", {}).get("COMPRA", [])) +
            len(info.get("señales", {}).get("COMPRA_FUERTE", []))
        )
        evitar.append(len(info.get("señales", {}).get("EVITAR", [])))

    plt.figure(figsize=(10, 6))
    if mercados:
        plt.bar(mercados, compras, color="green", label="Compras")
        plt.bar(mercados, evitar, bottom=compras, color="red", label="Evitar")
        plt.legend()
    else:
        plt.text(0.5, 0.5, "Sin datos de señales", ha="center", va="center", fontsize=14)
        plt.xticks([])
        plt.yticks([])

    plt.title("Señales por Mercado", fontsize=16)
    plt.xlabel("Mercado")
    plt.ylabel("Cantidad de señales")
    plt.tight_layout()

    ruta = os.path.join(BASE_PATH, "grafico_senales.png")
    plt.savefig(ruta)
    plt.close()

    return ruta


# ============================================================
# GRÁFICO — ALFA POR MERCADO
# ============================================================
def generar_graficos_alfa_por_mercado(resumen):
    rutas = []

    for mercado, info in resumen.items():
        señales = info.get("señales", {})
        activos = []
        scores = []

        for tipo, lista in señales.items():
            for s in lista:
                activos.append(s["ticker"])
                scores.append(s["score"])

        if not activos:
            continue

        plt.figure(figsize=(10, 6))
        plt.barh(activos, scores, color="#1f77b4")
        plt.title(f"Ranking de Alfa - {info['mercado']}", fontsize=16, pad=15)
        plt.xlabel("Score Final")
        plt.tight_layout()

        ruta = os.path.join(BASE_PATH, f"alfa_{info['mercado']}.png")
        plt.savefig(ruta)
        plt.close()

        rutas.append((info["mercado"], ruta))

    return rutas


# ============================================================
# GRÁFICO — TOP 10 ALFA GLOBAL
# ============================================================
def generar_top10_alfa(resumen):
    lista = []

    for mercado, info in resumen.items():
        for tipo, señales in info.get("señales", {}).items():
            for s in señales:
                ticker = s.get("ticker")
                score = s.get("score")
                if ticker is None or score is None:
                    continue
                lista.append((ticker, score))

    lista.sort(key=lambda x: x[1], reverse=True)
    top10 = lista[:10]

    plt.figure(figsize=(10, 6))
    if top10:
        activos = [x[0] for x in top10]
        scores = [x[1] for x in top10]
        plt.barh(activos, scores, color="#003366")
    else:
        plt.text(0.5, 0.5, "Sin señales para Top 10", ha="center", va="center", fontsize=14)
        plt.xticks([])
        plt.yticks([])

    plt.title("Top 10 Alfa Global", fontsize=16, pad=15)
    plt.xlabel("Score Final")
    plt.tight_layout()

    ruta = os.path.join(BASE_PATH, "top10_alfa.png")
    plt.savefig(ruta)
    plt.close()

    return ruta


# ============================================================
# FOOTER
# ============================================================
def agregar_footer(c):
    fecha = datetime.now().strftime("%d/%m/%Y %H:%M")
    c.setFont("Helvetica", 10)
    c.drawString(50, 20, f"Market AI Prediction — {fecha}")


# ============================================================
# PDF PRO — COMPATIBLE CON ONEDRIVE + LOGO + FOOTER + SEÑALES
# ============================================================
def generar_pdf_pro(resumen, rutas_alfa, ruta_top10, ruta_senales):
    pdf_final = os.path.join(BASE_PATH, "dashboard_global_graficos_PRO.pdf")
    pdf_temp = os.path.join(BASE_PATH, "temp_output.pdf")

    c = canvas.Canvas(pdf_temp, pagesize=A4)
    width, height = A4

    # PORTADA CON LOGO
    if os.path.exists(LOGO_PATH):
        c.drawImage(LOGO_PATH, 50, height - 200, width=150, preserveAspectRatio=True)

    c.setFont("Helvetica-Bold", 30)
    c.drawString(50, height - 250, "Reporte Global de Mercados")

    c.setFont("Helvetica", 18)
    c.drawString(50, height - 290, "Market AI Prediction")

    fecha = datetime.now().strftime("%d/%m/%Y %H:%M")
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 320, f"Generado el {fecha}")

    agregar_footer(c)
    c.showPage()

    # ============================
    # GRÁFICO DE SEÑALES (PRIMERO)
    # ============================
    c.setFont("Helvetica-Bold", 22)
    c.drawString(50, height - 70, "0. Señales por Mercado")

    c.drawImage(ruta_senales, 50, height - 650, width=500, preserveAspectRatio=True)

    agregar_footer(c)
    c.showPage()

    # ============================
    # RESUMEN EJECUTIVO
    # ============================
    c.setFont("Helvetica-Bold", 22)
    c.drawString(50, height - 70, "1. Resumen Ejecutivo")

    c.setFont("Helvetica", 12)
    y = height - 120

    for mercado, info in resumen.items():
        compras = len(info.get("señales", {}).get("COMPRA", [])) + len(info.get("señales", {}).get("COMPRA_FUERTE", []))
        evitar = len(info.get("señales", {}).get("EVITAR", []))

        texto = f"{info.get('mercado', str(mercado).upper())}: {compras} compras, {evitar} evitar"
        c.drawString(50, y, texto)
        y -= 20

    agregar_footer(c)
    c.showPage()

    # ============================
    # ALFA POR MERCADO
    # ============================
    for mercado, ruta in rutas_alfa:
        c.setFont("Helvetica-Bold", 22)
        c.drawString(50, height - 70, f"2. Ranking de Alfa - {mercado}")
        c.drawImage(ruta, 50, height - 650, width=500, preserveAspectRatio=True)
        agregar_footer(c)
        c.showPage()

    # ============================
    # TOP 10 GLOBAL
    # ============================
    c.setFont("Helvetica-Bold", 22)
    c.drawString(50, height - 70, "3. Top 10 Alfa Global")
    c.drawImage(ruta_top10, 50, height - 650, width=500, preserveAspectRatio=True)
    agregar_footer(c)
    c.showPage()

    c.save()

    # Reemplazo compatible con OneDrive
    try:
        os.replace(pdf_temp, pdf_final)
        print("✔ PDF PRO generado:", pdf_final)
    except PermissionError:
        print("❌ OneDrive está bloqueando el archivo final.")
        print("Cerrá el PDF si está abierto y volvé a ejecutar.")


# ============================================================
# MAIN
# ============================================================
def main():
    resumen = cargar_resumen()
    if resumen is None:
        print("❌ No se pudo generar el reporte.")
        return 1

    ruta_senales = generar_grafico_senales(resumen)
    rutas_alfa = generar_graficos_alfa_por_mercado(resumen)
    ruta_top10 = generar_top10_alfa(resumen)

    generar_pdf_pro(resumen, rutas_alfa, ruta_top10, ruta_senales)
    return 0


if __name__ == "__main__":
    sys.exit(main())
