import json
import os
import pdfkit

BASE_PATH = r"C:\Users\feder\OneDrive\Desktop\OneDrive\PycharmProjects\Market-AI-Prediction\reportes"

# ============================================================
# Cargar resumen global
# ============================================================
def cargar_resumen_global():
    ruta = os.path.join(BASE_PATH, "resumen_global.json")
    with open(ruta, "r") as f:
        return json.load(f)

# ============================================================
# Colores por tipo de señal
# ============================================================
def color_señal(señal):
    señal = señal.upper()
    if señal == "COMPRA_FUERTE":
        return "#00aa00"
    if señal == "COMPRA":
        return "#33cc33"
    if señal == "VIGILAR":
        return "#ffaa00"
    if señal == "NEUTRAL":
        return "#cccccc"
    if señal == "EVITAR":
        return "#ff4444"
    return "#ffffff"

# ============================================================
# Generar HTML del dashboard
# ============================================================
def generar_html(resumen):

    html = """
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Reporte Global de Mercados</title>
        <style>
            body { font-family: Arial; margin: 40px; }
            h1 { color: #333; }
            h2 { color: #444; margin-top: 40px; }
            table { width: 100%; border-collapse: collapse; margin-top: 10px; }
            th, td { padding: 8px; border: 1px solid #ddd; text-align: left; }
            th { background-color: #f2f2f2; }
            .resumen { background-color: #eef; padding: 15px; border-radius: 8px; }
        </style>
    </head>
    <body>
        <h1>Reporte Global de Mercados</h1>
        <p>Generado automáticamente por Market AI Prediction</p>
    """

    # ============================================================
    # Resumen ejecutivo (placeholder para expandir)
    # ============================================================
    html += """
    <div class="resumen">
        <h2>Resumen Ejecutivo</h2>
        <p>Este reporte resume las señales detectadas en todos los mercados procesados por la plataforma.</p>
        <p>Incluye modelos activos, señales por mercado y puntuaciones finales.</p>
    </div>
    """

    # ============================================================
    # Secciones por mercado
    # ============================================================
    for mercado_key, info in resumen.items():
        mercado = info["mercado"]
        modelo = info.get("modelo_activo", {})
        señales = info.get("señales", {})

        html += f"<h2>{mercado}</h2>"
        html += f"<p><b>Modelo activo:</b> {modelo.get('modelo')} (Score: {modelo.get('score_modelo'):.4f})</p>"

        html += "<table>"
        html += "<tr><th>Señal</th><th>Ticker</th><th>Score</th></tr>"

        for tipo, lista in señales.items():
            for s in lista:
                color = color_señal(tipo)
                html += f"""
                <tr style="background-color:{color}">
                    <td>{tipo}</td>
                    <td>{s['ticker']}</td>
                    <td>{s['score']:.2f}</td>
                </tr>
                """

        html += "</table>"

    html += """
    </body>
    </html>
    """

    return html

# ============================================================
# Guardar HTML
# ============================================================
def guardar_html(html):
    ruta = os.path.join(BASE_PATH, "dashboard_global.html")
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n✔ Dashboard HTML generado correctamente")
    print(f"Archivo: {ruta}\n")
    return ruta

# ============================================================
# Generar PDF
# ============================================================
def generar_pdf(html_path):
    pdf_path = os.path.join(BASE_PATH, "dashboard_global.pdf")

    config = pdfkit.configuration(
        wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
    )

    pdfkit.from_file(html_path, pdf_path, configuration=config)

    print("✔ PDF generado correctamente")
    print(f"Archivo: {pdf_path}\n")

# ============================================================
# MAIN
# ============================================================
def main():
    resumen = cargar_resumen_global()
    html = generar_html(resumen)
    html_path = guardar_html(html)
    generar_pdf(html_path)

if __name__ == "__main__":
    main()
