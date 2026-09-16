import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from config import MarketConfig


def clasificar_regimen(prob_media, prob_max, score_medio, compras, evitar, total):
    evitar_pct = evitar / total if total > 0 else 0
    compras_pct = compras / total if total > 0 else 0

    if compras_pct >= 0.35 and score_medio >= 65 and prob_media >= 0.10:
        return (
            "ALCISTA",
            "COMPRAR SELECTIVO",
            "Mercado favorable. Se permiten compras selectivas."
        )

    if score_medio >= 55 and prob_max >= 0.07:
        return (
            "NEUTRAL",
            "VIGILAR",
            "Mercado sin senal dominante. Conviene monitorear."
        )

    if evitar_pct >= 0.60 or score_medio < 40:
        return (
            "BAJISTA / RIESGO ALTO",
            "ESPERAR / EVITAR COMPRAS",
            "Mercado debil. Conviene priorizar liquidez y evitar nuevas compras."
        )

    return (
        "DEBIL",
        "ESPERAR",
        "Mercado debil o mixto. Solo considerar oportunidades excepcionales."
    )


def generar_market_regime(nombre_mercado="merval"):
    inicio = datetime.now()
    market = MarketConfig(nombre_mercado)

    reportes_dir = ROOT / "reportes" / market.market

    ranking_path = reportes_dir / "ranking_diario.csv"

    csv_out = reportes_dir / f"market_regime_{market.market}_v2.csv"
    txt_out = reportes_dir / f"market_regime_{market.market}_v2.txt"

    if not ranking_path.exists():
        raise FileNotFoundError(f"No existe ranking diario: {ranking_path}")

    ranking = pd.read_csv(ranking_path)

    if "Consensus_Probability" in ranking.columns:
        prob_col = "Consensus_Probability"
    elif "Prob_IA" in ranking.columns:
        prob_col = "Prob_IA"
    else:
        raise ValueError("No se encontro columna de probabilidad.")

    if "Score_Final" not in ranking.columns:
        raise ValueError("No se encontro columna Score_Final.")

    if "Señal" not in ranking.columns:
        raise ValueError("No se encontro columna Señal.")

    fecha = ranking["Date"].max() if "Date" in ranking.columns else datetime.now().date()

    prob_media = ranking[prob_col].mean()
    prob_max = ranking[prob_col].max()
    score_medio = ranking["Score_Final"].mean()

    compras = ranking["Señal"].isin(["COMPRA", "COMPRA FUERTE"]).sum()
    evitar = (ranking["Señal"] == "EVITAR").sum()
    total = len(ranking)

    regimen, accion, descripcion = clasificar_regimen(
        prob_media,
        prob_max,
        score_medio,
        compras,
        evitar,
        total
    )

    resultado = {
        "Fecha": fecha,
        "Mercado": market.market,
        "Regimen": regimen,
        "Accion_Recomendada": accion,
        "Descripcion": descripcion,
        "Prob_Media": prob_media,
        "Prob_Max": prob_max,
        "Score_Medio": score_medio,
        "Compras": compras,
        "Evitar": evitar,
        "Acciones_Analizadas": total,
    }

    df_resultado = pd.DataFrame([resultado])

    df_resultado.to_csv(csv_out, index=False, encoding="utf-8-sig")

    with open(txt_out, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"MARKET REGIME DETECTOR V2 - {market.display_name}\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Fecha              : {fecha}\n")
        f.write(f"Mercado            : {market.display_name}\n")
        f.write(f"Regimen            : {regimen}\n")
        f.write(f"Accion recomendada : {accion}\n")
        f.write(f"Descripcion        : {descripcion}\n\n")

        f.write(f"Prob media         : {prob_media:.4f}\n")
        f.write(f"Prob maxima        : {prob_max:.4f}\n")
        f.write(f"Score medio        : {score_medio:.2f}\n")
        f.write(f"Compras            : {compras}\n")
        f.write(f"Evitar             : {evitar}\n")
        f.write(f"Activos analizados : {total}\n")

    print("=" * 80)
    print(f"MARKET REGIME DETECTOR V2 - {market.display_name}")
    print("=" * 80)
    print()
    print("=" * 80)
    print("REGIMEN DETECTADO")
    print("=" * 80)
    print(f"Fecha              : {fecha}")
    print(f"Mercado            : {market.display_name}")
    print(f"Regimen            : {regimen}")
    print(f"Accion recomendada : {accion}")
    print(f"Descripcion        : {descripcion}")
    print()
    print(f"Prob media         : {prob_media:.4f}")
    print(f"Prob maxima        : {prob_max:.4f}")
    print(f"Score medio        : {score_medio:.2f}")
    print(f"Compras            : {compras}")
    print(f"Evitar             : {evitar}")
    print()
    print(f"CSV generado       : {csv_out}")
    print(f"TXT generado       : {txt_out}")
    print(f"Tiempo             : {datetime.now() - inicio}")

    return df_resultado


def detectar_regimen_mercado(nombre_mercado="merval"):
    return generar_market_regime(nombre_mercado)


if __name__ == "__main__":
    import sys

    mercado = sys.argv[1].lower() if len(sys.argv) > 1 else "merval"
    generar_market_regime(mercado)
