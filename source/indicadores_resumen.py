from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent


INDICADORES_POSIBLES = [
    "ADX_14",
    "CMF_20",
    "OBV",
    "VWAP",
    "Volatility_Annualized",
    "Stoch_%K",
    "Stoch_%D",
    "CCI_20",
    "Williams_%R",
    "CRSI",
]


def generar_indicadores_resumen(mercado: str):
    mercado = mercado.lower().strip()
    mercado_upper = mercado.upper()

    data_dir = ROOT / "datos" / mercado / "procesados"
    report_dir = ROOT / "reportes" / mercado

    indicadores_path = (
        data_dir / f"{mercado_upper}_INDICADORES.csv"
    )

    output_csv = (
        report_dir / f"indicadores_resumen_{mercado}.csv"
    )

    output_excel = (
        report_dir / f"indicadores_resumen_{mercado}.xlsx"
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print(
        f"RESUMEN INDICADORES TÉCNICOS - "
        f"{mercado_upper}"
    )
    print("=" * 80)
    print(f"Archivo origen : {indicadores_path}")

    if not indicadores_path.exists():
        pd.DataFrame().to_csv(
            output_csv,
            index=False,
        )

        print(
            "⚠ No hay archivo de indicadores "
            f"para {mercado_upper}."
        )
        return None

    df = pd.read_csv(indicadores_path)

    if df.empty:
        pd.DataFrame().to_csv(
            output_csv,
            index=False,
        )

        print(
            f"⚠ El archivo de indicadores de "
            f"{mercado_upper} está vacío."
        )
        return None

    resumen = {
        "Mercado": mercado_upper,
        "Registros": len(df),
    }

    ticker_col = None

    for candidato in [
        "Ticker",
        "ticker",
        "Symbol",
        "symbol",
    ]:
        if candidato in df.columns:
            ticker_col = candidato
            break

    if ticker_col:
        resumen["Activos"] = int(
            df[ticker_col].nunique()
        )
    else:
        resumen["Activos"] = None

    for indicador in INDICADORES_POSIBLES:
        if indicador in df.columns:
            serie = pd.to_numeric(
                df[indicador],
                errors="coerce",
            )

            resumen[indicador] = (
                serie.mean()
                if serie.notna().any()
                else None
            )
        else:
            resumen[indicador] = None

    resumen_df = pd.DataFrame(
        [resumen]
    )

    resumen_df.to_csv(
        output_csv,
        index=False,
    )

    resumen_df.to_excel(
        output_excel,
        index=False,
    )

    print("-" * 80)
    print(
        f"Mercado          : {mercado_upper}"
    )
    print(
        f"Registros        : {resumen['Registros']:,}"
    )
    print(
        f"Activos          : {resumen['Activos']}"
    )

    print("-" * 80)

    for indicador in INDICADORES_POSIBLES:
        valor = resumen[indicador]

        if pd.isna(valor):
            texto = "NO DISPONIBLE"
        else:
            texto = f"{valor:.6f}"

        print(
            f"{indicador:<24}: {texto}"
        )

    print("-" * 80)
    print(f"CSV generado     : {output_csv}")
    print(f"Excel generado   : {output_excel}")
    print("=" * 80)

    return resumen_df


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit(
            "Uso: python "
            "source\\indicadores_resumen.py "
            "<mercado>"
        )

    generar_indicadores_resumen(
        sys.argv[1]
    )