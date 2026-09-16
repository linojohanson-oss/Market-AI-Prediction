import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import pandas as pd
import numpy as np

from config import MarketConfig

TARGETS = [1, 5, 10, 20, 60]


# ============================================================
# VOLATILIDAD REALIZADA (RV)
# ============================================================
def add_realized_volatility(df, windows=(7, 14, 30)):
    df = df.sort_values("Date").copy()
    df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))

    for w in windows:
        df[f"RV_{w}"] = (
            df["log_ret"]
            .rolling(window=w)
            .apply(lambda x: np.sqrt((x ** 2).sum()), raw=True)
        )

    return df


# ============================================================
# RSI
# ============================================================
def calcular_rsi(close, window=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ============================================================
# ATR
# ============================================================
def calcular_atr(df, window=14):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window).mean()


# ============================================================
# INDICADORES POR ACTIVO
# ============================================================
def calcular_indicadores_activo(df):
    df = df.sort_values("Date").copy()

    for col in ["Close", "Open", "High", "Low"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Volume"] = pd.to_numeric(df.get("Volume", 0), errors="coerce")

    # Retornos
    df["Return_1d"] = df["Close"].pct_change(1)
    df["Return_5d"] = df["Close"].pct_change(5)
    df["Return_10d"] = df["Close"].pct_change(10)
    df["Return_20d"] = df["Close"].pct_change(20)

    # Medias simples
    for w in [5, 10, 20, 50, 100, 200]:
        df[f"SMA_{w}"] = df["Close"].rolling(w).mean()

    # Medias exponenciales
    for w in [12, 20, 26, 50, 100, 200]:
        df[f"EMA_{w}"] = df["Close"].ewm(span=w, adjust=False).mean()

    # Momentum
    df["RSI_14"] = calcular_rsi(df["Close"], 14)
    df["ROC_10"] = df["Close"].pct_change(10)
    df["Momentum_10"] = df["Close"] - df["Close"].shift(10)

    # MACD
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # Bollinger Bands
    sma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    df["BB_Upper"] = sma20 + 2 * std20
    df["BB_Lower"] = sma20 - 2 * std20
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / sma20

    # Volatilidad clÃ¡sica
    df["ATR_14"] = calcular_atr(df, 14)
    df["Volatility_20"] = df["Return_1d"].rolling(20).std()

    # ðŸ”¥ Volatilidad Realizada (RV)
    df = add_realized_volatility(df)

    # Volumen
    df["Volume_SMA20"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA20"].replace(0, np.nan)

    # Tendencia
    df["Close_vs_SMA20"] = df["Close"] / df["SMA_20"] - 1
    df["Close_vs_SMA50"] = df["Close"] / df["SMA_50"] - 1
    df["Close_vs_SMA200"] = df["Close"] / df["SMA_200"] - 1

    df["Trend_SMA20_50"] = np.where(df["SMA_20"] > df["SMA_50"], 1, 0)
    df["Trend_SMA50_200"] = np.where(df["SMA_50"] > df["SMA_200"], 1, 0)

    # ADX
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    plus_dm = high.diff()
    minus_dm = low.diff()

    plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
    minus_dm = (-minus_dm).where((-minus_dm > 0) & (-minus_dm > plus_dm), 0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr14 = tr.rolling(14).mean()

    df["DI_plus"] = 100 * (plus_dm.rolling(14).mean() / atr14)
    df["DI_minus"] = 100 * (minus_dm.rolling(14).mean() / atr14)

    dx = (abs(df["DI_plus"] - df["DI_minus"]) /
          (df["DI_plus"] + df["DI_minus"])).replace(0, np.nan) * 100

    df["ADX_14"] = dx.rolling(14).mean()

    # OBV
    df["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()

    # VWAP
    df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].replace(0, np.nan).cumsum()

    # Stochastic
    low14 = df["Low"].rolling(14).min()
    high14 = df["High"].rolling(14).max()
    df["Stoch_%K"] = 100 * (df["Close"] - low14) / (high14 - low14)
    df["Stoch_%D"] = df["Stoch_%K"].rolling(3).mean()

    # CCI
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    sma_tp = tp.rolling(20).mean()
    mad = (tp - sma_tp).abs().rolling(20).mean()
    df["CCI_20"] = (tp - sma_tp) / (0.015 * mad)

    # Williams %R
    df["Williams_%R"] = -100 * (high14 - df["Close"]) / (high14 - low14)

    # Keltner Channels
    ema20 = df["Close"].ewm(span=20, adjust=False).mean()
    df["Keltner_Upper"] = ema20 + 2 * df["ATR_14"]
    df["Keltner_Lower"] = ema20 - 2 * df["ATR_14"]

    # Chaikin Money Flow
    mf_mult = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"]).replace(0, np.nan)
    mf_vol = mf_mult * df["Volume"]
    df["CMF_20"] = mf_vol.rolling(20).sum() / df["Volume"].rolling(20).sum().replace(0, np.nan)

    # Volatilidad avanzada
    df["Volatility_60"] = df["Return_1d"].rolling(60).std()
    df["Volatility_120"] = df["Return_1d"].rolling(120).std()
    df["Volatility_Annualized"] = df["Return_1d"].rolling(60).std() * np.sqrt(252)

    # Connors RSI
    rsi2 = calcular_rsi(df["Close"], 2)
    rsi7 = calcular_rsi(df["Close"], 7)
    up_days = (df["Close"].diff() > 0).astype(int)
    streak = up_days.groupby((up_days != up_days.shift()).cumsum()).cumsum()
    df["CRSI"] = (rsi2 + rsi7 + streak) / 3

    # Targets
    for t in TARGETS:
        df[f"Future_Close_{t}d"] = df["Close"].shift(-t)
        df[f"Future_Return_{t}d"] = df[f"Future_Close_{t}d"] / df["Close"] - 1
        future_return = df[f"Future_Return_{t}d"]

        # Una observación sin precio futuro todavía no tiene etiqueta.
        # np.where(future_return > 0, 1, 0) convertía los NaN en cero y
        # contaminaba el entrenamiento FULL con falsas caídas recientes.
        df[f"Target_{t}d"] = np.where(
            future_return.isna(),
            np.nan,
            np.where(future_return > 0, 1.0, 0.0),
        )

    return df


# ============================================================
# CORRELACIONES DINÃMICAS (UNA SOLA VEZ)
# ============================================================
def add_rolling_correlations_market(base, market, windows=(30, 60)):
    tickers = base["Ticker"].unique()

    if len(tickers) == 0:
        return base

    benchmark = market.benchmark

    if benchmark not in tickers:
        print(f"Benchmark {benchmark} no encontrado en datos.")
        return base

    print(f"Benchmark para correlaciones dinÃ¡micas: {benchmark}")

    pivot = base.pivot(index="Date", columns="Ticker", values="Close").sort_index()

    corr_frames = []

    for w in windows:
        corr_matrix = pivot.rolling(w).corr(pivot[benchmark])

        corr_df = corr_matrix.reset_index().rename(columns={0: "Date"})
        corr_df = corr_df.melt(id_vars="Date", var_name="Ticker", value_name=f"Corr_vs_{benchmark}_{w}")

        corr_frames.append(corr_df)

    corr_final = corr_frames[0]
    for extra in corr_frames[1:]:
        corr_final = corr_final.merge(extra, on=["Date", "Ticker"], how="left")

    base = base.merge(corr_final, on=["Date", "Ticker"], how="left")

    return base


# ============================================================
# GENERAR INDICADORES POR MERCADO
# ============================================================
def generar_indicadores(nombre_mercado="merval"):
    inicio = datetime.now()
    market = MarketConfig(nombre_mercado)

    print("\n" + "=" * 80)
    print(f"MARKET INDICATORS - {market.display_name}")
    print("=" * 80)
    print(f"Inicio: {inicio}")
    print(f"Archivo origen: {market.full_raw_csv}")

    if not market.full_raw_csv.exists():
        raise FileNotFoundError(
            f"\nâŒ ERROR: No existe archivo base: {market.full_raw_csv}\n"
            f"EjecutÃ¡ primero market_downloader.py"
        )

    df = pd.read_csv(market.full_raw_csv)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    print(f"Registros base: {len(df):,}")
    print(f"Activos detectados: {df['Ticker'].nunique()}")
    print(f"Rango temporal: {df['Date'].min()} â†’ {df['Date'].max()}")
    print("-" * 80)

    resultados = []
    fallidos = []

    tickers = df["Ticker"].unique()

    for i, ticker in enumerate(tickers, start=1):
        print(f"[{i}/{len(tickers)}] Procesando {ticker}...")

        try:
            df_ticker = df[df["Ticker"] == ticker].copy()
            procesado = calcular_indicadores_activo(df_ticker)
            resultados.append(procesado)

        except Exception as e:
            print(f"âŒ ERROR en {ticker}: {e}")
            fallidos.append(ticker)

    base = pd.concat(resultados, ignore_index=True)
    base.sort_values(["Ticker", "Date"], inplace=True)

    # Control de integridad: nunca puede existir target cuando no existe
    # el retorno futuro que lo determina.
    for t in TARGETS:
        retorno = f"Future_Return_{t}d"
        target = f"Target_{t}d"
        invalidos = base[retorno].isna() & base[target].notna()
        if invalidos.any():
            raise RuntimeError(
                f"Targets futuros inválidos en horizonte {t}d: "
                f"{int(invalidos.sum())} filas"
            )

    # ðŸ”¥ Correlaciones dinÃ¡micas (una sola vez)
    base = add_rolling_correlations_market(base, market)

    base.to_csv(market.indicators_csv, index=False, encoding="utf-8-sig")
    base.to_excel(market.indicators_excel, index=False)

    print("\n" + "=" * 80)
    print(f"RESUMEN FINAL - {market.display_name}")
    print("=" * 80)
    print(f"Activos procesados: {base['Ticker'].nunique()}")
    print(f"Registros totales: {len(base):,}")
    print(f"Target 5d conocido: {base['Target_5d'].notna().sum():,}")
    print(f"Target 5d pendiente: {base['Target_5d'].isna().sum():,}")
    print(f"Fallidos: {len(fallidos)} â†’ {fallidos if fallidos else 'Ninguno'}")
    print(f"CSV generado: {market.indicators_csv}")
    print(f"Excel generado: {market.indicators_excel}")
    print(f"Tiempo total: {datetime.now() - inicio}")
    print("=" * 80)

    return base


if __name__ == "__main__":
    generar_indicadores("merval")
