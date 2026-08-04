import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import pandas as pd
import numpy as np

from config import MarketConfig, get_json_config_value


TARGETS = get_json_config_value("pipeline.target_horizons", [1, 5, 10, 20, 60])
TARGET_5D_MIN_RETURN = get_json_config_value("pipeline.target_min_return_5d", 0.0)


def resolver_columna(df, posibles):
    for col in posibles:
        if col in df.columns:
            return col
    return None


def normalizar_columnas_indicadores(df):
    df = df.copy()

    col_ticker = resolver_columna(df, ["Ticker", "ticker", "Symbol", "symbol"])
    col_fecha = resolver_columna(df, ["Date", "date", "Fecha", "fecha", "Datetime", "datetime"])

    if col_ticker is not None and col_ticker != "Ticker":
        df.rename(columns={col_ticker: "Ticker"}, inplace=True)

    if col_fecha is not None and col_fecha != "Date":
        df.rename(columns={col_fecha: "Date"}, inplace=True)

    return df


def calcular_rsi(close, window=14):
    delta = close.diff()

    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()

    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calcular_atr(df, window=14):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()

    return atr


def calcular_adx(df, window=14):
    up_move = df["High"].diff()
    down_move = -df["Low"].diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs(),
    ], axis=1).max(axis=1)

    tr14 = tr.rolling(window).sum()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(window).sum() / tr14.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(window).sum() / tr14.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.rolling(window).mean()

    return adx.fillna(0)


def calcular_indicadores_adicionales(df):
    df = df.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # OBV
    price_change = close.diff()
    obv = pd.Series(0, index=df.index, dtype=float)
    obv[price_change > 0] = volume[price_change > 0]
    obv[price_change < 0] = -volume[price_change < 0]
    obv = obv.cumsum()
    df["OBV"] = obv

    # VWAP
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum().replace(0, np.nan)
    df["VWAP"] = vwap

    # Stochastic %K and %D
    lowest_low = low.rolling(14).min()
    highest_high = high.rolling(14).max()
    k_pct = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    d_pct = k_pct.rolling(3).mean()
    df["Stoch_K"] = k_pct
    df["Stoch_D"] = d_pct

    # CCI
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(20).mean()
    mean_dev = (tp - sma_tp).abs().rolling(20).mean()
    cci = (tp - sma_tp) / (0.015 * mean_dev.replace(0, np.nan))
    df["CCI"] = cci

    # Williams %R
    highest_high = high.rolling(14).max()
    lowest_low = low.rolling(14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low).replace(0, np.nan)
    df["Williams_R"] = williams_r

    # Keltner Channels
    ema_mid = close.ewm(span=20, adjust=False).mean()
    atr20 = calcular_atr(df, 14)
    upper = ema_mid + (2 * atr20)
    lower = ema_mid - (2 * atr20)
    df["Keltner_Upper"] = upper
    df["Keltner_Lower"] = lower
    df["Keltner_Width"] = (upper - lower) / ema_mid.replace(0, np.nan)

    # CMF
    money_flow_multiplier = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    money_flow_volume = money_flow_multiplier * volume
    cmf = money_flow_volume.rolling(20).sum() / volume.rolling(20).sum().replace(0, np.nan)
    df["CMF"] = cmf

    # Volatilidad avanzada
    returns = close.pct_change()
    df["Volatility_Std_20"] = returns.rolling(20).std()
    df["Volatility_GARCH_Proxy"] = returns.rolling(20).std().rolling(5).mean()
    df["Volatility_Amplitude"] = (high - low) / close.replace(0, np.nan)
    df["Volatility_Range_20"] = (high - low).rolling(20).mean()

    return df


def calcular_patrones_candlestick(df):
    df = df.copy()

    if df.empty:
        return df

    body = (df["Close"] - df["Open"]).abs()
    range_total = (df["High"] - df["Low"]).replace(0, np.nan)
    upper_shadow = df["High"] - df[["Open", "Close"]].max(axis=1)
    lower_shadow = df[["Open", "Close"]].min(axis=1) - df["Low"]

    body_ratio = body.div(range_total.replace(0, np.nan)).fillna(1)
    bullish_candle = (df["Close"] >= df["Open"]).astype(int)
    bearish_candle = (df["Close"] < df["Open"]).astype(int)

    prev_open = df["Open"].shift(1)
    prev_close = df["Close"].shift(1)
    prev_high = df["High"].shift(1)
    prev_low = df["Low"].shift(1)
    prev_body = (prev_close - prev_open).abs()
    prev_bearish = prev_close < prev_open
    prev_bullish = prev_close > prev_open

    current_open = df["Open"]
    current_close = df["Close"]
    current_high = df["High"]
    current_low = df["Low"]

    df["Candle_Doji"] = (body_ratio <= 0.1).astype(int)
    df["Candle_Hammer"] = (
        (lower_shadow > 2 * body)
        & (upper_shadow < body)
        & bullish_candle.astype(bool)
    ).astype(int)
    df["Candle_Hanging_Man"] = (
        (lower_shadow < body)
        & (upper_shadow > 2 * body)
        & bearish_candle.astype(bool)
    ).astype(int)
    df["Candle_Shooting_Star"] = (
        (upper_shadow > 2 * body)
        & (lower_shadow < body)
        & bearish_candle.astype(bool)
    ).astype(int)
    df["Candle_Bearish_Belt_Hold"] = (
        bearish_candle.astype(bool)
        & (current_open <= current_close + 0.01)
        & (current_close < current_open)
        & (current_low < prev_low)
    ).astype(int)
    df["Candle_Inverted_Hammer"] = (
        (upper_shadow > 2 * body)
        & (lower_shadow < body)
        & bullish_candle.astype(bool)
    ).astype(int)
    df["Candle_Bullish_Engulfing"] = (
        prev_bearish
        & bullish_candle.astype(bool)
        & (current_open <= prev_close)
        & (current_close >= prev_open)
        & (body > prev_body)
    ).astype(int)
    df["Candle_Bearish_Engulfing"] = (
        prev_bullish
        & bearish_candle.astype(bool)
        & (current_open >= prev_close)
        & (current_close <= prev_open)
        & (body > prev_body)
    ).astype(int)
    df["Candle_Tweezer_Top"] = (
        prev_high.eq(current_high)
        & prev_bearish
        & bearish_candle.astype(bool)
    ).astype(int)
    df["Candle_Tweezer_Bottom"] = (
        prev_low.eq(current_low)
        & prev_bullish
        & bullish_candle.astype(bool)
    ).astype(int)
    df["Candle_Piercing"] = (
        prev_bearish
        & bullish_candle.astype(bool)
        & (current_open < prev_close)
        & (current_close > (prev_open + prev_close) / 2)
    ).astype(int)
    df["Candle_Bullish_Separating_Line"] = (
        prev_bearish
        & bullish_candle.astype(bool)
        & (current_open > prev_open)
        & (current_close > prev_close)
    ).astype(int)
    df["Candle_Bullish_Kicker"] = (
        prev_bearish
        & bullish_candle.astype(bool)
        & (current_open > prev_high)
        & (current_close > prev_open)
    ).astype(int)
    df["Candle_Bearish_Kicker"] = (
        prev_bullish
        & bearish_candle.astype(bool)
        & (current_open < prev_low)
        & (current_close < prev_open)
    ).astype(int)
    df["Candle_Dark_Cloud_Cover"] = (
        prev_bullish
        & bearish_candle.astype(bool)
        & (current_open > prev_open)
        & (current_close < prev_close * 0.5 + prev_open * 0.5)
    ).astype(int)
    df["Candle_Morning_Star"] = (
        prev_bearish
        & (df["Close"].shift(2) < df["Close"].shift(1))
        & bullish_candle.astype(bool)
    ).astype(int)
    df["Candle_Evening_Star"] = (
        prev_bullish
        & (df["Close"].shift(2) > df["Close"].shift(1))
        & bearish_candle.astype(bool)
    ).astype(int)
    df["Candle_Evening_Doji_Star"] = (
        prev_bullish
        & (df["Close"].shift(2) > df["Close"].shift(1))
        & df["Candle_Doji"].shift(1).astype(bool)
        & bearish_candle.astype(bool)
    ).astype(int)
    df["Candle_Three_White_Soldiers"] = (
        (df["Close"] > df["Open"]).rolling(3, min_periods=3).sum().eq(3)
    ).astype(int)
    df["Candle_Three_Black_Crows"] = (
        (df["Close"] < df["Open"]).rolling(3, min_periods=3).sum().eq(3)
    ).astype(int)
    df["Candle_Three_White_Strike"] = (
        (df["Close"] > df["Open"]).rolling(3, min_periods=3).sum().eq(3)
    ).astype(int)
    df["Candle_Inside_Bar"] = (
        (current_high < prev_high)
        & (current_low > prev_low)
    ).astype(int)
    df["Candle_Bullish_Abandoned_Baby"] = (
        df["Candle_Doji"].shift(1).astype(bool)
        & prev_bearish
        & bullish_candle.astype(bool)
    ).astype(int)
    df["Candle_Bearish_Abandoned_Baby"] = (
        df["Candle_Doji"].shift(1).astype(bool)
        & prev_bullish
        & bearish_candle.astype(bool)
    ).astype(int)
    df["Candle_Three_Inside_Up"] = (
        prev_bearish
        & df["Candle_Inside_Bar"].shift(1).astype(bool)
        & bullish_candle.astype(bool)
    ).astype(int)
    df["Candle_Three_Outside_Up"] = (
        prev_bearish
        & df["Candle_Inside_Bar"].shift(1).astype(bool)
        & bullish_candle.astype(bool)
    ).astype(int)
    df["Candle_Upside_Gap_Two_Crows"] = (
        prev_bullish
        & bearish_candle.astype(bool)
        & (current_open > prev_high)
    ).astype(int)
    df["Candle_Bullish_Harami"] = (
        prev_bearish
        & bullish_candle.astype(bool)
        & (current_close < prev_open)
        & (current_open > prev_close)
    ).astype(int)
    df["Candle_Bearish_Harami"] = (
        prev_bullish
        & bearish_candle.astype(bool)
        & (current_close > prev_open)
        & (current_open < prev_close)
    ).astype(int)
    df["Candle_Spinning_Top"] = (
        (body_ratio > 0.3) & (body_ratio < 0.7)
    ).astype(int)
    df["Candle_Concealing_Baby_Swallow"] = (
        prev_bearish
        & bullish_candle.astype(bool)
        & (current_close > prev_open)
    ).astype(int)
    df["Candle_Rising_Three_Methods"] = (
        prev_bullish
        & df["Close"].rolling(4, min_periods=4).mean().gt(df["Close"].shift(1).rolling(4, min_periods=4).mean())
    ).astype(int)
    df["Candle_Mat_Hold"] = (
        prev_bullish
        & bullish_candle.astype(bool)
        & (current_close > prev_close)
    ).astype(int)
    df["Candle_Ladder_Bottom"] = (
        prev_bearish
        & bullish_candle.astype(bool)
        & (current_close > prev_close)
    ).astype(int)

    return df


def calcular_indicadores_activo(df):
    df = df.sort_values("Date").copy()

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
    df["High"] = pd.to_numeric(df["High"], errors="coerce")
    df["Low"] = pd.to_numeric(df["Low"], errors="coerce")

    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    else:
        df["Volume"] = 0

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

    # Volatilidad
    df["ATR_14"] = calcular_atr(df, 14)
    df["ADX_14"] = calcular_adx(df, 14)
    df["Volatility_20"] = df["Return_1d"].rolling(20).std()

    # Indicadores adicionales de volumen y momentum
    df = calcular_indicadores_adicionales(df)

    # Patrones de velas
    df = calcular_patrones_candlestick(df)

    # Volumen
    df["Volume_SMA20"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA20"].replace(0, np.nan)

    # Tendencia / posición relativa
    df["Close_vs_SMA20"] = df["Close"] / df["SMA_20"] - 1
    df["Close_vs_SMA50"] = df["Close"] / df["SMA_50"] - 1
    df["Close_vs_SMA200"] = df["Close"] / df["SMA_200"] - 1

    df["Trend_SMA20_50"] = np.where(df["SMA_20"] > df["SMA_50"], 1, 0)
    df["Trend_SMA50_200"] = np.where(df["SMA_50"] > df["SMA_200"], 1, 0)

    # Targets multihorizonte
    for t in TARGETS:
        df[f"Future_Close_{t}d"] = df["Close"].shift(-t)
        df[f"Future_Return_{t}d"] = df[f"Future_Close_{t}d"] / df["Close"] - 1
        threshold = TARGET_5D_MIN_RETURN if t == 5 else 0
        df[f"Target_{t}d"] = np.where(df[f"Future_Return_{t}d"] > threshold, 1, 0)

    return df


def generar_indicadores(nombre_mercado="merval"):
    inicio = datetime.now()
    market = MarketConfig(nombre_mercado)

    print("=" * 70)
    print(f"MARKET INDICATORS - {market.display_name}")
    print("=" * 70)

    if not market.full_raw_csv.exists():
        raise FileNotFoundError(
            f"No existe archivo base: {market.full_raw_csv}. "
            f"Primero ejecutá market_downloader.py"
        )

    df = pd.read_csv(market.full_raw_csv)
    df = normalizar_columnas_indicadores(df)

    if "Date" not in df.columns:
        raise ValueError(f"No encontré columna de fecha en {market.full_raw_csv}")

    if "Ticker" not in df.columns:
        raise ValueError(f"No encontré columna de ticker en {market.full_raw_csv}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()

    print(f"Mercado        : {market.market}")
    print(f"Archivo origen : {market.full_raw_csv}")
    print(f"Registros base : {len(df):,}")
    print(f"Activos        : {df['Ticker'].nunique()}")

    resultados = []

    for ticker, df_ticker in df.groupby("Ticker"):
        print(f"Procesando {ticker}...")
        procesado = calcular_indicadores_activo(df_ticker)
        resultados.append(procesado)

    base = pd.concat(resultados, ignore_index=True)
    base.sort_values(["Ticker", "Date"], inplace=True)

    base.to_csv(market.indicators_csv, index=False, encoding="utf-8-sig")
    base.to_excel(market.indicators_excel, index=False)

    columnas_indicadores = [
        c for c in base.columns
        if c.startswith((
            "Return_",
            "SMA_",
            "EMA_",
            "RSI_",
            "ROC_",
            "Momentum_",
            "MACD",
            "BB_",
            "ATR_",
            "ADX_",
            "OBV",
            "VWAP",
            "Stoch_",
            "CCI",
            "Williams_",
            "Keltner_",
            "CMF",
            "Volatility_",
            "Volume_",
            "Candle_",
            "Close_vs_",
            "Trend_",
        ))
    ]

    columnas_targets = [c for c in base.columns if c.startswith("Target_")]

    print()
    print("=" * 70)
    print(f"RESUMEN INDICADORES - {market.display_name}")
    print("=" * 70)
    print(f"Activos procesados      : {base['Ticker'].nunique()}")
    print(f"Registros totales       : {len(base):,}")
    print(f"Indicadores generados   : {len(columnas_indicadores)}")
    print(f"Targets generados       : {len(columnas_targets)}")
    print(f"Columnas finales        : {len(base.columns)}")
    print(f"Desde                   : {base['Date'].min()}")
    print(f"Hasta                   : {base['Date'].max()}")
    print(f"CSV generado            : {market.indicators_csv}")
    print(f"Excel generado          : {market.indicators_excel}")
    print(f"Tiempo total            : {datetime.now() - inicio}")

    return base


if __name__ == "__main__":
    generar_indicadores("merval")