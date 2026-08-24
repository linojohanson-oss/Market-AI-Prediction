from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List
import json


# ============================================================
# BASE DEL PROYECTO
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

JSON_CONFIG_FILE = BASE_DIR / "app_config.json"

DATOS_BASE = BASE_DIR / "datos"
MODELOS_BASE = BASE_DIR / "modelos"
REPORTES_BASE = BASE_DIR / "reportes"
DASHBOARDS_BASE = BASE_DIR / "dashboards"
LOGS_BASE = BASE_DIR / "logs"
BACKUPS = BASE_DIR / "backups"
MARKETS_DIR = BASE_DIR / "markets"
SOURCE_DIR = BASE_DIR / "source"
TESTS_DIR = BASE_DIR / "tests"


DEFAULT_JSON_CONFIG = {
    "app": {
        "name": "Market-AI-Prediction",
        "default_market": "merval",
        "timezone": "UTC",
    },
    "paths": {
        "datos": "datos",
        "modelos": "modelos",
        "reportes": "reportes",
        "logs": "logs",
    },
    "pipeline": {
        "target_principal": "Target_5d",
        "target_horizons": [1, 5, 10, 20, 60,120,240],
        "target_min_return_5d": 0.02,
        "min_rows_per_ticker": 200,
        "fillna_model_features": 0,
        "enable_optimization": False,
        "publish_stable_ranking": True,
        "sqlite_incremental_cache_enabled": True,
        "indicators_recalc_lookback_bars": 320,
        "context_recalc_lookback_bars": 320,
    },
    "training": {
        "random_state": 42,
        "test_size": 0.2,
    },
    "optimizer_simulator": {
        "target_days_list": [5, 10, 15],
        "target_min_returns": [0.01, 0.015, 0.03, 0.05],
        "stop_loss_pct_list": [0.015, 0.02, 0.03],
    },
    "logging": {
        "level": "INFO",
        "info_verbosity": 3,
    },
    "portfolio": {
        "neutral_score": 50.0,
        "alpha_per_score_point_pct": 0.03,
        "min_edge_buffer_pct": 0.20,
        "min_trade_value": 250.0,
        "stop_loss_pct": 0.015,
        "markets": {
            "crypto": {
                "capital_inicial": 100000.0,
                "costs": {
                    "buy_fee_pct": 0.10,
                    "sell_fee_pct": 0.10,
                    "slippage_pct": 0.05,
                    "tax_sell_pct": 0.00,
                    "fixed_cost_per_trade": 0.0,
                },
                "current_positions": {
                    "cash": 100000.0,
                    "positions": []
                }
            }
        }
    },
}

_JSON_CONFIG_CACHE = None


def _merge_dicts(base, override):
    """Merge dictionaries recursively, preserving defaults when keys are missing."""
    merged = dict(base)

    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value

    return merged


def ensure_json_config_exists():
    """Create app_config.json with defaults if it does not exist."""
    if JSON_CONFIG_FILE.exists():
        return JSON_CONFIG_FILE

    with JSON_CONFIG_FILE.open("w", encoding="utf-8") as f:
        json.dump(DEFAULT_JSON_CONFIG, f, indent=2, ensure_ascii=False)

    return JSON_CONFIG_FILE


def load_json_config(refresh=False):
    """Load project JSON config once and keep it cached for module reuse."""
    global _JSON_CONFIG_CACHE

    if _JSON_CONFIG_CACHE is not None and not refresh:
        return _JSON_CONFIG_CACHE

    ensure_json_config_exists()

    with JSON_CONFIG_FILE.open("r", encoding="utf-8-sig") as f:
        user_config = json.load(f)

    _JSON_CONFIG_CACHE = _merge_dicts(DEFAULT_JSON_CONFIG, user_config)
    return _JSON_CONFIG_CACHE


def get_json_config_value(path, default=None):
    """Read configuration value by dotted path. Example: pipeline.target_principal"""
    config = load_json_config()
    current = config

    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]

    return current


# ============================================================
# MERCADOS SOPORTADOS
# ============================================================

SUPPORTED_MARKETS = [
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


# ============================================================
# CONFIGURACIÓN DE MERCADOS
# ============================================================

MARKET_DEFINITIONS = {
    "merval": {
        "display_name": "MERVAL",
        "description": "Acciones líderes del mercado argentino",
        "currency": "ARS",
        "timezone": "America/Argentina/Buenos_Aires",
        "benchmark": "^MERV",
        "target_days": 5,
        "period": "10y",
        "interval": "1d",
        "source": "yfinance",
        "calendar": "BYMA",
        "symbols": {
            "GGAL.BA": "Grupo Financiero Galicia",
            "YPFD.BA": "YPF",
            "PAMP.BA": "Pampa Energía",
            "ALUA.BA": "Aluar",
            "BBAR.BA": "Banco BBVA Argentina",
            "BMA.BA": "Banco Macro",
            "CEPU.BA": "Central Puerto",
            "COME.BA": "Sociedad Comercial del Plata",
            "CRES.BA": "Cresud",
            "EDN.BA": "Edenor",
            "LOMA.BA": "Loma Negra",
            "MIRG.BA": "Mirgor",
            "SUPV.BA": "Grupo Supervielle",
            "TECO2.BA": "Telecom Argentina",
            "TGNO4.BA": "Transportadora Gas del Norte",
            "TGSU2.BA": "Transportadora Gas del Sur",
            "TRAN.BA": "Transener",
            "TXAR.BA": "Ternium Argentina",
            "VALO.BA": "Grupo Financiero Valores",
        },
        "sectors": {
            "GGAL.BA": "Bancos",
            "BBAR.BA": "Bancos",
            "BMA.BA": "Bancos",
            "SUPV.BA": "Bancos",
            "YPFD.BA": "Energía",
            "PAMP.BA": "Energía",
            "CEPU.BA": "Energía",
            "EDN.BA": "Energía",
            "TGNO4.BA": "Energía",
            "TGSU2.BA": "Energía",
            "TRAN.BA": "Energía",
            "ALUA.BA": "Industria",
            "TXAR.BA": "Industria",
            "LOMA.BA": "Construcción",
            "CRES.BA": "Real Estate",
            "COME.BA": "Holding",
            "MIRG.BA": "Tecnología / Consumo",
            "TECO2.BA": "Telecomunicaciones",
            "VALO.BA": "Financiero",
        },
    },

    "crypto": {
        "display_name": "CRYPTO",
        "description": "Principales criptomonedas por capitalización y liquidez",
        "currency": "USD",
        "timezone": "UTC",
        "benchmark": "BTC-USD",
        "target_days": 5,
        "period": "10y",
        "interval": "1d",
        "source": "yfinance",
        "calendar": "24/7",
        "symbols": {
            "BTC-USD": "Bitcoin",
            "BCH-USD": "Bitcoin Cash",
            "ETH-USD": "Ethereum",
            "SOL-USD": "Solana",
            "BNB-USD": "BNB",
            "XRP-USD": "XRP",
            "ADA-USD": "Cardano",
            "DOGE-USD": "Dogecoin",
            "AVAX-USD": "Avalanche",
            "LINK-USD": "Chainlink",
            "DOT-USD": "Polkadot",
            "XLM-USD": "Stellar",
            "LTC-USD": "Litecoin",
            "SHIB-USD": "Shiba Inu",
            "SUI-USD": "Sui",
            "AAVE-USD": "Aave",
            "ALGO-USD": "Algorand",
            "TRUMP-USD": "Official Trump",
            "SUSHI-USD": "SushiSwap",
            "LPT-USD": "Livepeer",
            "ATOM-USD": "Cosmos",
        },
        "sectors": {
            "BTC-USD": "Reserva de valor",
            "BCH-USD": "Reserva de valor",
            "ETH-USD": "Smart Contracts",
            "SOL-USD": "Smart Contracts",
            "BNB-USD": "Exchange / Ecosistema",
            "XRP-USD": "Pagos",
            "ADA-USD": "Smart Contracts",
            "DOGE-USD": "Meme / Comunidad",
            "AVAX-USD": "Smart Contracts",
            "LINK-USD": "Oráculos",
            "DOT-USD": "Interoperabilidad",
            "XLM-USD": "Pagos / Infraestructura",
            "LTC-USD": "Moneda digital / Store of value",
            "SHIB-USD": "Meme / Comunidad",
            "SUI-USD": "Smart Contracts / Layer 1",
            "AAVE-USD": "DeFi",
            "ALGO-USD": "Layer 1 / Smart Contracts",
            "TRUMP-USD": "Meme / Comunidad",
            "SUSHI-USD": "DeFi",
            "LPT-USD": "Streaming / Infraestructura",
            "ATOM-USD": "Interoperabilidad / Cosmos",
        },
    },

    "nasdaq": {
        "display_name": "NASDAQ",
        "description": "Acciones tecnológicas líderes de Estados Unidos",
        "currency": "USD",
        "timezone": "America/New_York",
        "benchmark": "^IXIC",
        "target_days": 5,
        "period": "10y",
        "interval": "1d",
        "source": "yfinance",
        "calendar": "NASDAQ",
        "symbols": {
            "AAPL": "Apple",
            "MSFT": "Microsoft",
            "NVDA": "NVIDIA",
            "AMZN": "Amazon",
            "GOOGL": "Alphabet",
            "META": "Meta Platforms",
            "TSLA": "Tesla",
            "NFLX": "Netflix",
            "AMD": "Advanced Micro Devices",
            "INTC": "Intel",
            "CMCSA": "Comcast",
            "PINS": "Pinterest",
            "YELP": "Yelp",
            "T": "AT&T",
            "F": "Ford",
            "SNAP": "Snap",
            "AAL": "American Airlines Group",
            "DY": "Dycom Industries",
            "MELI": "MercadoLibre",
            "GPRO": "GoPro",
            "NDLS": "Noodles & Company",
            "ROOT": "Root",
            "SIRI": "Sirius XM",
            "EBAY": "eBay",
            "ARLO": "Arlo Technologies",
            "MCHP": "Microchip Technology",
            "ADBE": "Adobe",
            "COST": "Costco Wholesale",
            "CRM": "Salesforce",
            "CHWY": "Chewy",
            "GILD": "Gilead Sciences",
            "SPSC": "SPS Commerce",
            "PLD": "Prologis",
            "BOX": "Box",
            "SBUX": "Starbucks",
            "SFIX": "Stitch Fix",
            "PYPL": "PayPal Holdings",
            "DOCU": "DocuSign",
            "TWLO": "Twilio",
            "ROKU": "Roku",
            "ETSY": "Etsy",
            "NKE": "Nike",
        },
        "sectors": {
            "AAPL": "Tecnología",
            "MSFT": "Tecnología",
            "NVDA": "Semiconductores",
            "AMZN": "Consumo / Cloud",
            "GOOGL": "Tecnología / Publicidad",
            "META": "Tecnología / Redes",
            "TSLA": "Autos / Tecnología",
            "NFLX": "Streaming",
            "AMD": "Semiconductores",
            "INTC": "Semiconductores",
            "CMCSA": "Comunicación",
            "PINS": "Comunicación",
            "YELP": "Comunicación",
            "T": "Telecomunicaciones",
            "F": "Automotriz",
            "SNAP": "Comunicación",
            "AAL": "Aerolíneas",
            "DY": "Industriales",
            "MELI": "Consumo / E-commerce",
            "GPRO": "Consumo / Tecnología",
            "NDLS": "Consumo",
            "ROOT": "Finanzas",
            "SIRI": "Comunicación",
            "EBAY": "Consumo / E-commerce",
            "ARLO": "Tecnología",
            "MCHP": "Semiconductores",
            "ADBE": "Software",
            "COST": "Consumo defensivo",
            "CRM": "Software",
            "CHWY": "Consumo / E-commerce",
            "GILD": "Salud / Biotecnología",
            "SPSC": "Software",
            "PLD": "Real Estate",
            "BOX": "Software",
            "SBUX": "Consumo",
            "SFIX": "Consumo",
            "PYPL": "Fintech",
            "DOCU": "Software",
            "TWLO": "Software",
            "ROKU": "Comunicación / Streaming",
            "ETSY": "Consumo / E-commerce",
            "NKE": "Consumo",
        },
    },

    "cedears": {
        "display_name": "CEDEARS",
        "description": "Certificados de Depósito Argentinos",
        "currency": "ARS",
        "timezone": "America/Argentina/Buenos_Aires",
        "benchmark": "^MERV",
        "target_days": 5,
        "period": "10y",
        "interval": "1d",
        "source": "yfinance",
        "calendar": "BYMA",
        "symbols": {},
        "sectors": {},
    },

    "etf": {
        "display_name": "ETF",
        "description": "Fondos cotizados internacionales",
        "currency": "USD",
        "timezone": "America/New_York",
        "benchmark": "SPY",
        "target_days": 5,
        "period": "10y",
        "interval": "1d",
        "source": "yfinance",
        "calendar": "NYSE",
        "symbols": {
            "SPY": "S&P 500 ETF",
            "QQQ": "Nasdaq 100 ETF",
            "IWM": "Russell 2000 ETF",
            "DIA": "Dow Jones ETF",
            "GLD": "Gold ETF",
            "SLV": "Silver ETF",
            "TLT": "20+ Year Treasury ETF",
            "HYG": "High Yield Corporate Bond ETF",
        },
        "sectors": {},
    },

    "forex": {
        "display_name": "FOREX",
        "description": "Principales pares de divisas",
        "currency": "USD",
        "timezone": "UTC",
        "benchmark": "DX-Y.NYB",
        "target_days": 5,
        "period": "10y",
        "interval": "1d",
        "source": "yfinance",
        "calendar": "24/5",
        "symbols": {
            "EURUSD=X": "EUR/USD",
            "GBPUSD=X": "GBP/USD",
            "JPY=X": "USD/JPY",
            "ARS=X": "USD/ARS",
            "BRL=X": "USD/BRL",
            "CLP=X": "USD/CLP",
        },
        "sectors": {},
    },

    "commodities": {
        "display_name": "COMMODITIES",
        "description": "Principales commodities globales",
        "currency": "USD",
        "timezone": "America/New_York",
        "benchmark": "GC=F",
        "target_days": 5,
        "period": "10y",
        "interval": "1d",
        "source": "yfinance",
        "calendar": "CME",
        "symbols": {
            "GC=F": "Oro",
            "SI=F": "Plata",
            "CL=F": "Petróleo WTI",
            "BZ=F": "Petróleo Brent",
            "NG=F": "Gas Natural",
            "HG=F": "Cobre",
            "ZC=F": "Maíz",
            "ZS=F": "Soja",
            "ZW=F": "Trigo",
        },
        "sectors": {},
    },

    "bonos": {
        "display_name": "BONOS",
        "description": "Bonos soberanos y renta fija",
        "currency": "USD",
        "timezone": "America/New_York",
        "benchmark": "TLT",
        "target_days": 5,
        "period": "10y",
        "interval": "1d",
        "source": "yfinance",
        "calendar": "NYSE",
        "symbols": {
            "TLT": "iShares 20+ Year Treasury Bond ETF",
            "IEF": "iShares 7-10 Year Treasury Bond ETF",
            "SHY": "iShares 1-3 Year Treasury Bond ETF",
            "HYG": "High Yield Corporate Bond ETF",
            "LQD": "Investment Grade Corporate Bond ETF",
            "EMB": "Emerging Markets Bond ETF",
        },
        "sectors": {},
    },

    "indices": {
        "display_name": "INDICES",
        "description": "Índices globales y variables de contexto",
        "currency": "USD",
        "timezone": "UTC",
        "benchmark": "^GSPC",
        "target_days": 5,
        "period": "10y",
        "interval": "1d",
        "source": "yfinance",
        "calendar": "GLOBAL",
        "symbols": {
            "^MERV": "Merval",
            "^GSPC": "S&P 500",
            "^IXIC": "Nasdaq Composite",
            "^DJI": "Dow Jones",
            "^RUT": "Russell 2000",
            "^VIX": "VIX",
            "DX-Y.NYB": "DXY",
            "GC=F": "Oro",
            "CL=F": "Petróleo WTI",
            "BTC-USD": "Bitcoin",
        },
        "sectors": {},
    },
}


# Límite de lookback para intervalos intradía en Yahoo Finance.
# Valores conservadores para evitar errores de rango al solicitar históricos largos.
INTRADAY_INTERVAL_MAX_PERIOD = {
    "1m": "7d",
    "2m": "60d",
    "5m": "60d",
    "15m": "60d",
    "30m": "60d",
    "60m": "730d",
    "90m": "60d",
    "1h": "730d",
    "4h": "730d",
}


def _period_to_days(period: str):
    if not isinstance(period, str):
        return None

    value = period.strip().lower()

    if value.endswith("d") and value[:-1].isdigit():
        return int(value[:-1])

    if value.endswith("wk") and value[:-2].isdigit():
        return int(value[:-2]) * 7

    if value.endswith("mo") and value[:-2].isdigit():
        return int(value[:-2]) * 30

    if value.endswith("y") and value[:-1].isdigit():
        return int(value[:-1]) * 365

    return None


def ajustar_periodo_yahoo(period: str, interval: str):
    if not isinstance(interval, str):
        return period

    interval_value = interval.strip().lower()
    max_period = INTRADAY_INTERVAL_MAX_PERIOD.get(interval_value)

    if not max_period:
        return period

    period_days = _period_to_days(period)
    max_days = _period_to_days(max_period)

    if period_days is None or max_days is None:
        return max_period

    return max_period if period_days > max_days else period


# ============================================================
# CLASE MARKET CONFIG
# ============================================================

@dataclass
class MarketConfig:
    market: str
    version: str = "1.0"
    definition: Dict = field(init=False)
    execution_timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))

    def __post_init__(self):
        market = self.market.lower().strip()

        if market not in MARKET_DEFINITIONS:
            raise ValueError(
                f"Mercado no soportado: {market}. "
                f"Opciones válidas: {', '.join(SUPPORTED_MARKETS)}"
            )

        self.market = market
        self.definition = MARKET_DEFINITIONS[market]

        self._crear_directorios()

    # -------------------------
    # Metadata
    # -------------------------

    @property
    def display_name(self):
        return self.definition["display_name"]

    @property
    def description(self):
        return self.definition["description"]

    @property
    def currency(self):
        return self.definition["currency"]

    @property
    def timezone(self):
        return self.definition["timezone"]

    @property
    def benchmark(self):
        return self.definition["benchmark"]

    @property
    def target_days(self):
        return self.definition["target_days"]

    @property
    def configured_period(self):
        return self.definition["period"]

    @property
    def effective_period(self):
        return ajustar_periodo_yahoo(self.configured_period, self.interval)

    @property
    def period(self):
        return self.effective_period

    @property
    def period_was_adjusted(self):
        return self.configured_period != self.effective_period

    @property
    def period_adjustment_message(self):
        if not self.period_was_adjusted:
            return ""

        return (
            "WARNING Yahoo intraday limit applied: "
            f"period configured={self.configured_period} -> used={self.effective_period} "
            f"(interval={self.interval})"
        )

    @property
    def interval(self):
        return self.definition["interval"]

    @property
    def source(self):
        return self.definition["source"]

    @property
    def calendar(self):
        return self.definition["calendar"]

    @property
    def symbols(self):
        return self.definition["symbols"]

    @property
    def tickers(self) -> List[str]:
        return list(self.symbols.keys())

    @property
    def sectors(self):
        return self.definition.get("sectors", {})

    # -------------------------
    # Directorios del mercado
    # -------------------------

    @property
    def base_data_dir(self):
        return DATOS_BASE / self.market

    @property
    def raw_dir(self):
        return self.base_data_dir / "brutos"

    @property
    def processed_dir(self):
        return self.base_data_dir / "procesados"

    @property
    def cache_dir(self):
        return self.base_data_dir / "cache"

    @property
    def history_dir(self):
        return self.base_data_dir / "historicos"

    @property
    def temp_dir(self):
        return self.base_data_dir / "temporales"

    @property
    def model_dir(self):
        return MODELOS_BASE / self.market

    @property
    def trained_models_dir(self):
        return self.model_dir / "entrenados"

    @property
    def metrics_dir(self):
        return self.model_dir / "metricas"

    @property
    def report_dir(self):
        return REPORTES_BASE / self.market

    @property
    def charts_dir(self):
        return self.report_dir / "graficos"

    @property
    def dashboard_dir(self):
        return DASHBOARDS_BASE / self.market

    @property
    def log_dir(self):
        return LOGS_BASE / self.market

    # -------------------------
    # Archivos estándar
    # -------------------------

    def _output_path(self, directory, name, extension):
        return directory / f"{name}_{self.execution_timestamp}{extension}"

    @property
    def full_raw_csv(self):
        return self.raw_dir / f"{self.market.upper()}_COMPLETO.csv"

    @property
    def full_raw_excel(self):
        return self.raw_dir / f"{self.market.upper()}_COMPLETO.xlsx"

    @property
    def indicators_csv(self):
        return self._output_path(self.processed_dir, f"{self.market.upper()}_INDICADORES", ".csv")

    @property
    def indicators_excel(self):
        return self._output_path(self.processed_dir, f"{self.market.upper()}_INDICADORES", ".xlsx")

    @property
    def dataset_csv(self):
        return self._output_path(self.processed_dir, f"{self.market.upper()}_DATASET_IA", ".csv")

    @property
    def dataset_excel(self):
        return self._output_path(self.processed_dir, f"{self.market.upper()}_DATASET_IA", ".xlsx")

    @property
    def dataset_macro_csv(self):
        return self._output_path(self.processed_dir, f"{self.market.upper()}_DATASET_IA_MACRO", ".csv")

    @property
    def dataset_macro_excel(self):
        return self._output_path(self.processed_dir, f"{self.market.upper()}_DATASET_IA_MACRO", ".xlsx")

    @property
    def dataset_features_csv(self):
        return self._output_path(self.processed_dir, f"{self.market.upper()}_DATASET_FEATURES_AVANZADAS", ".csv")

    @property
    def dataset_features_excel(self):
        return self._output_path(self.processed_dir, f"{self.market.upper()}_DATASET_FEATURES_AVANZADAS", ".xlsx")

    @property
    def ranking_csv(self):
        return self._output_path(self.report_dir, "ranking_diario", ".csv")

    @property
    def ranking_excel(self):
        return self._output_path(self.report_dir, "ranking_diario", ".xlsx")

    @property
    def predictions_csv(self):
        return self._output_path(self.report_dir, "predicciones_futuras", ".csv")

    @property
    def predictions_excel(self):
        return self._output_path(self.report_dir, "predicciones_futuras", ".xlsx")

    @property
    def decision_engine_csv(self):
        return self._output_path(self.report_dir, "decision_engine", ".csv")

    @property
    def decision_engine_excel(self):
        return self._output_path(self.report_dir, "decision_engine", ".xlsx")

    @property
    def investor_report_txt(self):
        return self._output_path(self.report_dir, "informe_inversor", ".txt")

    @property
    def investor_report_md(self):
        return self._output_path(self.report_dir, "informe_inversor", ".md")

    @property
    def backtesting_excel(self):
        return self._output_path(self.report_dir, "backtesting", ".xlsx")

    @property
    def model_active_json(self):
        return self._output_path(self.report_dir, "modelo_activo", ".json")

    @property
    def ensemble_json(self):
        return self._output_path(self.report_dir, "modelos_ensemble", ".json")

    @property
    def ranking_models_csv(self):
        return self._output_path(self.report_dir, "ranking_modelos", ".csv")

    @property
    def ranking_models_excel(self):
        return self._output_path(self.report_dir, "ranking_modelos", ".xlsx")

    @property
    def features_pkl(self):
        return self._output_path(self.trained_models_dir, "features", ".pkl")

    @property
    def features_macro_pkl(self):
        return self._output_path(self.trained_models_dir, "features_macro", ".pkl")

    @property
    def features_advanced_pkl(self):
        return self._output_path(self.trained_models_dir, "features_avanzadas", ".pkl")

    # -------------------------
    # Métodos
    # -------------------------

    def _crear_directorios(self):
        carpetas = [
            DATOS_BASE,
            MODELOS_BASE,
            REPORTES_BASE,
            DASHBOARDS_BASE,
            LOGS_BASE,
            BACKUPS,
            MARKETS_DIR,
            SOURCE_DIR,
            TESTS_DIR,
            self.raw_dir,
            self.processed_dir,
            self.cache_dir,
            self.history_dir,
            self.temp_dir,
            self.trained_models_dir,
            self.metrics_dir,
            self.report_dir,
            self.charts_dir,
            self.dashboard_dir,
            self.log_dir,
        ]

        for carpeta in carpetas:
            carpeta.mkdir(parents=True, exist_ok=True)

    def resumen(self):
        print("=" * 70)
        print(f"MARKET CONFIG - {self.display_name}")
        print("=" * 70)
        print(f"Mercado       : {self.market}")
        print(f"Descripción   : {self.description}")
        print(f"Moneda        : {self.currency}")
        print(f"Zona horaria  : {self.timezone}")
        print(f"Benchmark     : {self.benchmark}")
        print(f"Target días   : {self.target_days}")
        if self.period_was_adjusted:
            print(f"Periodo cfg   : {self.configured_period}")
            print(f"Periodo usado : {self.effective_period}")
            print(self.period_adjustment_message)
        else:
            print(f"Periodo       : {self.period}")
        print(f"Intervalo     : {self.interval}")
        print(f"Fuente        : {self.source}")
        print(f"Calendario    : {self.calendar}")
        print(f"Activos       : {len(self.tickers)}")
        print()
        print("Directorios")
        print("-" * 70)
        print(f"Datos brutos  : {self.raw_dir}")
        print(f"Procesados    : {self.processed_dir}")
        print(f"Modelos       : {self.model_dir}")
        print(f"Reportes      : {self.report_dir}")
        print(f"Gráficos      : {self.charts_dir}")
        print(f"Dashboard     : {self.dashboard_dir}")


# ============================================================
# MERCADO ACTIVO POR DEFECTO
# ============================================================

DEFAULT_MARKET = get_json_config_value("app.default_market", "merval")
ACTIVE_MARKET = MarketConfig(DEFAULT_MARKET)


# ============================================================
# COMPATIBILIDAD TEMPORAL CON CÓDIGO VIEJO
# ============================================================

# Aliases principales
DATOS = DATOS_BASE
MODELOS = MODELOS_BASE
REPORTES = ACTIVE_MARKET.report_dir
DASHBOARDS = DASHBOARDS_BASE
LOGS = LOGS_BASE

# Aliases Merval legacy
TICKERS = ACTIVE_MARKET.tickers
PERIODO = ACTIVE_MARKET.period
INTERVALO = ACTIVE_MARKET.interval

HISTORICOS = ACTIVE_MARKET.raw_dir
INDICADORES = ACTIVE_MARKET.processed_dir
PROCESADOS = ACTIVE_MARKET.processed_dir

MODELOS_ENTRENADOS = ACTIVE_MARKET.trained_models_dir
MODELOS_METRICAS = ACTIVE_MARKET.metrics_dir

GRAFICOS = ACTIVE_MARKET.charts_dir
GRAFICOS.mkdir(parents=True, exist_ok=True)

# Macro / índices globales
MACRO = DATOS_BASE / "indices" / "brutos"
MACRO.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ACTIVE_MARKET.resumen()