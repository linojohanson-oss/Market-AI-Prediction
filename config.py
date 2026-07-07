from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List


# ============================================================
# BASE DEL PROYECTO
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATOS_BASE = BASE_DIR / "datos"
MODELOS_BASE = BASE_DIR / "modelos"
REPORTES_BASE = BASE_DIR / "reportes"
DASHBOARDS_BASE = BASE_DIR / "dashboards"
LOGS_BASE = BASE_DIR / "logs"
BACKUPS = BASE_DIR / "backups"
MARKETS_DIR = BASE_DIR / "markets"
SOURCE_DIR = BASE_DIR / "source"
TESTS_DIR = BASE_DIR / "tests"


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
            "ETH-USD": "Ethereum",
            "SOL-USD": "Solana",
            "BNB-USD": "BNB",
            "XRP-USD": "XRP",
            "ADA-USD": "Cardano",
            "DOGE-USD": "Dogecoin",
            "AVAX-USD": "Avalanche",
            "LINK-USD": "Chainlink",
            "DOT-USD": "Polkadot",
        },
        "sectors": {
            "BTC-USD": "Reserva de valor",
            "ETH-USD": "Smart Contracts",
            "SOL-USD": "Smart Contracts",
            "BNB-USD": "Exchange / Ecosistema",
            "XRP-USD": "Pagos",
            "ADA-USD": "Smart Contracts",
            "DOGE-USD": "Meme / Comunidad",
            "AVAX-USD": "Smart Contracts",
            "LINK-USD": "Oráculos",
            "DOT-USD": "Interoperabilidad",
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


# ============================================================
# CLASE MARKET CONFIG
# ============================================================

@dataclass
class MarketConfig:
    market: str
    version: str = "1.0"
    definition: Dict = field(init=False)

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
    def period(self):
        return self.definition["period"]

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

    @property
    def full_raw_csv(self):
        return self.raw_dir / f"{self.market.upper()}_COMPLETO.csv"

    @property
    def full_raw_excel(self):
        return self.raw_dir / f"{self.market.upper()}_COMPLETO.xlsx"

    @property
    def indicators_csv(self):
        return self.processed_dir / f"{self.market.upper()}_INDICADORES.csv"

    @property
    def indicators_excel(self):
        return self.processed_dir / f"{self.market.upper()}_INDICADORES.xlsx"

    @property
    def dataset_csv(self):
        return self.processed_dir / f"{self.market.upper()}_DATASET_IA.csv"

    @property
    def dataset_excel(self):
        return self.processed_dir / f"{self.market.upper()}_DATASET_IA.xlsx"

    @property
    def dataset_macro_csv(self):
        return self.processed_dir / f"{self.market.upper()}_DATASET_IA_MACRO.csv"

    @property
    def dataset_macro_excel(self):
        return self.processed_dir / f"{self.market.upper()}_DATASET_IA_MACRO.xlsx"

    @property
    def dataset_features_csv(self):
        return self.processed_dir / f"{self.market.upper()}_DATASET_FEATURES_AVANZADAS.csv"

    @property
    def dataset_features_excel(self):
        return self.processed_dir / f"{self.market.upper()}_DATASET_FEATURES_AVANZADAS.xlsx"

    @property
    def ranking_csv(self):
        return self.report_dir / "ranking_diario.csv"

    @property
    def ranking_excel(self):
        return self.report_dir / "ranking_diario.xlsx"

    @property
    def predictions_csv(self):
        return self.report_dir / "predicciones_futuras.csv"

    @property
    def predictions_excel(self):
        return self.report_dir / "predicciones_futuras.xlsx"

    @property
    def decision_engine_csv(self):
        return self.report_dir / "decision_engine.csv"

    @property
    def decision_engine_excel(self):
        return self.report_dir / "decision_engine.xlsx"

    @property
    def investor_report_txt(self):
        return self.report_dir / "informe_inversor.txt"

    @property
    def investor_report_md(self):
        return self.report_dir / "informe_inversor.md"

    @property
    def backtesting_excel(self):
        return self.report_dir / "backtesting.xlsx"

    @property
    def model_active_json(self):
        return self.report_dir / "modelo_activo.json"

    @property
    def ensemble_json(self):
        return self.report_dir / "modelos_ensemble.json"

    @property
    def ranking_models_csv(self):
        return self.report_dir / "ranking_modelos.csv"

    @property
    def ranking_models_excel(self):
        return self.report_dir / "ranking_modelos.xlsx"

    @property
    def features_pkl(self):
        return self.trained_models_dir / "features.pkl"

    @property
    def features_macro_pkl(self):
        return self.trained_models_dir / "features_macro.pkl"

    @property
    def features_advanced_pkl(self):
        return self.trained_models_dir / "features_avanzadas.pkl"

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

DEFAULT_MARKET = "merval"
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