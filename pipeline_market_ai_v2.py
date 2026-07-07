from datetime import datetime

# =============================================================================
# IMPORTS
# =============================================================================

from source.descargar_datos import descargar_merval
from source.indicadores import procesar_indicadores
from source.dataset import generar_dataset

from source.descargar_macro import descargar_macro
from source.integrar_macro import integrar_macro_dataset

from source.features_engine import generar_features_avanzadas

from source.model_manager import entrenar_model_manager

from source.scoring_merval_v2 import generar_scoring_v2

from source.market_regime_detector_v2 import detectar_regimen_mercado

from source.decision_engine import generar_decision

from source.portfolio_optimizer import construir_portafolio


# =============================================================================
# PIPELINE
# =============================================================================

def ejecutar_pipeline():

    inicio = datetime.now()

    print("=" * 80)
    print("MARKET AI PREDICTION 2.0")
    print("=" * 80)

    pasos = [

        ("1. Descargar datos históricos",
         descargar_merval),

        ("2. Indicadores técnicos",
         procesar_indicadores),

        ("3. Dataset IA",
         generar_dataset),

        ("4. Descargar variables macro",
         descargar_macro),

        ("5. Integrar contexto macro",
         integrar_macro_dataset),

        ("6. Feature Engineering",
         generar_features_avanzadas),

        ("7. Model Manager",
         entrenar_model_manager),

        ("8. Scoring Engine",
         generar_scoring_v2),

        ("9. Market Regime Detector",
         detectar_regimen_mercado),

        ("10. Decision Engine",
         generar_decision),

        ("11. Portfolio Optimizer",
         construir_portafolio),

        # ("12. Backtesting",
        #  ejecutar_backtesting_global),

    ]

    for nombre, funcion in pasos:

        print()
        print("=" * 80)
        print(nombre)
        print("=" * 80)

        try:

            funcion()

        except Exception as e:

            print()
            print("=" * 80)
            print("ERROR")
            print("=" * 80)

            print(nombre)
            print(e)

            return

    print()
    print("=" * 80)
    print("PIPELINE FINALIZADO CORRECTAMENTE")
    print("=" * 80)

    print(f"Tiempo total : {datetime.now()-inicio}")


# =============================================================================

if __name__ == "__main__":

    ejecutar_pipeline()