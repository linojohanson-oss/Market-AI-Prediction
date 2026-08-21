"""
=========================================================
MARKET AI PREDICTION 2.0
Investment Policy Engine
=========================================================

Este módulo define la política de inversión del sistema.

NO contiene modelos de IA.
NO contiene predicciones.

Simplemente establece cómo debe actuar el sistema
según el régimen detectado.

Toda modificación de la estrategia de inversión
debe hacerse aquí.

=========================================================
"""

from source.app_logger import log_print as print

POLITICAS_INVERSION = {

    # =====================================================
    # Mercado claramente alcista
    # =====================================================

    "ALCISTA": {

        "nivel_riesgo": 20,

        "cash_pct": 10,
        "acciones_pct": 90,

        "max_posiciones": 8,

        "min_prob_compra": 0.70,
        "min_score_final": 70,

        "stop_loss": 8,
        "take_profit": 20,

        "rebalanceo": "SEMANAL",

        "accion_principal": "COMPRAR",

        "descripcion":
            "Mercado alcista confirmado. "
            "Se permite alta exposición en acciones."
    },

    # =====================================================

    "LATERAL / SELECTIVO": {

        "nivel_riesgo": 45,

        "cash_pct": 40,
        "acciones_pct": 60,

        "max_posiciones": 5,

        "min_prob_compra": 0.65,
        "min_score_final": 65,

        "stop_loss": 6,
        "take_profit": 15,

        "rebalanceo": "SEMANAL",

        "accion_principal": "COMPRAR SELECTIVO",

        "descripcion":
            "Mercado lateral. Operar únicamente activos de alta calidad."
    },

    # =====================================================

    "NEUTRAL": {

        "nivel_riesgo": 60,

        "cash_pct": 60,
        "acciones_pct": 40,

        "max_posiciones": 3,

        "min_prob_compra": 0.65,
        "min_score_final": 70,

        "stop_loss": 6,
        "take_profit": 12,

        "rebalanceo": "QUINCENAL",

        "accion_principal": "VIGILAR",

        "descripcion":
            "Mercado sin tendencia definida. "
            "Mantener exposición reducida."
    },

    # =====================================================

    "DÉBIL": {

        "nivel_riesgo": 80,

        "cash_pct": 80,
        "acciones_pct": 20,

        "max_posiciones": 2,

        "min_prob_compra": 0.75,
        "min_score_final": 75,

        "stop_loss": 5,
        "take_profit": 10,

        "rebalanceo": "MENSUAL",

        "accion_principal": "ESPERAR",

        "descripcion":
            "Mercado débil. "
            "Sólo operar oportunidades excepcionales."
    },

    # =====================================================

    "BAJISTA / RIESGO ALTO": {

        "nivel_riesgo": 95,

        "cash_pct": 100,
        "acciones_pct": 0,

        "max_posiciones": 0,

        "min_prob_compra": 0.90,
        "min_score_final": 90,

        "stop_loss": 0,
        "take_profit": 0,

        "rebalanceo": "NO APLICA",

        "accion_principal": "NO ABRIR POSICIONES",

        "descripcion":
            "Mercado bajista confirmado. "
            "Priorizar liquidez y preservar capital."
    }

}


# ==========================================================
# Política por defecto
# ==========================================================

POLITICA_DEFAULT = {

    "nivel_riesgo": 90,

    "cash_pct": 90,
    "acciones_pct": 10,

    "max_posiciones": 1,

    "min_prob_compra": 0.90,
    "min_score_final": 90,

    "stop_loss": 5,
    "take_profit": 10,

    "rebalanceo": "NO DEFINIDO",

    "accion_principal": "ESPERAR",

    "descripcion":
        "Régimen desconocido. Política defensiva aplicada."
}


# ==========================================================

def obtener_politica(regimen: str) -> dict:
    """
    Devuelve la política correspondiente al régimen detectado.
    """

    return POLITICAS_INVERSION.get(regimen, POLITICA_DEFAULT)


# ==========================================================

def listar_politicas():
    """
    Devuelve todas las políticas disponibles.
    """

    return POLITICAS_INVERSION


# ==========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("INVESTMENT POLICY ENGINE")
    print("=" * 70)

    for nombre, politica in POLITICAS_INVERSION.items():

        print()

        print(nombre)
        print("-" * len(nombre))

        for clave, valor in politica.items():
            print(f"{clave:20}: {valor}")