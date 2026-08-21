import re
import builtins
import sys
from datetime import datetime

from config import get_json_config_value


_LEVELS = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "NONE": 50,
}

_ACTIVE_LEVEL = _LEVELS["INFO"]
_ACTIVE_INFO_VERBOSITY = 3
_CONFIG_LOADED = False

_WARNING_PATTERNS = [
    re.compile(r"\bwarn(ing)?\b", re.IGNORECASE),
]

_ERROR_PATTERNS = [
    re.compile(r"\berror\b", re.IGNORECASE),
    re.compile(r"traceback", re.IGNORECASE),
    re.compile(r"exception", re.IGNORECASE),
    re.compile(r"\bfailed\b", re.IGNORECASE),
]

_INFO_V2_PATTERNS = [
    re.compile(r"resumen", re.IGNORECASE),
    re.compile(r"finalizado", re.IGNORECASE),
    re.compile(r"pipeline", re.IGNORECASE),
    re.compile(r"top oportunidades", re.IGNORECASE),
    re.compile(r"r[ée]gimen detectado", re.IGNORECASE),
    re.compile(r"decisi[óo]n del sistema", re.IGNORECASE),
    re.compile(r"portafolio sugerido", re.IGNORECASE),
    re.compile(r"daily report", re.IGNORECASE),
]

_INFO_V3_PATTERNS = [
    re.compile(r"^mercado\s*:", re.IGNORECASE),
    re.compile(r"^activos\s*:", re.IGNORECASE),
    re.compile(r"^registros\s*:", re.IGNORECASE),
    re.compile(r"csv generado", re.IGNORECASE),
    re.compile(r"excel generado", re.IGNORECASE),
    re.compile(r"tiempo total", re.IGNORECASE),
    re.compile(r"^fecha\s*:", re.IGNORECASE),
]

_INFO_V4_PATTERNS = [
    re.compile(r"guardando", re.IGNORECASE),
    re.compile(r"procesando", re.IGNORECASE),
    re.compile(r"variables", re.IGNORECASE),
    re.compile(r"modelos", re.IGNORECASE),
    re.compile(r"estado", re.IGNORECASE),
]


def normalize_level(level):
    if level is None:
        return "INFO"

    text = str(level).strip().upper()
    if text not in _LEVELS:
        return "INFO"

    return text


def configure_from_config():
    global _ACTIVE_LEVEL, _ACTIVE_INFO_VERBOSITY, _CONFIG_LOADED

    configured = get_json_config_value("logging.level", "INFO")
    level_name = normalize_level(configured)
    _ACTIVE_LEVEL = _LEVELS[level_name]
    configured_info = get_json_config_value("logging.info_verbosity", 3)
    _ACTIVE_INFO_VERBOSITY = normalize_info_verbosity(configured_info)
    _CONFIG_LOADED = True
    return level_name


def _ensure_config_loaded():
    if _CONFIG_LOADED:
        return
    configure_from_config()


def get_active_level_name():
    for name, value in _LEVELS.items():
        if value == _ACTIVE_LEVEL:
            return name
    return "INFO"


def normalize_info_verbosity(value):
    try:
        num = int(value)
    except Exception:
        return 3

    if num < 1:
        return 1
    if num > 5:
        return 5
    return num


def get_active_info_verbosity():
    return _ACTIVE_INFO_VERBOSITY


def should_log(level="INFO", info_verbosity=3):
    _ensure_config_loaded()
    level_name = normalize_level(level)

    if _LEVELS[level_name] < _ACTIVE_LEVEL:
        return False

    if level_name == "INFO":
        return normalize_info_verbosity(info_verbosity) <= _ACTIVE_INFO_VERBOSITY

    return True


def log(message, level="INFO", with_ts=False, info_verbosity=3):
    if not should_log(level, info_verbosity=info_verbosity):
        return

    if with_ts:
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] {message}")
    else:
        print(message)


def log_print(*args, sep=" ", end="\n", file=None, flush=False, level="INFO", info_verbosity=3):
    """Drop-in replacement for print() routed through configured log filtering."""
    target = file if file is not None else sys.stdout

    # Respect explicit non-console outputs (e.g. custom streams/files).
    if target not in (sys.stdout, sys.stderr):
        builtins.print(*args, sep=sep, end=end, file=target, flush=flush)
        return

    if not should_log(level=level, info_verbosity=info_verbosity):
        return

    message = sep.join(str(a) for a in args)
    builtins.print(message, end=end, file=target, flush=flush)


def _is_warning_line(line):
    return any(p.search(line) for p in _WARNING_PATTERNS)


def _is_error_line(line):
    return any(p.search(line) for p in _ERROR_PATTERNS)


def _matches_any(line, patterns):
    return any(p.search(line) for p in patterns)


def _should_emit_info_line(line):
    if not line:
        return False

    if _ACTIVE_INFO_VERBOSITY >= 5:
        return True

    if _is_warning_line(line) or _is_error_line(line):
        return True

    if _ACTIVE_INFO_VERBOSITY >= 4 and _matches_any(line, _INFO_V4_PATTERNS):
        return True

    if _ACTIVE_INFO_VERBOSITY >= 3 and _matches_any(line, _INFO_V3_PATTERNS):
        return True

    if _ACTIVE_INFO_VERBOSITY >= 2 and _matches_any(line, _INFO_V2_PATTERNS):
        return True

    return False


def emit_captured_output(text):
    if not text:
        return

    level_name = get_active_level_name()

    if level_name == "DEBUG":
        print(text, end="")
        return

    if level_name == "INFO" and _ACTIVE_INFO_VERBOSITY >= 5:
        print(text, end="")
        return

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line:
            continue

        if level_name == "INFO":
            if _should_emit_info_line(line):
                print(line)
            continue

        if level_name == "WARNING":
            if _is_warning_line(line) or _is_error_line(line):
                print(line)
            continue

        if level_name == "ERROR":
            if _is_error_line(line):
                print(line)
            continue

        # NONE
        if level_name == "NONE":
            continue
