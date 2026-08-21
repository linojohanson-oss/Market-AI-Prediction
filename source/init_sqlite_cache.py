from pathlib import Path
import argparse
import sqlite3
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import MarketConfig


def build_db_path(market: str, explicit_db_path: str | None) -> Path:
    if explicit_db_path:
        return Path(explicit_db_path)
    cfg = MarketConfig(market)
    return cfg.cache_dir / f"{market}_cache.sqlite"


def create_or_update_schema(db_path: Path, schema_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    schema_sql = schema_path.read_text(encoding="utf-8")

    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript(schema_sql)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create or update the market SQLite cache schema from template (no data)."
    )
    parser.add_argument(
        "market",
        nargs="?",
        default="crypto",
        help="Market name (used to resolve default cache path).",
    )
    parser.add_argument(
        "--db-path",
        dest="db_path",
        default=None,
        help="Optional explicit SQLite DB path.",
    )
    parser.add_argument(
        "--schema",
        dest="schema",
        default="docs/sql/market_cache_schema.sql",
        help="Path to SQL schema template.",
    )
    args = parser.parse_args()

    db_path = build_db_path(args.market, args.db_path)
    schema_path = Path(args.schema)

    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    create_or_update_schema(db_path, schema_path)
    print(f"SQLite cache schema ready at: {db_path}")
