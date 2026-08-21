import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd


class MarketSQLiteCache:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _ensure_schema(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS indicator_rows (
                    market TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    date TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (market, interval, ticker, date)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_indicator_lookup
                ON indicator_rows (market, interval, ticker, date)
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS context_rows (
                    market TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    serie TEXT NOT NULL,
                    date TEXT NOT NULL,
                    value REAL NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (market, interval, serie, date)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_context_lookup
                ON context_rows (market, interval, serie, date)
                """
            )

    @staticmethod
    def _to_iso(v):
        return pd.to_datetime(v, errors="coerce").isoformat()

    def get_indicator_max_date(self, market: str, interval: str, ticker: str):
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT MAX(date) FROM indicator_rows
                WHERE market = ? AND interval = ? AND ticker = ?
                """,
                (market, interval, ticker),
            ).fetchone()

        if not row or not row[0]:
            return None
        return pd.to_datetime(row[0], errors="coerce")

    def upsert_indicator_rows(self, market: str, interval: str, ticker: str, df: pd.DataFrame):
        if df is None or df.empty:
            return 0

        records = []
        updated_at = datetime.now().isoformat()

        for _, row in df.iterrows():
            d = row.to_dict()
            date_iso = self._to_iso(d.get("Date"))
            if not date_iso or date_iso == "NaT":
                continue

            payload = dict(d)
            payload.pop("Date", None)
            payload.pop("Ticker", None)

            records.append(
                (
                    market,
                    interval,
                    ticker,
                    date_iso,
                    json.dumps(payload, ensure_ascii=False, default=str),
                    updated_at,
                )
            )

        if not records:
            return 0

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO indicator_rows (market, interval, ticker, date, payload, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(market, interval, ticker, date)
                DO UPDATE SET payload=excluded.payload, updated_at=excluded.updated_at
                """,
                records,
            )

        return len(records)

    def load_indicators_market(self, market: str, interval: str):
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT ticker, date, payload
                FROM indicator_rows
                WHERE market = ? AND interval = ?
                ORDER BY ticker, date
                """,
                (market, interval),
            ).fetchall()

        if not rows:
            return pd.DataFrame()

        out = []
        for ticker, date_iso, payload_json in rows:
            payload = json.loads(payload_json)
            payload["Ticker"] = ticker
            payload["Date"] = pd.to_datetime(date_iso, errors="coerce")
            out.append(payload)

        df = pd.DataFrame(out)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        return df

    def get_context_max_date(self, market: str, interval: str, serie: str):
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT MAX(date) FROM context_rows
                WHERE market = ? AND interval = ? AND serie = ?
                """,
                (market, interval, serie),
            ).fetchone()

        if not row or not row[0]:
            return None
        return pd.to_datetime(row[0], errors="coerce")

    def upsert_context_series(self, market: str, interval: str, serie: str, df: pd.DataFrame):
        if df is None or df.empty or "Date" not in df.columns or serie not in df.columns:
            return 0

        working = df[["Date", serie]].copy()
        working["Date"] = pd.to_datetime(working["Date"], errors="coerce")
        working[serie] = pd.to_numeric(working[serie], errors="coerce")
        working.dropna(subset=["Date", serie], inplace=True)

        if working.empty:
            return 0

        updated_at = datetime.now().isoformat()
        records = [
            (
                market,
                interval,
                serie,
                self._to_iso(r["Date"]),
                float(r[serie]),
                updated_at,
            )
            for _, r in working.iterrows()
        ]

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO context_rows (market, interval, serie, date, value, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(market, interval, serie, date)
                DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                """,
                records,
            )

        return len(records)

    def load_context_series(self, market: str, interval: str, serie: str):
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT date, value
                FROM context_rows
                WHERE market = ? AND interval = ? AND serie = ?
                ORDER BY date
                """,
                (market, interval, serie),
            ).fetchall()

        if not rows:
            return pd.DataFrame(columns=["Date", serie])

        df = pd.DataFrame(rows, columns=["Date", serie])
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df[serie] = pd.to_numeric(df[serie], errors="coerce")
        df.dropna(subset=["Date", serie], inplace=True)
        return df
