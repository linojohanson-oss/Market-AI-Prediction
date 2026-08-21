-- SQLite schema template for incremental market cache.
-- This file is intentionally data-free and can be versioned safely.

PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS indicator_rows (
    market TEXT NOT NULL,
    interval TEXT NOT NULL,
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    payload TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (market, interval, ticker, date)
);

CREATE INDEX IF NOT EXISTS idx_indicator_lookup
ON indicator_rows (market, interval, ticker, date);

CREATE TABLE IF NOT EXISTS context_rows (
    market TEXT NOT NULL,
    interval TEXT NOT NULL,
    serie TEXT NOT NULL,
    date TEXT NOT NULL,
    value REAL NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (market, interval, serie, date)
);

CREATE INDEX IF NOT EXISTS idx_context_lookup
ON context_rows (market, interval, serie, date);
