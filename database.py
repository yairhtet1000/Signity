"""SQLite schema, migrations, and connection helpers for Signity."""

import os
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = Path(os.environ.get("SIGNITY_DATABASE", BASE_DIR / "signity.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS "Admin" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS "User" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL,
    is_approved INTEGER NOT NULL DEFAULT 0 CHECK (is_approved IN (0, 1)),
    approval_status TEXT NOT NULL DEFAULT 'pending' CHECK (approval_status IN ('pending', 'approved', 'rejected')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS "AdminActivity" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    adminId INTEGER NOT NULL,
    userId INTEGER NOT NULL,
    action TEXT NOT NULL,
    timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (adminId) REFERENCES "Admin" (id),
    FOREIGN KEY (userId) REFERENCES "User" (id)
);
CREATE TABLE IF NOT EXISTS "UserApprove" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    adminId INTEGER NOT NULL,
    userId INTEGER NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN ('approved', 'rejected')),
    timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (adminId) REFERENCES "Admin" (id),
    FOREIGN KEY (userId) REFERENCES "User" (id)
);
CREATE TABLE IF NOT EXISTS "UserHistory" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    userId INTEGER NOT NULL,
    interpretedText TEXT NOT NULL,
    timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (userId) REFERENCES "User" (id)
);
"""


def get_connection():
    """Return a transaction-safe connection with foreign keys enforced."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def _columns(connection, table):
    return {row["name"] for row in connection.execute(f'PRAGMA table_info("{table}")')}


def _add_column(connection, table, definition):
    name = definition.split()[0]
    if name not in _columns(connection, table):
        connection.execute(f'ALTER TABLE "{table}" ADD COLUMN {definition}')


def initialize_database():
    """Create or safely migrate the account schema without deleting records."""
    with get_connection() as connection:
        connection.executescript(SCHEMA)
        for table in ("Admin", "User"):
            _add_column(connection, table, "created_at TEXT")
        _add_column(connection, "User", "is_approved INTEGER NOT NULL DEFAULT 0")
        _add_column(connection, "User", "approval_status TEXT NOT NULL DEFAULT 'pending'")
        _add_column(connection, "AdminActivity", "timestamp TEXT")
        _add_column(connection, "UserApprove", "decision TEXT")
        _add_column(connection, "UserApprove", "timestamp TEXT")
        _add_column(connection, "UserHistory", "interpretedText TEXT")
        _add_column(connection, "UserHistory", "timestamp TEXT")

        history_columns = _columns(connection, "UserHistory")
        if "interpretedTexts" in history_columns:
            connection.execute(
                'UPDATE "UserHistory" SET interpretedText = interpretedTexts WHERE interpretedText IS NULL'
            )
            connection.execute(
                'UPDATE "UserHistory" SET interpretedTexts = interpretedText WHERE interpretedTexts IS NULL'
            )
        connection.execute(
            'UPDATE "User" SET is_approved = 1, approval_status = "approved" '
            'WHERE id IN (SELECT userId FROM "UserApprove" WHERE COALESCE(decision, "approved") = "approved")'
        )
        connection.execute('UPDATE "User" SET approval_status = CASE WHEN is_approved = 1 THEN "approved" ELSE COALESCE(approval_status, "pending") END')
        connection.execute('UPDATE "UserHistory" SET timestamp = COALESCE(timestamp, CURRENT_TIMESTAMP)')
        connection.execute('UPDATE "AdminActivity" SET timestamp = COALESCE(timestamp, CURRENT_TIMESTAMP)')
        connection.execute('UPDATE "UserApprove" SET decision = COALESCE(decision, "approved"), timestamp = COALESCE(timestamp, CURRENT_TIMESTAMP)')
        connection.execute('CREATE INDEX IF NOT EXISTS idx_user_status ON "User" (approval_status)')
        connection.execute('CREATE INDEX IF NOT EXISTS idx_history_user_time ON "UserHistory" (userId, timestamp DESC)')
        connection.execute('CREATE INDEX IF NOT EXISTS idx_activity_time ON "AdminActivity" (timestamp DESC)')
