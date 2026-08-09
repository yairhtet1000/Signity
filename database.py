"""SQLite persistence for Signity accounts, approvals, and history."""

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
    password TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS "User" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS "AdminActivity" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    adminId INTEGER NOT NULL,
    userId INTEGER NOT NULL,
    action TEXT NOT NULL,
    FOREIGN KEY (adminId) REFERENCES "Admin" (id),
    FOREIGN KEY (userId) REFERENCES "User" (id)
);
CREATE TABLE IF NOT EXISTS "UserApprove" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    adminId INTEGER NOT NULL,
    userId INTEGER NOT NULL UNIQUE,
    FOREIGN KEY (adminId) REFERENCES "Admin" (id),
    FOREIGN KEY (userId) REFERENCES "User" (id)
);
CREATE TABLE IF NOT EXISTS "UserHistory" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    userId INTEGER NOT NULL,
    interpretedTexts TEXT NOT NULL,
    FOREIGN KEY (userId) REFERENCES "User" (id)
);
"""


def get_connection():
    """Return a SQLite connection with row access by column name."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database():
    """Create tables without overwriting existing account data."""
    with get_connection() as connection:
        connection.executescript(SCHEMA)
