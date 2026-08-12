#!/usr/bin/env python3
"""Temporary emergency script to reset the primary admin password.

Usage:
    python reset_admin.py

This script is intended for one-time emergency use only. Delete or secure
it after recovery.
"""

import sqlite3
from pathlib import Path

from database import get_connection
from werkzeug.security import generate_password_hash

ADMIN_EMAIL = "admin@gmail.com"
NEW_PASSWORD = "admin123"


def reset_admin():
    base_dir = Path(__file__).resolve().parent
    db_path = base_dir / "signity.db"
    if not db_path.exists():
        print(f"Database not found at {db_path}")
        return

    with get_connection() as connection:
        row = connection.execute(
            'SELECT id, email FROM "Admin" WHERE email = ?', (ADMIN_EMAIL,)
        ).fetchone()
        if row is None:
            print(f"No admin found with email {ADMIN_EMAIL}")
            return

        connection.execute(
            'UPDATE "Admin" SET password = ? WHERE id = ?',
            (generate_password_hash(NEW_PASSWORD), row["id"]),
        )
    print(f"Password for admin {ADMIN_EMAIL} has been reset.")


if __name__ == "__main__":
    reset_admin()
