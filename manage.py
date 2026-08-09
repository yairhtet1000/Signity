"""Administrative commands for a local Signity installation."""

import argparse
from getpass import getpass

from werkzeug.security import generate_password_hash

from database import get_connection, initialize_database


def create_admin(args):
    initialize_database()
    password = getpass("Admin password: ")
    confirmation = getpass("Confirm password: ")
    if password != confirmation:
        raise SystemExit("Passwords do not match.")
    if len(password) < 8:
        raise SystemExit("Use a password with at least 8 characters.")
    try:
        with get_connection() as connection:
            connection.execute(
                'INSERT INTO "Admin" (name, email, password) VALUES (?, ?, ?)',
                (args.name.strip(), args.email.strip().lower(), generate_password_hash(password)),
            )
    except Exception as exc:
        raise SystemExit(f"Could not create admin: {exc}") from exc
    print(f"Created admin account for {args.email.strip().lower()}.")


def main():
    parser = argparse.ArgumentParser(description="Manage Signity accounts.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create-admin", help="Create an administrator account.")
    create.add_argument("--name", required=True)
    create.add_argument("--email", required=True)
    create.set_defaults(handler=create_admin)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
