"""Shared utilities for Signity blueprints."""

import secrets
from functools import wraps

from flask import (
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from database import get_connection


def api_response(data=None, status=200):
    return jsonify({"success": True, "data": data}), status


def api_error(message, status=400):
    return jsonify({"success": False, "data": None, "error": message}), status


def current_account():
    role, account_id = session.get("role"), session.get("account_id")
    if role not in {"admin", "user"} or not account_id:
        return None
    table = "Admin" if role == "admin" else "User"
    with get_connection() as connection:
        row = connection.execute(
            f'SELECT id, name, email FROM "{table}" WHERE id = ?', (account_id,)
        ).fetchone()
        if row is None:
            session.clear()
            return None
        status = (
            "approved"
            if role == "admin"
            else connection.execute(
                'SELECT approval_status FROM "User" WHERE id = ?', (account_id,)
            ).fetchone()["approval_status"]
        )
    return {
        **dict(row),
        "role": role,
        "approved": status == "approved",
        "approval_status": status,
    }


def _auth_failure(api, message, status):
    if api:
        return api_error(message, status)
    flash(message, "warning")
    return redirect(url_for("auth.login_choice"))


def login_required(api=False):
    def decorator(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            account = current_account()
            if account is None:
                return _auth_failure(api, "Sign in to continue.", 401)
            g.account = account
            return view(*args, **kwargs)

        return wrapped

    return decorator


def admin_required(api=False):
    def decorator(view):
        @login_required(api=api)
        @wraps(view)
        def wrapped(*args, **kwargs):
            if g.account["role"] != "admin":
                return _auth_failure(api, "Administrator access is required.", 403)
            return view(*args, **kwargs)

        return wrapped

    return decorator


def approved_user_required(api=False):
    def decorator(view):
        @login_required(api=api)
        @wraps(view)
        def wrapped(*args, **kwargs):
            if g.account["role"] != "admin" and not g.account["approved"]:
                return _auth_failure(
                    api, "Your account is awaiting admin approval.", 403
                )
            return view(*args, **kwargs)

        return wrapped

    return decorator


def csrf_token():
    return session.setdefault("csrf_token", secrets.token_urlsafe(24))


def csrf_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if request.method in {"GET", "HEAD", "OPTIONS"}:
            return view(*args, **kwargs)
        token = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token")
        if not token or not secrets.compare_digest(
            token, session.get("csrf_token", "")
        ):
            return (
                api_error("Invalid CSRF token.", 400)
                if request.is_json
                else ("Invalid CSRF token.", 400)
            )
        return view(*args, **kwargs)

    return wrapped


def record_history(user_id, interpreted_text):
    if not interpreted_text or not interpreted_text.strip():
        return
    with get_connection() as connection:
        connection.execute(
            'INSERT INTO "UserHistory" (userId, interpretedText, interpretedTexts) VALUES (?, ?, ?)',
            (user_id, interpreted_text.strip(), interpreted_text.strip()),
        )
