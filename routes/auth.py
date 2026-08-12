"""Authentication routes blueprint."""

from flask import Blueprint, flash, redirect, render_template, request, session, url_for

from database import get_connection
from shared import (
    csrf_required,
    current_account,
)
from werkzeug.security import check_password_hash, generate_password_hash

bp = Blueprint("auth", __name__)


@bp.route("/register", methods=["GET", "POST"])
@csrf_required
def register():
    if session.get("account_id"):
        return redirect(url_for("main.index"))
    if session.get("account_id"):
        return redirect(url_for("main.index"))
    account = current_account()
    if request.method == "POST":
        from werkzeug.security import generate_password_hash

        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not name or not email or len(password) < 8:
            flash(
                "Enter a name and email, and use a password with at least 8 characters.",
                "error",
            )
        else:
            try:
                with get_connection() as connection:
                    connection.execute(
                        'INSERT INTO "User" (name, email, password, is_approved, approval_status) VALUES (?, ?, ?, 0, "pending")',
                        (name, email, generate_password_hash(password)),
                    )
            except Exception:
                flash("An account with that email already exists.", "error")
            else:
                flash(
                    "Registration received. An admin must approve your account before access is enabled.",
                    "success",
                )
                return redirect(url_for("auth.login_choice"))
    return render_template("register.html", account=account)


@bp.route("/login", methods=["GET"])
def login_choice():
    return render_template("login.html", account=current_account())


@bp.route("/login/user", methods=["GET", "POST"])
@csrf_required
def login_user():
    if request.method == "POST":
        from werkzeug.security import check_password_hash

        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        with get_connection() as connection:
            account = connection.execute(
                'SELECT id, password FROM "User" WHERE email = ?', (email,)
            ).fetchone()
        if account is None or not check_password_hash(account["password"], password):
            flash("Invalid email or password.", "error")
            return render_template("login_user.html", account=None)
        session.clear()
        session.update(account_id=account["id"], role="user")
        if not current_account()["approved"]:
            status = current_account()["approval_status"]
            flash(
                f"Your account is {status}; an admin must approve it before access is enabled.",
                "warning",
            )
            return redirect(url_for("main.index"))
        return redirect(url_for("predict.live"))
    return render_template("login_user.html", account=current_account())


@bp.route("/login/admin", methods=["GET", "POST"])
@csrf_required
def login_admin():
    if request.method == "POST":
        from werkzeug.security import check_password_hash

        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        with get_connection() as connection:
            account = connection.execute(
                'SELECT id, password FROM "Admin" WHERE email = ?', (email,)
            ).fetchone()
        if account is None or not check_password_hash(account["password"], password):
            flash("Invalid email or password.", "error")
            return render_template("login_admin.html", account=None)
        session.clear()
        session.update(account_id=account["id"], role="admin")
        return redirect(url_for("admin.admin_dashboard"))
    return render_template("login_admin.html", account=current_account())


@bp.post("/logout")
@csrf_required
def logout():
    session.clear()
    return redirect(url_for("main.index"))
