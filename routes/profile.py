"""Profile management routes blueprint."""

from flask import Blueprint, flash, g, redirect, render_template, request, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from database import get_connection
from shared import csrf_required, login_required

bp = Blueprint("profile", __name__)


@bp.route("/profile/edit", methods=["GET", "POST"])
@login_required()
@csrf_required
def edit_profile():
    account = g.account
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        if not name or not email:
            flash("Name and email are required.", "error")
        else:
            table = "Admin" if account["role"] == "admin" else "User"
            try:
                with get_connection() as connection:
                    connection.execute(
                        f'UPDATE "{table}" SET name = ?, email = ? WHERE id = ?',
                        (name, email, account["id"]),
                    )
            except Exception:
                flash("That email is already in use.", "error")
            else:
                flash("Profile updated.", "success")
                return redirect(url_for("main.index"))
    return render_template("edit_profile.html", account=account)


@bp.route("/profile/password", methods=["GET", "POST"])
@login_required()
@csrf_required
def change_password():
    account = g.account
    if request.method == "POST":
        current_password = request.form.get("current_password", "")
        new_password = request.form.get("new_password", "")
        confirm_password = request.form.get("confirm_password", "")
        if not current_password or not new_password:
            flash("All password fields are required.", "error")
        elif len(new_password) < 8:
            flash("New password must be at least 8 characters.", "error")
        elif new_password != confirm_password:
            flash("New passwords do not match.", "error")
        else:
            table = "Admin" if account["role"] == "admin" else "User"
            with get_connection() as connection:
                row = connection.execute(
                    f'SELECT password FROM "{table}" WHERE id = ?', (account["id"],)
                ).fetchone()
            if row is None or not check_password_hash(
                row["password"], current_password
            ):
                flash("Current password is incorrect.", "error")
            else:
                with get_connection() as connection:
                    connection.execute(
                        f'UPDATE "{table}" SET password = ? WHERE id = ?',
                        (generate_password_hash(new_password), account["id"]),
                    )
                flash("Password changed successfully.", "success")
                return redirect(url_for("main.index"))
    return render_template("change_password.html", account=account)
