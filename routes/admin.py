"""Admin routes blueprint."""

from flask import Blueprint, flash, g, redirect, render_template, request, url_for

from database import get_connection
from shared import admin_required, csrf_required

bp = Blueprint("admin", __name__)


@bp.route("/admin")
@admin_required()
def admin_dashboard():
    account = g.account
    page = max(request.args.get("page", 1, type=int), 1)
    page_size = 20
    offset = (page - 1) * page_size
    with get_connection() as connection:
        pending = connection.execute(
            'SELECT id, name, email, approval_status, created_at FROM "User" WHERE approval_status = "pending" ORDER BY created_at ASC, id ASC'
        ).fetchall()
        activity = connection.execute(
            'SELECT a.action, a.timestamp, admin.name AS admin_name, user.name AS user_name FROM "AdminActivity" a LEFT JOIN "Admin" admin ON admin.id = a.adminId LEFT JOIN "User" user ON user.id = a.userId ORDER BY a.timestamp DESC, a.id DESC LIMIT ? OFFSET ?',
            (page_size, offset),
        ).fetchall()
        activity_count = connection.execute(
            'SELECT COUNT(*) AS count FROM "AdminActivity"'
        ).fetchone()["count"]
    return render_template(
        "admin.html",
        account=account,
        pending=pending,
        activity=activity,
        page=page,
        has_next=offset + page_size < activity_count,
    )


@bp.post("/admin/users/<int:user_id>/approve")
@csrf_required
@admin_required()
def approve_user(user_id):
    account = g.account
    with get_connection() as connection:
        user = connection.execute(
            'SELECT id, name FROM "User" WHERE id = ?', (user_id,)
        ).fetchone()
        if user is None:
            return ("User not found.", 404)
        connection.execute(
            'UPDATE "User" SET is_approved = 1, approval_status = "approved" WHERE id = ?',
            (user_id,),
        )
        connection.execute(
            'INSERT INTO "UserApprove" (adminId, userId, decision) VALUES (?, ?, "approved")',
            (account["id"], user_id),
        )
        connection.execute(
            'INSERT INTO "AdminActivity" (adminId, userId, action) VALUES (?, ?, ?)',
            (account["id"], user_id, f"Approved {user['name']}"),
        )
        flash(f"Approved {user['name']}.", "success")
    return redirect(url_for("admin.admin_dashboard"))


@bp.post("/admin/users/<int:user_id>/reject")
@csrf_required
@admin_required()
def reject_user(user_id):
    account = g.account
    with get_connection() as connection:
        user = connection.execute(
            'SELECT id, name FROM "User" WHERE id = ?', (user_id,)
        ).fetchone()
        if user is None:
            return ("User not found.", 404)
        connection.execute(
            'UPDATE "User" SET is_approved = 0, approval_status = "rejected" WHERE id = ?',
            (user_id,),
        )
        connection.execute(
            'INSERT INTO "UserApprove" (adminId, userId, decision) VALUES (?, ?, "rejected")',
            (account["id"], user_id),
        )
        connection.execute(
            'INSERT INTO "AdminActivity" (adminId, userId, action) VALUES (?, ?, ?)',
            (account["id"], user_id, f"Rejected {user['name']}"),
        )
        flash(f"Rejected {user['name']}.", "success")
    return redirect(url_for("admin.admin_dashboard"))
