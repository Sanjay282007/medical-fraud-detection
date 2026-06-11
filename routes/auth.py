"""
Authentication blueprint: register / login / logout.

Phase 2 hardening:
  * Passwords are 8+ chars, must contain letters and digits
  * Login is rate-limited in-memory (3/min per username)
  * Session is set `permanent` so the lifetime config applies
  * Generic error messages so we don't leak whether a username exists
"""
from __future__ import annotations

import re
import time
from collections import defaultdict
from functools import wraps
from datetime import datetime, timezone, timedelta

from flask import (
    Blueprint, flash, redirect, render_template, request,
    session, url_for, current_app
)
from werkzeug.security import generate_password_hash, check_password_hash

from models import User, db, AuditEvent

auth_bp = Blueprint("auth", __name__)

# Rate limit variables removed (now database-backed)


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------
def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "username" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("auth.login", next=request.full_path))
        return view(*args, **kwargs)
    return wrapped


def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for("auth.login"))
        if not session.get("is_admin"):
            flash("Admin privileges required.", "danger")
            return redirect(url_for("main.dashboard"))
        return view(*args, **kwargs)
    return wrapped


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_PASSWORD_RE = re.compile(r"^(?=.*[A-Za-z])(?=.*\d).{8,}$")


def _is_strong(pw: str) -> bool:
    return bool(_PASSWORD_RE.match(pw))


DUMMY_HASH = "scrypt:32768:8:1$dummy_salt$781ac94f61f77d337a7cdbc7953288f3bfcd23d8c8309df5370d0cf6e3556cc6"


def _is_rate_limited(username: str) -> bool:
    """Return True if rate-limited (too many failed attempts)."""
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=60)
    failed_attempts = AuditEvent.query.filter(
        AuditEvent.action == "login_fail",
        AuditEvent.actor == username,
        AuditEvent.created_at >= cutoff
    ).count()
    return failed_attempts >= 5


def _is_reg_rate_limited(ip: str) -> bool:
    """Return True if rate-limited (too many registration requests)."""
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=60)
    reg_count = AuditEvent.query.filter(
        AuditEvent.action == "register_attempt",
        AuditEvent.detail == ip,
        AuditEvent.created_at >= cutoff
    ).count()
    return reg_count >= 3


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if "username" in session:
        return redirect(url_for("main.dashboard"))

    if request.method == "POST":
        ip = request.remote_addr or "unknown"
        if _is_reg_rate_limited(ip):
            flash("Too many registration requests. Please wait a minute.", "danger")
            return render_template("register.html")

        db.session.add(AuditEvent(
            actor="system", action="register_attempt", target=ip, detail=ip
        ))
        db.session.commit()

        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        if not username or not password:
            flash("Username and password are required.", "danger")
            return render_template("register.html")

        if password != confirm:
            flash("Passwords do not match.", "danger")
            return render_template("register.html")

        if not _is_strong(password):
            flash(
                "Password must be 8+ characters and include letters and digits.",
                "danger",
            )
            return render_template("register.html")

        if User.query.filter_by(username=username).first():
            # Generic message - don't leak account existence
            flash("Unable to create account. Try a different username.", "danger")
            return render_template("register.html")

        user = User(
            username=username,
            password=generate_password_hash(password),
        )
        db.session.add(user)
        db.session.commit()
        db.session.add(AuditEvent(
            actor=username, action="register", target=username, detail=ip
        ))
        db.session.commit()
        flash("Registration successful. Please log in.", "success")
        return redirect(url_for("auth.login"))

    return render_template("register.html")


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if "username" in session:
        return redirect(url_for("main.dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if _is_rate_limited(username):
            flash("Too many attempts. Please wait a minute.", "danger")
            return render_template("login.html")

        user = User.query.filter_by(username=username).first()
        if user:
            is_valid = check_password_hash(user.password, password)
        else:
            check_password_hash(DUMMY_HASH, password)
            is_valid = False

        if user and is_valid:
            session.permanent = True
            session["username"] = user.username
            session["is_admin"] = bool(user.is_admin)
            db.session.add(AuditEvent(
                actor=user.username, action="login", target=user.username
            ))
            db.session.commit()
            flash(f"Welcome back, {user.username}!", "success")
            next_url = request.args.get("next")
            if next_url and next_url.startswith("/") and not next_url.startswith("//"):
                return redirect(next_url)
            return redirect(url_for("main.dashboard"))

        # Log failed login attempt
        db.session.add(AuditEvent(
            actor=username, action="login_fail", target="login_attempt", detail=request.remote_addr
        ))
        db.session.commit()
        flash("Invalid username or password.", "danger")
    return render_template("login.html")


@auth_bp.route("/logout")
def logout():
    user = session.get("username")
    session.clear()
    if user:
        try:
            db.session.add(AuditEvent(actor=user, action="logout", target=user))
            db.session.commit()
        except Exception:
            current_app.logger.exception("Failed to log logout event")
    flash("Logged out successfully.", "info")
    return redirect(url_for("main.home"))
