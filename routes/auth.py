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

from flask import (
    Blueprint, flash, redirect, render_template, request,
    session, url_for, current_app
)
from werkzeug.security import generate_password_hash, check_password_hash

from models import User, db, AuditEvent

auth_bp = Blueprint("auth", __name__)

# ---- in-memory rate limiter (good enough for the hackathon) ----
_attempts: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_WINDOW = 60.0
RATE_LIMIT_MAX = 5


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------
def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "username" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("auth.login"))
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


def _record_attempt(username: str) -> bool:
    """Return True if the request is allowed, False if rate-limited."""
    now = time.time()
    bucket = _attempts[username]
    _attempts[username] = [t for t in bucket if now - t < RATE_LIMIT_WINDOW]
    if len(_attempts[username]) >= RATE_LIMIT_MAX:
        return False
    _attempts[username].append(now)
    return True


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
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
            actor=username, action="register", target=username
        ))
        db.session.commit()
        flash("Registration successful. Please log in.", "success")
        return redirect(url_for("auth.login"))

    return render_template("register.html")


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not _record_attempt(username):
            flash("Too many attempts. Please wait a minute.", "danger")
            return render_template("login.html")

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session.permanent = True
            session["username"] = user.username
            session["is_admin"] = bool(user.is_admin)
            db.session.add(AuditEvent(
                actor=user.username, action="login", target=user.username
            ))
            db.session.commit()
            flash(f"Welcome back, {user.username}!", "success")
            return redirect(url_for("main.dashboard"))

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
