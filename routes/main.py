"""
Main / dashboard / analytics routes.

Phase 2 fix: `dashboard` no longer loads every claim into Python.
It uses SQL aggregates (count / filtered count) so the query stays
constant-time with the number of claims.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from flask import Blueprint, current_app, flash, redirect, render_template, request, session, url_for
from sqlalchemy import func, or_, and_

import plotly
import plotly.express as px

from models import ClaimRecord, Hospital, User, db
from routes.auth import login_required, admin_required
from utils.model_loader import compute_dataset_metrics, get_model
from utils.risk_engine import calculate_multi_layer_risk

main_bp = Blueprint("main", __name__)


# ---------------------------------------------------------------------------
# Public pages
# ---------------------------------------------------------------------------
@main_bp.route("/")
def home():
    return render_template("home.html")


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
@main_bp.route("/dashboard")
@login_required
def dashboard():
    # SQL aggregates - O(1) round trip
    total = db.session.query(func.count(ClaimRecord.id)).scalar() or 0
    fraud = db.session.query(func.count(ClaimRecord.id))\
        .filter(
            or_(
                ClaimRecord.status == "Fraud Confirmed",
                and_(ClaimRecord.prediction == "Fraud", ClaimRecord.status != "Cleared")
            )
        ).scalar() or 0
    under_review = db.session.query(func.count(ClaimRecord.id))\
        .filter(ClaimRecord.status.in_(["Under Review", "Investigating"])).scalar() or 0
    total_amount = db.session.query(
        func.coalesce(func.sum(ClaimRecord.claim_amount), 0.0)
    ).scalar() or 0.0
    fraud_amount = db.session.query(
        func.coalesce(func.sum(ClaimRecord.claim_amount), 0.0)
    ).filter(
        or_(
            ClaimRecord.status == "Fraud Confirmed",
            and_(ClaimRecord.prediction == "Fraud", ClaimRecord.status != "Cleared")
        )
    ).scalar() or 0.0

    # Recent claims for the current user
    user_claims = (
        ClaimRecord.query
        .filter_by(username=session["username"])
        .order_by(ClaimRecord.timestamp.desc())
        .limit(5)
        .all()
    )

    # Risk distribution for the executive pie chart
    risk_counts = dict(
        db.session.query(ClaimRecord.risk_level, func.count(ClaimRecord.id))
        .group_by(ClaimRecord.risk_level).all()
    )
    if total > 0:
        fig = px.pie(
            names=["Fraud", "No Fraud"],
            values=[fraud, total - fraud],
            title="Claims Distribution",
            color_discrete_sequence=["#ef553b", "#636efa"],
            hole=0.55,
        )
        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    else:
        graph_json = "null"

    # 30-day trend for the small line chart on dashboard
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    rows = (
        db.session.query(
            func.date(ClaimRecord.timestamp).label("d"),
            func.count(ClaimRecord.id).label("n"),
        )
        .filter(ClaimRecord.timestamp >= cutoff)
        .group_by("d")
        .order_by("d")
        .all()
    )
    trend_labels = [str(r.d) for r in rows]
    trend_values = [int(r.n) for r in rows]
    if not trend_labels:
        trend_labels = [datetime.now(timezone.utc).strftime("%Y-%m-%d")]
        trend_values = [0]

    return render_template(
        "dashboard.html",
        total=total,
        fraud=fraud,
        under_review=under_review,
        total_amount=total_amount,
        fraud_amount=fraud_amount,
        loss_prevented=round(fraud_amount, 2),
        graph_json=graph_json,
        recent_claims=user_claims,
        risk_counts={k or "Low": int(v) for k, v in risk_counts.items()},
        trend_labels=trend_labels,
        trend_values=trend_values,
    )


# ---------------------------------------------------------------------------
# Model metrics
# ---------------------------------------------------------------------------
@main_bp.route("/chart")
@login_required
def chart():
    metrics = compute_dataset_metrics() or {
        "total_records": 0, "fraud_records": 0, "legit_records": 0,
        "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
        "confusion_matrix": [[0, 0], [0, 0]],
        "legit_precision": 0.0, "legit_recall": 0.0, "legit_f1": 0.0,
        "fraud_precision": 0.0, "fraud_recall": 0.0, "fraud_f1": 0.0,
        "macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0,
        "weighted_precision": 0.0, "weighted_recall": 0.0, "weighted_f1": 0.0,
    }
    return render_template("chart.html", metrics=metrics)
