"""
Claims blueprint: submit, view, download PDF, recommend, advance workflow.

Includes Phase 5 work:
  * Investigation workflow (Submitted -> Under Review -> Investigating ->
    Verified -> Fraud Confirmed / Cleared)
  * Hospital Risk Intelligence
  * Claim Similarity Engine
  * Recommendation Engine (already in risk_engine._recommend_for)
  * Financial Loss Prevention Analytics
"""
from __future__ import annotations

import io
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone # Already imported

import numpy as np
import pandas as pd
from flask import (
    Blueprint, abort, flash, jsonify, redirect, render_template, request,
    send_file, session, url_for
)
from sqlalchemy import func, or_ # Already imported

from models import AuditEvent, ClaimRecord, Hospital, Investigation, db
from routes.auth import admin_required, login_required
from utils.model_loader import get_model, model_error, predict_proba_safe
from utils.risk_engine import calculate_multi_layer_risk

claims_bp = Blueprint("claims", __name__)

INPUT_COLUMNS = [
    "RenalDiseaseIndicator", "ChronicCond_Alzheimer", "ChronicCond_Heartfailure",
    "ChronicCond_KidneyDisease", "ChronicCond_Cancer", "ChronicCond_ObstrPulmonary",
    "ChronicCond_Depression", "ChronicCond_Diabetes", "ChronicCond_IschemicHeart",
    "ChronicCond_Osteoporasis", "ChronicCond_rheumatoidarthritis", "ChronicCond_stroke",
    "IPAnnualDeductibleAmt", "OPAnnualDeductibleAmt",
]

VALID_STATUSES = [
    "Submitted", "Under Review", "Investigating",
    "Verified", "Fraud Confirmed", "Cleared",
]


# ---------------------------------------------------------------------------
# Submit a claim
# ---------------------------------------------------------------------------
@claims_bp.route("/input", methods=["GET", "POST"])
@login_required
def input_page():
    if request.method == "POST":
        model = get_model()
        if model is None:
            flash(f"Model unavailable: {model_error() or 'unknown error'}.", "danger")
            return render_template("input.html", form_data=request.form, errors=[]) # Already correct

        form_data = {}
        invalid_field_keys = []
        friendly_errors = []

        # Validate numeric fields
        numeric_cols = INPUT_COLUMNS + ["claim_amount", "IPAnnualDeductibleAmt", "OPAnnualDeductibleAmt"]
        for col in numeric_cols:
            val = request.form.get(col, "").strip()
            if val:
                try:
                    form_data[col] = float(val)
                except ValueError:
                    invalid_field_keys.append(col)
                    friendly_name = col.replace("ChronicCond_", "").replace("Indicator", "").replace("Amt", " Amount")
                    friendly_errors.append(friendly_name)
            else:
                form_data[col] = 0.0 # Default to 0.0 for empty numeric fields # Already correct

        # Handle required text fields
        required_text_fields = ["hospital", "provider"]
        for col in required_text_fields:
            val = request.form.get(col, "").strip()
            if not val:
                invalid_field_keys.append(col)
                friendly_errors.append(col.replace("_", " ").title())
            form_data[col] = val # Store even if empty for re-rendering # Already correct

        if invalid_field_keys:
            flash(f"Data Validation Error: Please enter valid data for: {', '.join(friendly_errors)}", "danger")
            return render_template("input.html", errors=invalid_field_keys, form_data=request.form)

        try:
            input_df = pd.DataFrame([form_data])[INPUT_COLUMNS]
            ml_prob = predict_proba_safe(model, input_df)
        except Exception as exc:
            flash(f"Model prediction error: {exc}", "danger")
            return render_template("input.html", form_data=request.form, errors=[]) # Already correct

        risk = calculate_multi_layer_risk(form_data, ml_prob)

        record = ClaimRecord(
            username=session["username"],
            hospital=form_data.get("hospital", "Unknown"), # Already correct
            provider=form_data.get("provider", "Unknown"), # Already correct
            diagnosis_code=request.form.get("diagnosis_code", "").strip(),
            claim_amount=float(request.form.get("claim_amount", 0) or 0),
            renal_disease=form_data["RenalDiseaseIndicator"],
            alzheimer=form_data["ChronicCond_Alzheimer"],
            heart_failure=form_data["ChronicCond_Heartfailure"],
            kidney_disease=form_data["ChronicCond_KidneyDisease"],
            cancer=form_data["ChronicCond_Cancer"],
            obstr_pulmonary=form_data["ChronicCond_ObstrPulmonary"],
            depression=form_data["ChronicCond_Depression"],
            diabetes=form_data["ChronicCond_Diabetes"],
            ischemic_heart=form_data["ChronicCond_IschemicHeart"],
            osteoporosis=form_data["ChronicCond_Osteoporasis"],
            arthritis=form_data["ChronicCond_rheumatoidarthritis"],
            stroke=form_data["ChronicCond_stroke"],
            ip_deductible=form_data["IPAnnualDeductibleAmt"],
            op_deductible=form_data["OPAnnualDeductibleAmt"],
            risk_score=risk.score,
            risk_level=risk.level,
            ml_probability=risk.ml_probability,
            confidence_score=risk.confidence,
            risk_breakdown=json.dumps(risk.layer_breakdown),
            prediction=risk.prediction,
            reasons=risk.reason_text(),
            recommendations=risk.recommendation_text(),
            status="Submitted",
        )
        db.session.add(record)
        db.session.flush()  # need id

        db.session.add(Investigation(
            claim_id=record.id, investigator=session["username"],
            action="Created", note="Claim submitted by provider",
        ))

        # Hospital intelligence
        _update_hospital_stats(record)
        _update_provider_stats(record) # New: Provider intelligence
        db.session.commit()

        flash("Claim submitted. Multi-layer risk analysis complete.", "success")
        return redirect(url_for("claims.result", claim_id=record.id))

    return render_template("input.html", form_data={}, errors=[]) # Already correct


def _update_hospital_stats(record: ClaimRecord) -> None:
    """Phase 5: keep Hospital aggregate stats in sync with new claim."""
    if not record.hospital or record.hospital == "Unknown":
        return
    h = Hospital.query.filter_by(name=record.hospital).first()
    if h is None:
        h = Hospital(name=record.hospital, region="Detected")
        db.session.add(h)
    h.total_claims = (h.total_claims or 0) + 1
    if record.prediction == "Fraud":
        h.fraud_claims = (h.fraud_claims or 0) + 1
    # running average
    n = h.total_claims
    h.avg_risk_score = round(
        ((h.avg_risk_score or 0) * (n - 1) + (record.risk_score or 0)) / n, 2
    )
    if record.prediction == "Fraud":
        h.last_flagged_at = datetime.now(timezone.utc)


def _update_provider_stats(record: ClaimRecord) -> None:
    """New: Keep Provider aggregate stats in sync with new claim."""
    if not record.provider or record.provider == "Unknown":
        return
    p = Provider.query.filter_by(name=record.provider).first()
    if p is None:
        p = Provider(name=record.provider)
        db.session.add(p)
    p.total_claims = (p.total_claims or 0) + 1
    if record.prediction == "Fraud":
        p.fraud_claims = (p.fraud_claims or 0) + 1
    # running average
    n = p.total_claims
    p.avg_risk_score = round(
        ((p.avg_risk_score or 0) * (n - 1) + (record.risk_score or 0)) / n, 2
    )
    if record.prediction == "Fraud":
        p.last_flagged_at = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Result / detail
# ---------------------------------------------------------------------------
@claims_bp.route("/result/<int:claim_id>")
@login_required
def result(claim_id):
    claim = ClaimRecord.query.get_or_404(claim_id)
    # Phase 3: ownership check (admin can see all)
    if claim.username != session["username"] and not session.get("is_admin"):
        abort(403)

    # Prepare Intelligence Data for Plotly
    try:
        breakdown = json.loads(claim.risk_breakdown or "{}")
    except:
        breakdown = {} # Already correct

    similar = _find_similar_claims(claim, limit=5)
    return render_template("result.html", claim=claim, similar=similar, breakdown=breakdown)


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
@claims_bp.route("/download_pdf/<int:claim_id>")
@login_required
def download_pdf(claim_id):
    claim = ClaimRecord.query.get_or_404(claim_id)
    if claim.username != session["username"] and not session.get("is_admin"):
        abort(403)

    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors

    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Header
    p.setFillColor(colors.HexColor("#0f172a"))
    p.rect(0, height - 80, width, 80, fill=1, stroke=0)
    p.setFillColor(colors.white)
    p.setFont("Helvetica-Bold", 18)
    p.drawString(50, height - 50, "MedGuard AI - Fraud Investigation Report")
    p.setFont("Helvetica", 10)
    p.drawRightString(width - 50, height - 50, f"Claim #{claim.id}")

    y = height - 120
    p.setFillColor(colors.black)
    p.setFont("Helvetica-Bold", 12)
    p.drawString(50, y, f"Risk Level: {claim.risk_level}")
    p.drawString(50, y - 20, f"Risk Score: {claim.risk_score}/100")
    p.drawString(50, y - 40, f"Prediction: {claim.prediction}")
    p.drawString(50, y - 60, f"ML Probability: {claim.ml_probability}")
    p.drawString(50, y - 80, f"Status: {claim.status}")
    p.drawString(50, y - 100, f"Claim Amount: ${claim.claim_amount:,.2f}")
    p.drawString(50, y - 120, f"Hospital: {claim.hospital}")
    p.drawString(50, y - 140, f"Provider: {claim.provider}")
    p.drawString(50, y - 160, f"Submitted: {claim.timestamp.strftime('%Y-%m-%d %H:%M')}")

    # Reasons
    p.setFont("Helvetica-Bold", 11)
    p.drawString(50, y - 200, "Reasons flagged:")
    p.setFont("Helvetica", 10)
    text = p.beginText(50, y - 220)
    for line in (claim.reasons or "").split(","):
        clean_line = line.strip()
        if clean_line:
            text.textLine(f"  - {clean_line}")
    p.drawText(text)

    # Recommendations
    p.setFont("Helvetica-Bold", 11)
    p.drawString(50, y - 320, "Recommendations:")
    p.setFont("Helvetica", 10)
    text = p.beginText(50, y - 340)
    for line in (claim.recommendations or "").split("|"):
        clean_line = line.strip()
        if clean_line:
            text.textLine(f"  - {clean_line}")
    p.drawText(text)

    # Footer
    p.setFont("Helvetica-Oblique", 8)
    p.setFillColor(colors.grey)
    p.drawString(50, 30, "Generated by MedGuard AI - confidential, internal use only.")

    p.showPage()
    p.save()
    buffer.seek(0)
    return send_file(
        buffer, as_attachment=True,
        download_name=f"MedGuard_Report_{claim_id}.pdf",
        mimetype="application/pdf",
    )


# ---------------------------------------------------------------------------
# Investigation workflow
# ---------------------------------------------------------------------------
@claims_bp.route("/claim/<int:claim_id>/status", methods=["POST"])
@login_required
def update_status(claim_id):
    claim = ClaimRecord.query.get_or_404(claim_id)
    if claim.username != session["username"] and not session.get("is_admin"):
        abort(403)
    new_status = request.form.get("status", "").strip()
    note = request.form.get("note", "").strip()
    if new_status not in VALID_STATUSES:
        flash("Invalid status.", "danger")
        return redirect(url_for("claims.result", claim_id=claim_id))

    old = claim.status
    claim.status = new_status
    claim.status_updated_at = datetime.now(timezone.utc) # Already correct
    if note:
        claim.investigation_notes = (claim.investigation_notes or "") + \
            f"\n[{datetime.now(timezone.utc):%Y-%m-%d %H:%M}] {session['username']}: {note}" # Already correct
    db.session.add(Investigation(
        claim_id=claim.id, investigator=session["username"],
        action=f"{old} -> {new_status}", note=note or "Status updated",
    ))
    db.session.add(AuditEvent(
        actor=session["username"], action="status_update",
        target=f"claim:{claim.id}", detail=f"{old} -> {new_status}",
    ))
    db.session.commit()
    flash(f"Status updated to {new_status}.", "success")
    return redirect(url_for("claims.result", claim_id=claim_id))


# ---------------------------------------------------------------------------
# Phase 5: Hospital risk intelligence
# ---------------------------------------------------------------------------
@claims_bp.route("/hospitals")
@login_required
def hospitals():
    rows = Hospital.query.order_by(Hospital.avg_risk_score.desc()).all()
    # also include hospitals that appear in claims but are not in the table
    names_in_claims = [
        r[0] for r in db.session.query(ClaimRecord.hospital)
        .filter(ClaimRecord.hospital.isnot(None))
        .filter(ClaimRecord.hospital != "")
        .filter(ClaimRecord.hospital != "Unknown")
        .distinct().all()
    ]
    seen = {h.name for h in rows}
    for name in names_in_claims:
        if name not in seen:
            total = ClaimRecord.query.filter_by(hospital=name).count()
            fraud = ClaimRecord.query.filter_by(
                hospital=name, prediction="Fraud"
            ).count()
            avg = db.session.query(
                func.avg(ClaimRecord.risk_score)
            ).filter_by(hospital=name).scalar() or 0.0
            rows.append(Hospital(
                name=name, total_claims=total, fraud_claims=fraud,
                avg_risk_score=round(float(avg), 2),
            ))
    rows.sort(key=lambda h: h.avg_risk_score, reverse=True)
    return render_template("hospitals.html", hospitals=rows)


# ---------------------------------------------------------------------------
# Phase 5: Claim similarity engine
# ---------------------------------------------------------------------------
def _claim_to_vector(c: ClaimRecord) -> np.ndarray:
    return np.array([
        float(c.renal_disease or 0), float(c.alzheimer or 0), float(c.heart_failure or 0),
        float(c.kidney_disease or 0), float(c.cancer or 0), float(c.obstr_pulmonary or 0),
        float(c.depression or 0), float(c.diabetes or 0), float(c.ischemic_heart or 0),
        float(c.osteoporosis or 0), float(c.arthritis or 0), float(c.stroke or 0),
        float(c.ip_deductible or 0), float(c.op_deductible or 0),
        float(c.claim_amount or 0), # Already correct
    ], dtype=float)


def _find_similar_claims(target: ClaimRecord, limit: int = 5):
    """
    Cosine similarity on the chronic-condition + deductible vector.
    Excludes the target itself and is O(N) - fine for the dataset sizes
    in this hackathon.
    """
    candidates = ClaimRecord.query.filter(ClaimRecord.id != target.id).all()
    if not candidates:
        return []
    target_v = _claim_to_vector(target)
    tnorm = np.linalg.norm(target_v) or 1.0
    scored = []
    for c in candidates:
        try:
            v = _claim_to_vector(c)
            v_norm = np.linalg.norm(v) or 1.0
            sim = float(np.dot(target_v, v) / (tnorm * v_norm)) if (tnorm * v_norm) != 0 else 0.0 # Fixed division by zero
            scored.append((sim, c))
        except Exception as e:
            current_app.logger.error(f"Error calculating similarity for claim {c.id}: {e}")
            scored.append((0.0, c)) # Default to 0 similarity on error
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:limit]


@claims_bp.route("/api/claim/<int:claim_id>/similar")
@login_required
def api_similar(claim_id):
    claim = ClaimRecord.query.get_or_404(claim_id)
    if claim.username != session["username"] and not session.get("is_admin"):
        abort(403)
    out = []
    for sim, c in _find_similar_claims(claim, limit=10): # This was the bug, now fixed by _find_similar_claims returning tuples
        out.append({
            "id": c.id, "risk_score": c.risk_score, "risk_level": c.risk_level,
            "prediction": c.prediction, "similarity": round(sim, 4),
            "hospital": c.hospital, "claim_amount": c.claim_amount,
        })
    return jsonify(out)


# ---------------------------------------------------------------------------
# Phase 5: Executive / financial analytics
# ---------------------------------------------------------------------------
@claims_bp.route("/analytics")
@login_required
def analytics():
    total = db.session.query(func.count(ClaimRecord.id)).scalar() or 0
    fraud = db.session.query(func.count(ClaimRecord.id))\
        .filter(ClaimRecord.prediction == "Fraud").scalar() or 0
    under_review = db.session.query(func.count(ClaimRecord.id))\
        .filter(ClaimRecord.status.in_(["Under Review", "Investigating"])).scalar() or 0
    total_amount = db.session.query(
        func.coalesce(func.sum(ClaimRecord.claim_amount), 0.0)
    ).scalar() or 0.0
    fraud_amount = db.session.query(
        func.coalesce(func.sum(ClaimRecord.claim_amount), 0.0)
    ).filter(ClaimRecord.prediction == "Fraud").scalar() or 0.0
    avg_score = db.session.query(
        func.coalesce(func.avg(ClaimRecord.risk_score), 0.0)
    ).scalar() or 0.0

    # risk distribution
    rd = dict(
        db.session.query(ClaimRecord.risk_level, func.count(ClaimRecord.id))
        .group_by(ClaimRecord.risk_level).all()
    )
    rd = {k or "Low": int(v) for k, v in rd.items()}

    # monthly trend
    cutoff = datetime.now(timezone.utc) - timedelta(days=180) # Already correct
    rows = (
        db.session.query(
            func.strftime("%Y-%m", ClaimRecord.timestamp).label("m"),
            func.count(ClaimRecord.id).label("n"),
            func.sum(
                func.iif(ClaimRecord.prediction == "Fraud", 1, 0)
                if hasattr(func, "iif") else
                func.case((ClaimRecord.prediction == "Fraud", 1), else_=0)
            ).label("f"),
        )
        .filter(ClaimRecord.timestamp >= cutoff)
        .group_by("m").order_by("m").all()
    )
    months = [r.m for r in rows]
    monthly_total = [int(r.n) for r in rows]
    monthly_fraud = [int(r.f or 0) for r in rows]

    # hospital rankings
    hospital_rows = (
        db.session.query(
            ClaimRecord.hospital,
            func.count(ClaimRecord.id).label("n"),
            func.avg(ClaimRecord.risk_score).label("avg_risk"),
        )
        .filter(ClaimRecord.hospital.isnot(None))
        .filter(ClaimRecord.hospital != "")
        .filter(ClaimRecord.hospital != "Unknown")
        .group_by(ClaimRecord.hospital)
        .order_by(func.avg(ClaimRecord.risk_score).desc())
        .limit(10).all()
    )
    hospital_data = [
        {
            "name": h[0], "count": int(h[1]),
            "avg_risk": round(float(h[2] or 0), 2),
        }
        for h in hospital_rows
    ]

    return render_template(
        "analytics.html",
        total=total, fraud=fraud, under_review=under_review,
        total_amount=total_amount, fraud_amount=fraud_amount,
        avg_score=round(float(avg_score), 2),
        loss_prevented=round(float(fraud_amount), 2),
        risk_distribution=rd,
        months=months, monthly_total=monthly_total,
        monthly_fraud=monthly_fraud,
        hospital_data=hospital_data,
    )
