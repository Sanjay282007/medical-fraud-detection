"""
MedGuard AI - Healthcare Insurance Fraud Prevention Platform.

Phase 2-7 entry point.

Run:
    python app.py
"""
from __future__ import annotations

import json
import logging
import os
import random
import warnings
from datetime import datetime, timedelta, timezone

from flask import Flask, render_template
from flask_wtf.csrf import CSRFProtect, generate_csrf
from sqlalchemy import event
from sqlalchemy.engine import Engine

from config import Config
from models import (
    AuditEvent, ClaimRecord, Hospital, Investigation, Provider, User, db
)
from routes.auth import auth_bp
from routes.claims import claims_bp
from routes.main import main_bp
from utils.model_loader import get_model, predict_proba_safe


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app(config_class: type = Config) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config_class)

    _configure_logging(app)
    _configure_db(app)
    _configure_security(app)
    _register_blueprints(app)
    _register_context(app)
    _register_error_handlers(app)

    with app.app_context():
        # db.create_all() # Handled by _maybe_seed for schema updates
        _maybe_seed(app)

    return app


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------
def _configure_logging(app: Flask) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _configure_db(app: Flask) -> None:
    db.init_app(app)

    # SQLite: enable foreign keys + WAL for stability under load
    @event.listens_for(Engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, _):
        try:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()
        except Exception:
            pass


def _configure_security(app: Flask) -> None:
    CSRFProtect(app)

    @app.after_request
    def _security_headers(resp):
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        resp.headers["Permissions-Policy"] = "geolocation=(), microphone=()"
        # Permissive but safer default; tighten in production.
        resp.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "img-src 'self' data: https:; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com https://fonts.googleapis.com; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdn.plot.ly https://cdnjs.cloudflare.com https://code.jquery.com https://cdn.jsdelivr.net; "
            "font-src 'self' https://fonts.gstatic.com https://cdnjs.cloudflare.com; "
            "connect-src 'self' https://cdn.plot.ly; "
            "frame-ancestors 'none';"
        )
        return resp


def _register_blueprints(app: Flask) -> None:
    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(claims_bp)


def _register_context(app: Flask) -> None:
    @app.context_processor
    def inject_globals():
        return {
            "csrf_token": generate_csrf,
            "now": lambda: datetime.now(timezone.utc),
            "app_name": "MedGuard Intelligence",
        }


def _register_error_handlers(app: Flask) -> None:
    @app.errorhandler(400)
    def _bad(e):
        return render_template("error.html", code=400,
                               message="Bad request."), 400

    @app.errorhandler(403)
    def _forbidden(e):
        return render_template("error.html", code=403,
                               message="You do not have permission to view this resource."), 403

    @app.errorhandler(404)
    def _not_found(e):
        return render_template("error.html", code=404,
                               message="The page you are looking for could not be found."), 404

    @app.errorhandler(500)
    def _server(e):
        app.logger.exception("Internal Server Error")
        return render_template("error.html", code=500,
                               message="The Intelligence Engine encountered an unexpected operational failure."), 500

    @app.errorhandler(Exception)
    def _generic(e):
        # Let HTTPExceptions use their real status; only swallow true exceptions
        from werkzeug.exceptions import HTTPException
        if isinstance(e, HTTPException):
            return e
        app.logger.exception("Unhandled operational exception")
        return render_template("error.html", code=500,
                               message="A critical system anomaly has occurred. Please contact system administration."), 500


# ---------------------------------------------------------------------------
# Seeding (Phase 7 - demo data so judges see a populated system)
# ---------------------------------------------------------------------------
def _maybe_seed(app: Flask) -> None:
    """Seed only when the DB is essentially empty AND seed is requested."""
    force = os.environ.get("MEDGUARD_SEED", "").lower() in ("1", "true", "yes")
    try:
        user_count = User.query.count()
        claim_count = ClaimRecord.query.count()
    except Exception as exc:
        # Create missing tables safely (db.create_all() will not overwrite existing tables)
        if "no such table" in str(exc).lower():
            app.logger.warning("Database tables missing. Creating tables...")
            db.create_all()
            user_count, claim_count = 0, 0
            force = True  # Populate after creation
        else:
            raise exc

    if user_count > 0 and claim_count > 0 and not force:
        return
    _seed_demo_claims(app)


def _seed_demo_claims(app: Flask) -> None:
    """Create 15 demo claims across hospitals so the analytics page has life."""
    model = get_model()
    if model is None:
        app.logger.warning("Seed skipped: model unavailable")
        return

    from werkzeug.security import generate_password_hash
    random.seed(42)
    np_seed_user = "demo_user"
    np_seed_admin = "admin"

    # Seed regular demo user
    if not User.query.filter_by(username=np_seed_user).first():
        db.session.add(User(
            username=np_seed_user,
            password=generate_password_hash("Demo12345"),
            is_admin=False,
        ))
    # Seed admin user (Fix: Missing admin user during demo seeding)
    if not User.query.filter_by(username=np_seed_admin).first():
        db.session.add(User(username=np_seed_admin, password=generate_password_hash("Admin12345"), is_admin=True))
    db.session.commit()

    import pandas as pd
    from utils.risk_engine import calculate_multi_layer_risk
    from routes.claims import recalculate_hospital_stats, recalculate_provider_stats

    hospitals = [
        ("City General Hospital", "Dr. Smith"),
        ("Apollo Medical Center", "Dr. Patel"),
        ("St. Mary's Hospital", "Dr. Johnson"),
        ("Sunrise Health Clinic", "Dr. Lee"),
        ("Riverside Memorial", "Dr. Garcia"),
    ]
    diagnoses = ["I10", "E11", "J45", "M19", "F32", "N18"]

    # load 20 rows from the training CSV to get realistic chronic-condition profiles
    try:
        sample = pd.read_csv(Config.TRAINING_DATA_PATH, nrows=20)
    except Exception:
        sample = None

    unique_hospitals = set()
    unique_providers = set()

    for i in range(15):
        # ... (rest of the loop content is similar but I'll make it cleaner)
        if sample is not None and i < len(sample):
            row = sample.iloc[i]
            form_data = {
                "RenalDiseaseIndicator": float(row["RenalDiseaseIndicator"]),
                "ChronicCond_Alzheimer": float(row["ChronicCond_Alzheimer"]),
                "ChronicCond_Heartfailure": float(row["ChronicCond_Heartfailure"]),
                "ChronicCond_KidneyDisease": float(row["ChronicCond_KidneyDisease"]),
                "ChronicCond_Cancer": float(row["ChronicCond_Cancer"]),
                "ChronicCond_ObstrPulmonary": float(row["ChronicCond_ObstrPulmonary"]),
                "ChronicCond_Depression": float(row["ChronicCond_Depression"]),
                "ChronicCond_Diabetes": float(row["ChronicCond_Diabetes"]),
                "ChronicCond_IschemicHeart": float(row["ChronicCond_IschemicHeart"]),
                "ChronicCond_Osteoporasis": float(row["ChronicCond_Osteoporasis"]),
                "ChronicCond_rheumatoidarthritis": float(row["ChronicCond_rheumatoidarthritis"]),
                "ChronicCond_stroke": float(row["ChronicCond_stroke"]),
                "IPAnnualDeductibleAmt": float(row["IPAnnualDeductibleAmt"]),
                "OPAnnualDeductibleAmt": float(row["OPAnnualDeductibleAmt"]),
            }
        else:
            form_data = {
                "RenalDiseaseIndicator": float(random.randint(0, 1)),
                "ChronicCond_Alzheimer": float(random.randint(0, 1)),
                "ChronicCond_Heartfailure": float(random.randint(0, 1)),
                "ChronicCond_KidneyDisease": float(random.randint(0, 1)),
                "ChronicCond_Cancer": float(random.randint(0, 1)),
                "ChronicCond_ObstrPulmonary": float(random.randint(0, 1)),
                "ChronicCond_Depression": float(random.randint(0, 1)),
                "ChronicCond_Diabetes": float(random.randint(0, 1)),
                "ChronicCond_IschemicHeart": float(random.randint(0, 1)),
                "ChronicCond_Osteoporasis": float(random.randint(0, 1)),
                "ChronicCond_rheumatoidarthritis": float(random.randint(0, 1)),
                "ChronicCond_stroke": float(random.randint(0, 1)),
                "IPAnnualDeductibleAmt": float(random.randint(0, 12000)),
                "OPAnnualDeductibleAmt": float(random.randint(0, 4000)),
            }

        df = pd.DataFrame([form_data])[
            ["RenalDiseaseIndicator", "ChronicCond_Alzheimer",
             "ChronicCond_Heartfailure", "ChronicCond_KidneyDisease",
             "ChronicCond_Cancer", "ChronicCond_ObstrPulmonary",
             "ChronicCond_Depression", "ChronicCond_Diabetes",
             "ChronicCond_IschemicHeart", "ChronicCond_Osteoporasis",
             "ChronicCond_rheumatoidarthritis", "ChronicCond_stroke",
             "IPAnnualDeductibleAmt", "OPAnnualDeductibleAmt"]
        ]
        ml_prob = predict_proba_safe(model, df)
        ml_prob = float(ml_prob) if ml_prob is not None else 0.0
        risk = calculate_multi_layer_risk(form_data, ml_prob)

        h, p = random.choice(hospitals)
        unique_hospitals.add(h)
        unique_providers.add(p)
        amount = round(random.uniform(500, 25000), 2)
        # bias amount with risk: higher risk => larger amount
        amount = round(amount * (0.7 + (risk.score / 100.0)), 2)
        status = random.choice([
            "Submitted", "Under Review", "Investigating",
            "Verified", "Cleared", "Fraud Confirmed",
        ])
        ts = datetime.now(timezone.utc) - timedelta(days=random.randint(0, 90))

        rec = ClaimRecord(
            username=np_seed_user, hospital=h, provider=p,
            diagnosis_code=random.choice(diagnoses),
            claim_amount=amount,
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
            risk_score=risk.score, risk_level=risk.level,
            ml_probability=risk.ml_probability,
            prediction=risk.prediction,
            confidence_score=risk.confidence,
            risk_breakdown=json.dumps(risk.layer_breakdown),
            reasons=risk.reason_text(),
            recommendations=risk.recommendation_text(),
            status=status, status_updated_at=ts,
            timestamp=ts,
        )
        db.session.add(rec)
        db.session.flush()

        db.session.add(Investigation(
            claim_id=rec.id, investigator="auto-triage",
            action="Created", note="Seeded for demo", created_at=ts,
        ))

    db.session.commit() # Commit claims first so recalculation sees them

    # Recalculate stats for all seeded entities
    for h_name in unique_hospitals:
        recalculate_hospital_stats(h_name)
    for p_name in unique_providers:
        recalculate_provider_stats(p_name)
    
    db.session.add(AuditEvent(
        actor="system", action="seed", target="claims",
        detail="Demo data populated",
    ))
    db.session.commit()
    app.logger.info("Seeded 15 demo claims and updated aggregate stats.")
    app.logger.info("Seeded 15 demo claims for demo_user.")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
app = create_app()

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # Production safety: use environment variable for debug mode, default to False
    debug_mode = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true", "yes")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=debug_mode)
