"""
SQLAlchemy models for MedGuard AI.

Kept in one file for the hackathon but logically grouped:
* User                     - authentication
* ClaimRecord              - submitted claim + risk verdict
* Hospital                 - provider we track for risk intelligence
* Investigation            - workflow on a suspect claim
* AuditEvent               - immutable log for compliance / judging
"""
from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    password = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, default=False, nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "is_admin": self.is_admin,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------
class ClaimRecord(db.Model):
    __tablename__ = "claims"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False, index=True)

    # --- Hospital / provider (Indexed for analytics) ---
    hospital = db.Column(db.String(120), default="Unknown")
    provider = db.Column(db.String(120), default="Unknown")
    diagnosis_code = db.Column(db.String(20), default="")
    claim_amount = db.Column(db.Float, default=0.0)

    # --- Chronic conditions (1/0) ---
    renal_disease = db.Column(db.Float, default=0.0)
    alzheimer = db.Column(db.Float, default=0.0)
    heart_failure = db.Column(db.Float, default=0.0)
    kidney_disease = db.Column(db.Float, default=0.0)
    cancer = db.Column(db.Float, default=0.0)
    obstr_pulmonary = db.Column(db.Float, default=0.0)
    depression = db.Column(db.Float, default=0.0)
    diabetes = db.Column(db.Float, default=0.0)
    ischemic_heart = db.Column(db.Float, default=0.0)
    osteoporosis = db.Column(db.Float, default=0.0)
    arthritis = db.Column(db.Float, default=0.0)
    stroke = db.Column(db.Float, default=0.0)

    ip_deductible = db.Column(db.Float, default=0.0)
    op_deductible = db.Column(db.Float, default=0.0)

    # --- Verdict ---
    risk_score = db.Column(db.Float, default=0.0)
    risk_level = db.Column(db.String(20), default="Low")
    ml_probability = db.Column(db.Float, default=0.0)
    prediction = db.Column(db.String(20), default="No Fraud", index=True)
    reasons = db.Column(db.Text, default="")
    recommendations = db.Column(db.Text, default="")
    risk_breakdown = db.Column(db.Text, default="{}") # JSON storage for XAI
    confidence_score = db.Column(db.Float, default=0.0)

    # --- Workflow ---
    status = db.Column(
        db.String(30), default="Submitted", nullable=False, index=True
    )
    status_updated_at = db.Column(
        db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    investigator = db.Column(db.String(80), default="")
    investigation_notes = db.Column(db.Text, default="")

    timestamp = db.Column(
        db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True
    )

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


# ---------------------------------------------------------------------------
# Hospitals
# ---------------------------------------------------------------------------
class Hospital(db.Model):
    __tablename__ = "hospitals"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), unique=True, nullable=False, index=True)
    region = db.Column(db.String(80), default="Unknown")
    total_claims = db.Column(db.Integer, default=0)
    fraud_claims = db.Column(db.Integer, default=0)
    avg_risk_score = db.Column(db.Float, default=0.0)
    last_flagged_at = db.Column(db.DateTime)

    @property
    def fraud_rate(self):
        if not self.total_claims:
            return 0.0
        return round(100.0 * self.fraud_claims / self.total_claims, 1)


# ---------------------------------------------------------------------------
# Investigations (workflow on a claim)
# ---------------------------------------------------------------------------
class Investigation(db.Model):
    __tablename__ = "investigations"

    id = db.Column(db.Integer, primary_key=True)
    claim_id = db.Column(db.Integer, db.ForeignKey("claims.id"), nullable=False, index=True)
    investigator = db.Column(db.String(80), default="")
    action = db.Column(db.String(40), default="Created")
    note = db.Column(db.Text, default="")
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False) # Already correct

    claim = db.relationship("ClaimRecord", backref="investigations")


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------
class AuditEvent(db.Model):
    __tablename__ = "audit_events"

    id = db.Column(db.Integer, primary_key=True)
    actor = db.Column(db.String(80), default="system")
    action = db.Column(db.String(80), nullable=False)
    target = db.Column(db.String(120), default="")
    detail = db.Column(db.Text, default="")
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False) # Already correct


# ---------------------------------------------------------------------------
# Providers (New: for provider intelligence)
# ---------------------------------------------------------------------------
class Provider(db.Model):
    __tablename__ = "providers"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), unique=True, nullable=False, index=True)
    total_claims = db.Column(db.Integer, default=0)
    fraud_claims = db.Column(db.Integer, default=0)
    avg_risk_score = db.Column(db.Float, default=0.0)
    last_flagged_at = db.Column(db.DateTime)

    @property
    def fraud_rate(self):
        if not self.total_claims:
            return 0.0
        return round(100.0 * self.fraud_claims / self.total_claims, 1)
