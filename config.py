"""
Centralized configuration for MedGuard AI.

Reads sensible defaults but allows env-var overrides so the same code
can run in development, demo and production.
"""
import os


def _truthy(name: str, default: str = "0") -> bool:
    """Return True if the env var is a truthy string (1/true/yes/on)."""
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


class Config:
    # --- Flask ---
    SECRET_KEY = os.environ.get(
        "MEDGUARD_SECRET_KEY",
        # Demo / dev fallback. Override in production.
        "med_fraud_demo_key_change_me",
    )
    SESSION_COOKIE_HTTPONLY = True
    # SESSION_COOKIE_SECURE: Default to False to prevent breaking login on HTTP 
    # (Critical fix for dev environments). We only enable it if MEDGUARD_HTTPS 
    # is truthy and we are not in debug mode, or if explicitly requested.
    SESSION_COOKIE_SECURE = _truthy("MEDGUARD_HTTPS", "0") and not _truthy("FLASK_DEBUG", "0")
    
    SESSION_COOKIE_SAMESITE = "Lax"
    PERMANENT_SESSION_LIFETIME = 60 * 60 * 8  # 8 hours
    JSON_SORT_KEYS = False

    # --- Database ---
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "MEDGUARD_DB_URI", "sqlite:///fraud_prevention.db"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # --- Model ---
    MODEL_PATH = os.environ.get(
        "MEDGUARD_MODEL_PATH", "fraud_detection_model4.pkl"
    )
    TRAINING_DATA_PATH = os.environ.get(
        "MEDGUARD_DATA_PATH", "medi_fraud3.csv"
    )

    # --- Demo seeding ---
    # If true (default for fresh installs), create demo_user + admin accounts.
    SEED_DEMO_USERS = _truthy("MEDGUARD_SEED_USERS", "1")

    # --- Risk engine thresholds (the brief's "multi-layer" knobs) ---
    FRAUD_DECISION_THRESHOLD = 50.0   # score >= this -> "Fraud"
    IP_DEDUCTIBLE_HIGH = 5000.0       # inpatient > this -> +15
    CHRONIC_COUNT_HIGH = 5            # > this many conditions -> +10
    ANOMALY_THRESHOLD = 0.8           # Confidence above this is "High Confidence"

    # Recommendations thresholds (kept here so tuning is one place)
    ML_PROB_PRIORITY = 0.8            # > this -> "ML model high-confidence fraud"
    IP_DEDUCTIBLE_REVIEW = 10000.0    # > this -> "verify necessity"
