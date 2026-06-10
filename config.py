"""
Centralized configuration for MedGuard AI.

Reads sensible defaults but allows env-var overrides so the same code
can run in development, demo and production.
"""
import os


class Config:
    # --- Flask ---
    SECRET_KEY = os.environ.get(
        "MEDGUARD_SECRET_KEY",
        # Demo / dev fallback. Override in production.
        "med_fraud_demo_key_change_me",
    )
    SESSION_COOKIE_HTTPONLY = True
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

    # --- Risk engine thresholds (the brief's "multi-layer" knobs) ---
    FRAUD_DECISION_THRESHOLD = 50.0   # score >= this -> "Fraud"
    IP_DEDUCTIBLE_HIGH = 5000.0       # inpatient > this -> +15
    CHRONIC_COUNT_HIGH = 5            # > this many conditions -> +10
    ANOMALY_THRESHOLD = 0.8           # Confidence above this is "High Confidence" # Already correct
    ANOMALY_THRESHOLD = 0.8           # Confidence above this is "High Confidence"
