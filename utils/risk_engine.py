"""
Multi-layer risk scoring engine.

Three signals combined:
  1. Rule-based     - domain heuristics on deductible amounts and chronic counts
  2. ML probability - the pickled classifier in fraud_detection_model4.pkl
  3. Pattern        - deviation vs. typical claim profile (chronic count, etc.)

The function returns a *structured* result so the UI can render a breakdown
without re-deriving the layers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json # Already imported
from typing import Dict, List

from config import Config

# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------
@dataclass
class RiskResult:
    score: float
    level: str
    prediction: str
    reasons: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    ml_probability: float = 0.0
    confidence: float = 0.0
    layer_breakdown: Dict[str, float] = field(default_factory=dict)

    def reason_text(self) -> str:
        return ", ".join(self.reasons) if self.reasons else "Standard clinical profile"

    def recommendation_text(self) -> str:
        return (
            " | ".join(self.recommendations)
            if self.recommendations
            else "Standard verification protocol"
        )

    def get_confidence_wording(self) -> str:
        """Returns human-readable confidence levels."""
        if self.confidence >= 0.85:
            return "High Certainty"
        if self.confidence >= 0.70:
            return "Moderate Certainty"
        return "Low Certainty (Manual Review Recommended)"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
CHRONIC_KEYS = (
    "RenalDiseaseIndicator",
    "ChronicCond_Alzheimer",
    "ChronicCond_Heartfailure",
    "ChronicCond_KidneyDisease",
    "ChronicCond_Cancer",
    "ChronicCond_ObstrPulmonary",
    "ChronicCond_Depression",
    "ChronicCond_Diabetes",
    "ChronicCond_IschemicHeart",
    "ChronicCond_Osteoporasis",
    "ChronicCond_rheumatoidarthritis",
    "ChronicCond_stroke",
)


def _chronic_count(data: Dict[str, float]) -> int:
    count = 0
    for k in CHRONIC_KEYS:
        try:
            count += int(round(float(data.get(k, 0) or 0)))
        except (ValueError, TypeError):
            continue
    return count


def _score_to_level(score: float) -> str:
    if score < 30:
        return "Low"
    if score < 60:
        return "Medium"
    if score < 85:
        return "High"
    return "Critical"


def _recommend_for(risk_level: str, ml_prob: float, ip_amt: float) -> List[str]:
    """Actionable next-step recommendations for the investigator."""
    recs: List[str] = []
    if risk_level == "Low":
        recs.append("Approve claim - standard processing")
    elif risk_level == "Medium":
        recs.append("Schedule routine review within 7 days")
        recs.append("Verify supporting documents")
    elif risk_level == "High":
        recs.append("Flag for manual review")
        recs.append("Request audit of provider records")
        recs.append("Cross-check with historical claims")
    else:  # Critical
        recs.append("Immediate investigation required")
        recs.append("Escalate to senior fraud analyst")
        recs.append("Verify all submitted documents")
        recs.append("Place payment on hold pending review")

    if ml_prob > 0.8:
        recs.append("ML model indicates high fraud probability - prioritize audit")

    if ip_amt > 10000:
        recs.append("High inpatient deductible - verify necessity")

    return recs


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def calculate_multi_layer_risk(
    input_data: Dict[str, float], ml_prob: float
) -> RiskResult:
    """
    Combine rule-based, ML, and pattern signals into a final 0-100 risk score.
    """
    ml_prob_f = float(ml_prob)
    ml_score = ml_prob_f * 70.0
    score = ml_score
    reasons: List[str] = []
    breakdown: Dict[str, float] = {"ml_layer": round(ml_score, 2)}

    # --- Layer 1: rule-based ---
    ip_amt = float(input_data.get("IPAnnualDeductibleAmt", 0) or 0)
    op_amt = float(input_data.get("OPAnnualDeductibleAmt", 0) or 0)
    chronic = _chronic_count(input_data)

    rule_add = 0.0
    if ip_amt > Config.IP_DEDUCTIBLE_HIGH:
        rule_add += 15
        reasons.append("High Inpatient Deductible Flag")
    if chronic > Config.CHRONIC_COUNT_HIGH:
        rule_add += 10
        reasons.append(f"Multiple Chronic Conditions ({chronic})")
    if op_amt < 0:
        rule_add += 5
        reasons.append("Irregular Negative Deductible Value")
    if ip_amt == 0 and op_amt == 0:
        rule_add += 3
        reasons.append("Zero deductible on both sides")
    score += rule_add
    breakdown["rule_layer"] = round(rule_add, 2)

    # --- Layer 2: pattern (small adjustment, fully explainable) ---
    pattern_add = 0.0
    if chronic >= 8:
        pattern_add += 5
        reasons.append("Unusually high chronic-condition density")
    if ip_amt > 0 and op_amt == 0 and chronic == 0:
        pattern_add += 3
        reasons.append("Unbalanced deductible pattern")
    score += pattern_add
    breakdown["pattern_layer"] = round(pattern_add, 2)

    final_score = min(round(float(score), 2), 100.0)
    level = _score_to_level(final_score)
    prediction = "Fraud" if final_score >= Config.FRAUD_DECISION_THRESHOLD else "No Fraud"
    
    # Confidence Calculation (XAI Component)
    confidence = ml_prob_f if ml_prob_f > 0.5 else (1.0 - ml_prob_f)
    
    # Map feature names to scores for the contribution chart
    feature_importance = {
        "Deductibles": round(rule_add, 2),
        "Chronic Conditions": round(pattern_add, 2),
        "ML Signal": round(ml_score, 2)
    }
    breakdown["feature_importance"] = feature_importance
    
    res = RiskResult(
        score=final_score,
        level=level,
        prediction=prediction,
        reasons=reasons,
        recommendations=_recommend_for(level, ml_prob_f, ip_amt),
        ml_probability=round(ml_prob_f, 4),
        confidence=round(float(confidence), 4),
        layer_breakdown=breakdown
    )
    # Inject wording into breakdown for UI consumption
    breakdown["confidence_desc"] = res.get_confidence_wording()

    return res
