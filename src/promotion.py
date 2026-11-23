"""Promotion criteria for retrained candidate models.

Both models are scored on the *same fixed validation split*. The test split is
never used here, so repeated retraining cannot tune against final-test results.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.labels import CLASS_NAMES


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    candidate_score: float
    active_score: float | None
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"promote": self.promote, "candidate_score": self.candidate_score, "active_score": self.active_score, "reasons": self.reasons}


def decide_promotion(candidate: dict, active: dict | None, config: dict) -> PromotionDecision:
    """Promote when the candidate is non-inferior overall and for every class.

    * validation macro-F1 >= active macro-F1 - ``non_inferiority_margin``;
    * no class's recall drops by more than ``max_class_recall_drop``.

    With no active model the candidate is promoted if it beats chance (macro-F1 > 0.25).
    """
    section = config["promotion"]
    metric = section["metric"]
    score = float(candidate[metric])
    if active is None:
        ok = score > 1.0 / len(CLASS_NAMES)
        return PromotionDecision(ok, score, None, [f"no active model; candidate {metric}={score:.4f} {'>' if ok else '<='} chance"])

    reference = float(active[metric])
    margin = float(section["non_inferiority_margin"])
    reasons = []
    overall_ok = score >= reference - margin
    reasons.append(f"{metric}: candidate {score:.4f} vs active {reference:.4f} (allowed drop {margin}) -> {'ok' if overall_ok else 'too low'}")
    max_drop = float(section["max_class_recall_drop"])
    recall_ok = True
    for name in CLASS_NAMES:
        drop = active["per_class"][name]["recall"] - candidate["per_class"][name]["recall"]
        if drop > max_drop:
            recall_ok = False
            reasons.append(f"recall for {name} fell by {drop:.4f} (> {max_drop})")
    if recall_ok:
        reasons.append(f"no class recall fell by more than {max_drop}")
    return PromotionDecision(overall_ok and recall_ok, score, reference, reasons)
