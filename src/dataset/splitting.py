"""Group-aware, stratified splitting.

Whole groups (an image plus its near-duplicates / related slices) are assigned
to one split, so related images can never straddle training and evaluation.
Strata are ``label|plane`` so class and anatomical-plane mixes stay comparable.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

TRAIN, VALIDATION, TEST, RETRAIN_DEMO = "train", "validation", "test", "retrain_demo"
EVALUATION_SPLITS = (VALIDATION, TEST)
ALL_SPLITS = (TRAIN, VALIDATION, TEST, RETRAIN_DEMO)


@dataclass(frozen=True)
class SplitGroup:
    group_id: int
    size: int
    label: str
    stratum: str


def reserve_retrain_demo(groups: list[SplitGroup], per_class: int, rng: np.random.Generator) -> set[int]:
    """Pick ``per_class`` singleton groups per label for the retraining demo.

    Only singletons are eligible, so demo images have no near-duplicate or
    related image anywhere else in the dataset.
    """
    chosen: set[int] = set()
    by_label: dict[str, list[int]] = defaultdict(list)
    for group in groups:
        if group.size == 1:
            by_label[group.label].append(group.group_id)
    for label in sorted(by_label):
        candidates = sorted(by_label[label])
        if len(candidates) < per_class:
            raise ValueError(f"Only {len(candidates)} singleton images for {label!r}; cannot reserve {per_class}.")
        chosen.update(int(g) for g in rng.choice(candidates, size=per_class, replace=False))
    return chosen


def assign_groups(groups: list[SplitGroup], fractions: dict[str, float], rng: np.random.Generator) -> dict[int, str]:
    """Greedy stratified assignment of whole groups to splits.

    Largest groups are placed first (ties in random order); each goes to the
    split whose per-stratum quota is furthest from being met.
    """
    stratum_totals = Counter()
    for group in groups:
        stratum_totals[group.stratum] += group.size
    targets = {split: {s: n * share for s, n in stratum_totals.items()} for split, share in fractions.items()}
    filled = {split: Counter() for split in fractions}

    order = rng.permutation(len(groups))
    ordered = sorted((groups[i] for i in order), key=lambda g: -g.size)  # stable sort keeps random tie order
    assignment: dict[int, str] = {}
    for group in ordered:
        # Relative shortfall of each split for this group's stratum; the largest wins.
        deficits = {
            split: (targets[split][group.stratum] - filled[split][group.stratum]) / targets[split][group.stratum]
            if targets[split][group.stratum] else float("-inf")
            for split in fractions
        }
        split = max(deficits, key=deficits.get)
        assignment[group.group_id] = split
        filled[split][group.stratum] += group.size
    return assignment


def split_groups(groups: list[SplitGroup], config: dict) -> dict[int, str]:
    """Reserve the retraining demo first, then split the remainder into train/validation/test."""
    section = config["dataset"]
    rng = np.random.default_rng(int(section["seed"]))
    demo = reserve_retrain_demo(groups, int(section["retrain_demo_per_class"]), rng)
    validation, test = float(section["validation_fraction"]), float(section["test_fraction"])
    fractions = {TRAIN: 1.0 - validation - test, VALIDATION: validation, TEST: test}
    remaining = [group for group in groups if group.group_id not in demo]
    assignment = assign_groups(remaining, fractions, rng)
    assignment.update({group_id: RETRAIN_DEMO for group_id in demo})
    return assignment
