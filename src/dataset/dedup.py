"""Exact and near-duplicate detection, and grouping of related images.

Policy (thresholds live in ``config/pipeline.yaml``):

* exact duplicates (identical decoded greyscale pixels) keep one representative;
  if their labels disagree, every copy is rejected because the label is untrustworthy;
* near-duplicates (resized / re-encoded copies) and related images (likely
  neighbouring slices of one scan) are *kept* but linked, so each linked group
  lands in a single split and cannot leak between training and evaluation;
* looser "review" matches are only reported for manual inspection.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from utils.hashing import pairs_within_distance

NEAR_DUPLICATE = "near_duplicate"
RELATED = "related"
REVIEW = "review"
LINKING_KINDS = frozenset({NEAR_DUPLICATE, RELATED})


@dataclass(frozen=True)
class PairRule:
    max_hamming: int
    min_correlation: float

    def matches(self, hamming: int, correlation: float) -> bool:
        return hamming <= self.max_hamming and correlation >= self.min_correlation


@dataclass(frozen=True)
class DuplicateRules:
    """Ordered pair rules: the first (strictest) matching rule names the pair."""

    near_duplicate: PairRule
    related: PairRule
    review: PairRule

    @classmethod
    def from_config(cls, config: dict) -> DuplicateRules:
        section = config["duplicates"]
        return cls(**{kind: PairRule(**section[kind]) for kind in (NEAR_DUPLICATE, RELATED, REVIEW)})

    @property
    def search_radius(self) -> int:
        return max(self.near_duplicate.max_hamming, self.related.max_hamming, self.review.max_hamming)

    def classify(self, hamming: int, correlation: float) -> str | None:
        for kind in (NEAR_DUPLICATE, RELATED, REVIEW):
            if getattr(self, kind).matches(hamming, correlation):
                return kind
        return None


@dataclass(frozen=True)
class SimilarPair:
    first: int
    second: int
    hamming: int
    correlation: float
    kind: str


def thumbnail_correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Pearson correlation of two thumbnails (1.0 = same structure up to brightness/contrast)."""
    a = first.astype(np.float64).ravel()
    b = second.astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    denominator = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / denominator) if denominator else float(np.array_equal(first, second))


def exact_duplicate_groups(pixel_hashes: Sequence[str]) -> list[list[int]]:
    """Indices sharing identical decoded pixels, as groups of two or more."""
    groups: dict[str, list[int]] = defaultdict(list)
    for index, value in enumerate(pixel_hashes):
        groups[value].append(index)
    return [members for members in groups.values() if len(members) > 1]


def find_similar_pairs(phashes: Sequence[int], thumbnails: Sequence[np.ndarray], rules: DuplicateRules) -> list[SimilarPair]:
    """Candidate pairs by perceptual hash, confirmed and named by thumbnail correlation."""
    hashes = np.asarray(phashes, dtype=np.uint64)
    pairs = []
    for i, j, distance in pairs_within_distance(hashes, rules.search_radius):
        correlation = thumbnail_correlation(thumbnails[i], thumbnails[j])
        kind = rules.classify(distance, correlation)
        if kind:
            pairs.append(SimilarPair(i, j, distance, round(correlation, 4), kind))
    return pairs


def link_groups(count: int, pairs: Sequence[SimilarPair]) -> list[int]:
    """Union-find over linking pairs; returns a group id (smallest member index) per item."""
    parent = list(range(count))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for pair in pairs:
        if pair.kind in LINKING_KINDS:
            root_a, root_b = find(pair.first), find(pair.second)
            parent[max(root_a, root_b)] = min(root_a, root_b)
    return [find(item) for item in range(count)]
