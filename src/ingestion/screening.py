"""Duplicate and leakage screening of uploads against the dataset and earlier uploads.

Rules (same fingerprint and pair rules as dataset preparation):

* identical to, or a near-duplicate / related image of, a validation or test
  image -> **rejected** (admitting it would leak evaluation data into training);
* identical to a training image or to another kept upload -> **duplicate** (skipped);
* identical to a reserved retrain-demo image -> allowed (that batch exists to be uploaded);
* weaker similarities -> accepted, with a note for the reviewer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.dataset.dedup import LINKING_KINDS, DuplicateRules, thumbnail_correlation
from src.dataset.fingerprint import Fingerprint
from src.dataset.splitting import EVALUATION_SPLITS, RETRAIN_DEMO, TRAIN
from utils.hashing import hamming_distances

EXACT = "exact"


@dataclass(frozen=True)
class Match:
    sample_id: str
    split: str
    kind: str
    hamming: int
    correlation: float

    def describe(self) -> str:
        return f"{self.kind} match with {self.split} image {self.sample_id} (hash distance {self.hamming}, r={self.correlation:.3f})"


@dataclass(frozen=True)
class Verdict:
    status: str          # "ok", "rejected" or "duplicate"
    reason: str | None
    notes: list[str]


class DatasetIndex:
    """Fingerprints of every retained dataset image (written by dataset preparation)."""

    def __init__(self, arrays: dict[str, np.ndarray], rules: DuplicateRules) -> None:
        self.sample_ids = arrays["sample_ids"]
        self.splits = arrays["splits"]
        self.pixel_hashes = arrays["pixel_hashes"]
        self.phashes = arrays["phashes"].astype(np.uint64)
        self.thumbnails = arrays["thumbnails"]
        self.rules = rules

    def matches(self, fp: Fingerprint) -> list[Match]:
        """Exact matches plus hash-near candidates confirmed by thumbnail correlation."""
        found = [
            Match(str(self.sample_ids[i]), str(self.splits[i]), EXACT, 0, 1.0)
            for i in np.flatnonzero(self.pixel_hashes == fp.pixel_hash)
        ]
        exact_ids = {m.sample_id for m in found}
        distances = hamming_distances(fp.phash, self.phashes)
        for i in np.flatnonzero(distances <= self.rules.search_radius):
            if str(self.sample_ids[i]) in exact_ids:
                continue
            correlation = thumbnail_correlation(fp.thumbnail, self.thumbnails[i])
            kind = self.rules.classify(int(distances[i]), correlation)
            if kind:
                found.append(Match(str(self.sample_ids[i]), str(self.splits[i]), kind, int(distances[i]), round(correlation, 4)))
        return found


def judge(dataset_matches: list[Match], duplicate_upload_id: str | None) -> Verdict:
    """Turn matches into an admission verdict (see module docstring for the rules)."""
    notes = []
    for match in dataset_matches:
        if match.split in EVALUATION_SPLITS and (match.kind == EXACT or match.kind in LINKING_KINDS):
            return Verdict("rejected", f"Evaluation leakage risk: {match.describe()}.", notes)
    for match in dataset_matches:
        if match.split == TRAIN and match.kind == EXACT:
            return Verdict("duplicate", f"Already in the training set as {match.sample_id}.", notes)
    if duplicate_upload_id:
        return Verdict("duplicate", f"Identical to upload {duplicate_upload_id}.", notes)
    for match in dataset_matches:
        if match.split == RETRAIN_DEMO and match.kind == EXACT:
            notes.append(f"Matches reserved retrain-demo image {match.sample_id} (held out from initial training).")
        else:
            notes.append(f"Similar to dataset image: {match.describe()}.")
    return Verdict("ok", None, notes)
