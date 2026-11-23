"""Duplicate detection, group-aware splitting and leakage checks on the prepared data."""

from __future__ import annotations

import io
from collections import defaultdict

import numpy as np
import pytest
from PIL import Image

from src.dataset.dedup import DuplicateRules, PairRule, SimilarPair, exact_duplicate_groups, link_groups, thumbnail_correlation
from src.dataset.fingerprint import fingerprint
from src.dataset.splitting import RETRAIN_DEMO, SplitGroup, split_groups
from tests.helpers import make_image, require_artifacts
from utils.files import read_csv
from utils.hashing import pairs_within_distance, perceptual_hash

RULES = DuplicateRules(PairRule(12, 0.98), PairRule(8, 0.93), PairRule(10, 0.90))


def test_pairs_within_distance_matches_brute_force() -> None:
    rng = np.random.default_rng(0)
    hashes = rng.integers(0, 2**63, 300, dtype=np.uint64)
    hashes[5] = hashes[7] ^ np.uint64(0b111)  # distance 3
    expected = {
        (i, j) for i in range(300) for j in range(i + 1, 300) if bin(int(hashes[i]) ^ int(hashes[j])).count("1") <= 20
    }
    found = {(i, j) for i, j, _ in pairs_within_distance(hashes, 20, chunk=64)}
    assert found == expected and (5, 7) in found


def test_resized_copy_is_classified_near_duplicate() -> None:
    original = Image.open(io.BytesIO(make_image(10, size=(256, 256))))
    resized = original.resize((180, 180))
    a, b = fingerprint(original, 64), fingerprint(resized, 64)
    distance = bin(a.phash ^ b.phash).count("1")
    correlation = thumbnail_correlation(a.thumbnail, b.thumbnail)
    assert a.pixel_hash != b.pixel_hash  # not an exact duplicate ...
    assert RULES.classify(distance, correlation) == "near_duplicate"  # ... but caught as a near-duplicate


def test_unrelated_images_are_not_linked() -> None:
    a = fingerprint(Image.open(io.BytesIO(make_image(11, label_index=0))), 64)
    b = fingerprint(Image.open(io.BytesIO(make_image(12, label_index=3))), 64)
    assert RULES.classify(bin(a.phash ^ b.phash).count("1"), thumbnail_correlation(a.thumbnail, b.thumbnail)) is None


def test_exact_duplicates_use_decoded_pixels_not_file_bytes() -> None:
    png = Image.open(io.BytesIO(make_image(13, fmt="PNG")))
    rgb_png = Image.open(io.BytesIO(make_image(13, fmt="PNG", mode="RGB")))
    hashes = [fingerprint(png, 64).pixel_hash, fingerprint(rgb_png, 64).pixel_hash, "other"]
    assert exact_duplicate_groups(hashes) == [[0, 1]]


def test_link_groups_is_transitive_and_ignores_review_pairs() -> None:
    pairs = [SimilarPair(0, 1, 4, 0.95, "related"), SimilarPair(1, 2, 2, 0.99, "near_duplicate"), SimilarPair(3, 4, 9, 0.91, "review")]
    groups = link_groups(5, pairs)
    assert groups[0] == groups[1] == groups[2]
    assert len({groups[3], groups[4], groups[0]}) == 3


def test_perceptual_hash_is_stable() -> None:
    image = Image.open(io.BytesIO(make_image(14)))
    assert perceptual_hash(image) == perceptual_hash(image.copy())


def _synthetic_groups() -> list[SplitGroup]:
    groups, gid = [], 0
    for label in ("glioma", "meningioma", "pituitary", "no_tumor"):
        for plane in ("axial", "coronal"):
            for size in [1] * 60 + [2] * 10 + [5] * 3:
                groups.append(SplitGroup(gid, size, label, f"{label}|{plane}"))
                gid += 1
    return groups


def test_split_keeps_groups_whole_and_reserves_singleton_demo(config: dict) -> None:
    config["dataset"]["retrain_demo_per_class"] = 5
    groups = _synthetic_groups()
    assignment = split_groups(groups, config)
    assert set(assignment) == {g.group_id for g in groups}  # every group gets exactly one split
    demo = [g for g in groups if assignment[g.group_id] == RETRAIN_DEMO]
    assert all(g.size == 1 for g in demo)
    assert sorted(g.label for g in demo) == sorted(["glioma", "meningioma", "pituitary", "no_tumor"] * 5)
    sizes = defaultdict(int)
    for group in groups:
        sizes[assignment[group.group_id]] += group.size
    total = sum(sizes.values()) - sizes[RETRAIN_DEMO]
    assert abs(sizes["validation"] / total - 0.15) < 0.03 and abs(sizes["test"] / total - 0.15) < 0.03


def test_split_is_reproducible(config: dict) -> None:
    groups = _synthetic_groups()
    assert split_groups(groups, config) == split_groups(groups, config)


@pytest.mark.needs_artifacts
def test_prepared_splits_have_no_leakage(real_settings) -> None:
    require_artifacts(real_settings)
    samples = read_csv(real_settings.manifests_dir / "samples.csv")
    retained = [s for s in samples if s["status"] == "retained"]
    splits_per_hash, splits_per_group = defaultdict(set), defaultdict(set)
    for sample in retained:
        splits_per_hash[sample["pixel_hash"]].add(sample["split"])
        splits_per_group[sample["group_id"]].add(sample["split"])
    assert all(len(s) == 1 for s in splits_per_hash.values()), "identical pixels appear in two splits"
    assert all(len(s) == 1 for s in splits_per_group.values()), "a linked group straddles splits"
    # every rejected exact duplicate points at a retained representative with the same pixels
    kept = {s["sample_id"]: s["pixel_hash"] for s in retained}
    for sample in samples:
        if sample["reason"] == "exact_duplicate":
            assert kept[sample["duplicate_of"]] == sample["pixel_hash"]
    ids = {split: {r["sample_id"] for r in read_csv(real_settings.manifests_dir / f"{split}.csv")} for split in ("train", "validation", "test", "retrain_demo")}
    assert not ids["train"] & (ids["validation"] | ids["test"] | ids["retrain_demo"])
    assert not ids["validation"] & ids["test"]
