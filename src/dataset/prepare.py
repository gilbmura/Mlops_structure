"""End-to-end dataset preparation from the read-only BRISC ZIP.

Steps: verify manifests -> decode and fingerprint every classification image ->
resolve exact duplicates -> link near-duplicate / related images -> reserve the
retraining demo and split by group -> write manifests, images and a dataset card.
"""

from __future__ import annotations

import logging
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path


from config.settings import Settings
from src.dataset.dedup import LINKING_KINDS, DuplicateRules, SimilarPair, exact_duplicate_groups, find_similar_pairs, link_groups
from src.dataset.fingerprint import Fingerprint, fingerprint
from src.dataset.manifests import SPLIT_FIELDS, write_similarity_index
from src.dataset.source import ArchiveError, BriscArchive, SourceImage
from src.dataset.splitting import ALL_SPLITS, SplitGroup, split_groups
from src.labels import CLASS_NAMES
from utils.files import write_csv, write_json
from utils.hashing import hash_to_hex, sha256_bytes, sha256_file
from utils.images import ImageDecodeError, ImageLimits, decode_image

logger = logging.getLogger(__name__)

MIN_UNIQUE_IMAGES = 5000
_MAX_IMAGE_BYTES = 32 * 1024 * 1024
SAMPLE_FIELDS = [
    "sample_id", "source_path", "original_split", "label", "plane", "width", "height", "mode",
    "file_sha256", "pixel_hash", "phash", "status", "reason", "duplicate_of", "group_id", "group_size", "flags", "split",
]


@dataclass
class SampleRecord:
    """Mutable bookkeeping for one source image while it moves through the pipeline."""

    image: SourceImage
    status: str = "retained"
    reason: str = ""
    duplicate_of: str = ""
    fp: Fingerprint | None = None
    group_id: str = ""
    group_size: int = 0
    flags: set[str] = field(default_factory=set)
    split: str = ""

    def reject(self, reason: str, duplicate_of: str = "") -> None:
        self.status, self.reason, self.duplicate_of = "rejected", reason, duplicate_of

    def as_row(self) -> dict:
        image, fp = self.image, self.fp
        return {
            "sample_id": image.sample_id, "source_path": image.zip_path, "original_split": image.original_split,
            "label": image.label, "plane": image.plane,
            "width": fp.width if fp else image.width, "height": fp.height if fp else image.height,
            "mode": fp.mode if fp else "", "file_sha256": image.sha256,
            "pixel_hash": fp.pixel_hash if fp else "", "phash": hash_to_hex(fp.phash) if fp else "",
            "status": self.status, "reason": self.reason, "duplicate_of": self.duplicate_of,
            "group_id": self.group_id, "group_size": self.group_size,
            "flags": ";".join(sorted(self.flags)), "split": self.split,
        }


def _inspect(archive: BriscArchive, images: list[SourceImage], limits: ImageLimits, thumb_size: int) -> list[SampleRecord]:
    """Read, checksum, decode and fingerprint every image; failures are recorded, not raised."""
    records = []
    for position, image in enumerate(images, start=1):
        record = SampleRecord(image)
        records.append(record)
        try:
            data = archive.read(image.zip_path, _MAX_IMAGE_BYTES)
        except ArchiveError as exc:
            record.reject(f"unreadable_archive_entry: {exc}")
            continue
        if sha256_bytes(data) != image.sha256:
            record.reject("checksum_mismatch_with_manifest")
            continue
        try:
            decoded = decode_image(data, limits)
        except ImageDecodeError as exc:
            record.reject(f"undecodable_image: {exc}")
            continue
        if decoded.size != (image.width, image.height):
            record.flags.add("dimensions_differ_from_manifest")
        record.fp = fingerprint(decoded, thumb_size)
        if position % 1000 == 0:
            logger.info("Inspected %d / %d images", position, len(images))
    return records


def _resolve_exact_duplicates(records: list[SampleRecord]) -> list[dict]:
    """Keep one representative per identical-pixel group; reject all copies if labels conflict."""
    candidates = [r for r in records if r.status == "retained"]
    rows = []
    for number, members in enumerate(exact_duplicate_groups([r.fp.pixel_hash for r in candidates]), start=1):
        group = sorted((candidates[i] for i in members), key=lambda r: r.image.sample_id)
        labels = sorted({r.image.label for r in group})
        splits = sorted({r.image.original_split for r in group})
        if len(labels) > 1:
            for record in group:
                record.reject("conflicting_labels_among_exact_duplicates")
            resolution = "all_rejected_conflicting_labels"
        else:
            keeper = group[0]
            keeper.flags.add("had_exact_duplicates")
            for record in group[1:]:
                record.reject("exact_duplicate", duplicate_of=keeper.image.sample_id)
            resolution = f"kept {keeper.image.sample_id}"
        rows.append({
            "duplicate_group": f"x{number:04d}", "size": len(group),
            "members": ";".join(r.image.sample_id for r in group), "labels": ";".join(labels),
            "original_splits": ";".join(splits), "crosses_original_split": len(splits) > 1, "resolution": resolution,
        })
    return rows


def _link_related(retained: list[SampleRecord], rules: DuplicateRules) -> list[SimilarPair]:
    """Find similar pairs among retained images, flag them, and assign group ids."""
    pairs = find_similar_pairs([r.fp.phash for r in retained], [r.fp.thumbnail for r in retained], rules)
    for pair in pairs:
        for index in (pair.first, pair.second):
            retained[index].flags.add(f"{pair.kind}_match")
    roots = link_groups(len(retained), pairs)
    members: dict[int, list[SampleRecord]] = defaultdict(list)
    for record, root in zip(retained, roots):
        members[root].append(record)
    for root, group in members.items():
        conflict = len({r.image.label for r in group}) > 1
        for record in group:
            record.group_id, record.group_size = f"g{root:05d}", len(group)
            if conflict:
                record.flags.add("label_conflict_in_linked_group")
    return pairs


def _assign_splits(retained: list[SampleRecord], config: dict) -> None:
    by_group: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in retained:
        by_group[record.group_id].append(record)
    groups = []
    for group_id, members in by_group.items():
        label = Counter(r.image.label for r in members).most_common(1)[0][0]
        plane = Counter(r.image.plane for r in members).most_common(1)[0][0]
        groups.append(SplitGroup(int(group_id[1:]), len(members), label, f"{label}|{plane}"))
    assignment = split_groups(groups, config)
    for record in retained:
        record.split = assignment[int(record.group_id[1:])]


def _materialize(archive: BriscArchive, retained: list[SampleRecord], data_dir: Path) -> dict[str, list[dict]]:
    """Copy original image bytes into ``data/<split>/<label>/`` and return split manifest rows."""
    for split in ALL_SPLITS:  # generated folders are rebuilt from scratch
        for label in CLASS_NAMES:
            shutil.rmtree(data_dir / split / label, ignore_errors=True)
    rows: dict[str, list[dict]] = defaultdict(list)
    for record in retained:
        image = record.image
        relative = Path(record.split) / image.label / Path(image.zip_path).name
        target = data_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(image.zip_path, _MAX_IMAGE_BYTES))
        rows[record.split].append({
            "sample_id": image.sample_id, "label": image.label, "plane": image.plane,
            "path": relative.as_posix(), "group_id": record.group_id, "flags": ";".join(sorted(record.flags)),
        })
    return rows


def _pair_rows(retained: list[SampleRecord], pairs: list[SimilarPair]) -> list[dict]:
    return [{
        "first": retained[p.first].image.sample_id, "second": retained[p.second].image.sample_id,
        "hamming": p.hamming, "correlation": p.correlation, "kind": p.kind,
        "action": "same_split" if p.kind in LINKING_KINDS else "flag_for_review",
        "same_label": retained[p.first].image.label == retained[p.second].image.label,
        "first_original_split": retained[p.first].image.original_split,
        "second_original_split": retained[p.second].image.original_split,
    } for p in pairs]


def prepare_dataset(archive_path: Path, settings: Settings, config: dict) -> dict:
    """Run the full preparation pipeline and return the dataset card that was written."""
    rules = DuplicateRules.from_config(config)
    upload = config["uploads"]
    limits = ImageLimits(upload["image_min_side"], upload["image_max_side"], upload["image_max_side"] ** 2)
    manifests_dir = settings.manifests_dir

    with BriscArchive(archive_path, config["dataset"]["archive_root"]) as archive:
        sidecars = archive.verify_manifest_sidecars()
        if not all(sidecars.values()):
            raise ArchiveError(f"Manifest checksum sidecars do not match: {sidecars}")
        manifests_agree = archive.manifests_agree()
        images, issues = archive.classification_images()
        images.sort(key=lambda image: image.sample_id)
        logger.info("Manifest lists %d classification images (%d issues)", len(images), len(issues))

        records = _inspect(archive, images, limits, int(config["duplicates"]["thumbnail_size"]))
        duplicate_rows = _resolve_exact_duplicates(records)
        retained = [r for r in records if r.status == "retained"]
        pairs = _link_related(retained, rules)
        _assign_splits(retained, config)
        split_rows = _materialize(archive, retained, settings.data_dir)
        segmentation = archive.segmentation_overlap({image.sha256 for image in images})

    write_csv(manifests_dir / "samples.csv", [r.as_row() for r in records], SAMPLE_FIELDS)
    write_csv(manifests_dir / "manifest_issues.csv", [vars(i) for i in issues], ["zip_path", "reason"])
    write_csv(manifests_dir / "exact_duplicate_groups.csv", duplicate_rows,
              ["duplicate_group", "size", "members", "labels", "original_splits", "crosses_original_split", "resolution"])
    pair_rows = _pair_rows(retained, pairs)
    write_csv(manifests_dir / "similar_pairs.csv", pair_rows, list(pair_rows[0]) if pair_rows else ["first", "second"])
    for split in ALL_SPLITS:
        write_csv(manifests_dir / f"{split}.csv", sorted(split_rows[split], key=lambda r: r["sample_id"]), SPLIT_FIELDS)
    write_similarity_index(settings.cache_dir / "similarity_index.npz", [
        (r.image.sample_id, r.split, r.image.label, r.fp) for r in retained
    ])

    card = _dataset_card(archive_path, config, sidecars, manifests_agree, issues, records, duplicate_rows, pairs, retained, segmentation)
    write_json(manifests_dir / "dataset_card.json", card)
    return card


def _dataset_card(archive_path, config, sidecars, manifests_agree, issues, records, duplicate_rows, pairs, retained, segmentation) -> dict:
    """Machine-readable provenance and cleaning summary (also rendered in the notebook and UI)."""
    archive_path = Path(archive_path)
    group_sizes = Counter(r.group_id for r in retained)
    kinds = Counter(p.kind for p in pairs)
    splits = {split: dict(sorted(Counter(r.image.label for r in retained if r.split == split).items())) for split in ALL_SPLITS}
    return {
        "dataset_version": config["dataset"]["version"],
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": config["dataset"]["source"],
        "archive": {"filename": archive_path.name, "bytes": archive_path.stat().st_size, "sha256": sha256_file(archive_path)},
        "manifest_checks": {"sidecar_checksums_match": sidecars, "csv_and_json_agree": manifests_agree, "issues": len(issues)},
        "counts": {
            "classification_files": len(records),
            "by_label": dict(sorted(Counter(r.image.label for r in records).items())),
            "by_original_split": dict(sorted(Counter(r.image.original_split for r in records).items())),
            "decoded_and_checksum_ok": sum(r.fp is not None for r in records),
            "rejected_by_reason": dict(Counter(r.reason.split(":")[0] for r in records if r.status == "rejected")),
            "exact_duplicate_groups": len(duplicate_rows),
            "exact_groups_crossing_original_split": sum(row["crosses_original_split"] for row in duplicate_rows),
            "exact_groups_with_conflicting_labels": sum(row["resolution"].startswith("all_rejected") for row in duplicate_rows),
            "retained_unique_images": len(retained),
            "meets_minimum_unique_images": len(retained) >= MIN_UNIQUE_IMAGES,
            "similar_pairs_by_kind": dict(sorted(kinds.items())),
            "linked_groups": sum(size > 1 for size in group_sizes.values()),
            "images_in_linked_groups": sum(size for size in group_sizes.values() if size > 1),
            "largest_linked_group": max(group_sizes.values(), default=0),
            "linked_groups_with_label_conflicts": len({r.group_id for r in retained if "label_conflict_in_linked_group" in r.flags}),
        },
        "splits": splits,
        "split_totals": {split: sum(counts.values()) for split, counts in splits.items()},
        "segmentation_task": segmentation,
        "config": {key: config[key] for key in ("dataset", "duplicates", "preprocessing")},
        "limitations": [
            "No patient identifiers are provided, so patient-level separation between splits cannot be guaranteed; "
            "linking near-duplicates and similar neighbouring slices reduces but does not eliminate this risk.",
            "Clinical label accuracy has not been independently verified.",
            "No explicit licence file is included in the archive; see README 'Dataset licence and attribution'.",
        ],
    }
