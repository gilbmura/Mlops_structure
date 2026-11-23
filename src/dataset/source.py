"""Read the supplied BRISC 2025 ZIP and its manifests without extracting or modifying it."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from src.labels import CLASS_NAMES, FILENAME_CODES
from utils.hashing import sha256_bytes

_MAX_MANIFEST_BYTES = 64 * 1024 * 1024


class ArchiveError(RuntimeError):
    """Raised when the archive is missing or structurally unusable."""


@dataclass(frozen=True)
class SourceImage:
    """One classification image as described by the manifest."""

    sample_id: str
    zip_path: str
    label: str
    plane: str
    original_split: str
    sha256: str
    file_size: int
    width: int
    height: int


@dataclass(frozen=True)
class ManifestIssue:
    """A manifest row or archive entry that failed a consistency check."""

    zip_path: str
    reason: str


class BriscArchive:
    """Read-only accessor for the BRISC archive (the ZIP is opened in read mode only)."""

    def __init__(self, path: Path, root: str = "brisc2025") -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise ArchiveError(f"Dataset archive not found at {self.path}. Pass --archive or set BRISC_ARCHIVE.")
        try:
            self._zip = zipfile.ZipFile(self.path, mode="r")
        except zipfile.BadZipFile as exc:
            raise ArchiveError(f"{self.path} is not a valid ZIP archive.") from exc
        self.root = root
        self._infos = {info.filename: info for info in self._zip.infolist()}

    def close(self) -> None:
        self._zip.close()

    def __enter__(self) -> BriscArchive:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def read(self, zip_path: str, max_bytes: int) -> bytes:
        """Read one member, refusing anything larger than ``max_bytes`` once decompressed."""
        info = self._infos.get(zip_path)
        if info is None:
            raise KeyError(zip_path)
        if info.file_size > max_bytes:
            raise ArchiveError(f"{zip_path} declares {info.file_size} bytes, above the {max_bytes} limit.")
        with self._zip.open(info) as handle:
            data = handle.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ArchiveError(f"{zip_path} decompressed beyond its declared size.")
        return data

    def _root_file(self, name: str) -> bytes:
        return self.read(f"{self.root}/{name}", _MAX_MANIFEST_BYTES)

    def verify_manifest_sidecars(self) -> dict[str, bool]:
        """Check manifest.csv / manifest.json against their .sha256 sidecar files."""
        results = {}
        for name in ("manifest.csv", "manifest.json"):
            expected = self._root_file(f"{name}.sha256").decode("ascii").split()[0].lower()
            results[name] = sha256_bytes(self._root_file(name)) == expected
        return results

    def manifest_rows(self) -> list[dict[str, str]]:
        """Rows of manifest.csv (the canonical manifest)."""
        text = self._root_file("manifest.csv").decode("utf-8")
        return list(csv.DictReader(io.StringIO(text)))

    def manifests_agree(self) -> bool:
        """True when manifest.csv and manifest.json list the same paths and checksums."""
        from_csv = {(row["relative_path"], row["sha256"]) for row in self.manifest_rows()}
        from_json = {(row["relative_path"], row["sha256"]) for row in json.loads(self._root_file("manifest.json"))}
        return from_csv == from_json

    def classification_images(self) -> tuple[list[SourceImage], list[ManifestIssue]]:
        """Return manifest-listed classification images plus any consistency problems.

        Checks that each listed file exists, that folder, filename code and
        manifest label agree, that the split matches the path, that declared
        sizes match the archive, and that no classification file is unlisted.
        """
        images: list[SourceImage] = []
        issues: list[ManifestIssue] = []
        listed: set[str] = set()
        for row in self.manifest_rows():
            if row["task"] != "classification":
                continue
            zip_path = f"{self.root}/{row['relative_path'].replace(chr(92), '/')}"
            listed.add(zip_path)
            problem = self._row_problem(row, zip_path)
            if problem:
                issues.append(ManifestIssue(zip_path, problem))
                continue
            images.append(
                SourceImage(
                    sample_id=PurePosixPath(zip_path).stem,
                    zip_path=zip_path,
                    label=row["tumor_label"],
                    plane=row["plane_label"],
                    original_split=row["split"],
                    sha256=row["sha256"].lower(),
                    file_size=int(row["file_size_bytes"]),
                    width=int(row["width"]),
                    height=int(row["height"]),
                )
            )
        prefix = f"{self.root}/classification_task/"
        for name, info in self._infos.items():
            if name.startswith(prefix) and not info.is_dir() and name not in listed:
                issues.append(ManifestIssue(name, "file present in archive but missing from manifest"))
        return images, issues

    def _row_problem(self, row: dict[str, str], zip_path: str) -> str | None:
        """Describe the first inconsistency in a classification manifest row, if any."""
        info = self._infos.get(zip_path)
        if info is None:
            return "listed in manifest but missing from archive"
        parts = PurePosixPath(zip_path).parts  # root, classification_task, split, label, file
        if len(parts) != 5:
            return "unexpected folder depth"
        _, _, split, folder_label, filename = parts
        name_parts = PurePosixPath(filename).stem.split("_")
        if row["is_mask"] != "False":
            return "classification row marked as mask"
        if row["tumor_label"] not in CLASS_NAMES:
            return f"unknown manifest label {row['tumor_label']!r}"
        if folder_label != row["tumor_label"]:
            return f"folder label {folder_label!r} disagrees with manifest label {row['tumor_label']!r}"
        if len(name_parts) < 4 or FILENAME_CODES.get(name_parts[3]) != row["tumor_label"]:
            return "filename tumor code disagrees with manifest label"
        if split != row["split"] or name_parts[1] != row["split"]:
            return "path split disagrees with manifest split"
        if info.file_size != int(row["file_size_bytes"]):
            return "archive file size differs from manifest"
        return None

    def segmentation_overlap(self, classification_hashes: set[str]) -> dict[str, int]:
        """Count segmentation images that are byte-identical copies of classification images.

        Uses manifest checksums only; segmentation files are never used for training.
        """
        rows = [r for r in self.manifest_rows() if r["task"] == "segmentation"]
        images = [r for r in rows if r["is_mask"] == "False"]
        return {
            "segmentation_images": len(images),
            "segmentation_masks": len(rows) - len(images),
            "images_identical_to_classification": sum(r["sha256"].lower() in classification_hashes for r in images),
        }
