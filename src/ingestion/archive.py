"""Defensive reading of uploaded ZIP files.

Accepted layout (class folder names are case-insensitive)::

    glioma/a.jpg                       class folder at the top level, or
    new_data/glioma/a.jpg              inside ONE optional wrapper folder

Files outside a class folder, or in an unknown folder, are kept for review as
"needs label" (they are never assigned a class automatically).

Whole-archive rejections (unsafe): path traversal, absolute or drive paths,
symbolic links, too many entries, excessive declared or actual decompressed
size, suspicious compression ratios (ZIP bombs), or sizes that lie in the header.
Entry-level skips: nested archives, encrypted entries, OS metadata files, and
images above the per-file size limit.
"""

from __future__ import annotations

import io
import stat
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterator

from src.ingestion.limits import UploadLimits
from src.labels import UnknownLabelError, normalize_label

_NESTED_ARCHIVE_SUFFIXES = {".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar", ".jar"}
_NESTED_ARCHIVE_MAGIC = (b"PK\x03\x04", b"\x1f\x8b", b"7z\xbc\xaf\x27\x1c", b"Rar!", b"BZh", b"\xfd7zXZ")
_SYSTEM_FILES = {".ds_store", "thumbs.db", "desktop.ini"}


class UnsafeArchiveError(ValueError):
    """The whole archive is refused; nothing from it is staged."""


@dataclass(frozen=True)
class ArchiveEntry:
    path: str
    data: bytes
    label: str | None
    label_problem: str | None


@dataclass(frozen=True)
class SkippedEntry:
    path: str
    reason: str


def _check_name(name: str) -> PurePosixPath:
    """Reject names that could escape an extraction directory."""
    if "\x00" in name or "\\" in name:
        raise UnsafeArchiveError(f"Entry {name!r} contains a backslash or NUL character.")
    path = PurePosixPath(name)
    if not path.parts:
        raise UnsafeArchiveError("ZIP contains an empty entry path.")
    if path.is_absolute() or (path.parts and ":" in path.parts[0]):
        raise UnsafeArchiveError(f"Entry {name!r} uses an absolute or drive path.")
    if ".." in path.parts:
        raise UnsafeArchiveError(f"Entry {name!r} attempts path traversal.")
    return path


def label_from_path(path: PurePosixPath) -> tuple[str | None, str | None]:
    """Return ``(label, None)`` or ``(None, problem)`` for an entry path."""
    folders = path.parts[:-1]
    if not folders:
        return None, "file is not inside a class folder"
    if len(folders) > 2:
        return None, "folder nesting too deep; use <class>/<image> or <wrapper>/<class>/<image>"
    folder = folders[-1]
    try:
        return normalize_label(folder), None
    except UnknownLabelError:
        return None, f"unknown class folder {folder!r}"


def _skip_reason(path: PurePosixPath, info: zipfile.ZipInfo, limits: UploadLimits) -> str | None:
    if path.parts[0] == "__MACOSX" or path.name.lower() in _SYSTEM_FILES or path.name.startswith("._"):
        return "operating-system metadata file"
    if info.flag_bits & 0x1:
        return "encrypted entries are not accepted"
    if path.suffix.lower() in _NESTED_ARCHIVE_SUFFIXES:
        return "nested archives are not accepted"
    if info.file_size > limits.max_file_bytes:
        return f"file is larger than the {limits.max_file_bytes // (1024 * 1024)} MiB per-image limit"
    return None


class ZipReader:
    """Validate metadata up front, then decompress at most one bounded image at a time."""

    def __init__(self, source: Path | BinaryIO, limits: UploadLimits, *,
                 max_files: int | None = None, infer_labels: bool = True):
        self.source, self.limits = source, limits
        self.max_files, self.infer_labels = max_files, infer_labels

    def __enter__(self):
        if isinstance(self.source, Path):
            size = self.source.stat().st_size
        else:
            size = self.source.seek(0, io.SEEK_END)
            self.source.seek(0)
        if size > self.limits.max_batch_bytes:
            raise UnsafeArchiveError("ZIP exceeds the compressed upload size limit.")
        try:
            self.archive = zipfile.ZipFile(self.source)
        except zipfile.BadZipFile as exc:
            raise UnsafeArchiveError("The upload is not a valid ZIP file.") from exc
        try:
            self.infos = self._validate()
        except BaseException:
            self.archive.close()
            raise
        return self

    def __exit__(self, *_):
        self.archive.close()

    def _validate(self) -> list[zipfile.ZipInfo]:
        infos = self.archive.infolist()
        if len(infos) > self.limits.max_zip_entries:
            raise UnsafeArchiveError(f"ZIP has {len(infos)} entries; the limit is {self.limits.max_zip_entries}.")
        for info in infos:
            _check_name(info.filename)
            if stat.S_ISLNK(info.external_attr >> 16):
                raise UnsafeArchiveError(f"Entry {info.filename!r} is a symbolic link.")
        files = [info for info in infos if not info.is_dir()]
        if self.max_files is not None and len(files) > self.max_files:
            raise UnsafeArchiveError(f"ZIP has {len(files)} files; the batch limit is {self.max_files}.")
        declared = sum(info.file_size for info in files)
        if declared > self.limits.max_zip_uncompressed_bytes:
            raise UnsafeArchiveError(f"ZIP would expand to {declared:,} bytes; the limit is {self.limits.max_zip_uncompressed_bytes:,}.")
        for info in files:
            if info.compress_size and info.file_size / info.compress_size > self.limits.max_compression_ratio:
                raise UnsafeArchiveError(f"Entry {info.filename!r} has a suspicious compression ratio (possible ZIP bomb).")
        return files

    def __iter__(self) -> Iterator[ArchiveEntry | SkippedEntry]:
        total = 0
        for info in self.infos:
            path = _check_name(info.filename)
            reason = _skip_reason(path, info, self.limits)
            if reason:
                yield SkippedEntry(info.filename, reason)
                continue
            try:
                with self.archive.open(info) as handle:
                    content = handle.read(info.file_size + 1)
            except (zipfile.BadZipFile, RuntimeError, NotImplementedError, EOFError, zlib.error) as exc:
                raise UnsafeArchiveError(f"Cannot read ZIP entry {info.filename!r}: corrupt or unsupported archive.") from exc
            if len(content) != info.file_size:
                raise UnsafeArchiveError(f"Entry {info.filename!r} does not match its declared size.")
            total += len(content)
            if total > self.limits.max_zip_uncompressed_bytes:
                raise UnsafeArchiveError("ZIP content exceeds the decompressed size limit.")
            if content.startswith(_NESTED_ARCHIVE_MAGIC):
                yield SkippedEntry(info.filename, "nested archives are not accepted")
                continue
            label, problem = label_from_path(path) if self.infer_labels else (None, None)
            yield ArchiveEntry(info.filename, content, label, problem)


def read_zip(data: bytes, limits: UploadLimits, *, infer_labels: bool = True,
             max_files: int | None = None) -> tuple[list[ArchiveEntry], list[SkippedEntry]]:
    """Collect validated entries for synchronous callers; jobs use ZipReader directly."""
    entries, skipped = [], []
    with ZipReader(io.BytesIO(data), limits, infer_labels=infer_labels, max_files=max_files) as reader:
        for entry in reader:
            (skipped if isinstance(entry, SkippedEntry) else entries).append(entry)
    return entries, skipped
