"""Content hashing for integrity checks and duplicate detection.

Perceptual hashes are computed with a plain DCT (no pretrained networks), so
near-duplicate detection stays transparent and reproducible.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

_CHUNK_BYTES = 1 << 20


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file, streamed in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def pixel_hash(pixels: np.ndarray) -> str:
    """Hash decoded pixel values together with shape and dtype.

    Two files with different encodings but identical decoded pixels get the same
    hash, which is what "exact duplicate" means throughout this project.
    """
    digest = hashlib.sha256(f"{pixels.shape}|{pixels.dtype}".encode())
    digest.update(np.ascontiguousarray(pixels).tobytes())
    return digest.hexdigest()


def _dct_matrix(size: int) -> np.ndarray:
    """Orthonormal DCT-II basis, so ``M @ x @ M.T`` is a 2-D DCT."""
    n = np.arange(size)
    matrix = np.cos(np.pi * (2 * n[None, :] + 1) * n[:, None] / (2 * size))
    matrix[0] *= 1 / np.sqrt(2)
    return matrix * np.sqrt(2 / size)


def perceptual_hash(gray: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    """Compute a 64-bit DCT perceptual hash (pHash) of a greyscale image.

    The image is shrunk to 32x32, transformed with a 2-D DCT, and the lowest
    8x8 frequencies are thresholded at their median. Small edits such as
    re-compression or resizing flip only a few bits.
    """
    size = hash_size * highfreq_factor
    pixels = np.asarray(gray.convert("L").resize((size, size), Image.Resampling.LANCZOS), dtype=np.float64)
    basis = _dct_matrix(size)
    low_freq = (basis @ pixels @ basis.T)[:hash_size, :hash_size]
    bits = (low_freq > np.median(low_freq)).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def hash_to_hex(value: int) -> str:
    """Format a 64-bit hash as fixed-width hex for CSV storage."""
    return f"{value:016x}"


def hex_to_hash(text: str) -> int:
    """Parse a hash written by :func:`hash_to_hex`."""
    return int(text, 16)


def hamming_distances(query: int, hashes: np.ndarray) -> np.ndarray:
    """Hamming distance from one 64-bit hash to an array of hashes."""
    return np.bitwise_count(hashes ^ np.uint64(query))


def pairs_within_distance(hashes: np.ndarray, max_distance: int, chunk: int = 512) -> list[tuple[int, int, int]]:
    """Return ``(i, j, distance)`` for all ``i < j`` with Hamming distance <= ``max_distance``.

    Works in row chunks so 6,000 hashes (18M pairs) fit comfortably in memory.
    """
    hashes = np.asarray(hashes, dtype=np.uint64)
    pairs: list[tuple[int, int, int]] = []
    for start in range(0, len(hashes), chunk):
        block = np.bitwise_count(hashes[start : start + chunk, None] ^ hashes[None, :])
        rows, cols = np.nonzero(block <= max_distance)
        rows = rows + start
        keep = cols > rows
        pairs.extend(zip(rows[keep].tolist(), cols[keep].tolist(), block[rows[keep] - start, cols[keep]].tolist()))
    return pairs
