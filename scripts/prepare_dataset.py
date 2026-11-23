"""Prepare the cleaned, de-duplicated and split BRISC dataset from the original ZIP.

Usage:
    python scripts/prepare_dataset.py --archive "C:/path/to/archive.zip"

The ZIP is only read. Outputs: data/manifests/*.csv, data/manifests/dataset_card.json,
data/{train,validation,test,retrain_demo}/<class>/ images, data/cache/similarity_index.npz.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings, load_pipeline_config  # noqa: E402
from src.dataset.prepare import prepare_dataset  # noqa: E402
from utils.log import configure_logging  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", type=Path, default=os.environ.get("BRISC_ARCHIVE"), help="Path to the BRISC 2025 ZIP")
    args = parser.parse_args()
    if not args.archive:
        parser.error("Provide --archive or set BRISC_ARCHIVE.")
    settings = get_settings()
    configure_logging(settings.log_level)
    card = prepare_dataset(args.archive, settings, load_pipeline_config())
    print(json.dumps({key: card[key] for key in ("counts", "split_totals", "splits")}, indent=2))


if __name__ == "__main__":
    main()
