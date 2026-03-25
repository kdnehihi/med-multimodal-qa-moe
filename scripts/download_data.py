"""Data download script scaffold."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    """Entry point for data download or data registration."""
    parser = argparse.ArgumentParser(description="Download or register raw data sources.")
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw",
        help="Placeholder output path for raw data registration.",
    )
    args = parser.parse_args()

    # TODO: add raw data download or local data registration logic.
    # Possible options:
    # - create a manifest of downloaded files
    # - copy raw files into project structure
    # - document expected dataset placement
    print(f"Download scaffold ready. Target location: {PROJECT_ROOT / args.output}")


if __name__ == "__main__":
    main()
