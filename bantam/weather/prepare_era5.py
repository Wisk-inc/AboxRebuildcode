"""Kept for old command lines: `python weather/prepare_era5.py ...` is `bantam prepare-era5 ...`."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bantamtrain.cli.prepare_era5 import main  # noqa: E402

if __name__ == "__main__":
    main()
