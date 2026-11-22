"""Kept for old command lines: `python weather/train.py ...` is `bantam weather ...`."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bantamtrain.cli.weather import main  # noqa: E402

if __name__ == "__main__":
    main()
