"""Kept for old command lines: `python llm/pack_data.py ...` is `bantam pack-text ...`."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bantamtrain.cli.pack_text import main  # noqa: E402

if __name__ == "__main__":
    main()
