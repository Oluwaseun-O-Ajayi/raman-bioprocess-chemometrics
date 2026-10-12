"""Download both data sets from Hugging Face into data/ (about 95 MB).

    pip install huggingface_hub
    python scripts/download_data.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ramanpat import data  # noqa: E402

if __name__ == "__main__":
    data.download(ROOT / "data")
    print("saved to", ROOT / "data")
