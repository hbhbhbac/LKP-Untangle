"""
JSONL
JSONL
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from core import LKML_JSONL_DIR


def load_email_jsonl_files(epoch: Optional[int] = None) -> Iterable[Tuple[Path, Dict]]:
    """ lkml_jsonl JSONL .
    
    Args:
        epoch: epochJSONLepoch
    
    Yields:
        (file_path, email_data_dict)
    """
    if not LKML_JSONL_DIR.exists():
        return

    if epoch is not None:
        epoch_dir = LKML_JSONL_DIR / str(epoch)
        if epoch_dir.exists():
            jsonl_files = sorted(epoch_dir.glob("*.jsonl.gz"))
        else:
            return
    else:
        # epochJSONL
        jsonl_files = sorted(LKML_JSONL_DIR.glob("*/*.jsonl.gz"))

    for f in jsonl_files:
        try:
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                for line_num, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        yield f, data
                    except json.JSONDecodeError as e:
                        print(f"[WARN] JSONL {f}:{line_num}: {e}")
        except Exception as e:
            print(f"[WARN] JSONL {f}: {e}")

