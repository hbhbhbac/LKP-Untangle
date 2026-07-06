"""
LKML Data
Import
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


# Path
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw_data"
LKML_DIR = RAW_DIR / "lkml"
LKML_EMAILS_DIR = RAW_DIR / "lkml_emails" #
LKML_JSONL_DIR = RAW_DIR / "lkml_jsonl" # JSONLDirectory
STRUCTURE_DIR = DATA_DIR / "structure_data"
DB_PATH = STRUCTURE_DIR / "lkml.sqlite" # epoch


@dataclass
class LKMLEmail:
    """ LKML ."""

    epoch: int
    seq_in_epoch: int

    #
    message_id: str
    subject: str
    from_addr: str
    to_addr: str
    cc: str
    date_raw: str
    timestamp: Optional[float]

    in_reply_to: str
    references: str
    list_id: str

    #
    body: str
    body_length: int

    # Patch
    is_patch: bool
    patch_version: Optional[str]
    patch_current: Optional[int]
    patch_total: Optional[int]
    is_cover: Optional[bool]
    patchwork_id: Optional[str]

    #
    raw_headers: Dict[str, str]

