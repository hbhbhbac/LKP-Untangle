"""
SQLite
JSONLSQLite
"""

from __future__ import annotations

import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

from core import LKML_JSONL_DIR, STRUCTURE_DIR, BASE_DIR
from processors.jsonl.jsonl_loader import load_email_jsonl_files
from processors.sqlite.db_utils import get_db_path, init_db


def import_jsonl_to_db_for_epoch(epoch: int) -> Tuple[int, int, float]:
    """epochJSONLSQLite.
    
    Returns:
        (inserted, skipped, duration)
    """
    db_path = get_db_path(epoch)
    STRUCTURE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    init_db(conn)

    inserted = 0
    skipped = 0

    start = time.perf_counter()
    print(f" epoch {epoch} JSONL SQLite ...")

    with conn:
        for idx, (path, data) in enumerate(load_email_jsonl_files(epoch=epoch)):
            try:
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO emails (
                        epoch, seq_in_epoch,
                        message_id, subject, from_addr, to_addr, cc,
                        date_raw, timestamp,
                        in_reply_to, references_raw, list_id,
                        body, body_length,
                        is_patch, patch_version, patch_current, patch_total, is_cover, patchwork_id,
                        raw_headers_json, jsonl_path
                    ) VALUES (
                        :epoch, :seq_in_epoch,
                        :message_id, :subject, :from_addr, :to_addr, :cc,
                        :date_raw, :timestamp,
                        :in_reply_to, :references_raw, :list_id,
                        :body, :body_length,
                        :is_patch, :patch_version, :patch_current, :patch_total, :is_cover, :patchwork_id,
                        :raw_headers_json, :jsonl_path
                    )
                    """,
                    {
                        "epoch": data.get("epoch"),
                        "seq_in_epoch": data.get("seq_in_epoch"),
                        "message_id": data.get("message_id"),
                        "subject": data.get("subject"),
                        "from_addr": data.get("from_addr"),
                        "to_addr": data.get("to_addr"),
                        "cc": data.get("cc"),
                        "date_raw": data.get("date_raw"),
                        "timestamp": data.get("timestamp"),
                        "in_reply_to": data.get("in_reply_to"),
                        "references_raw": data.get("references"),
                        "list_id": data.get("list_id"),
                        "body": data.get("body"),
                        "body_length": data.get("body_length"),
                        "is_patch": 1 if data.get("is_patch") else 0,
                        "patch_version": data.get("patch_version"),
                        "patch_current": data.get("patch_current"),
                        "patch_total": data.get("patch_total"),
                        "is_cover": None if data.get("is_cover") is None else int(bool(data.get("is_cover"))),
                        "patchwork_id": data.get("patchwork_id"),
                        "raw_headers_json": json.dumps(data.get("raw_headers", {}), ensure_ascii=False),
                        "jsonl_path": str(path.relative_to(BASE_DIR)),
                    },
                )
                # rowcount INSERT OR IGNORE10
                if cursor.rowcount > 0:
                    inserted += 1
                else:
                    skipped += 1
                
                # 1000
                if (inserted + skipped) % 1000 == 0:
                    print(f" epoch {epoch} : {inserted + skipped} {inserted} {skipped} ")
            except Exception as e:
                print(f"[WARN] epoch={epoch} {path}: {e}")

    duration = time.perf_counter() - start
    print(f"epoch {epoch} : {inserted} , () {skipped} {duration:.2f}s")
    print(f": {db_path}")
    return inserted, skipped, duration


def import_jsonl_to_db(selected_epochs: Optional[Set[int]] = None) -> None:
    """epochJSONLSQLite."""
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    if not LKML_JSONL_DIR.exists():
        print("JSONL")
        return

    # epoch
    if selected_epochs:
        epochs_to_process = sorted(selected_epochs)
    else:
        # JSONLepoch
        epochs_to_process = sorted([
            int(d.name) for d in LKML_JSONL_DIR.iterdir()
            if d.is_dir() and d.name.isdigit()
        ])

    if not epochs_to_process:
        print("epoch")
        return

    print(f" {len(epochs_to_process)} epochJSONLSQLite...")
    
    max_workers = min(8, len(epochs_to_process), (os.cpu_count() or 4))
    total_inserted = 0
    total_skipped = 0
    start_all = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(import_jsonl_to_db_for_epoch, epoch): epoch
            for epoch in epochs_to_process
        }

        for future in as_completed(future_map):
            epoch = future_map[future]
            try:
                inserted, skipped, _ = future.result()
                total_inserted += inserted
                total_skipped += skipped
            except Exception as e:
                print(f"[WARN] epoch {epoch} : {e}")

    duration_all = time.perf_counter() - start_all
    print(f"\nepoch: {total_inserted} , {total_skipped} {duration_all:.2f}s")

