"""
JSONL
ProcessJSONL
"""

from __future__ import annotations

import gzip
import json
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

from core import LKMLEmail, LKML_JSONL_DIR


class JSONLWriter:
    """JSONL."""
    
    def __init__(self, epoch: int, max_lines_per_file: int = 10000):
        self.epoch = epoch
        self.max_lines = max_lines_per_file
        self.epoch_dir = LKML_JSONL_DIR / str(epoch)
        self.epoch_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_file_index = 0
        self.current_line_count = 0
        self.current_file: Optional[gzip.GzipFile] = None
        self.lock = threading.Lock()
        self._open_new_file()
    
    def _open_new_file(self):
        """JSONLFile."""
        if self.current_file:
            self.current_file.close()
        
        start_seq = self.current_file_index * self.max_lines
        end_seq = start_seq + self.max_lines - 1
        fname = f"lkml_{self.epoch:02d}_{start_seq:07d}_{end_seq:07d}.jsonl.gz"
        fpath = self.epoch_dir / fname
        
        # errors='replace'Process
        self.current_file = gzip.open(fpath, "wt", encoding="utf-8", errors='replace')
        self.current_line_count = 0
        self.current_file_index += 1
    
    def write(self, mail: LKMLEmail):
        """."""
        with self.lock:
            if self.current_line_count >= self.max_lines:
                self._open_new_file()
            
            data_dict = asdict(mail)
            # ensure_ascii=Falseerrors='replace'Process
            try:
                json_line = json.dumps(data_dict, ensure_ascii=False, default=str)
                #
                json_line.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
            except (UnicodeEncodeError, UnicodeDecodeError) as e:
                #
                json_line = json.dumps(data_dict, ensure_ascii=True, default=str)
            self.current_file.write(json_line + "\n")
            self.current_line_count += 1
    
    def close(self):
        """File."""
        with self.lock:
            if self.current_file:
                self.current_file.close()
                self.current_file = None


# JSONLepoch
_jsonl_writers: Dict[int, JSONLWriter] = {}
_jsonl_writers_lock = threading.Lock()


def get_jsonl_writer(epoch: int) -> JSONLWriter:
    """GetCreateepochJSONL."""
    with _jsonl_writers_lock:
        if epoch not in _jsonl_writers:
            _jsonl_writers[epoch] = JSONLWriter(epoch)
        return _jsonl_writers[epoch]


def close_all_jsonl_writers():
    """JSONL."""
    with _jsonl_writers_lock:
        for writer in _jsonl_writers.values():
            writer.close()
        _jsonl_writers.clear()


def export_email_to_jsonl(mail: LKMLEmail) -> None:
    """JSONLFile."""
    writer = get_jsonl_writer(mail.epoch)
    writer.write(mail)

