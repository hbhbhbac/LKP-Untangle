"""
LKML

1. data/raw_data/lkml/{epoch}/m mbox
   epochgit
2. data/raw_data/lkml_emails
3. SQLite data/structure_data/lkml.sqlite
"""

from __future__ import annotations

import argparse
import email
import email.utils
import gzip
import json
import mailbox
import os
import re
import sqlite3
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


# core
from core import (
    BASE_DIR,
    DATA_DIR,
    RAW_DIR,
    LKML_DIR,
    LKML_EMAILS_DIR,
    LKML_JSONL_DIR,
    STRUCTURE_DIR,
    DB_PATH,
    LKMLEmail,
)


def is_git_repo(path: Path) -> bool:
    """git."""
    git_dir = path / ".git"
    return git_dir.exists() and git_dir.is_dir()


def get_git_commits(repo_path: Path) -> List[str]:
    """githash."""
    try:
        result = subprocess.run(
            ["git", "rev-list", "--all", "--reverse"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        )
        commits = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
        return commits
    except subprocess.CalledProcessError as e:
        print(f"[WARN] git {repo_path}: {e}")
        return []
    except Exception as e:
        print(f"[WARN] git {repo_path}: {e}")
        return []


def get_file_from_git_commit(repo_path: Path, commit_hash: str, file_path: str = "m") -> Optional[bytes]:
    """git."""
    try:
        result = subprocess.run(
            ["git", "show", f"{commit_hash}:{file_path}"],
            cwd=str(repo_path),
            capture_output=True,
            check=True,
            timeout=30,
        )
        return result.stdout
    except subprocess.CalledProcessError:
        return None
    except Exception as e:
        print(f"[WARN] {commit_hash} : {e}")
        return None


def iter_epoch_mboxes() -> Iterable[Tuple[int, Path, bool]]:
    """ epoch (epoch, path, is_git_repo).
    
    Returns:
        epoch: epoch
        path: epoch
        is_git_repo: git
    """
    if not LKML_DIR.exists():
        return

    for entry in sorted(LKML_DIR.iterdir()):
        if not entry.is_dir():
            continue
        try:
            epoch = int(entry.name)
        except ValueError:
            continue

        is_git = is_git_repo(entry)
        yield epoch, entry, is_git


def parse_epoch_spec(spec: str) -> Set[int]:
    """ epoch .

    :
    - "14-18"       -> {14, 15, 16, 17, 18}
    - "14,15,18"    -> {14, 15, 18}
    - "14-14,18,20" -> {14, 18, 20}
    """
    result: Set[int] = set()
    if not spec:
        return result

    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            try:
                start_s, end_s = part.split("-", 1)
                start = int(start_s.strip())
                end = int(end_s.strip())
                if start > end:
                    start, end = end, start
                result.update(range(start, end + 1))
            except ValueError:
                print(f"[WARN] epoch: '{part}'")
        else:
            try:
                result.add(int(part))
            except ValueError:
                print(f"[WARN] epoch: '{part}'")

    return result


def _process_single_commit(epoch: int, repo_path: Path, commit_hash: str, seq: int) -> Optional[LKMLEmail]:
    """git."""
    raw_content = get_file_from_git_commit(repo_path, commit_hash, "m")
    if raw_content is None:
        return None
    
    try:
        msg = email.message_from_bytes(raw_content)
        return parse_single_message(epoch, seq, msg)
    except Exception as e:
        print(f"[WARN] git epoch={epoch} commit={commit_hash[:8]}: {e}")
        return None


def parse_mbox_from_git_history(epoch: int, repo_path: Path, max_workers: int = 8) -> Iterable[LKMLEmail]:
    """gitcommits.
    
    gitm
    commits
    """
    print(f" epoch {epoch} git: {repo_path}")
    
    commits = get_git_commits(repo_path)
    if not commits:
        print(f"[WARN] epoch {epoch} git")
        return
    
    total_commits = len(commits)
    print(f"epoch {epoch}: {total_commits} git {max_workers} ")
    
    processed_count = 0
    processed_lock = threading.Lock()
    step = max(1, total_commits // 40) # 40
    
    def update_progress():
        nonlocal processed_count
        with processed_lock:
            processed_count += 1
            if processed_count % step == 0 or processed_count == total_commits:
                ratio = processed_count / total_commits
                bar_len = 30
                filled = int(bar_len * ratio)
                bar = "#" * filled + "." * (bar_len - filled)
                percent = ratio * 100.0
                print(f" epoch {epoch} : [{bar}] {processed_count}/{total_commits} ({percent:5.1f}%)")
    
    # commits
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_seq = {}
        seq = 0
        for commit_hash in commits:
            future = executor.submit(_process_single_commit, epoch, repo_path, commit_hash, seq)
            future_to_seq[future] = seq
            seq += 1
        
        # seq
        results = {}
        for future in as_completed(future_to_seq):
            seq_num = future_to_seq[future]
            try:
                mail = future.result()
                if mail is not None:
                    results[seq_num] = mail
            except Exception as e:
                print(f"[WARN] commit epoch={epoch} seq={seq_num}: {e}")
            update_progress()
    
    # seqyield
    for seq_num in sorted(results.keys()):
        yield results[seq_num]


def parse_mbox(epoch: int, mbox_path: Path) -> Iterable[LKMLEmail]:
    """ mbox LKMLEmail .

    :
    - mbox mailbox.mbox
    - 0 RFC822
    """
    print(f" epoch {epoch} mbox: {mbox_path}")

    mbox = mailbox.mbox(mbox_path)
    count = 0
    for i, msg in enumerate(mbox):
        count += 1
        try:
            yield parse_single_message(epoch, i, msg)
        except Exception as e:
            print(f"[WARN] epoch={epoch} idx={i}: {e}")

    # mbox
    if count == 0:
        try:
            raw = mbox_path.read_bytes()
            msg = email.message_from_bytes(raw)
            yield parse_single_message(epoch, 0, msg) # seq_in_epoch = 0
        except Exception as e:
            print(f"[WARN] epoch={epoch}: {e}")


def _safe_get_header(msg: email.message.Message, header_name: str, default: str = "") -> str:
    """Header."""
    try:
        value = msg.get(header_name, default)
        if value is None:
            return default
        # Header
        if hasattr(value, 'encode'):
            return str(value)
        else:
            # Header
            return str(value)
    except Exception:
        return default


def _safe_strip(value: str) -> str:
    """stripNone."""
    if not value:
        return ""
    try:
        return str(value).strip()
    except Exception:
        return ""


def parse_single_message(epoch: int, idx: int, msg: mailbox.mboxMessage) -> LKMLEmail:
    """ mailbox.Message LKMLEmail."""
    # msg email.message.Message
    # headerHeader
    message_id_raw = _safe_get_header(msg, "Message-ID", "")
    message_id = _safe_strip(message_id_raw).strip("<>")
    
    subject = _safe_get_header(msg, "Subject", "")
    from_addr = _safe_get_header(msg, "From", "")
    to_addr = _safe_get_header(msg, "To", "")
    cc = _safe_get_header(msg, "Cc", "")
    date_raw = _safe_get_header(msg, "Date", "")
    
    in_reply_to_raw = _safe_get_header(msg, "In-Reply-To", "")
    in_reply_to = _safe_strip(in_reply_to_raw).strip("<>")
    
    references = _safe_get_header(msg, "References", "")
    list_id = _safe_get_header(msg, "List-Id", "")

    # unix
    timestamp: Optional[float] = None
    if date_raw:
        try:
            dt_tuple = email.utils.parsedate_tz(date_raw)
            if dt_tuple:
                timestamp = email.utils.mktime_tz(dt_tuple)
        except Exception:
            timestamp = None

    body = extract_body(msg)
    # body
    try:
        # bodysurrogateerrors='replace'
        body = body.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
    except Exception:
        body = ""
    
    body_length = len(body)

    is_patch, patch_meta = extract_patch_info(subject, body)

    # raw_headers
    raw_headers = {}
    for k, v in msg.items():
        try:
            key = str(k)
            # Header
            if v is None:
                value = ""
            else:
                value = str(v)
                # surrogate
                value = value.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
            raw_headers[key] = value
        except Exception:
            raw_headers[str(k)] = "[encoding_error]"

    return LKMLEmail(
        epoch=epoch,
        seq_in_epoch=idx,
        message_id=message_id,
        subject=subject,
        from_addr=from_addr,
        to_addr=to_addr,
        cc=cc,
        date_raw=date_raw,
        timestamp=timestamp,
        in_reply_to=in_reply_to,
        references=references,
        list_id=list_id,
        body=body,
        body_length=body_length,
        is_patch=is_patch,
        patch_version=patch_meta.get("version"),
        patch_current=patch_meta.get("current"),
        patch_total=patch_meta.get("total"),
        is_cover=patch_meta.get("is_cover"),
        patchwork_id=patch_meta.get("patchwork_id"),
        raw_headers=raw_headers,
    )


def extract_body(msg: email.message.Message) -> str:
    """ text/plain ."""
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == "text/plain":
                try:
                    payload = part.get_payload(decode=True)
                    if payload is None:
                        continue
                    body = payload.decode(part.get_content_charset() or "utf-8", errors="ignore")
                    break
                except Exception:
                    continue
    else:
        try:
            payload = msg.get_payload(decode=True)
            if payload is not None:
                body = payload.decode(msg.get_content_charset() or "utf-8", errors="ignore")
            else:
                body = str(msg.get_payload() or "")
        except Exception:
            body = str(msg.get_payload() or "")
    return body


def extract_patch_info(subject: str, body: str) -> Tuple[bool, Dict[str, Optional[object]]]:
    """.
    
     subject
    - [PATCH v2 0/5] xxx -
    - [PATCH v7 loongarch-next 0/4] - N/M
    - [PATCH v2 RFC 1/3] xxx - RFC
    - [PATCH 1/3] xxx -
    """
    info: Dict[str, Optional[object]] = {
        "version": None,
        "current": None,
        "total": None,
        "is_cover": None,
        "patchwork_id": None,
    }

    is_patch = any(kw in subject for kw in ("[PATCH", "PATCH ", "patch "))

    # [PATCH v2 0/5] [PATCH v7 loongarch-next 0/4]
    # N/M subsystem RFC
    patch_pattern = r"\[PATCH\s+(v\d+)?(?:\s+[^\]]*?)?\s*(\d+)/(\d+)\]"
    m = re.search(patch_pattern, subject, flags=re.IGNORECASE)
    if m:
        version = m.group(1)
        current = m.group(2)
        total = m.group(3)
        info["version"] = version
        info["current"] = int(current) if current else None
        info["total"] = int(total) if total else None
        info["is_cover"] = info["current"] == 0
        is_patch = True

    # patchwork
    pw_pattern = r"patchwork\.kernel\.org[^\n]*?\bid=(\d+)"
    m2 = re.search(pw_pattern, body)
    if m2:
        info["patchwork_id"] = m2.group(1)
        is_patch = True

    return is_patch, info


# ------------- JSONL -------------
# JSONL processors.jsonl.jsonl_writer

from processors.jsonl.jsonl_writer import (
    export_email_to_jsonl,
    close_all_jsonl_writers,
)


# JSON
def export_email_to_json(mail: LKMLEmail) -> Path:
    """ JSON ."""
    LKML_EMAILS_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"lkml_{mail.epoch:02d}_{mail.seq_in_epoch:07d}.json"
    fpath = LKML_EMAILS_DIR / fname
    data_dict = asdict(mail)
    json_text = json.dumps(data_dict, ensure_ascii=False, indent=2, default=str)
    with fpath.open("w", encoding="utf-8") as f:
        f.write(json_text)
    return fpath


def export_emails_to_json(emails: Iterable[LKMLEmail]) -> List[Path]:
    """ JSON ."""
    paths: List[Path] = []
    for mail in emails:
        paths.append(export_email_to_json(mail))
    return paths


def _process_single_epoch_to_jsonl(epoch: int, epoch_path: Path, is_git: bool, commit_workers: int = 8) -> Tuple[int, int, float]:
    """epoch (epoch, , ).
    
    JSONLcommits
    """
    start = time.perf_counter()

    if is_git:
        email_iter: Iterable[LKMLEmail] = parse_mbox_from_git_history(epoch, epoch_path, max_workers=commit_workers)
    else:
        mbox_path = epoch_path / "m"
        if mbox_path.exists() and mbox_path.is_file():
            email_iter = parse_mbox(epoch, mbox_path)
        else:
            print(f"[WARN] epoch {epoch} gitm")
            return epoch, 0, 0.0

    count = 0
    for mail in email_iter:
        export_email_to_jsonl(mail) # JSONL
        count += 1

    if count == 0:
        print(f"[WARN] epoch {epoch}: ")

    duration = time.perf_counter() - start
    print(f"epoch {epoch}: {count} JSONL {duration:.2f}s")
    return epoch, count, duration


def export_all_epochs_to_jsonl(selected_epochs: Optional[Set[int]] = None, epoch_workers: int = 4, commit_workers: int = 8) -> None:
    """ epoch mbox JSONL.

    epochgitgit
    mbox
    
    - epochepoch
    - commitepochcommits

    Args:
        selected_epochs: Noneepoch
        epoch_workers: epoch
        commit_workers: epochcommits
    """
    LKML_JSONL_DIR.mkdir(parents=True, exist_ok=True)

    epochs = list(iter_epoch_mboxes())
    if not epochs:
        print(" data/raw_data/lkml epoch ")
        return

    # epoch
    if selected_epochs:
        epochs = [(e, p, g) for (e, p, g) in epochs if e in selected_epochs]
        if not epochs:
            print(f" epoch: {sorted(selected_epochs)}")
            return

    total_epochs = len(epochs)
    print(f" {total_epochs} epochepoch {epoch_workers} epochcommits {commit_workers} ...")

    total_mails = 0
    start_all = time.perf_counter()

    completed_epochs = 0
    completed_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=epoch_workers) as executor:
        future_map = {
            executor.submit(_process_single_epoch_to_jsonl, epoch, epoch_path, is_git, commit_workers): epoch
            for (epoch, epoch_path, is_git) in epochs
        }

        for future in as_completed(future_map):
            epoch = future_map[future]
            try:
                _, count, duration = future.result()
                with completed_lock:
                    total_mails += count
                    completed_epochs += 1
                    ratio = completed_epochs / total_epochs
                    bar_len = 40
                    filled = int(bar_len * ratio)
                    bar = "#" * filled + "." * (bar_len - filled)
                    percent = ratio * 100.0
                    print(
                        f"[] [{bar}] {completed_epochs}/{total_epochs} epoch ({percent:5.1f}%)"
                        f": {total_mails}"
                    )
            except Exception as e:
                print(f"[WARN] epoch {epoch} : {e}")
                with completed_lock:
                    completed_epochs += 1

    # JSONL
    close_all_jsonl_writers()

    duration_all = time.perf_counter() - start_all
    print(f" epoch {total_mails} {duration_all:.2f}s")


# ------------- SQLite -------------
# SQLite processors.sqlite

from processors.sqlite.db_importer import import_jsonl_to_db


def run_all(
    epoch_spec: Optional[str] = None,
    epoch_workers: int = 4,
    commit_workers: int = 8,
    skip_jsonl: bool = False,
    skip_sqlite: bool = False,
):
    """mbox/git -> JSONL -> SQLite -> .

    Args:
        epoch_spec: epoch:
            - "14-18"
            - "14,15,18"
            - "14-14,18,20"
            Noneepoch
        epoch_workers: epoch
        commit_workers: epochcommits
        skip_jsonl: mbox/git -> JSONL
        skip_sqlite: JSONL -> SQLite
    """
    all_start = time.perf_counter()

    if epoch_spec:
        selected_epochs = parse_epoch_spec(epoch_spec)
        print(f" epoch: {sorted(selected_epochs)}")
    else:
        selected_epochs = None
        print("epochepoch")

    if not skip_jsonl:
        print("=== : mbox/git -> JSONL () ===")
        export_all_epochs_to_jsonl(
            selected_epochs=selected_epochs,
            epoch_workers=epoch_workers,
            commit_workers=commit_workers
        )
    else:
        print("=== : mbox/git -> JSONL ===")
    
    if not skip_sqlite:
        print("\n=== : JSONL -> SQLite (epoch) ===")
        import_jsonl_to_db(selected_epochs=selected_epochs)
    else:
        print("\n=== : JSONL -> SQLite ===")

    all_duration = time.perf_counter() - all_start
    print(f"\n {all_duration:.2f}s")


def main():
    parser = argparse.ArgumentParser(
        description="LKML epoch -> JSONL -> SQLite (epoch)"
    )
    parser.add_argument(
        "--epochs",
        type=str,
        default="",
        help=(
            "epoch: "
            "'14-18' '14,15,18' '14-14,18,20'"
            "epoch"
        ),
    )
    parser.add_argument(
        "--epoch-workers",
        type=int,
        default=4,
        help="epoch: 4",
    )
    parser.add_argument(
        "--commit-workers",
        type=int,
        default=7,
        help="epochcommits: 7",
    )
    parser.add_argument(
        "--skip-jsonl",
        action="store_true",
        help="mbox/git -> JSONL: False",
    )
    parser.add_argument(
        "--skip-sqlite",
        action="store_true",
        help="JSONL -> SQLite: False",
    )
    args = parser.parse_args()

    epoch_spec = args.epochs.strip() or None
    run_all(
        epoch_spec=epoch_spec,
        epoch_workers=args.epoch_workers,
        commit_workers=args.commit_workers,
        skip_jsonl=args.skip_jsonl,
        skip_sqlite=args.skip_sqlite,
    )


if __name__ == "__main__":
    main()


# python lkml_preprocess.py --epochs 0-18 --skip-jsonl --skip-sqlite --analyze-versions 