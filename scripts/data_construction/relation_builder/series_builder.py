"""
series_builder.py

 + series is_patch / is_cover / series_id
 series patch comment

- series lkml_series_global.sqlite epoch
- epoch (0-18)
-

- Re: comment is_patch=1
- series series patch patch_version comment None
- cover series.has_cover = 0 1/n is_cover=1
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    from tqdm import tqdm
except ImportError:
    # tqdm
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, unit="", ncols=100, **kwargs):
            self.iterable = iterable
            self.total = total
            self.desc = desc or ""
            self.unit = unit
            self.n = 0
            
        def __iter__(self):
            if self.iterable is None:
                return iter([])
            return iter(self.iterable)
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass
        
        def set_postfix(self, **kwargs):
            pass
        
        def close(self):
            pass

from core import STRUCTURE_DIR
from processors.sqlite.db_utils import get_db_path


# ----------------- series -----------------

def get_global_series_db_path() -> Path:
    """ series """
    return STRUCTURE_DIR / "lkml_series_global.sqlite"


# ----------------- -----------------

def ensure_global_series_schema(conn: sqlite3.Connection) -> None:
    """ series """
    # series epoch series
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS series (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            representative_email_id INTEGER,
            representative_epoch INTEGER,
            has_cover INTEGER,
            patch_email_ids TEXT,
            comment_email_ids TEXT,
            previous_version_series_id_body_match INTEGER,
            previous_version_series_id_subject_match INTEGER,
            previous_version_series_id_thread_match INTEGER,
            patch_version TEXT
        )
        """
    )
    
    cursor = conn.execute("PRAGMA table_info(series)")
    existing_columns = {row[1] for row in cursor.fetchall()}
    
    columns_added = False
    if 'representative_epoch' not in existing_columns:
        try:
            conn.execute("ALTER TABLE series ADD COLUMN representative_epoch INTEGER")
            columns_added = True
        except sqlite3.OperationalError as e:
            print(f"[WARN] representative_epoch : {e}")
    
    if 'previous_version_series_id_body_match' not in existing_columns:
        try:
            conn.execute("ALTER TABLE series ADD COLUMN previous_version_series_id_body_match INTEGER")
            columns_added = True
        except sqlite3.OperationalError as e:
            print(f"[WARN] previous_version_series_id_body_match : {e}")
    
    if 'previous_version_series_id_subject_match' not in existing_columns:
        try:
            conn.execute("ALTER TABLE series ADD COLUMN previous_version_series_id_subject_match INTEGER")
            columns_added = True
        except sqlite3.OperationalError as e:
            print(f"[WARN] previous_version_series_id_subject_match : {e}")
    
    if 'previous_version_series_id_thread_match' not in existing_columns:
        try:
            conn.execute("ALTER TABLE series ADD COLUMN previous_version_series_id_thread_match INTEGER")
            columns_added = True
        except sqlite3.OperationalError as e:
            print(f"[WARN] previous_version_series_id_thread_match : {e}")
    
    if 'patch_version' not in existing_columns:
        try:
            conn.execute("ALTER TABLE series ADD COLUMN patch_version TEXT")
            columns_added = True
        except sqlite3.OperationalError as e:
            print(f"[WARN] patch_version : {e}")
    
    if columns_added:
        conn.commit()
    
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_series_rep_epoch ON series(representative_epoch)"
    )
    
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_series_rep_email ON series(representative_email_id, representative_epoch)"
    )
    
    cursor = conn.execute("PRAGMA table_info(series)")
    current_columns = {row[1] for row in cursor.fetchall()}
    
    if 'previous_version_series_id_body_match' in current_columns:
        try:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_series_prev_version_body ON series(previous_version_series_id_body_match)"
            )
        except sqlite3.OperationalError:
            pass
    
    if 'previous_version_series_id_subject_match' in current_columns:
        try:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_series_prev_version_subject ON series(previous_version_series_id_subject_match)"
            )
        except sqlite3.OperationalError:
            pass
    
    if 'previous_version_series_id_thread_match' in current_columns:
        try:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_series_prev_version_thread ON series(previous_version_series_id_thread_match)"
            )
        except sqlite3.OperationalError:
            pass


def ensure_emails_series_id_column(epoch: int) -> None:
    """ epoch emails series_id """
    db_path = get_db_path(epoch)
    if not db_path.exists():
        return
    
    conn = sqlite3.connect(db_path)
    try:
        try:
            conn.execute("ALTER TABLE emails ADD COLUMN series_id INTEGER")
        except sqlite3.OperationalError:
            pass
        
        try:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_emails_series_id ON emails(series_id)"
            )
        except sqlite3.OperationalError:
            pass
        
        conn.commit()
    finally:
        conn.close()


def is_comment_email(subject: str) -> bool:
    """
    ""
    - Re: / rE: / RE: / re: → comment
    """
    if not subject:
        return False
    s = subject.lstrip()
    if re.match(r"(?i)^re\s*:", s):
        return True
    return False


def determine_is_patch(subject: str, body: str) -> bool:
    """
     patch is_patch
    
    1. Subject [PATCH PATCH patch → patch
    2. Subject [PATCH vX N/M] → patch
    3. Body patchwork.kernel.org → patch
    
     subject Re: is_comment_email
    """
    if not subject:
        return False
    
    # 1: Subject PATCH
    if any(kw in subject for kw in ("[PATCH", "PATCH ", "patch ")):
        return True
    
    # 2: Subject [PATCH vX N/M]
    patch_pattern = r"\[PATCH\s+(v\d+)?\s*(\d+)/(\d+)?\]"
    if re.search(patch_pattern, subject, flags=re.IGNORECASE):
        return True
    
    # 3: Body patchwork
    if body:
        pw_pattern = r"patchwork\.kernel\.org[^\n]*?\bid=(\d+)"
        if re.search(pw_pattern, body):
            return True
    
    return False


def normalize_version(version: Optional[str]) -> Optional[str]:
    """ 'vN' None"""
    if not version:
        return None
    v = version.strip().lower()
    m = re.search(r"v?(\d+)", v)
    if not m:
        return None
    return f"v{m.group(1)}"


def extract_version_from_subject(subject: str) -> Optional[str]:
    """ subject
    
    - [PATCH v7 0/4]
    - [PATCH v7 loongarch-next 0/4]
    - [PATCH v2 RFC 1/3]
    """
    if not subject:
        return None
    # [PATCH vN ... N/M] N/M
    pattern = r'\[PATCH\s+(v\d+)\s+.*?\d+/\d+\]'
    match = re.search(pattern, subject, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return None


def parse_references(refs: Optional[str]) -> List[str]:
    """ references_raw message-id """
    if not refs:
        return []
    ids = re.findall(r"<([^>]+)>", refs)
    return [i.strip() for i in ids if i.strip()]


def load_single_epoch_emails(epoch: int) -> Tuple[int, Dict[Tuple[int, int], Dict]]:
    """
     epoch
    
    Returns:
        (epoch, {(epoch, email_id): {id, epoch, message_id, ...}})
         (epoch, email_id) epoch email_id
    """
    db_path = get_db_path(epoch)
    if not db_path.exists():
        return epoch, {}
    
    epoch_emails: Dict[Tuple[int, int], Dict] = {}
    
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT
                id,
                message_id,
                in_reply_to,
                references_raw,
                subject,
                body,
                from_addr,
                timestamp,
                patch_version,
                patch_current,
                patch_total
            FROM emails
            ORDER BY COALESCE(timestamp, 0), id
        """)
        
        rows = cur.fetchall()
        for (
            eid,
            message_id,
            in_reply_to,
            references_raw,
            subject,
            body,
            from_addr,
            ts,
            patch_version,
            patch_current,
            patch_total,
        ) in rows:
            # is_patch
            subject_str = subject or ""
            body_str = body or ""
            
            is_comment = is_comment_email(subject_str)
            if is_comment:
                determined_is_patch = False
            else:
                determined_is_patch = determine_is_patch(subject_str, body_str)
            
            # (epoch, email_id) epoch email_id
            composite_key = (epoch, eid)
            epoch_emails[composite_key] = {
                "id": eid,
                "epoch": epoch,
                "composite_id": composite_key, #
                "message_id": (message_id or "").strip(),
                "in_reply_to": (in_reply_to or "").strip().strip("<>"),
                "references_raw": references_raw or "",
                "subject": subject_str,
                "body": body_str,
                "from_addr": from_addr or "",
                "timestamp": ts,
                "determined_is_patch": determined_is_patch,
                "patch_version": patch_version,
                "patch_current": patch_current,
                "patch_total": patch_total,
            }
    finally:
        conn.close()
    
    return epoch, epoch_emails


def load_all_emails(epochs: List[int], workers: int = 4) -> Dict[Tuple[int, int], Dict]:
    """
     epoch
    
    Args:
        epochs: epoch
        workers:
    
    Returns:
        {(epoch, email_id): {id, epoch, message_id, in_reply_to, references_raw, subject, body, ...}}
         (epoch, email_id) epoch email_id
    """
    all_emails: Dict[Tuple[int, int], Dict] = {}
    
    print(f" {len(epochs)} epoch ...")
    
    pbar = tqdm(
        total=len(epochs),
        desc="[]",
        unit="epoch",
        ncols=100
    )
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(load_single_epoch_emails, epoch): epoch for epoch in epochs}
        
        for future in as_completed(futures):
            try:
                epoch, epoch_emails = future.result()
                all_emails.update(epoch_emails)
                pbar.update(1)
                pbar.set_postfix({
                    'total_emails': len(all_emails),
                    'epoch': epoch,
                    'epoch_emails': len(epoch_emails)
                })
            except Exception as e:
                epoch = futures[future]
                print(f"[ERROR] epoch {epoch} : {e}")
                pbar.update(1)
    
    pbar.close()
    print(f" {len(all_emails)} ")
    return all_emails


def build_threads_global(all_emails: Dict[Tuple[int, int], Dict]) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    """
     in_reply_to / references root_id -> [email_ids]
     epoch
    
    Args:
        all_emails: {(epoch, email_id): {id, epoch, message_id, ...}}
    
    Returns:
         {root_id: [composite_ids]}
         (epoch, email_id)
    """
    # message_id -> composite_id
    msgid_to_composite_id: Dict[str, Tuple[int, int]] = {}
    for composite_id, info in all_emails.items():
        mid = (info.get("message_id") or "").strip()
        if mid and mid not in msgid_to_composite_id:
            msgid_to_composite_id[mid] = composite_id

    # parent
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}
    for composite_id, info in all_emails.items():
        irt = (info.get("in_reply_to") or "").strip().strip("<>")
        p: Optional[Tuple[int, int]] = None

        # in_reply_to
        if irt and irt in msgid_to_composite_id:
            p = msgid_to_composite_id[irt]

        # References thread
        # References
        # if not p:
        #     refs = parse_references(info.get("references_raw"))
        #     for mid in reversed(refs):
        #         mid_clean = mid.strip().strip("<>")
        #         if mid_clean in msgid_to_composite_id:
        #             p = msgid_to_composite_id[mid_clean]
        #             break

        parent[composite_id] = p

    root_cache: Dict[Tuple[int, int], Tuple[int, int]] = {}

    def find_root(composite_id: Tuple[int, int]) -> Tuple[int, int]:
        """ in_reply_to """
        if composite_id in root_cache:
            return root_cache[composite_id]
        path: List[Tuple[int, int]] = []
        cur = composite_id
        while True:
            if cur in root_cache:
                root = root_cache[cur]
                break
            path.append(cur)
            p = parent.get(cur)
            if p is None or p not in all_emails:
                root = cur
                break
            if p in path:
                # composite_id (epoch, id)
                cycle_nodes = path[path.index(p):]
                root = min(cycle_nodes)
                break
            cur = p
        for node in path:
            root_cache[node] = root
        return root

    threads: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for composite_id in all_emails.keys():
        r = find_root(composite_id)
        threads.setdefault(r, []).append(composite_id)
    
    return threads


def assign_version_for_thread(
    thread_emails: List[Dict], 
    all_emails: Dict[Tuple[int, int], Dict],
    msgid_to_composite_id: Optional[Dict[str, Tuple[int, int]]] = None
) -> Dict[Tuple[int, int], Optional[str]]:
    """
    "" series
    
    Args:
        thread_emails: composite_id
        all_emails:
        msgid_to_composite_id: message_id -> composite_id
                             None
    
    Returns:
        {composite_id: version}
    """
    # msgid_to_composite_id
    if msgid_to_composite_id is None:
        msgid_to_composite_id = {
            (e.get("message_id") or "").strip().strip("<>"): (e.get("composite_id") or (e["epoch"], e["id"]))
            for e in thread_emails
            if e.get("message_id")
        }
    
    # patch
    patch_versions: Dict[Tuple[int, int], Optional[str]] = {}
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}
    
    for e in thread_emails:
        composite_id = e.get("composite_id") or (e["epoch"], e["id"])
        v = normalize_version(e.get("patch_version"))
        # patch_version subject
        if not v:
            subject = e.get("subject", "")
            if subject:
                v_from_subject = extract_version_from_subject(subject)
                if v_from_subject:
                    v = normalize_version(v_from_subject)
        
        determined_is_patch = e.get("determined_is_patch", False)
        
        if determined_is_patch and v:
            patch_versions[composite_id] = v
        elif not determined_is_patch and v:
            patch_versions[composite_id] = v
        
        # parent
        irt = (e.get("in_reply_to") or "").strip().strip("<>")
        if irt:
            parent_composite_id = msgid_to_composite_id.get(irt)
            parent[composite_id] = parent_composite_id
        else:
            parent[composite_id] = None

    version_for_email: Dict[Tuple[int, int], Optional[str]] = {}

    # patch
    for composite_id, v in patch_versions.items():
        version_for_email[composite_id] = v

    # composite_id -> row
    composite_id_to_row: Dict[Tuple[int, int], Dict] = {
        (e.get("composite_id") or (e["epoch"], e["id"])): e for e in thread_emails
    }

    def resolve_version(composite_id: Tuple[int, int]) -> Optional[str]:
        """"""
        if composite_id in version_for_email:
            return version_for_email[composite_id]

        path: List[Tuple[int, int]] = []
        cur = composite_id
        found_version: Optional[str] = None

        while True:
            if cur in version_for_email:
                found_version = version_for_email[cur]
                break
            
            node = composite_id_to_row.get(cur)
            if node is None:
                break
            
            v_self = normalize_version(node.get("patch_version"))
            if v_self:
                found_version = v_self
                break
            
            if not v_self:
                subject = node.get("subject", "")
                if subject:
                    v_from_subject = extract_version_from_subject(subject)
                    if v_from_subject:
                        v_self = normalize_version(v_from_subject)
                        if v_self:
                            found_version = v_self
                            break
            
            p = parent.get(cur)
            if p is None or p in path:
                break
            
            path.append(cur)
            cur = p

        for nid in path:
            version_for_email[nid] = found_version
        version_for_email[composite_id] = found_version
        return found_version

    for e in thread_emails:
        composite_id = e.get("composite_id") or (e["epoch"], e["id"])
        if composite_id not in version_for_email:
            resolve_version(composite_id)

    # None None
    existing_versions = {v for v in version_for_email.values() if v is not None}
    if len(existing_versions) == 1:
        the_v = next(iter(existing_versions))
        for composite_id, v in version_for_email.items():
            if v is None:
                version_for_email[composite_id] = the_v
    
    # subject
    for e in thread_emails:
        composite_id = e.get("composite_id") or (e["epoch"], e["id"])
        if composite_id not in version_for_email or version_for_email[composite_id] is None:
            subject = e.get("subject", "")
            if subject:
                v_from_subject = extract_version_from_subject(subject)
                if v_from_subject:
                    v_normalized = normalize_version(v_from_subject)
                    if v_normalized:
                        version_for_email[composite_id] = v_normalized
                        if len(existing_versions) == 1 and v_normalized == next(iter(existing_versions)):
                            for composite_id2, v2 in version_for_email.items():
                                if v2 is None:
                                    version_for_email[composite_id2] = v_normalized

    return version_for_email


def process_single_thread(
    thread_data: Tuple[Tuple[int, int], List[Tuple[int, int]], Dict[Tuple[int, int], Dict], Dict[str, Tuple[int, int]]]
) -> List[Tuple[Dict, List[Tuple[int, int, int, int]]]]:
    """
     series
    
    Args:
        thread_data: (root_composite_id, member_composite_ids, all_emails, global_msgid_to_composite_id)
         (epoch, email_id)
        global_msgid_to_composite_id: message_id -> composite_id
    
    Returns:
        [(series_info, email_updates), ...] # series
        series_info: {representative_id, representative_epoch, has_cover, patch_ids, comment_ids, version}
        email_updates: [(email_id, epoch, is_patch, is_cover, series_id)]
    """
    root_composite_id, member_composite_ids, all_emails, global_msgid_to_composite_id = thread_data
    
    thread_emails: List[Dict] = [all_emails[composite_id] for composite_id in member_composite_ids if composite_id in all_emails]
    
    if not thread_emails:
        return [] # series
    
    # composite_id
    for e in thread_emails:
        if "composite_id" not in e:
            e["composite_id"] = (e["epoch"], e["id"])
    
    # parent_id
    # message_id
    msgid_to_composite_id_thread = {
        (e["message_id"] or "").strip().strip("<>"): e["composite_id"]
        for e in thread_emails
        if e["message_id"]
    }
    
    combined_msgid_map = {**global_msgid_to_composite_id, **msgid_to_composite_id_thread}
    
    for e in thread_emails:
        parent_composite_id = None
        irt = (e["in_reply_to"] or "").strip().strip("<>")
        if irt and irt in combined_msgid_map:
            parent_composite_id = combined_msgid_map[irt]

        # References thread
        # References
        # else:
        #     refs = parse_references(e.get("references_raw"))
        #     for mid in reversed(refs):
        #         mid_clean = mid.strip().strip("<>")
        #         if mid_clean in combined_msgid_map:
        #             parent_composite_id = combined_msgid_map[mid_clean]
        #             break
        e["parent_composite_id"] = parent_composite_id

    version_for_email = assign_version_for_thread(thread_emails, all_emails, combined_msgid_map)

    version_to_emails: Dict[Optional[str], List[Dict]] = {}
    for e in thread_emails:
        composite_id = e.get("composite_id") or (e["epoch"], e["id"])
        v = version_for_email.get(composite_id)
        version_to_emails.setdefault(v, []).append(e)

    results = []
    
    # series
    for v, emails_in_version in version_to_emails.items():
        patch_ids: List[int] = []
        comment_ids: List[int] = []

        for e in emails_in_version:
            subj = e["subject"]
            determined_is_patch = e.get("determined_is_patch", False)
            
            if is_comment_email(subj):
                new_is_patch = 0
                comment_ids.append(e["id"]) # email_id
            else:
                new_is_patch = 1 if determined_is_patch else 0
                if new_is_patch:
                    patch_ids.append(e["id"]) # email_id
                else:
                    comment_ids.append(e["id"]) # email_id
            
            e["new_is_patch"] = new_is_patch

        if not patch_ids and not comment_ids:
            continue

        # cover
        cover_ids = [
            e["id"]
            for e in emails_in_version
            if (e.get("new_is_patch") and (e.get("patch_current") == 0))
        ]
        has_cover = 1 if cover_ids else 0

        representative_id: Optional[int] = None
        representative_epoch: Optional[int] = None
        
        # 1: cover letter
        if cover_ids:
            covers = [e for e in emails_in_version if e["id"] in cover_ids]
            if covers:
                covers.sort(
                    key=lambda x: (
                        x.get("timestamp") if x.get("timestamp") is not None else 0,
                        x["id"],
                    )
                )
                representative_id = covers[0]["id"]
                representative_epoch = covers[0]["epoch"]
        
        # 2: patch
        if not representative_id and patch_ids:
            patches = [e for e in emails_in_version if e["id"] in patch_ids]
            if patches:
                patches.sort(
                    key=lambda x: (
                        x.get("patch_current")
                        if x.get("patch_current") is not None
                        else 999999,
                        x.get("timestamp") if x.get("timestamp") is not None else 0,
                        x["id"],
                    )
                )
                representative_id = patches[0]["id"]
                representative_epoch = patches[0]["epoch"]
        
        # 3: comment
        if not representative_id and comment_ids:
            comments = [e for e in emails_in_version if e["id"] in comment_ids]
            if comments:
                comments.sort(
                    key=lambda x: (
                        x.get("timestamp") if x.get("timestamp") is not None else 0,
                        x["id"],
                    )
                )
                representative_id = comments[0]["id"]
                representative_epoch = comments[0]["epoch"]
        
        if representative_id is None:
            continue

        # series
        series_version = v
        if not series_version and patch_ids:
            first_patch = next((e for e in emails_in_version if e["id"] in patch_ids), None)
            if first_patch:
                series_version = normalize_version(first_patch.get("patch_version"))
                if not series_version:
                    series_version = extract_version_from_subject(first_patch.get("subject", ""))

        # series
        series_info = {
            "representative_id": representative_id,
            "representative_epoch": representative_epoch,
            "has_cover": has_cover,
            "patch_ids": sorted(patch_ids),
            "comment_ids": sorted(comment_ids),
            "version": series_version,
        }
        
        # series_id
        email_updates = []
        for e in emails_in_version:
            eid = e["id"]
            epoch = e["epoch"]
            new_is_patch = int(e.get("new_is_patch", 0))
            new_is_cover = 1 if (new_is_patch and e.get("patch_current") == 0) else 0
            email_updates.append((eid, epoch, new_is_patch, new_is_cover, None)) # series_id
        
        results.append((series_info, email_updates))
    
    # series
    return results


def build_series_global(epochs: List[int], clear_relations: bool = False, workers: int = 8) -> None:
    """
     series epoch
    
    Args:
        epochs: epoch [0, 1, 2, ..., 18]
        clear_relations: series
        workers:
    """
    print(f"\n=== series {len(epochs)} epoch ===")
    
    # epoch emails series_id
    print(" epoch emails series_id ...")
    for epoch in epochs:
        ensure_emails_series_id_column(epoch)
    
    all_emails = load_all_emails(epochs, workers=workers)
    
    if not all_emails:
        print("[WARN] ")
        return
    
    print("\n...")
    threads = build_threads_global(all_emails)
    print(f" {len(threads)} ")
    
    # message_id -> composite_id assign_version_for_thread
    # build_threads_global
    # process_single_thread
    print(" message_id ...")
    global_msgid_to_composite_id: Dict[str, Tuple[int, int]] = {}
    for composite_id, info in all_emails.items():
        mid = (info.get("message_id") or "").strip().strip("<>")
        if mid and mid not in global_msgid_to_composite_id:
            global_msgid_to_composite_id[mid] = composite_id
    print(f" {len(global_msgid_to_composite_id):,} message_id ")
    
    # series
    global_db_path = get_global_series_db_path()
    global_conn = sqlite3.connect(global_db_path)
    try:
        ensure_global_series_schema(global_conn)
        
        if clear_relations:
            print(" series ...")
            
            # series
            global_conn.execute("DELETE FROM series")
            global_conn.commit()
            
            # epoch emails series_id
            def clear_epoch_series(epoch: int) -> Tuple[int, int]:
                """ epoch series_id"""
                db_path = get_db_path(epoch)
                if not db_path.exists():
                    return epoch, 0
                
                conn = sqlite3.connect(db_path)
                try:
                    cur = conn.cursor()
                    cur.execute("SELECT COUNT(*) FROM emails WHERE series_id IS NOT NULL")
                    count_before = cur.fetchone()[0]
                    
                    cur.execute("UPDATE emails SET series_id = NULL, is_patch = 0, is_cover = 0")
                    conn.commit()
                    return epoch, count_before
                finally:
                    conn.close()
            
            pbar_clear = tqdm(
                total=len(epochs),
                desc="[ series ]",
                unit="epoch",
                ncols=100
            )
            
            total_cleared = 0
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(clear_epoch_series, epoch): epoch for epoch in epochs}
                
                for future in as_completed(futures):
                    try:
                        epoch, cleared_count = future.result()
                        total_cleared += cleared_count
                        pbar_clear.update(1)
                        pbar_clear.set_postfix({
                            'total_cleared': total_cleared,
                            'epoch': epoch,
                            'epoch_cleared': cleared_count
                        })
                    except Exception as e:
                        epoch = futures[future]
                        print(f"[ERROR] epoch {epoch} series : {e}")
                        pbar_clear.update(1)
            
            pbar_clear.close()
            print(f" {total_cleared} series_id")
        
        print(f"\n {workers} ...")
        thread_items = list(threads.items())
        total_threads = len(thread_items)
        
        # message_id
        thread_data_list = [
            (root_id, member_ids, all_emails, global_msgid_to_composite_id)
            for root_id, member_ids in thread_items
        ]
        
        all_series_results = [] # series
        email_to_series: Dict[Tuple[int, int], int] = {} # (epoch, email_id) -> series_id
        
        pbar = tqdm(
            total=total_threads,
            desc="[]",
            unit="thread",
            ncols=100
        )
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_single_thread, data): data for data in thread_data_list}
            
            for future in as_completed(futures):
                try:
                    thread_results = future.result()
                    
                    # thread_results (series_info, email_updates)
                    for series_info, email_updates in thread_results:
                        if series_info:
                            all_series_results.append((series_info, email_updates))
                    
                    pbar.update(1)
                except Exception as e:
                    print(f"[ERROR] : {e}")
                    import traceback
                    traceback.print_exc()
                    pbar.update(1)
        
        pbar.close()
        
        # series
        print(f"\n {len(all_series_results)} series ...")
        global_cur = global_conn.cursor()
        
        # (epoch, email_id) -> series_id
        pbar_insert = tqdm(
            total=len(all_series_results),
            desc="[ series]",
            unit="series",
            ncols=100
        )
        
        for series_info, email_updates in all_series_results:
            global_cur.execute("""
                INSERT INTO series (
                    representative_email_id,
                    representative_epoch,
                    has_cover,
                    patch_email_ids,
                    comment_email_ids,
                    patch_version
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                series_info["representative_id"],
                series_info["representative_epoch"],
                series_info["has_cover"],
                json.dumps(series_info["patch_ids"], ensure_ascii=False),
                json.dumps(series_info["comment_ids"], ensure_ascii=False),
                series_info["version"],
            ))
            series_id = global_cur.lastrowid
            
            # series_id
            for eid, epoch, is_patch, is_cover, _ in email_updates:
                email_to_series[(epoch, eid)] = series_id
            
            pbar_insert.update(1)
        
        pbar_insert.close()
        global_conn.commit()
        
        # epoch emails series_id
        print("\n epoch emails series_id...")
        
        # epoch
        updates_by_epoch: Dict[int, List[Tuple[int, int, int, int]]] = {}  # epoch -> [(email_id, series_id, is_patch, is_cover)]
        
        for series_info, email_updates in all_series_results:
            for eid, epoch, is_patch, is_cover, _ in email_updates:
                series_id = email_to_series.get((epoch, eid))
                if series_id:
                    if epoch not in updates_by_epoch:
                        updates_by_epoch[epoch] = []
                    updates_by_epoch[epoch].append((eid, series_id, is_patch, is_cover))
        
        # epoch
        total_emails_to_update = sum(len(updates) for updates in updates_by_epoch.values())
        pbar_update = tqdm(
            total=total_emails_to_update,
            desc="[ emails]",
            unit="email",
            ncols=100
        )
        
        for epoch, updates in updates_by_epoch.items():
            db_path = get_db_path(epoch)
            if not db_path.exists():
                continue
            
            conn = sqlite3.connect(db_path)
            try:
                cur = conn.cursor()
                for eid, series_id, is_patch, is_cover in updates:
                    cur.execute("""
                        UPDATE emails
                        SET series_id = ?,
                            is_patch = ?,
                            is_cover = ?
                        WHERE id = ?
                    """, (series_id, is_patch, is_cover, eid))
                    pbar_update.update(1)
                conn.commit()
            finally:
                conn.close()
        
        pbar_update.close()
        
        print(f"\n {len(all_series_results)} series")
        
    finally:
        global_conn.close()


def get_global_series_conn() -> sqlite3.Connection:
    """ series WAL """
    global_db_path = get_global_series_db_path()
    conn = sqlite3.connect(global_db_path, timeout=30.0) #
    conn.row_factory = sqlite3.Row
    # WAL
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError:
        # WAL
        pass
    ensure_global_series_schema(conn)
    return conn


def build_series(epoch_spec: Optional[str] = None, clear_relations: bool = False, workers: int = 8) -> None:
    """
     series
    
    Args:
        epoch_spec: epoch "0-18" epoch
        clear_relations: series
        workers:
    """
    from lkml_preprocess import parse_epoch_spec
    
    if epoch_spec:
        selected_epochs: Optional[List[int]] = sorted(parse_epoch_spec(epoch_spec))
        print(f" series epoch: {selected_epochs}")
    else:
        selected_epochs = None
        print(" epoch SQLite epoch series ")
    
    if selected_epochs is None:
        epochs: List[int] = []
        for db_file in STRUCTURE_DIR.glob("lkml_epoch_*.sqlite"):
            m = re.search(r"epoch_(\d+)", db_file.name)
            if m:
                epochs.append(int(m.group(1)))
        epochs = sorted(epochs)
    else:
        epochs = selected_epochs
    
    if not epochs:
        print("[ERROR] epoch")
        return
    
    build_series_global(epochs, clear_relations=clear_relations, workers=workers)
