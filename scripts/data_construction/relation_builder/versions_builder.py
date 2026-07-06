"""
versions_builder.py

 cover letter processors.version.link_based_analyzer
 body_matchsubject_match thread_match
"""

from __future__ import annotations

import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

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
from lkml_preprocess import parse_epoch_spec
from processors.sqlite.db_utils import get_db_path

# series_builder
from .series_builder import (
    get_global_series_conn,
    get_global_series_db_path,
    ensure_global_series_schema,
    is_comment_email,
    determine_is_patch,
    normalize_version,
    extract_version_from_subject,
)

# series_id -> representative_subject subject_match
_SERIES_SUBJECTS: Dict[int, Optional[str]] = {}

# (epoch, version) -> [series_id] subject_match
_SUBJECT_MATCH_INDEX: Dict[Tuple[int, Optional[str]], List[int]] = {}

# series_id -> (author, timestamp, base_subject) subject_fuzz_match
_SERIES_FUZZ_DATA: Dict[int, Tuple[Optional[str], Optional[int], Optional[str]]] = {}

# series_id -> (representative_email_id, representative_epoch, body) represent_body_match
_SERIES_REP_BODY: Dict[int, Tuple[int, int, Optional[str]]] = {}

MATCH_BODY = 0 # body_match - cover letter body
MATCH_SUBJECT = 1 # subject_match -
MATCH_THREAD = 2 # thread_match - in_reply_to
MATCH_SUBJECT_FUZZ = 3 # subject_fuzz_match -
MATCH_REP_BODY = 4 # represent_body_match - representative letter body

ALL_MATCH_METHODS = {MATCH_BODY, MATCH_SUBJECT, MATCH_THREAD, MATCH_SUBJECT_FUZZ, MATCH_REP_BODY}
MATCH_METHOD_NAMES = {
    MATCH_BODY: "body_match",
    MATCH_SUBJECT: "subject_match",
    MATCH_THREAD: "thread_match",
    MATCH_SUBJECT_FUZZ: "subject_fuzz_match",
    MATCH_REP_BODY: "represent_body_match",
}

# series schema
def ensure_series_schema(conn: sqlite3.Connection) -> None:
    """ series ensure_global_series_schema"""
    # series
    global_db_path = get_global_series_db_path()
    if hasattr(conn, 'database') and str(conn.database) == str(global_db_path):
        ensure_global_series_schema(conn)
    else:
        # epoch series schema
        pass


def get_previous_version(version: Optional[str]) -> Optional[str]:
    """
    
    v2 -> v1
    v1 -> None
    None -> None
    """
    if not version:
        return None
    v = normalize_version(version)
    if not v:
        return None
    m = re.search(r"v(\d+)", v)
    if not m:
        return None
    num = int(m.group(1))
    if num <= 1:
        return None
    return f"v{num - 1}"


def extract_base_subject(subject: str) -> str:
    """ [PATCH vX N/M] Re: """
    if not subject:
        return ""
    # Re: flags
    base = re.sub(r'^re\s*:\s*', '', subject, flags=re.IGNORECASE)
    # [PATCH vX N/M]
    base = re.sub(r'\[PATCH\s+v?\d*\s*\d*/\d*\]\s*', '', base, flags=re.IGNORECASE)
    base = ' '.join(base.split())
    return base.strip()


def body_match_previous_version(
    series_id: int,
    representative_epoch: int,
    cover_email_id: Optional[int],
    cover_body: Optional[str],
    current_version: Optional[str],
    all_epochs: List[int],
) -> Optional[int]:
    """
     body series
     series epoch
    
    Returns:
         series_id None
    """
    if not cover_email_id or not cover_body:
        return None
    
    # link_based_analyzer
    from processors.version.link_based_analyzer import extract_version_links_from_body
    
    version_links = extract_version_links_from_body(cover_body)
    if not version_links:
        return None
    
    prev_version = get_previous_version(current_version)
    
    # v2 v1 None
    candidate_versions = []
    if prev_version:
        candidate_versions.append(prev_version)
    
    # v2 None
    if current_version:
        normalized_current = normalize_version(current_version)
        if normalized_current == "v2":
            candidate_versions.append(None)
    
    if not candidate_versions:
        return None
    
    # epoch epoch
    search_epochs = []
    if representative_epoch > 0:
        search_epochs.append(representative_epoch - 1)
    # epoch epoch
    for ep in all_epochs:
        if ep != representative_epoch and ep not in search_epochs:
            search_epochs.append(ep)
    
    # link_based_analyzer.py
    from processors.version.link_based_analyzer import (
        find_email_by_patchwork_id,
        find_email_by_message_id,
        find_email_by_patchwork_url,
    )
    
    for candidate_version in candidate_versions:
        # None version_links
        if candidate_version is not None and candidate_version not in version_links:
            continue
        
        if candidate_version is not None:
            links = version_links[candidate_version]
        else:
            # None
            # version_links
            # None body_match body None
            continue
        
        # series
        for link in links:
            linked_email_id = None
            linked_epoch = None
            
            # epoch
            for search_epoch in search_epochs:
                search_db_path = get_db_path(search_epoch)
                if not search_db_path.exists():
                    continue
                
                search_conn = sqlite3.connect(search_db_path)
                search_conn.row_factory = sqlite3.Row
                
                try:
                    # patchwork URL ID
                    if 'patchwork.kernel.org' in link or link.isdigit():
                        linked_email_id = find_email_by_patchwork_url(search_epoch, search_conn, link)
                        if not linked_email_id and link.isdigit():
                            linked_email_id = find_email_by_patchwork_id(search_epoch, search_conn, link)
                    
                    # patch.msgid.link
                    if not linked_email_id and 'patch.msgid.link' in link:
                        from processors.version.link_based_analyzer import MSGID_LINK_PATTERN
                        msgid_match = MSGID_LINK_PATTERN.search(link)
                        if msgid_match:
                            msgid_path = msgid_match.group(1)
                            msgid = msgid_path
                            if not msgid.startswith('<'):
                                msgid = f'<{msgid}>'
                            linked_email_id = find_email_by_message_id(search_epoch, search_conn, msgid)
                    
                    # message ID
                    if not linked_email_id and link.startswith('<') and link.endswith('>'):
                        linked_email_id = find_email_by_message_id(search_epoch, search_conn, link)
                    
                    # link message-id @
                    if not linked_email_id and '@' in link and not link.startswith('http'):
                        msgid = link
                        if not msgid.startswith('<'):
                            msgid = f'<{msgid}>'
                        linked_email_id = find_email_by_message_id(search_epoch, search_conn, msgid)
                    
                    if linked_email_id:
                        linked_epoch = search_epoch
                        break
                finally:
                    search_conn.close()
            
            # series series_id
            if linked_email_id:
                search_db_path = get_db_path(linked_epoch)
                search_conn = sqlite3.connect(search_db_path)
                search_conn.row_factory = sqlite3.Row
                
                try:
                    search_cur = search_conn.cursor()
                    search_cur.execute("""
                        SELECT e.series_id, e.subject, e.patch_version
                        FROM emails e
                        WHERE e.id = ? AND e.series_id IS NOT NULL
                        LIMIT 1
                    """, (linked_email_id,))
                    row = search_cur.fetchone()
                    if row and row[0]:
                        found_series_id = row[0]
                        # series series
                        global_series_conn = get_global_series_conn()
                        try:
                            global_cur = global_series_conn.cursor()
                            global_cur.execute("""
                                SELECT patch_version FROM series
                                WHERE id = ?
                            """, (found_series_id,))
                            series_row = global_cur.fetchone()
                            series_version = series_row['patch_version'] if series_row else None
                        finally:
                            global_series_conn.close()
                        
                        email_subject = row[1] or ""
                        email_version = row[2]
                        email_version_from_subject = extract_version_from_subject(email_subject)
                        if email_version_from_subject:
                            email_version_from_subject = normalize_version(email_version_from_subject)
                        
                        effective_version = normalize_version(email_version) or email_version_from_subject
                        target_version = normalize_version(series_version) or effective_version
                        
                        # series
                        if candidate_version is not None:
                            if target_version == candidate_version or effective_version == candidate_version:
                                return found_series_id
                        else:
                            # None
                            if target_version is None and effective_version is None:
                                return found_series_id
                finally:
                    search_conn.close()
    
    return None


def subject_match_previous_version(
    series_id: int,
    representative_epoch: int,
    representative_subject: str,
    current_version: Optional[str],
    all_epochs: List[int],
) -> Optional[int]:
    """
     subject series
    
    Returns:
         series_id None
    """
    if not representative_subject:
        return None
    
    prev_version = get_previous_version(current_version)
    
    candidate_versions = []
    if prev_version:
        candidate_versions.append(prev_version)
    if current_version:
        normalized_current = normalize_version(current_version)
        if normalized_current == "v2":
            candidate_versions.append(None)
    
    if not candidate_versions:
        return None
    
    # subject
    base_subject = extract_base_subject(representative_subject)
    if not base_subject:
        return None
    
    escaped_base = re.escape(base_subject)
    
    version_patterns: Dict[Optional[str], List[re.Pattern]] = {}
    escaped_base = re.escape(base_subject)
    
    for candidate_version in candidate_versions:
        patterns = []
        if candidate_version is not None:
            escaped_version = re.escape(candidate_version)
            if candidate_version == "v1":
                # v1 [PATCH v1 N/M] [PATCH N/M]
                patterns.append(re.compile(
                    r"\[PATCH\s+" + escaped_version + r"\s+\d+/\d+\]\s*" + escaped_base,
                    re.IGNORECASE
                ))
                patterns.append(re.compile(
                    r"\[PATCH\s+\d+/\d+\]\s*" + escaped_base,
                    re.IGNORECASE
                ))
            else:
                patterns.append(re.compile(
                    r"\[PATCH\s+" + escaped_version + r"\s+\d+/\d+\]\s*" + escaped_base,
                    re.IGNORECASE
                ))
        else:
            # None subject
            patterns.append(re.compile(
                r"\[PATCH\s+\d+/\d+\]\s*" + escaped_base,
                re.IGNORECASE
            ))
        version_patterns[candidate_version] = patterns
    
    # epoch epoch
    search_epochs = []
    if representative_epoch > 0:
        search_epochs.append(representative_epoch - 1)
    for ep in all_epochs:
        if ep != representative_epoch and ep not in search_epochs:
            search_epochs.append(ep)
    
    # series SQL
    global _SUBJECT_MATCH_INDEX
    for search_epoch in search_epochs:
        for candidate_version in candidate_versions:
            # series
            key = (search_epoch, candidate_version)
            candidate_series_ids = _SUBJECT_MATCH_INDEX.get(key, [])
            
            patterns = version_patterns.get(candidate_version, [])
            if not patterns:
                continue
            
            # series
            for cid in candidate_series_ids:
                email_subject = _SERIES_SUBJECTS.get(cid) or ""
                if not email_subject:
                    continue
                
                for compiled_regex in patterns:
                    try:
                        if compiled_regex.search(email_subject):
                            return cid
                    except Exception:
                        continue
    
    return None


def thread_match_previous_version(
    series_id: int,
    representative_epoch: int,
    representative_email_id: int,
    current_version: Optional[str],
    all_epochs: List[int],
) -> Optional[int]:
    """
     thread in_reply_to series
    
    1. series representative
    2. in_reply_to thread
    3. epoch thread series
    4. series series_id
    
    Returns:
         series_id None
    """
    if not current_version:
        return None
    
    prev_version = get_previous_version(current_version)
    
    # v2 v1 None
    candidate_versions = []
    if prev_version:
        candidate_versions.append(prev_version)
    
    # v2 None
    normalized_current = normalize_version(current_version)
    if normalized_current == "v2":
        candidate_versions.append(None)
    
    if not candidate_versions:
        return None
    
    # epoch representative
    epoch_db_path = get_db_path(representative_epoch)
    if not epoch_db_path.exists():
        return None
    
    epoch_conn = sqlite3.connect(epoch_db_path)
    epoch_conn.row_factory = sqlite3.Row
    try:
        epoch_cur = epoch_conn.cursor()
        epoch_cur.execute("""
            SELECT message_id, in_reply_to, references_raw FROM emails WHERE id = ?
        """, (representative_email_id,))
        row = epoch_cur.fetchone()
        if not row:
            return None
        
        rep_message_id = row[0]
        rep_in_reply_to = row[1]
        rep_references = row[2]
        
        for candidate_version in candidate_versions:
            # 1: in_reply_to series_id epoch
            if rep_in_reply_to:
                in_reply_to_clean = rep_in_reply_to.strip().strip("<>")
                if in_reply_to_clean:
                    # epoch message_id
                    for search_epoch in all_epochs:
                        search_db_path = get_db_path(search_epoch)
                        if not search_db_path.exists():
                            continue
                        
                        search_conn = sqlite3.connect(search_db_path)
                        search_conn.row_factory = sqlite3.Row
                        try:
                            search_cur = search_conn.cursor()
                            search_cur.execute("""
                                SELECT series_id FROM emails 
                                WHERE (message_id = ? OR message_id = ?) AND series_id IS NOT NULL
                                LIMIT 1
                            """, (in_reply_to_clean, f"<{in_reply_to_clean}>"))
                            
                            target_email = search_cur.fetchone()
                            if target_email and target_email['series_id']:
                                # series
                                global_series_conn = get_global_series_conn()
                                try:
                                    global_cur = global_series_conn.cursor()
                                    if candidate_version is not None:
                                        global_cur.execute("""
                                            SELECT patch_version FROM series
                                            WHERE id = ? AND patch_version = ?
                                        """, (target_email['series_id'], candidate_version))
                                    else:
                                        global_cur.execute("""
                                            SELECT patch_version FROM series
                                            WHERE id = ? AND patch_version IS NULL
                                        """, (target_email['series_id'],))
                                    target_series = global_cur.fetchone()
                                    if target_series:
                                        return target_email['series_id']
                                finally:
                                    global_series_conn.close()
                        finally:
                            search_conn.close()
            
            # 2: thread thread series
            # in_reply_to thread
            thread_message_ids = set()
            visited_message_ids = set()
            
            # in_reply_to
            if rep_message_id:
                rep_msgid_clean = rep_message_id.strip().strip("<>")
                thread_message_ids.add(rep_msgid_clean)
            
            # in_reply_to
            current_msgid = rep_in_reply_to
            max_depth = 50 #
            depth = 0
            
            while current_msgid and depth < max_depth:
                depth += 1
                current_msgid_clean = current_msgid.strip().strip("<>")
                
                if current_msgid_clean in visited_message_ids:
                    break
                visited_message_ids.add(current_msgid_clean)
                thread_message_ids.add(current_msgid_clean)
                
                # epoch message_id
                found_email = False
                for search_epoch in all_epochs:
                    search_db_path = get_db_path(search_epoch)
                    if not search_db_path.exists():
                        continue
                    
                    search_conn = sqlite3.connect(search_db_path)
                    search_conn.row_factory = sqlite3.Row
                    try:
                        search_cur = search_conn.cursor()
                        search_cur.execute("""
                            SELECT in_reply_to FROM emails 
                            WHERE (message_id = ? OR message_id = ?)
                            LIMIT 1
                        """, (current_msgid_clean, f"<{current_msgid_clean}>"))
                        
                        row = search_cur.fetchone()
                        if row and row['in_reply_to']:
                            current_msgid = row['in_reply_to']
                            found_email = True
                            break
                    finally:
                        search_conn.close()
                
                if not found_email:
                    break
            
            if not thread_message_ids:
                continue
            
            # epoch thread series_id
            candidate_series_ids = set()
            for search_epoch in all_epochs:
                search_db_path = get_db_path(search_epoch)
                if not search_db_path.exists():
                    continue
                
                search_conn = sqlite3.connect(search_db_path)
                search_conn.row_factory = sqlite3.Row
                try:
                    search_cur = search_conn.cursor()
                    # message_id thread
                    placeholders = ','.join(['?'] * len(thread_message_ids))
                    msgid_list = []
                    for msgid in thread_message_ids:
                        msgid_list.append(msgid)
                        msgid_list.append(f"<{msgid}>")
                    
                    # message_id IN (?, ?, ...) OR message_id IN (?, ?, ...)
                    # SQLite OR
                    conditions = []
                    params = []
                    for msgid in thread_message_ids:
                        conditions.append("(message_id = ? OR message_id = ?)")
                        params.append(msgid)
                        params.append(f"<{msgid}>")
                    
                    query = f"""
                        SELECT series_id FROM emails 
                        WHERE ({' OR '.join(conditions)}) 
                          AND series_id IS NOT NULL
                    """
                    
                    search_cur.execute(query, params)
                    
                    for email_row in search_cur.fetchall():
                        if email_row['series_id'] and email_row['series_id'] != series_id:
                            candidate_series_ids.add(email_row['series_id'])
                finally:
                    search_conn.close()
            
            if not candidate_series_ids:
                continue
            
            # series series
            global_series_conn = get_global_series_conn()
            try:
                global_cur = global_series_conn.cursor()
                placeholders = ','.join(['?'] * len(candidate_series_ids))
                if candidate_version is not None:
                    global_cur.execute(f"""
                        SELECT id FROM series
                        WHERE id IN ({placeholders}) AND patch_version = ?
                    """, list(candidate_series_ids) + [candidate_version])
                else:
                    global_cur.execute(f"""
                        SELECT id FROM series
                        WHERE id IN ({placeholders}) AND patch_version IS NULL
                    """, list(candidate_series_ids))
                
                result = global_cur.fetchone()
                if result:
                    return result['id']
            finally:
                global_series_conn.close()
    finally:
        epoch_conn.close()
    
    return None


def subject_fuzz_match_previous_version(
    series_id: int,
    representative_epoch: int,
    representative_subject: str,
    representative_author: Optional[str],
    representative_timestamp: Optional[int],
    current_version: Optional[str],
    all_epochs: List[int],
    similarity_threshold: float = 0.9,
    max_time_diff_days: Optional[float] = None,
) -> Optional[int]:
    """
     subject series
    
    1.
    2. series series
    3. max_time_diff_days
    4. subject >= similarity_threshold
    
     epoch epoch
    
    Args:
        similarity_threshold: 0.9 (90%)
        max_time_diff_days: None
    
    Returns:
         series_id None
    """
    if not representative_subject:
        return None
    
    prev_version = get_previous_version(current_version)
    
    candidate_versions = []
    if prev_version:
        candidate_versions.append(prev_version)
    if current_version:
        normalized_current = normalize_version(current_version)
        if normalized_current == "v2":
            candidate_versions.append(None)
    
    if not candidate_versions:
        return None
    
    # subject
    base_subject = extract_base_subject(representative_subject)
    if not base_subject:
        return None
    
    current_author_normalized = representative_author.lower().strip() if representative_author else ""
    
    # difflib.SequenceMatcher
    from difflib import SequenceMatcher
    base_subject_lower = base_subject.lower()
    
    max_time_diff_seconds = max_time_diff_days * 86400 if max_time_diff_days is not None else None
    
    # epoch epoch
    search_epochs = []
    if representative_epoch > 0:
        search_epochs.append(representative_epoch - 1)
    search_epochs.append(representative_epoch) # epoch
    
    # series
    global _SUBJECT_MATCH_INDEX, _SERIES_FUZZ_DATA
    
    for search_epoch in search_epochs:
        for candidate_version in candidate_versions:
            key = (search_epoch, candidate_version)
            candidate_series_ids = _SUBJECT_MATCH_INDEX.get(key, [])
            
            for cid in candidate_series_ids:
                if cid == series_id:
                    continue
                
                # series
                fuzz_data = _SERIES_FUZZ_DATA.get(cid)
                if not fuzz_data:
                    continue
                
                cand_author, cand_timestamp, cand_base_subject = fuzz_data
                
                if current_author_normalized:
                    cand_author_normalized = cand_author.lower().strip() if cand_author else ""
                    if cand_author_normalized != current_author_normalized:
                        continue
                
                if representative_timestamp is not None and cand_timestamp is not None:
                    if cand_timestamp >= representative_timestamp:
                        continue
                    
                    if max_time_diff_seconds is not None:
                        time_diff = representative_timestamp - cand_timestamp
                        if time_diff > max_time_diff_seconds:
                            continue
                
                # subject
                if not cand_base_subject:
                    continue
                
                len_ratio = len(cand_base_subject) / len(base_subject) if base_subject else 0
                if len_ratio < 0.5 or len_ratio > 2.0:
                    continue
                
                similarity = SequenceMatcher(None, base_subject_lower, cand_base_subject.lower()).ratio()
                if similarity >= similarity_threshold:
                    return cid
    
    return None


def represent_body_match_previous_version(
    series_id: int,
    representative_epoch: int,
    representative_email_id: int,
    representative_body: Optional[str],
    current_version: Optional[str],
    all_epochs: List[int],
) -> Optional[int]:
    """
     representative letter body series
    
     body_match
    - body_match cover letter body
    - represent_body_match representative letter body cover letter cover patch
    - cover letter representative letter
    
    Returns:
         series_id None
    """
    if not representative_email_id or not representative_body:
        return None
    
    # link_based_analyzer
    from processors.version.link_based_analyzer import extract_version_links_from_body
    
    version_links = extract_version_links_from_body(representative_body)
    if not version_links:
        return None
    
    prev_version = get_previous_version(current_version)
    
    candidate_versions = []
    if prev_version:
        candidate_versions.append(prev_version)
    if current_version:
        normalized_current = normalize_version(current_version)
        if normalized_current == "v2":
            candidate_versions.append(None)
    
    if not candidate_versions:
        return None
    
    # epoch epoch
    search_epochs = []
    if representative_epoch > 0:
        search_epochs.append(representative_epoch - 1)
    search_epochs.append(representative_epoch) # epoch
    
    from processors.version.link_based_analyzer import (
        find_email_by_patchwork_id,
        find_email_by_message_id,
        find_email_by_patchwork_url,
    )
    
    for candidate_version in candidate_versions:
        if candidate_version is not None and candidate_version not in version_links:
            continue
        
        if candidate_version is not None:
            links = version_links[candidate_version]
        else:
            continue
        
        # series
        for link in links:
            linked_email_id = None
            linked_epoch = None
            
            # epoch
            for search_epoch in search_epochs:
                search_db_path = get_db_path(search_epoch)
                if not search_db_path.exists():
                    continue
                
                search_conn = sqlite3.connect(search_db_path)
                search_conn.row_factory = sqlite3.Row
                
                try:
                    # patchwork URL ID
                    if 'patchwork.kernel.org' in link or link.isdigit():
                        linked_email_id = find_email_by_patchwork_url(search_epoch, search_conn, link)
                        if not linked_email_id and link.isdigit():
                            linked_email_id = find_email_by_patchwork_id(search_epoch, search_conn, link)
                    
                    # patch.msgid.link
                    if not linked_email_id and 'patch.msgid.link' in link:
                        from processors.version.link_based_analyzer import MSGID_LINK_PATTERN
                        msgid_match = MSGID_LINK_PATTERN.search(link)
                        if msgid_match:
                            msgid_path = msgid_match.group(1)
                            msgid = msgid_path
                            if not msgid.startswith('<'):
                                msgid = f'<{msgid}>'
                            linked_email_id = find_email_by_message_id(search_epoch, search_conn, msgid)
                    
                    # message ID
                    if not linked_email_id and link.startswith('<') and link.endswith('>'):
                        linked_email_id = find_email_by_message_id(search_epoch, search_conn, link)
                    
                    if not linked_email_id and '@' in link and not link.startswith('http'):
                        msgid = link
                        if not msgid.startswith('<'):
                            msgid = f'<{msgid}>'
                        linked_email_id = find_email_by_message_id(search_epoch, search_conn, msgid)
                    
                    if linked_email_id:
                        linked_epoch = search_epoch
                        break
                finally:
                    search_conn.close()
            
            # series_id representative letter
            if linked_email_id and linked_epoch is not None:
                search_db_path = get_db_path(linked_epoch)
                search_conn = sqlite3.connect(search_db_path)
                search_conn.row_factory = sqlite3.Row
                
                try:
                    search_cur = search_conn.cursor()
                    search_cur.execute("""
                        SELECT e.series_id, e.subject, e.patch_version
                        FROM emails e
                        WHERE e.id = ? AND e.series_id IS NOT NULL
                        LIMIT 1
                    """, (linked_email_id,))
                    row = search_cur.fetchone()
                    if row and row[0]:
                        found_series_id = row[0]
                        # series
                        global_series_conn = get_global_series_conn()
                        try:
                            global_cur = global_series_conn.cursor()
                            global_cur.execute("""
                                SELECT patch_version FROM series
                                WHERE id = ?
                            """, (found_series_id,))
                            series_row = global_cur.fetchone()
                            series_version = series_row['patch_version'] if series_row else None
                        finally:
                            global_series_conn.close()
                        
                        email_subject = row[1] or ""
                        email_version = row[2]
                        email_version_from_subject = extract_version_from_subject(email_subject)
                        if email_version_from_subject:
                            email_version_from_subject = normalize_version(email_version_from_subject)
                        
                        effective_version = normalize_version(email_version) or email_version_from_subject
                        target_version = normalize_version(series_version) or effective_version
                        
                        if candidate_version is not None:
                            if target_version == candidate_version or effective_version == candidate_version:
                                return found_series_id
                        else:
                            if target_version is None and effective_version is None:
                                return found_series_id
                finally:
                    search_conn.close()
    
    return None


def parse_match_methods(methods_spec: str) -> set:
    """
    
    - "0,1,2" -> {0, 1, 2}
    - "0-2,4" -> {0, 1, 2, 4}
    - "0-4" -> {0, 1, 2, 3, 4}
    - "" None ->
    
    Returns:
    """
    if not methods_spec or methods_spec.strip() == "":
        return ALL_MATCH_METHODS.copy()
    
    result = set()
    parts = methods_spec.split(',')
    for part in parts:
        part = part.strip()
        if '-' in part:
            # "0-2"
            try:
                start, end = part.split('-')
                for i in range(int(start), int(end) + 1):
                    if i in ALL_MATCH_METHODS:
                        result.add(i)
            except ValueError:
                continue
        else:
            try:
                i = int(part)
                if i in ALL_MATCH_METHODS:
                    result.add(i)
            except ValueError:
                continue
    
    return result if result else ALL_MATCH_METHODS.copy()


def match_previous_version_series(
    series_id: int,
    representative_epoch: int,
    representative_email_id: int,
    has_cover: int,
    current_version: Optional[str],
    all_epochs: List[int],
    enabled_methods: Optional[set] = None,
    similarity_threshold: float = 0.9,
    max_time_diff_days: Optional[float] = None,
) -> Tuple[
    Optional[int], Optional[int], Optional[int], Optional[int], Optional[int],
    float, float, float, float, float
]:
    """
     series
    
    Args:
        enabled_methods: 0-4 None
        similarity_threshold: 0.9 (90%)
        max_time_diff_days: None
    
    Returns:
        (
            previous_series_id_body_match, # 0
            previous_series_id_subject_match, # 1
            previous_series_id_thread_match, # 2
            previous_series_id_subject_fuzz, # 3
            previous_series_id_rep_body, # 4
            time_body, time_subject, time_thread, time_subject_fuzz, time_rep_body,
        )
    """
    if enabled_methods is None:
        enabled_methods = ALL_MATCH_METHODS.copy()
    
    # epoch representative
    epoch_db_path = get_db_path(representative_epoch)
    if not epoch_db_path.exists():
        return None, None, None, None, None, 0.0, 0.0, 0.0, 0.0, 0.0
    
    epoch_conn = sqlite3.connect(epoch_db_path)
    epoch_conn.row_factory = sqlite3.Row
    try:
        epoch_cur = epoch_conn.cursor()
        # from_addr timestamp
        epoch_cur.execute("""
            SELECT subject, body, patch_current, from_addr, timestamp FROM emails WHERE id = ?
        """, (representative_email_id,))
        row = epoch_cur.fetchone()
        if not row:
            return None, None, None, None, None, 0.0, 0.0, 0.0, 0.0, 0.0
        
        subject = row[0]
        body = row[1]
        patch_current = row[2]
        from_addr = row[3]
        timestamp = row[4]
        
        # cover letter
        subject_str = subject or ""
        body_str = body or ""
        is_comment = is_comment_email(subject_str)
        determined_is_patch = False if is_comment else determine_is_patch(subject_str, body_str)
        is_cover_letter = (determined_is_patch and patch_current == 0)
        
        prev_series_id_body: Optional[int] = None
        prev_series_id_subject: Optional[int] = None
        prev_series_id_thread: Optional[int] = None
        prev_series_id_subject_fuzz: Optional[int] = None
        prev_series_id_rep_body: Optional[int] = None
        t_body = t_subject = t_thread = t_subject_fuzz = t_rep_body = 0.0
        
        # 0: body_match cover letter
        if MATCH_BODY in enabled_methods:
            t_body_start = time.time()
            if is_cover_letter:
                prev_series_id_body = body_match_previous_version(
                    series_id,
                    representative_epoch,
                    representative_email_id,
                    body,
                    current_version,
                    all_epochs,
                )
            t_body = time.time() - t_body_start
        
        # 1: subject_match
        if MATCH_SUBJECT in enabled_methods:
            t_subject_start = time.time()
            prev_series_id_subject = subject_match_previous_version(
                series_id, representative_epoch, subject, current_version, all_epochs
            )
            t_subject = time.time() - t_subject_start
        
        # 2: thread_match in_reply_to
        if MATCH_THREAD in enabled_methods:
            t_thread_start = time.time()
            prev_series_id_thread = thread_match_previous_version(
                series_id, representative_epoch, representative_email_id, current_version, all_epochs
            )
            t_thread = time.time() - t_thread_start
        
        # 3: subject_fuzz_match
        if MATCH_SUBJECT_FUZZ in enabled_methods:
            t_subject_fuzz_start = time.time()
            prev_series_id_subject_fuzz = subject_fuzz_match_previous_version(
                series_id,
                representative_epoch,
                subject,
                from_addr,
                timestamp,
                current_version,
                all_epochs,
                similarity_threshold=similarity_threshold,
                max_time_diff_days=max_time_diff_days,
            )
            t_subject_fuzz = time.time() - t_subject_fuzz_start
        
        # 4: represent_body_match representative letter body
        if MATCH_REP_BODY in enabled_methods:
            t_rep_body_start = time.time()
            prev_series_id_rep_body = represent_body_match_previous_version(
                series_id,
                representative_epoch,
                representative_email_id,
                body,
                current_version,
                all_epochs,
            )
            t_rep_body = time.time() - t_rep_body_start
        
        return (
            prev_series_id_body,
            prev_series_id_subject,
            prev_series_id_thread,
            prev_series_id_subject_fuzz,
            prev_series_id_rep_body,
            t_body,
            t_subject,
            t_thread,
            t_subject_fuzz,
            t_rep_body,
        )
    finally:
        epoch_conn.close()


# process_single_series_version_match
_ENABLED_METHODS: Optional[set] = None
_SIMILARITY_THRESHOLD: float = 0.9
_MAX_TIME_DIFF_DAYS: Optional[float] = None


def process_single_series_version_match(
    series_data: Tuple[int, int, int, Optional[str], int, List[int]]
) -> Tuple[int, Dict[str, Optional[int]], Dict[str, float]]:
    """
     series
    
    Args:
        series_data: (series_id, representative_epoch, representative_email_id, patch_version, has_cover, all_epochs)
    
    Returns:
        (series_id, match_results, stats)
        match_results: {
            'body': prev_series_id_body,
            'subject': prev_series_id_subject,
            'thread': prev_series_id_thread,
            'subject_fuzz': prev_series_id_subject_fuzz,
            'rep_body': prev_series_id_rep_body,
        }
        stats: {
            'version_gt_0': 0 or 1,
            'body_match': 0 or 1,
            'subject_match': 0 or 1,
            'thread_match': 0 or 1,
            'subject_fuzz_match': 0 or 1,
            'rep_body_match': 0 or 1,
            'found_previous': 0 or 1,
            'time_body': float,
            'time_subject': float,
            'time_thread': float,
            'time_subject_fuzz': float,
            'time_rep_body': float,
        }
    """
    global _ENABLED_METHODS, _SIMILARITY_THRESHOLD, _MAX_TIME_DIFF_DAYS
    
    series_id, representative_epoch, representative_email_id, version, has_cover, all_epochs = series_data
    
    try:
        # > 0 series
        version_gt_0 = 0
        if version:
            v_num = re.search(r'v?(\d+)', version, re.IGNORECASE)
            if v_num and int(v_num.group(1)) > 0:
                version_gt_0 = 1
        
        (
            prev_series_id_body,
            prev_series_id_subject,
            prev_series_id_thread,
            prev_series_id_subject_fuzz,
            prev_series_id_rep_body,
            time_body,
            time_subject,
            time_thread,
            time_subject_fuzz,
            time_rep_body,
        ) = match_previous_version_series(
            series_id,
            representative_epoch,
            representative_email_id,
            has_cover,
            version,
            all_epochs,
            enabled_methods=_ENABLED_METHODS,
            similarity_threshold=_SIMILARITY_THRESHOLD,
            max_time_diff_days=_MAX_TIME_DIFF_DAYS,
        )
        
        
        body_match = 1 if prev_series_id_body else 0
        subject_match = 1 if prev_series_id_subject else 0
        thread_match = 1 if prev_series_id_thread else 0
        subject_fuzz_match = 1 if prev_series_id_subject_fuzz else 0
        rep_body_match = 1 if prev_series_id_rep_body else 0
        
        found_previous = 1 if any([
            prev_series_id_body, prev_series_id_subject, prev_series_id_thread,
            prev_series_id_subject_fuzz, prev_series_id_rep_body
        ]) else 0
        
        return (
            series_id,
            {
                "body": prev_series_id_body,
                "subject": prev_series_id_subject,
                "thread": prev_series_id_thread,
                "subject_fuzz": prev_series_id_subject_fuzz,
                "rep_body": prev_series_id_rep_body,
            },
            {
                "version_gt_0": float(version_gt_0),
                "body_match": float(body_match),
                "subject_match": float(subject_match),
                "thread_match": float(thread_match),
                "subject_fuzz_match": float(subject_fuzz_match),
                "rep_body_match": float(rep_body_match),
                "found_previous": float(found_previous),
                "time_body": time_body,
                "time_subject": time_subject,
                "time_thread": time_thread,
                "time_subject_fuzz": time_subject_fuzz,
                "time_rep_body": time_rep_body,
            },
        )
        
    except Exception as e:
        print(f"[ERROR] series {series_id} (epoch {representative_epoch}) : {e}")
        import traceback
        traceback.print_exc()
        return (
            series_id,
            {"body": None, "subject": None, "thread": None, "subject_fuzz": None, "rep_body": None},
            {
                "version_gt_0": 0.0,
                "body_match": 0.0,
                "subject_match": 0.0,
                "thread_match": 0.0,
                "subject_fuzz_match": 0.0,
                "rep_body_match": 0.0,
                "found_previous": 0.0,
                "time_body": 0.0,
                "time_subject": 0.0,
                "time_thread": 0.0,
                "time_subject_fuzz": 0.0,
                "time_rep_body": 0.0,
            },
        )


def build_version_relations_global(
    epochs: List[int],
    workers: int = 4,
    enabled_methods: Optional[set] = None,
    similarity_threshold: float = 0.9,
    max_time_diff_days: Optional[float] = None,
) -> Dict[str, int]:
    """
     series epoch
    
    Args:
        epochs: epoch
        workers:
        enabled_methods: 0-4 None
        similarity_threshold: 0.9 (90%)
        max_time_diff_days: None
    
    Returns:
    """
    if enabled_methods is None:
        enabled_methods = ALL_MATCH_METHODS.copy()
    
    global _ENABLED_METHODS, _SIMILARITY_THRESHOLD, _MAX_TIME_DIFF_DAYS
    _ENABLED_METHODS = enabled_methods
    _SIMILARITY_THRESHOLD = similarity_threshold
    _MAX_TIME_DIFF_DAYS = max_time_diff_days
    
    print(f"[] : {[MATCH_METHOD_NAMES[m] for m in sorted(enabled_methods)]}")
    if MATCH_SUBJECT_FUZZ in enabled_methods:
        print(f"[] subject_fuzz_match : {similarity_threshold:.0%}")
        if max_time_diff_days is not None:
            print(f"[] subject_fuzz_match : {max_time_diff_days:.2f} ")
    
    # series
    global_series_conn = get_global_series_conn()
    
    try:
        cur = global_series_conn.cursor()
        
        try:
            cur.execute("ALTER TABLE series ADD COLUMN previous_version_series_id_subject_fuzz_match INTEGER")
        except sqlite3.OperationalError:
            pass #
        try:
            cur.execute("ALTER TABLE series ADD COLUMN previous_version_series_id_rep_body_match INTEGER")
        except sqlite3.OperationalError:
            pass #
        global_series_conn.commit()
        
        # series series
        cur.execute(
            """
            SELECT id, representative_email_id, has_cover, patch_version, representative_epoch
            FROM series
            WHERE patch_version IS NOT NULL
            ORDER BY id
            """
        )
        
        series_list = cur.fetchall()
        if not series_list:
            return {
                'total': 0,
                'found_previous': 0,
                'version_gt_0': 0,
                'body_match': 0,
                'subject_match': 0,
                'thread_match': 0,
                'subject_fuzz_match': 0,
                'rep_body_match': 0,
            }
        
        total = len(series_list)

        # series SQLite
        global _SERIES_SUBJECTS, _SUBJECT_MATCH_INDEX, _SERIES_FUZZ_DATA, _SERIES_REP_BODY
        _SERIES_SUBJECTS = {}
        _SUBJECT_MATCH_INDEX = {}
        _SERIES_FUZZ_DATA = {}
        _SERIES_REP_BODY = {}
        
        from collections import defaultdict
        epoch_to_series: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for row in series_list:
            sid = row["id"]
            rep_epoch = row["representative_epoch"]
            rep_email_id = row["representative_email_id"]
            if rep_epoch is not None and rep_email_id is not None:
                epoch_to_series[rep_epoch].append((sid, rep_email_id))

        # subject, from_addr, timestamp, body
        print("[] ...")
        for epoch, items in epoch_to_series.items():
            db_path = get_db_path(epoch)
            if not db_path.exists() or not items:
                continue
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            try:
                cur_ep = conn.cursor()
                rep_ids = [rep_id for (_, rep_id) in items]
                placeholders = ",".join(["?"] * len(rep_ids))
                cur_ep.execute(
                    f"SELECT id, subject, from_addr, timestamp, body FROM emails WHERE id IN ({placeholders})",
                    rep_ids,
                )
                rows = cur_ep.fetchall()
                id_to_data = {r["id"]: r for r in rows}
                for sid, rep_id in items:
                    row_data = id_to_data.get(rep_id)
                    if row_data:
                        _SERIES_SUBJECTS[sid] = row_data["subject"]
                        # subject_fuzz_match
                        base_subj = extract_base_subject(row_data["subject"] or "")
                        _SERIES_FUZZ_DATA[sid] = (
                            row_data["from_addr"],
                            row_data["timestamp"],
                            base_subj,
                        )
                        # represent_body_match
                        _SERIES_REP_BODY[sid] = (rep_id, epoch, row_data["body"])
            finally:
                conn.close()
        print(f"[] {len(_SERIES_SUBJECTS)} series")
        
        # series (epoch, version) -> [series_id]
        print("[] subject_match ...")
        cur.execute("""
            SELECT id, representative_epoch, patch_version
            FROM series
            WHERE representative_epoch IS NOT NULL
        """)
        all_series_for_index = cur.fetchall()
        
        for row in all_series_for_index:
            sid = row["id"]
            rep_epoch = row["representative_epoch"]
            patch_version = row["patch_version"]
            key = (rep_epoch, patch_version)
            if key not in _SUBJECT_MATCH_INDEX:
                _SUBJECT_MATCH_INDEX[key] = []
            _SUBJECT_MATCH_INDEX[key].append(sid)
        
        print(f"[] {len(_SUBJECT_MATCH_INDEX)} (epoch, version) ")
        
        # series all_epochs
        series_data_list = [
            (s['id'], s['representative_epoch'], s['representative_email_id'], s['patch_version'], s['has_cover'], epochs)
            for s in series_list
        ]
        
        body_match_series = set()
        subject_match_series = set()
        thread_match_series = set()
        subject_fuzz_match_series = set()
        rep_body_match_series = set()
        
        found_previous = 0
        version_gt_0 = 0
        body_match_count = 0
        subject_match_count = 0
        thread_match_count = 0
        subject_fuzz_match_count = 0
        rep_body_match_count = 0

        total_time_body = 0.0
        total_time_subject = 0.0
        total_time_thread = 0.0
        total_time_subject_fuzz = 0.0
        total_time_rep_body = 0.0
        
        # series
        processed_count = 0
        
        pbar = tqdm(
            total=total,
            desc="[]",
            unit="series",
            ncols=140,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]'
        )
        
        # + series_id
        # enabled_methods
        batch_updates: List[Tuple] = []
        
        # UPDATE
        method_to_column = {
            MATCH_BODY: 'previous_version_series_id_body_match',
            MATCH_SUBJECT: 'previous_version_series_id_subject_match',
            MATCH_THREAD: 'previous_version_series_id_thread_match',
            MATCH_SUBJECT_FUZZ: 'previous_version_series_id_subject_fuzz_match',
            MATCH_REP_BODY: 'previous_version_series_id_rep_body_match',
        }
        method_to_result_key = {
            MATCH_BODY: 'body',
            MATCH_SUBJECT: 'subject',
            MATCH_THREAD: 'thread',
            MATCH_SUBJECT_FUZZ: 'subject_fuzz',
            MATCH_REP_BODY: 'rep_body',
        }
        enabled_methods_sorted = sorted(enabled_methods)
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_single_series_version_match, data): data for data in series_data_list}
            
            for future in as_completed(futures):
                try:
                    series_id, match_results, stats = future.result()
                    
                    update_values = tuple(
                        match_results[method_to_result_key[m]] for m in enabled_methods_sorted
                    ) + (series_id,)
                    batch_updates.append(update_values)
                    
                    version_gt_0 += int(stats.get('version_gt_0', 0))
                    body_int = int(stats.get('body_match', 0))
                    subj_int = int(stats.get('subject_match', 0))
                    thread_int = int(stats.get('thread_match', 0))
                    subj_fuzz_int = int(stats.get('subject_fuzz_match', 0))
                    rep_body_int = int(stats.get('rep_body_match', 0))

                    body_match_count += body_int
                    subject_match_count += subj_int
                    thread_match_count += thread_int
                    subject_fuzz_match_count += subj_fuzz_int
                    rep_body_match_count += rep_body_int
                    
                    if int(stats.get('found_previous', 0)):
                        found_previous += 1
                    
                    if match_results['body']:
                        body_match_series.add(series_id)
                    if match_results['subject']:
                        subject_match_series.add(series_id)
                    if match_results['thread']:
                        thread_match_series.add(series_id)
                    if match_results['subject_fuzz']:
                        subject_fuzz_match_series.add(series_id)
                    if match_results['rep_body']:
                        rep_body_match_series.add(series_id)

                    total_time_body += float(stats.get('time_body', 0.0))
                    total_time_subject += float(stats.get('time_subject', 0.0))
                    total_time_thread += float(stats.get('time_thread', 0.0))
                    total_time_subject_fuzz += float(stats.get('time_subject_fuzz', 0.0))
                    total_time_rep_body += float(stats.get('time_rep_body', 0.0))
                    
                    processed_count += 1
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'B': body_match_count,
                        'S': subject_match_count,
                        'T': thread_match_count,
                        'SF': subject_fuzz_match_count,
                        'RB': rep_body_match_count,
                        'F': found_previous,
                    })
                except Exception as e:
                    print(f"[ERROR] series : {e}")
                    import traceback
                    traceback.print_exc()
                    pbar.update(1)
        
        pbar.close()
        
        # UPDATE
        if not enabled_methods_sorted:
            print("[WARN] ")
        else:
            update_columns = [method_to_column[m] for m in enabled_methods_sorted]
            set_clause = ", ".join(f"{col} = ?" for col in update_columns)
            update_sql = f"UPDATE series SET {set_clause} WHERE id = ?"
            
            print(f"\n[] {len(batch_updates)} series ...")
            print(f"[] : {update_columns}")
            
            global_series_conn = get_global_series_conn()
            try:
                cur = global_series_conn.cursor()
                # executemany
                cur.executemany(update_sql, batch_updates)
                global_series_conn.commit()
                print(f"[] {len(batch_updates)} series")
            except Exception as e:
                print(f"[ERROR] : {e}")
                import traceback
                traceback.print_exc()
                print("[] ...")
                success_count = 0
                for update_data in batch_updates:
                    try:
                        cur.execute(update_sql, update_data)
                        success_count += 1
                    except Exception as e2:
                        print(f"[WARN] series {update_data[-1]} : {e2}")
                global_series_conn.commit()
                print(f"[] {success_count}/{len(batch_updates)}")
            finally:
                global_series_conn.close()
        
        all_enabled_sets = []
        if MATCH_BODY in enabled_methods:
            all_enabled_sets.append(('body', body_match_series))
        if MATCH_SUBJECT in enabled_methods:
            all_enabled_sets.append(('subject', subject_match_series))
        if MATCH_THREAD in enabled_methods:
            all_enabled_sets.append(('thread', thread_match_series))
        if MATCH_SUBJECT_FUZZ in enabled_methods:
            all_enabled_sets.append(('subject_fuzz', subject_fuzz_match_series))
        if MATCH_REP_BODY in enabled_methods:
            all_enabled_sets.append(('rep_body', rep_body_match_series))

        print("\n=== ===")
        if MATCH_BODY in enabled_methods:
            print(f" body_match : {total_time_body:.2f}")
        if MATCH_SUBJECT in enabled_methods:
            print(f" subject_match : {total_time_subject:.2f}")
        if MATCH_THREAD in enabled_methods:
            print(f" thread_match : {total_time_thread:.2f}")
        if MATCH_SUBJECT_FUZZ in enabled_methods:
            print(f" subject_fuzz_match : {total_time_subject_fuzz:.2f}")
        if MATCH_REP_BODY in enabled_methods:
            print(f" rep_body_match : {total_time_rep_body:.2f}")
        
        return {
            'total': total,
            'found_previous': found_previous,
            'version_gt_0': version_gt_0,
            'body_match': body_match_count,
            'subject_match': subject_match_count,
            'thread_match': thread_match_count,
            'subject_fuzz_match': subject_fuzz_match_count,
            'rep_body_match': rep_body_match_count,
            'enabled_methods': list(enabled_methods),
        }
        
    except Exception as e:
        return {'error': str(e)}
    finally:
        global_series_conn.close()


def build_version_relations(
    epoch_spec: Optional[str] = None,
    workers: int = 4,
    match_methods_spec: Optional[str] = None,
    similarity_threshold: float = 0.9,
    max_time_diff_days: Optional[float] = None,
) -> None:
    """
     series epoch
    
    Args:
        epoch_spec: epoch "0-18" epoch
        workers:
        match_methods_spec: "0,1,2" "0-2,4"
        similarity_threshold: 0.9 (90%)
        max_time_diff_days: None
    """
    if epoch_spec:
        selected_epochs: Optional[List[int]] = sorted(parse_epoch_spec(epoch_spec))
        print(f" epoch: {selected_epochs}")
    else:
        selected_epochs = None
        print(" epoch SQLite epoch ")
    
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
    
    enabled_methods = parse_match_methods(match_methods_spec or "")
    
    print(f"\n=== {len(epochs)} epoch ===")
    
    try:
        stats = build_version_relations_global(
            epochs,
            workers=workers,
            enabled_methods=enabled_methods,
            similarity_threshold=similarity_threshold,
            max_time_diff_days=max_time_diff_days,
        )
        if 'error' in stats:
            print(f"[ERROR] : {stats['error']}")
            return
        
        print(f"\n=== ===")
        print(f" series : {stats['total']}")
        print(f" series : {stats['found_previous']}")
        print(f" > 0 series : {stats['version_gt_0']}")
        if stats['version_gt_0'] > 0:
            ratio = stats['found_previous'] / stats['version_gt_0'] * 100
            print(f": {ratio:.1f}%")
        
        print("\n:")
        if MATCH_BODY in enabled_methods:
            print(f"  body_match          : {stats['body_match']}")
        if MATCH_SUBJECT in enabled_methods:
            print(f"  subject_match       : {stats['subject_match']}")
        if MATCH_THREAD in enabled_methods:
            print(f"  thread_match        : {stats['thread_match']}")
        if MATCH_SUBJECT_FUZZ in enabled_methods:
            print(f"  subject_fuzz_match  : {stats['subject_fuzz_match']}")
        if MATCH_REP_BODY in enabled_methods:
            print(f"  rep_body_match      : {stats['rep_body_match']}")
    except Exception as e:
        print(f"[ERROR] : {e}")
        import traceback
        traceback.print_exc()

