"""

 "Link to V1:", "Changes in V2:"
cover letter
"""

from __future__ import annotations

import re
import sqlite3
from typing import Dict, List, Optional, Set, Tuple

from processors.sqlite.db_utils import get_db_path


VERSION_LINK_PATTERNS = [
    r'(?i)link\s+to\s+v(\d+)',
    r'(?i)changes?\s+in\s+v(\d+)',
    r'(?i)change?\s+in\s+v(\d+)',
    r'(?i)v(\d+)\s*:',
    r'(?i)version\s+(\d+)',
    r'(?i)v(\d+)\s+link',
]

# PatchworkURL
PATCHWORK_URL_PATTERN = re.compile(
    r'https?://(?:www\.)?patchwork\.kernel\.org[^\s<>]*?/cover/(\d+)/?',
    re.IGNORECASE
)
# URL
PATCHWORK_SIMPLE_PATTERN = re.compile(
    r'patchwork\.kernel\.org[^\s<>]*?/cover/(\d+)',
    re.IGNORECASE
)

# patch.msgid.link message-id
MSGID_LINK_PATTERN = re.compile(
    r'https?://(?:www\.)?patch\.msgid\.link/([^\s<>/]+)',
    re.IGNORECASE
)

# Message-ID
MESSAGE_ID_PATTERN = re.compile(
    r'<[^<>]+@[^<>]+>',
    re.IGNORECASE
)


def extract_version_links_from_body(body: str) -> Dict[str, List[str]]:
    """.
    
    - "Link to V1: https://..."
    - "Changes in V2: https://..."
    - "V3: https://..."
    
    200500
    
    Returns:
        Dict[version, List[link_urls]]
        : {'v1': ['https://patchwork.kernel.org/.../cover/12345/'], 'v2': [...]}
    """
    if not body:
        return {}
    
    version_links: Dict[str, List[str]] = {}
    
    for pattern_str in VERSION_LINK_PATTERNS:
        pattern = re.compile(pattern_str)
        for match in pattern.finditer(body):
            version_num = match.group(1)
            version = f'v{version_num}'
            
            start = max(0, match.start() - 200)
            end = min(len(body), match.end() + 500)
            context = body[start:end]
            
            # patchworkURL
            match_end_in_context = match.end() - start
            context_after_match = context[match_end_in_context:]
            
            pw_matches = PATCHWORK_URL_PATTERN.findall(context_after_match)
            pw_matches_simple = PATCHWORK_SIMPLE_PATTERN.findall(context_after_match)
            all_pw_ids = set(pw_matches + pw_matches_simple)
            
            for pw_id in all_pw_ids:
                if version not in version_links:
                    version_links[version] = []
                # contextURL
                url_match = re.search(rf'https?://[^\s<>]*?/cover/{pw_id}[^\s<>]*', context_after_match, re.IGNORECASE)
                if url_match:
                    url = url_match.group(0)
                    if url not in version_links[version]:
                        version_links[version].append(url)
                else:
                    # URLpatchwork ID
                    if pw_id not in version_links[version]:
                        version_links[version].append(pw_id)
            
            # patch.msgid.link message-id
            msgid_link_matches = MSGID_LINK_PATTERN.findall(context_after_match)
            for msgid_path in msgid_link_matches:
                if version not in version_links:
                    version_links[version] = []
                # contextURL
                url_match = re.search(rf'https?://[^\s<>]*?patch\.msgid\.link/{re.escape(msgid_path)}[^\s<>]*', context_after_match, re.IGNORECASE)
                if url_match:
                    url = url_match.group(0)
                    if url not in version_links[version]:
                        version_links[version].append(url)
                else:
                    # URL message-id
                    # msgid_path message-id
                    msgid = msgid_path
                    if not msgid.startswith('<'):
                        msgid = f'<{msgid}>'
                    if msgid not in version_links[version]:
                        version_links[version].append(msgid)
            
            # message-id
            msg_id_matches = MESSAGE_ID_PATTERN.findall(context_after_match)
            for msg_id in msg_id_matches:
                if version not in version_links:
                    version_links[version] = []
                if msg_id not in version_links[version]:
                    version_links[version].append(msg_id)
    
    return version_links


def find_email_by_patchwork_id(epoch: int, conn: sqlite3.Connection, patchwork_id: str) -> Optional[int]:
    """patchwork ID.
    
    Returns:
        IDNone
    """
    cursor = conn.execute(
        """
        SELECT id FROM emails
        WHERE epoch = ? AND patchwork_id = ?
        LIMIT 1
        """,
        (epoch, patchwork_id)
    )
    row = cursor.fetchone()
    if row:
        return row[0]
    return None


def find_email_by_message_id(epoch: int, conn: sqlite3.Connection, message_id: str) -> Optional[int]:
    """message ID.
    
    Returns:
        IDNone
    """
    # message-id
    message_id = message_id.strip('<>')
    
    cursor = conn.execute(
        """
        SELECT id FROM emails
        WHERE epoch = ? AND message_id LIKE ?
        LIMIT 1
        """,
        (epoch, f'%{message_id}%')
    )
    row = cursor.fetchone()
    if row:
        return row[0]
    return None


def find_email_by_patchwork_url(epoch: int, conn: sqlite3.Connection, url: str) -> Optional[int]:
    """patchwork URL.
    
    Returns:
        IDNone
    """
    # URLpatchwork IDURL
    # URL
    match = PATCHWORK_URL_PATTERN.search(url)
    if match:
        patchwork_id = match.group(1)
        return find_email_by_patchwork_id(epoch, conn, patchwork_id)
    
    # ID
    id_match = re.search(r'/cover/(\d+)', url)
    if id_match:
        patchwork_id = id_match.group(1)
        return find_email_by_patchwork_id(epoch, conn, patchwork_id)
    
    return None


def analyze_version_links_from_cover_letters(epoch: int) -> Tuple[Dict[int, Dict[str, Optional[int]]], Dict[int, Dict]]:
    """cover letter.
    
    Returns:
        (cover_version_links, cover_info)
        cover_version_links: Dict[cover_email_id, Dict[version, linked_email_id]]
        cover_info: Dict[cover_email_id, {subject, patch_version, patchwork_id}]
    """
    db_path = get_db_path(epoch)
    if not db_path.exists():
        print(f"[WARN] : {db_path}")
        return {}, {}
    
    conn = sqlite3.connect(db_path)
    try:
        # cover letter
        cursor = conn.execute(
            """
            SELECT id, message_id, subject, body, patch_version, patchwork_id
            FROM emails
            WHERE epoch = ?
              AND is_patch = 1
              AND is_cover = 1
              AND body IS NOT NULL
            ORDER BY timestamp ASC, id ASC
            """,
            (epoch,)
        )
        
        cover_version_links: Dict[int, Dict[str, Optional[int]]] = {}
        cover_info: Dict[int, Dict] = {}
        
        for row in cursor.fetchall():
            email_id, message_id, subject, body, patch_version, patchwork_id = row
            
            # cover letter
            cover_info[email_id] = {
                'subject': subject,
                'patch_version': patch_version,
                'patchwork_id': patchwork_id,
            }
            
            version_links = extract_version_links_from_body(body)
            
            if not version_links:
                continue
            
            linked_emails: Dict[str, Optional[int]] = {}
            for version, links in version_links.items():
                linked_id = None
                
                for link in links:
                    # patchwork URLID
                    if 'patchwork.kernel.org' in link or link.isdigit():
                        linked_id = find_email_by_patchwork_url(epoch, conn, link)
                        if linked_id:
                            break
                        # linkpatchwork_id
                        if link.isdigit():
                            linked_id = find_email_by_patchwork_id(epoch, conn, link)
                            if linked_id:
                                break
                    
                    # patch.msgid.link
                    if 'patch.msgid.link' in link:
                        # URL message-id
                        msgid_match = MSGID_LINK_PATTERN.search(link)
                        if msgid_match:
                            msgid_path = msgid_match.group(1)
                            # msgid_path message-id
                            msgid = msgid_path
                            if not msgid.startswith('<'):
                                msgid = f'<{msgid}>'
                            linked_id = find_email_by_message_id(epoch, conn, msgid)
                            if linked_id:
                                break
                    
                    # message-id
                    if link.startswith('<') and link.endswith('>'):
                        linked_id = find_email_by_message_id(epoch, conn, link)
                        if linked_id:
                            break
                    
                    # link message-id @
                    if '@' in link and not link.startswith('http'):
                        # patch.msgid.link message-id
                        msgid = link
                        if not msgid.startswith('<'):
                            msgid = f'<{msgid}>'
                        linked_id = find_email_by_message_id(epoch, conn, msgid)
                        if linked_id:
                            break
                
                linked_emails[version] = linked_id
            
            if linked_emails:
                cover_version_links[email_id] = linked_emails
        
        return cover_version_links, cover_info
    
    finally:
        conn.close()


def extract_version_from_subject(subject: str) -> Optional[str]:
    """subject."""
    pattern = r'\[PATCH\s+(v\d+)\s*\d*/\d*\]'
    match = re.search(pattern, subject, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return None


def build_version_chain_from_cover_links(cover_links: Dict[int, Dict[str, Optional[int]]], cover_info: Dict[int, Dict]) -> Dict[int, Tuple[Optional[int], Optional[int]]]:
    """cover letter.
    
    Args:
        cover_links: Dict[cover_email_id, Dict[version, linked_email_id]]
        cover_info: Dict[cover_email_id, {subject, patch_version, ...}]
    
    Returns:
        Dict[email_id, (previous_version_id, next_version_id)]
    """
    relations: Dict[int, Tuple[Optional[int], Optional[int]]] = {}
    
    # cover letter
    for cover_id, version_map in cover_links.items():
        if not version_map:
            continue
        
        # cover letter
        cover_subject = cover_info.get(cover_id, {}).get('subject', '')
        cover_version = cover_info.get(cover_id, {}).get('patch_version')
        if not cover_version:
            cover_version = extract_version_from_subject(cover_subject)
        
        sorted_versions = sorted(
            version_map.items(),
            key=lambda x: int(re.search(r'v?(\d+)', x[0], re.IGNORECASE).group(1)) if re.search(r'v?(\d+)', x[0], re.IGNORECASE) else 999
        )
        
        # cover letter
        current_version_num = None
        if cover_version:
            match = re.search(r'v?(\d+)', cover_version, re.IGNORECASE)
            if match:
                current_version_num = int(match.group(1))
        
        for i, (version, linked_id) in enumerate(sorted_versions):
            if linked_id is None:
                continue
            
            prev_id = None
            if i > 0:
                prev_id = sorted_versions[i-1][1]
            
            next_id = None
            if i < len(sorted_versions) - 1:
                next_id = sorted_versions[i+1][1]
            
            # cover lettercovernext
            if current_version_num:
                link_version_num = int(re.search(r'v?(\d+)', version, re.IGNORECASE).group(1)) if re.search(r'v?(\d+)', version, re.IGNORECASE) else 0
                if current_version_num > link_version_num:
                    # cover letter
                    if linked_id in relations:
                        old_prev, old_next = relations[linked_id]
                        relations[linked_id] = (old_prev, cover_id if old_next is None else old_next)
                    else:
                        relations[linked_id] = (prev_id, cover_id)
                    continue
            
            # previousnext
            if linked_id in relations:
                old_prev, old_next = relations[linked_id]
                new_prev = prev_id if prev_id else old_prev
                new_next = next_id if next_id else old_next
                relations[linked_id] = (new_prev, new_next)
            else:
                relations[linked_id] = (prev_id, next_id)
        
        # cover letter
        if cover_id not in relations and sorted_versions:
            if current_version_num:
                closest_prev = None
                closest_next = None
                for version, linked_id in sorted_versions:
                    if linked_id is None:
                        continue
                    link_version_num = int(re.search(r'v?(\d+)', version, re.IGNORECASE).group(1)) if re.search(r'v?(\d+)', version, re.IGNORECASE) else 0
                    if link_version_num < current_version_num:
                        closest_prev = linked_id
                    elif link_version_num > current_version_num and closest_next is None:
                        closest_next = linked_id
                        break
                
                if closest_prev or closest_next:
                    relations[cover_id] = (closest_prev, closest_next)
    
    return relations


def analyze_cover_letter_version_links(epoch: int) -> Dict[int, Tuple[Optional[int], Optional[int]]]:
    """cover letter.
    
    Returns:
        Dict[email_id, (previous_version_id, next_version_id)]
    """
    cover_links, cover_info = analyze_version_links_from_cover_letters(epoch)
    
    if not cover_links:
        return {}
    
    relations = build_version_chain_from_cover_links(cover_links, cover_info)
    
    return relations


def update_db_with_cover_version_relations(epoch: int, relations: Dict[int, Tuple[Optional[int], Optional[int]]], clear_existing: bool = False) -> None:
    """cover letter.
    
    cover letter
    previous_version_id/next_version_id
    """
    db_path = get_db_path(epoch)
    if not db_path.exists():
        print(f"[WARN] : {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    
    try:
        try:
            conn.execute("ALTER TABLE emails ADD COLUMN cover_previous_version_id INTEGER")
        except sqlite3.OperationalError:
            pass #
        
        try:
            conn.execute("ALTER TABLE emails ADD COLUMN cover_next_version_id INTEGER")
        except sqlite3.OperationalError:
            pass #
        
        conn.execute("CREATE INDEX IF NOT EXISTS idx_emails_cover_prev_version ON emails(cover_previous_version_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_emails_cover_next_version ON emails(cover_next_version_id)")
        
        if clear_existing:
            conn.execute("UPDATE emails SET cover_previous_version_id = NULL, cover_next_version_id = NULL WHERE epoch = ?", (epoch,))
        
        updated = 0
        for email_id, (prev_id, next_id) in relations.items():
            conn.execute("""
                UPDATE emails
                SET cover_previous_version_id = ?,
                    cover_next_version_id = ?
                WHERE id = ?
            """, (prev_id, next_id, email_id))
            updated += 1
        
        conn.commit()
        print(f"epoch {epoch}: {updated} cover letter")
        
    finally:
        conn.close()


def analyze_all_epochs_cover_links(epochs: Optional[List[int]] = None, clear_existing: bool = False) -> None:
    """epochcover letter.
    
    Args:
        epochs: epochNoneepoch
        clear_existing:
    """
    import re
    from core import STRUCTURE_DIR
    
    if epochs is None:
        # epoch
        epochs = []
        for db_file in STRUCTURE_DIR.glob("lkml_epoch_*.sqlite"):
            match = re.search(r'epoch_(\d+)', db_file.name)
            if match:
                epochs.append(int(match.group(1)))
        epochs = sorted(epochs)
    
    print(f" {len(epochs)} epochcover letter...")
    
    for epoch in epochs:
        print(f"\n epoch {epoch}...")
        try:
            relations = analyze_cover_letter_version_links(epoch)
            if relations:
                update_db_with_cover_version_relations(epoch, relations, clear_existing=clear_existing)
                print(f"epoch {epoch}: {len(relations)} ")
            else:
                print(f"epoch {epoch}: ")
        except Exception as e:
            print(f"[ERROR] epoch {epoch} : {e}")
            import traceback
            traceback.print_exc()
    
    print("\nepoch")


if __name__ == "__main__":
    import argparse
    from lkml_preprocess import parse_epoch_spec
    
    parser = argparse.ArgumentParser(description="cover letter")
    parser.add_argument(
        "--epochs",
        type=str,
        default="",
        help="epoch: '14-18' '14,15,18'",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="",
    )
    args = parser.parse_args()
    
    if args.epochs:
        selected_epochs = sorted(parse_epoch_spec(args.epochs))
    else:
        selected_epochs = None
    
    analyze_all_epochs_cover_links(selected_epochs, clear_existing=args.clear)

