"""
source_resolver.py — Linux git patch pre/post

Phase 1: diff index blob SHA bare clone
 bare clone fallback diff pre/post
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# diff index : index <pre-blob>..<post-blob> <mode>
_INDEX_RE = re.compile(r'^index\s+([0-9a-f]+)\.\.([0-9a-f]+)')
_DIFF_FILE_RE = re.compile(r'^diff --git a/(.*) b/(.*)')
_HUNK_RE = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@')


@dataclass
class FileSource:
    """ pre/post """
    filepath: str
    pre_content: Optional[str] = None # None
    post_content: Optional[str] = None # None


@dataclass
class ResolvedEntry:
    """ dataset entry """
    entry_id: int
    files: List[FileSource] = field(default_factory=list)
    resolved_from_git: bool = False


def parse_diff_index_lines(diff_text: str) -> List[Tuple[str, str, str]]:
    """
     diff index
     [(filepath, pre_blob, post_blob), ...]
    """
    results = []
    current_file = None

    for line in diff_text.splitlines():
        m = _DIFF_FILE_RE.match(line)
        if m:
            current_file = m.group(2)
            continue
        m = _INDEX_RE.match(line)
        if m and current_file:
            pre_blob = m.group(1)
            post_blob = m.group(2)
            results.append((current_file, pre_blob, post_blob))
            current_file = None

    return results


def git_cat_file(repo_path: Path, blob_sha: str) -> Optional[str]:
    """ git bare repo blob SHA (7-12 )"""
    if not repo_path.exists():
        return None
    if blob_sha.startswith('0000000'):
        return None
    try:
        result = subprocess.run(
            ['git', '--git-dir', str(repo_path), 'cat-file', '-p', blob_sha],
            capture_output=True, timeout=10
        )
        if result.returncode == 0:
            # UTF-8 latin-1
            try:
                return result.stdout.decode('utf-8')
            except UnicodeDecodeError:
                return result.stdout.decode('latin-1')
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def reconstruct_pre_post_from_diff(diff_text: str) -> List[FileSource]:
    """
    Fallback: unified diff pre/post
     git repo diff context + add/del
     diff
     AST
    """
    files = []
    current_file = None
    pre_lines = []
    post_lines = []
    in_diff = False

    for line in diff_text.splitlines():
        m = _DIFF_FILE_RE.match(line)
        if m:
            if current_file and (pre_lines or post_lines):
                files.append(FileSource(
                    filepath=current_file,
                    pre_content='\n'.join(pre_lines) if pre_lines else None,
                    post_content='\n'.join(post_lines) if post_lines else None,
                ))
            current_file = m.group(2)
            pre_lines = []
            post_lines = []
            in_diff = False
            continue

        if line.startswith('--- ') or line.startswith('+++ '):
            continue

        if _HUNK_RE.match(line):
            in_diff = True
            continue

        if in_diff:
            if line.startswith('+'):
                post_lines.append(line[1:])
            elif line.startswith('-'):
                pre_lines.append(line[1:])
            else:
                # context line
                content = line[1:] if line.startswith(' ') else line
                pre_lines.append(content)
                post_lines.append(content)

    if current_file and (pre_lines or post_lines):
        files.append(FileSource(
            filepath=current_file,
            pre_content='\n'.join(pre_lines) if pre_lines else None,
            post_content='\n'.join(post_lines) if post_lines else None,
        ))

    return files


def resolve_entry(entry: Dict, repo_path: Optional[Path] = None,
                  cache_dir: Optional[Path] = None) -> ResolvedEntry:
    """
     dataset entry
     git repo fallback diff

    1. git repo pre-blob
    2. pre-image diff post-image
    3. pre-blob fallback diff
    """
    entry_id = entry.get('id', 0)
    diff_text = entry['source']['diff']
    result = ResolvedEntry(entry_id=entry_id)

    # git repo
    if repo_path and repo_path.exists():
        index_info = parse_diff_index_lines(diff_text)
        resolved_count = 0
        total_count = len(index_info)

        # diff hunks apply patch
        file_patches = _parse_file_patches(diff_text)

        for filepath, pre_blob, post_blob in index_info:
            pre_content = None
            post_content = None

            # pre-image
            if pre_blob and not pre_blob.startswith('0000000'):
                pre_content = git_cat_file(repo_path, pre_blob)

            # post-image
            if post_blob and not post_blob.startswith('0000000'):
                post_content = git_cat_file(repo_path, post_blob)

            # pre post apply diff post
            if pre_content and not post_content and filepath in file_patches:
                post_content = _apply_patch(pre_content, file_patches[filepath])

            # post pre reverse diff pre
            if post_content and not pre_content and filepath in file_patches:
                pre_content = _reverse_patch(post_content, file_patches[filepath])

            if pre_content or post_content:
                resolved_count += 1

            result.files.append(FileSource(
                filepath=filepath,
                pre_content=pre_content,
                post_content=post_content,
            ))

        if resolved_count > 0 and resolved_count >= total_count * 0.5:
            result.resolved_from_git = True
            # diff
            diff_files = reconstruct_pre_post_from_diff(diff_text)
            diff_map = {fs.filepath: fs for fs in diff_files}
            for fs in result.files:
                if not fs.pre_content and not fs.post_content:
                    fallback = diff_map.get(fs.filepath)
                    if fallback:
                        fs.pre_content = fallback.pre_content
                        fs.post_content = fallback.post_content
            return result

    # Fallback: diff
    result.files = reconstruct_pre_post_from_diff(diff_text)
    result.resolved_from_git = False

    if cache_dir:
        _cache_sources(cache_dir, result)

    return result


def _parse_file_patches(diff_text: str) -> Dict[str, List[Dict]]:
    """ diff hunk apply/reverse patch"""
    file_patches = {}
    current_file = None
    current_hunks = []

    for line in diff_text.splitlines():
        m = _DIFF_FILE_RE.match(line)
        if m:
            if current_file and current_hunks:
                file_patches[current_file] = current_hunks
            current_file = m.group(2)
            current_hunks = []
            continue

        hm = _HUNK_RE.match(line)
        if hm:
            current_hunks.append({
                'pre_start': int(hm.group(1)),
                'pre_count': int(hm.group(2)) if hm.group(2) else 1,
                'post_start': int(hm.group(3)),
                'post_count': int(hm.group(4)) if hm.group(4) else 1,
                'lines': [],
            })
            continue

        if current_hunks:
            current_hunks[-1]['lines'].append(line)

    if current_file and current_hunks:
        file_patches[current_file] = current_hunks

    return file_patches


def _apply_patch(pre_content: str, hunks: List[Dict]) -> Optional[str]:
    """ pre-image diff hunks post-image"""
    try:
        pre_lines = pre_content.splitlines()
        result_lines = []
        pre_idx = 0  # 0-based index into pre_lines

        for hunk in hunks:
            hunk_pre_start = hunk['pre_start'] - 1  # convert to 0-based

            # hunk
            while pre_idx < hunk_pre_start:
                result_lines.append(pre_lines[pre_idx])
                pre_idx += 1

            # hunk
            for line in hunk['lines']:
                if line.startswith('+'):
                    result_lines.append(line[1:])
                elif line.startswith('-'):
                    pre_idx += 1  # skip deleted line
                elif line.startswith(' '):
                    result_lines.append(line[1:])
                    pre_idx += 1
                elif line.startswith('\\'):
                    continue  # "\ No newline at end of file"
                else:
                    # context line without prefix
                    result_lines.append(line)
                    pre_idx += 1

        while pre_idx < len(pre_lines):
            result_lines.append(pre_lines[pre_idx])
            pre_idx += 1

        return '\n'.join(result_lines)
    except (IndexError, ValueError):
        return None


def _reverse_patch(post_content: str, hunks: List[Dict]) -> Optional[str]:
    """ post-image diff hunks pre-image"""
    try:
        post_lines = post_content.splitlines()
        result_lines = []
        post_idx = 0

        for hunk in hunks:
            hunk_post_start = hunk['post_start'] - 1

            while post_idx < hunk_post_start:
                result_lines.append(post_lines[post_idx])
                post_idx += 1

            for line in hunk['lines']:
                if line.startswith('-'):
                    result_lines.append(line[1:])  # restore deleted line
                elif line.startswith('+'):
                    post_idx += 1  # skip added line
                elif line.startswith(' '):
                    result_lines.append(line[1:])
                    post_idx += 1
                elif line.startswith('\\'):
                    continue
                else:
                    result_lines.append(line)
                    post_idx += 1

        while post_idx < len(post_lines):
            result_lines.append(post_lines[post_idx])
            post_idx += 1

        return '\n'.join(result_lines)
    except (IndexError, ValueError):
        return None


def _cache_sources(cache_dir: Path, resolved: ResolvedEntry):
    """"""
    entry_dir = cache_dir / str(resolved.entry_id)
    for fs in resolved.files:
        if fs.pre_content:
            pre_path = entry_dir / 'pre' / fs.filepath
            pre_path.parent.mkdir(parents=True, exist_ok=True)
            pre_path.write_text(fs.pre_content, encoding='utf-8')
        if fs.post_content:
            post_path = entry_dir / 'post' / fs.filepath
            post_path.parent.mkdir(parents=True, exist_ok=True)
            post_path.write_text(fs.post_content, encoding='utf-8')
