"""
build_balanced_dataset.py — filtered JSON BuildLayerDeduplicateData

 atom
 L1: File — atom File
 L2: FileFunction — atom FileFunction
 L3: FileFunction — atom FileFunction
 L4: Function — atom Function
 L5: Function — atom Function

OptimizeStrategy：
 - JSON pretty-print
 - orjson (C) Parse
 -
 - Process < 5GB
 - ExtractFile/FunctionPath


  python build_balanced_dataset.py \
    --filter-dir data/lkp_synthesize/filter \
    --per-level 2000 \
    --neg-ratio 1.0 \
    --seed 42 \
    --out-dir data/lkp_synthesize/balanced

python build_balanced_dataset.py --filter-dir data/lkp_synthesize/filter --neg-ratio 1.0 --seed 42 --out-dir data/lkp_synthesize/balanced --level-counts L1:500,L2:250,L3:250,L4:250,L5:250
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import random
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import orjson


# ---------------------------------------------------------------------------
# Fast file/function extraction (regex on bytes for speed)
# ---------------------------------------------------------------------------

_DIFF_GIT_RE_B = re.compile(rb'^diff --git ', re.MULTILINE)
_PLUS_RE_B = re.compile(rb'^\+\+\+ (?:b/)?(.*)', re.MULTILINE)
_MINUS_RE_B = re.compile(rb'^--- (?:a/)?(.*)', re.MULTILINE)
_HUNK_RE_B = re.compile(rb'^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@(.*)', re.MULTILINE)


def _fast_get_file_paths(diff_text: str) -> set:
    """Extract diff FilePath on bytes"""
    dt = diff_text.encode() if isinstance(diff_text, str) else diff_text
    paths = set()
    sections = _DIFF_GIT_RE_B.split(dt)
    for section in sections[1:]:
        m = _PLUS_RE_B.search(section)
        if m:
            p = m.group(1).strip()
            if p and p != b'/dev/null':
                paths.add(p)
                continue
        m = _MINUS_RE_B.search(section)
        if m:
            p = m.group(1).strip()
            if p and p != b'/dev/null':
                paths.add(p)
    return paths


def _fast_get_func_signatures(diff_text: str) -> set:
    """Extract diff hunk header Function"""
    dt = diff_text.encode() if isinstance(diff_text, str) else diff_text
    sigs = set()
    sections = _DIFF_GIT_RE_B.split(dt)
    for section in sections[1:]:
        m = _PLUS_RE_B.search(section)
        if m:
            fpath = m.group(1).strip()
        else:
            m = _MINUS_RE_B.search(section)
            fpath = m.group(1).strip() if m else b''
        for hm in _HUNK_RE_B.finditer(section):
            suffix = hm.group(1).strip()
            if suffix:
                sigs.add(fpath + b'::' + suffix)
    return sigs


# ---------------------------------------------------------------------------
# Difficulty classification
# ---------------------------------------------------------------------------

LEVEL_ORDER = {'L1': 1, 'L2': 2, 'L3': 3, 'L4': 4, 'L5': 5}


def classify_difficulty_from_atoms(atoms: list) -> str:
    """ atom File/Function"""
    diffs = [a['diff'] for a in atoms]
    atom_files = [_fast_get_file_paths(d) for d in diffs]
    atom_funcs = [_fast_get_func_signatures(d) for d in diffs]

    pairs = list(combinations(range(len(atoms)), 2))
    pair_file = [bool(atom_files[i] & atom_files[j]) for i, j in pairs]
    pair_func = [bool(atom_funcs[i] & atom_funcs[j]) for i, j in pairs]

    any_file = any(pair_file)
    all_file = all(pair_file)
    any_func = any(pair_func)
    all_func = all(pair_func)

    if not any_file:
        return 'L1'
    if not all_file and not any_func:
        return 'L2'
    if all_file and not any_func:
        return 'L3'
    if any_func and not all_func:
        return 'L4'
    return 'L5'


# ---------------------------------------------------------------------------
# Multiprocessing worker for batch classification
# ---------------------------------------------------------------------------

def _classify_batch(raw_records: list[bytes]) -> list[tuple]:
    """
    Worker: parse raw JSON bytes, classify, return (kind, key, level).
    - Positive: ('pos', tuple_of_sorted_msg_ids, level_str)
    - Negative: ('neg', msg_id_str, None)
    Skips records that fail to parse (rare edge case from splitter).
    """
    results = []
    for raw in raw_records:
        try:
            case = orjson.loads(raw)
        except (orjson.JSONDecodeError, ValueError):
            continue  # skip malformed split
        if case['label'] != 'tangled':
            results.append(('neg', case['atoms'][0]['message_id'], None))
        else:
            atoms = case['atoms']
            msg_ids = tuple(sorted(a['message_id'] for a in atoms))
            level = classify_difficulty_from_atoms(atoms)
            results.append(('pos', msg_ids, level))
    return results


# ---------------------------------------------------------------------------
# Fast streaming JSON array splitter (separator-based)
# ---------------------------------------------------------------------------

# Separator between top-level objects in pretty-printed JSON arrays
# Format: "  },\r\n  {" or "  },\n  {" depending on OS
_SEPARATORS = [b'\r\n  },\r\n  {', b'\n  },\n  {']


def _detect_separator(filepath: Path) -> bytes:
    """Detect record separator from file header (handles different indent levels)."""
    with open(filepath, 'rb') as f:
        header = f.read(256)
    # Detect indent: "[\r\n  {" = 2-space indent, "[\r\n{" = 0-space indent
    bracket = header.find(b'[')
    after = header[bracket + 1:bracket + 10]
    if b'\r\n  {' in after:
        return b'\r\n  },\r\n  {'
    elif b'\n  {' in after:
        return b'\n  },\n  {'
    elif b'\r\n{' in after:
        return b'\r\n},\r\n{'
    elif b'\n{' in after:
        return b'\n},\n{'
    # Fallback
    if b'\r\n' in header:
        return b'\r\n},\r\n{'
    return b'\n},\n{'


def iter_json_records_fast(filepath: Path, separator: bytes = None):
    """
 JSON pretty-print
 256MB + bytes.split()Parse 100x
 JSON CR/LF literal CRLF/LF
 JSON Layer
    """
    if separator is None:
        separator = _detect_separator(filepath)

    CHUNK_SIZE = 256 * 1024 * 1024  # 256MB

    with open(filepath, 'rb') as f:
        first_chunk = f.read(CHUNK_SIZE)
        bracket_pos = first_chunk.find(b'[')
        if bracket_pos == -1:
            return
        remainder = first_chunk[bracket_pos + 1:]
        del first_chunk

        while True:
            new_data = f.read(CHUNK_SIZE)
            eof = (len(new_data) == 0)

            data = remainder + new_data if remainder else new_data
            del new_data

            if not data:
                break

            parts = data.split(separator)
            del data

            if not eof:
                remainder = parts[-1]
                parts = parts[:-1]
            else:
                remainder = b''

            for idx_part, part in enumerate(parts):
                part = part.strip()
                if not part:
                    continue

                # After splitting by separator (e.g. "\r\n  },\r\n  {"):
                # - The separator includes the closing "}" of current record
                #   AND the opening "{" of the next record.
                # - Exception: the very first part starts with "{" (from "[{")
                #   and the very last part (at EOF) ends with "}\r\n]".

                is_last_at_eof = eof and (idx_part == len(parts) - 1)

                if is_last_at_eof:
                    # Last record in file: has its own "}" plus trailing "]"
                    if not part.startswith(b'{'):
                        part = b'{' + part
                    # Strip trailing array close
                    idx = part.rfind(b'}')
                    if idx >= 0:
                        part = part[:idx + 1]
                else:
                    # Normal record: separator consumed closing "}"
                    if not part.startswith(b'{'):
                        part = b'{' + part
                    part = part + b'}'

                yield part

            if eof:
                break


# ---------------------------------------------------------------------------
# Pass 1: Collect keys and levels (memory-light, ~2-3GB)
# ---------------------------------------------------------------------------

def pass1_collect_keys(files: list[Path], workers: int) -> tuple[dict, set]:
    """
 Pass 1: File
 Return:
      - pos_key_to_level: dict[tuple_of_msg_ids -> best_level_str]
      - neg_keys: set[msg_id_str]
    """
    pos_key_to_level: dict[tuple, str] = {}
    neg_keys: set[str] = set()

    for fp in files:
        print(f"  Scanning {fp.name} ...", end=' ', flush=True)
        t_file = time.time()
        separator = _detect_separator(fp)

        batch: list[bytes] = []
        batch_size = 1000  # smaller batches to reduce memory per future
        total_in_file = 0
        futures = []

        with ProcessPoolExecutor(max_workers=workers) as executor:
            for raw in iter_json_records_fast(fp, separator):
                batch.append(raw)
                if len(batch) >= batch_size:
                    futures.append(executor.submit(_classify_batch, batch))
                    batch = []

                    # Drain completed futures periodically to limit memory
                    if len(futures) >= workers * 2:
                        for fut in futures:
                            for kind, key, level in fut.result():
                                if kind == 'pos':
                                    old = pos_key_to_level.get(key)
                                    if old is None or LEVEL_ORDER[level] > LEVEL_ORDER[old]:
                                        pos_key_to_level[key] = level
                                else:
                                    neg_keys.add(key)
                                total_in_file += 1
                        futures = []

            if batch:
                futures.append(executor.submit(_classify_batch, batch))

            for fut in futures:
                for kind, key, level in fut.result():
                    if kind == 'pos':
                        old = pos_key_to_level.get(key)
                        if old is None or LEVEL_ORDER[level] > LEVEL_ORDER[old]:
                            pos_key_to_level[key] = level
                    else:
                        neg_keys.add(key)
                    total_in_file += 1

        elapsed = time.time() - t_file
        rps = total_in_file / elapsed if elapsed > 0 else 0
        print(f"{total_in_file} records ({elapsed:.1f}s, {rps:.0f} rec/s)")

    return pos_key_to_level, neg_keys


# ---------------------------------------------------------------------------
# Pass 2: Collect needed records (reservoir sampling)
# ---------------------------------------------------------------------------

def pass2_collect_records(
    files: list[Path],
    pos_key_to_level: dict[tuple, str],
    neg_keys_orig: set[str],
    level_targets: dict[str, int],
    neg_target: int,
    seed: int,
) -> tuple[dict[str, list], list]:
    """
 Pass 2: level SampleSample
 Reservoir sampling
    level_targets: dict mapping level name to target count, e.g. {'L1': 3000, 'L2': 2000, ...}
    """
    rng = random.Random(seed)
    neg_keys = set(neg_keys_orig)  # copy so we can discard

    level_pools: dict[str, list] = {lvl: [] for lvl in level_targets}
    level_seen: dict[str, int] = {lvl: 0 for lvl in level_targets}
    neg_pool: list = []
    neg_seen = 0
    collected_pos_keys: set[tuple] = set()

    for fp in files:
        print(f"  Collecting from {fp.name} ...", end=' ', flush=True)
        t_file = time.time()
        separator = _detect_separator(fp)
        file_collected = 0

        for raw in iter_json_records_fast(fp, separator):
            case = orjson.loads(raw)

            if case['label'] == 'tangled':
                atoms = case['atoms']
                msg_ids = tuple(sorted(a['message_id'] for a in atoms))

                best_level = pos_key_to_level.get(msg_ids)
                if best_level is None or best_level not in level_targets:
                    continue
                if msg_ids in collected_pos_keys:
                    continue

                actual_level = classify_difficulty_from_atoms(atoms)
                if actual_level != best_level:
                    continue

                collected_pos_keys.add(msg_ids)
                case['difficulty_level'] = best_level

                # Reservoir sampling
                target = level_targets[best_level]
                level_seen[best_level] += 1
                pool = level_pools[best_level]
                if len(pool) < target:
                    pool.append(case)
                else:
                    j = rng.randint(0, level_seen[best_level] - 1)
                    if j < target:
                        pool[j] = case
                file_collected += 1
            else:
                neg_key = case['atoms'][0]['message_id']
                if neg_key not in neg_keys:
                    continue
                neg_keys.discard(neg_key)

                case['difficulty_level'] = 'N/A'
                neg_seen += 1
                if len(neg_pool) < neg_target:
                    neg_pool.append(case)
                else:
                    j = rng.randint(0, neg_seen - 1)
                    if j < neg_target:
                        neg_pool[j] = case
                file_collected += 1

        elapsed = time.time() - t_file
        print(f"{file_collected} ({elapsed:.1f}s)")

    return level_pools, neg_pool


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_balanced_dataset(
    filter_dir: Path,
    per_level: int,
    neg_ratio: float,
    seed: int,
    out_dir: Path,
    workers: int = 0,
    level_counts: dict[str, int] | None = None,
):
    if workers <= 0:
        workers = min(mp.cpu_count(), 16)

    # Build level_targets: per-level target counts
    if level_counts:
        level_targets = level_counts
    else:
        level_targets = {f'L{i}': per_level for i in range(1, 6)}

    t0 = time.time()

    files = sorted(filter_dir.glob('*_filtered.json'))
    if not files:
        print(f"[ERROR] No *_filtered.json files found in {filter_dir}")
        sys.exit(1)

    print("=" * 60)
    print("Files to process:")
    print("=" * 60)
    total_bytes = 0
    for fp in files:
        sz = fp.stat().st_size
        total_bytes += sz
        print(f"  {fp.name} ({sz / (1024**3):.1f} GB)")
    print(f"  Total: {total_bytes / (1024**3):.1f} GB | Workers: {workers}\n")

    # =========================================================
    # Pass 1: Collect dedup keys + difficulty levels
    # =========================================================
    print("=" * 60)
    print("Pass 1: Scanning for keys and difficulty levels")
    print("=" * 60)

    pos_key_to_level, neg_keys = pass1_collect_keys(files, workers)

    t1 = time.time()
    print(f"\n  Unique positive keys: {len(pos_key_to_level)}")
    print(f"  Unique negative keys: {len(neg_keys)}")
    print(f"  Pass 1 time: {t1 - t0:.1f}s")

    level_dist = defaultdict(int)
    for lvl in pos_key_to_level.values():
        level_dist[lvl] += 1
    print("  Level distribution (deduped):")
    for lvl in sorted(level_dist):
        print(f"    {lvl}: {level_dist[lvl]}")
    print()

    # =========================================================
    # Pass 2: Collect actual records
    # =========================================================
    total_pos_target = sum(level_targets.values())
    neg_target = int(neg_ratio * total_pos_target)
    print("=" * 60)
    targets_str = ", ".join(f"{k}:{v}" for k, v in sorted(level_targets.items()))
    print(f"Pass 2: Collecting records ({targets_str}, neg={neg_target})")
    print("=" * 60)

    level_pools, neg_pool = pass2_collect_records(
        files, pos_key_to_level, neg_keys, level_targets, neg_target, seed
    )

    t2 = time.time()
    print(f"\n  Pass 2 time: {t2 - t1:.1f}s")
    for lvl in sorted(level_targets):
        n = len(level_pools[lvl])
        target = level_targets[lvl]
        warn = f" [WARN: < {target}]" if n < target else ""
        print(f"    {lvl}: {n}{warn}")
    print(f"    Neg: {len(neg_pool)}")
    print()

    # =========================================================
    # Final sampling and output
    # =========================================================
    print("=" * 60)
    print("Final: Sampling and writing output")
    print("=" * 60)

    rng = random.Random(seed + 1)
    output = []

    for lvl in sorted(level_targets):
        pool = level_pools[lvl]
        target = level_targets[lvl]
        n = min(target, len(pool))
        if n < target:
            print(f"  [WARN] {lvl}: only {n} available, requested {target}")
        selected = rng.sample(pool, n) if len(pool) > n else pool
        output.extend(selected)
        print(f"  {lvl}: {n}")

    neg_n = min(neg_target, len(neg_pool))
    if neg_n < neg_target:
        print(f"  [WARN] Neg: only {neg_n} available, requested {neg_target}")
    neg_selected = rng.sample(neg_pool, neg_n) if len(neg_pool) > neg_n else neg_pool
    output.extend(neg_selected)
    print(f"  Neg: {neg_n}")

    rng.shuffle(output)

    out_dir.mkdir(parents=True, exist_ok=True)
    total_pos = sum(min(level_targets[l], len(level_pools[l])) for l in level_targets)
    out_file = out_dir / f"balanced_N{total_pos}.json"
    with open(out_file, 'wb') as f:
        f.write(orjson.dumps(output, option=orjson.OPT_INDENT_2))

    t_end = time.time()
    print(f"\n  Output: {out_file}")
    print(f"  Records: {len(output)}")
    print(f"  Total time: {t_end - t0:.1f}s\n")

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    final_levels = defaultdict(int)
    for c in output:
        final_levels[c['difficulty_level']] += 1
    for lvl in sorted(final_levels):
        print(f"  {lvl}: {final_levels[lvl]}")
    print(f"  Total: {len(output)}")


def parse_level_counts(s: str) -> dict[str, int]:
    """Parse level counts string like 'L1:3000,L2:2000,L3:1000,L4:500,L5:500'."""
    result = {}
    for item in s.split(','):
        item = item.strip()
        if ':' not in item:
            raise argparse.ArgumentTypeError(
                f"Invalid format '{item}', expected 'L1:N,L2:N,...'")
        level, count = item.split(':', 1)
        level = level.strip().upper()
        if level not in ('L1', 'L2', 'L3', 'L4', 'L5'):
            raise argparse.ArgumentTypeError(
                f"Unknown level '{level}', must be L1-L5")
        result[level] = int(count.strip())
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Build balanced multi-difficulty dataset from filtered JSONs"
    )
    parser.add_argument('--filter-dir', type=Path,
                        default=Path('data/lkp_synthesize/filter'))
    parser.add_argument('--per-level', type=int, default=5000,
                        help='Default count per level (overridden by --level-counts)')
    parser.add_argument('--level-counts', type=parse_level_counts, default=None,
                        help='Per-level counts, e.g. "L1:3000,L2:2000,L3:1000,L4:500,L5:500"')
    parser.add_argument('--neg-ratio', type=float, default=0.2,
                        help='Negatives = ratio * total_positives')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-dir', type=Path,
                        default=Path('data/lkp_synthesize/balanced'))
    parser.add_argument('--workers', type=int, default=0,
                        help='Parallel workers (0=auto)')
    args = parser.parse_args()

    build_balanced_dataset(
        filter_dir=args.filter_dir,
        per_level=args.per_level,
        neg_ratio=args.neg_ratio,
        seed=args.seed,
        out_dir=args.out_dir,
        workers=args.workers,
        level_counts=args.level_counts,
    )


if __name__ == '__main__':
    main()
