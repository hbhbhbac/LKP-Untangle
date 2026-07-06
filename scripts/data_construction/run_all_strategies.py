"""
run_all_strategies.py tangled

  ../data/lkp_synthesize/origin/ JSON + statistic.txt
  ../data/lkp_synthesize/filter/ JSON + statistic.txt

atoms 2-10stmts 2-4000
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

ORIGIN_DIR = _HERE / "data" / "lkp_synthesize" / "origin"
FILTER_DIR = _HERE / "data" / "lkp_synthesize" / "filter"

ORIGIN_DIR.mkdir(parents=True, exist_ok=True)
FILTER_DIR.mkdir(parents=True, exist_ok=True)

FILTER_MIN_ATOMS = 2
FILTER_MAX_ATOMS = 10
FILTER_MIN_STMTS = 2
FILTER_MAX_STMTS = 4000

# seriesratio 1 0
STRATEGIES = [
    '2wayfile',
    '2wayfunction',
    'connected_file',
    'connected_func',
    'all_file',
    'all_func',
    'all_direct',
]

STRATEGY_INDEX = {name: i for i, name in enumerate(STRATEGIES)}


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def build_one_strategy(strategy: str, series_limit: int = None) -> dict:
    """
    """
    import json
    from build_tangled_dataset import (
        build_dataset, filter_dataset,
        collect_dataset_stats, format_stats,
        STRATEGY_NAMES,
    )

    t0 = time.time()
    print(f"[{strategy}] ...", flush=True)

    # ratio 100%
    ratios = [0.0] * 7
    ratios[STRATEGY_INDEX[strategy]] = 1.0

    pos_cases, neg_cases = build_dataset(
        strategy_ratios=ratios,
        min_code_lines=5,
        neg_min_code_lines=10,
        series_limit=series_limit,
    )
    dataset = pos_cases + neg_cases

    origin_path = ORIGIN_DIR / f"tangled_{strategy}.json"
    with open(origin_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    print(f"[{strategy}] {len(dataset)} {elapsed:.0f}s {origin_path}", flush=True)

    origin_stats = collect_dataset_stats(dataset)
    origin_text = format_stats(origin_stats, title=f': {strategy} (origin)')

    filtered = filter_dataset(
        dataset,
        min_stmts=FILTER_MIN_STMTS,
        max_stmts=FILTER_MAX_STMTS,
        min_atoms=FILTER_MIN_ATOMS,
        max_atoms=FILTER_MAX_ATOMS,
    )

    filter_path = FILTER_DIR / f"tangled_{strategy}_filtered.json"
    with open(filter_path, 'w', encoding='utf-8') as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    print(f"[{strategy}] {len(filtered)} {filter_path}", flush=True)

    filter_stats = collect_dataset_stats(filtered)
    filter_text = format_stats(
        filter_stats,
        title=f': {strategy} (filter: atoms {FILTER_MIN_ATOMS}-{FILTER_MAX_ATOMS}, stmts {FILTER_MIN_STMTS}-{FILTER_MAX_STMTS})'
    )

    return {
        'strategy': strategy,
        'origin_count': len(dataset),
        'filter_count': len(filtered),
        'elapsed': elapsed,
        'origin_text': origin_text,
        'filter_text': filter_text,
    }


# ---------------------------------------------------------------------------
# statistic.txt
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description=" tangled ")
    parser.add_argument(
        '--strategies', nargs='+', default=STRATEGIES,
        choices=STRATEGIES,
        help=''
    )
    parser.add_argument(
        '--workers', type=int, default=3,
        help=' 2-4'
    )
    parser.add_argument(
        '--series-limit', type=int, default=None,
        help=' series '
    )
    args = parser.parse_args()

    target_strategies = args.strategies
    workers = min(args.workers, len(target_strategies))
    series_limit = args.series_limit

    print(f"[main] : {target_strategies}")
    print(f"[main] : {workers}")
    print(f"[main] : {ORIGIN_DIR}")
    print(f"[main] : {FILTER_DIR}")
    print(f"[main] : atoms {FILTER_MIN_ATOMS}-{FILTER_MAX_ATOMS}, "
          f"stmts {FILTER_MIN_STMTS}-{FILTER_MAX_STMTS}")

    results = {}
    t_total = time.time()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(build_one_strategy, s, series_limit): s
            for s in target_strategies
        }
        for future in as_completed(futures):
            strategy = futures[future]
            try:
                result = future.result()
                results[strategy] = result
                print(f"[main] OK {strategy} done "
                      f"(origin={result['origin_count']}, filter={result['filter_count']}, "
                      f"elapsed={result['elapsed']:.0f}s)", flush=True)
            except Exception as e:
                print(f"[main] FAIL {strategy}: {e}", flush=True)
                import traceback
                traceback.print_exc()

    elapsed_total = time.time() - t_total
    print(f"\n[main] {elapsed_total:.0f}s")

    # statistic.txt
    _write_statistic(results, target_strategies)


def _write_statistic(results: dict, strategy_order: list):
    """ statistic.txt"""
    # origin statistic.txt
    origin_lines = [
        "=" * 70,
        " LKP Synthesize Dataset Origin ",
        f" : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
    ]
    for s in strategy_order:
        if s in results:
            origin_lines.append(results[s]['origin_text'])
            origin_lines.append("")

    origin_lines.append("=" * 70)
    origin_lines.append(" ")
    origin_lines.append("=" * 70)
    origin_lines.append(f" {'':<20} {'':>8} {'':>8} {'':>8} {'(s)':>8}")
    origin_lines.append("  " + "-" * 56)
    for s in strategy_order:
        if s in results:
            r = results[s]
            from build_tangled_dataset import collect_dataset_stats
            import json
            origin_path = ORIGIN_DIR / f"tangled_{s}.json"
            try:
                with open(origin_path, encoding='utf-8') as f:
                    ds = json.load(f)
                pos_n = sum(1 for d in ds if d['label'] == 'tangled')
                neg_n = sum(1 for d in ds if d['label'] == 'no_split')
            except Exception:
                pos_n = neg_n = -1
            origin_lines.append(
                f"  {s:<20} {r['origin_count']:>8} {pos_n:>8} {neg_n:>8} {r['elapsed']:>8.0f}"
            )

    origin_stat_path = ORIGIN_DIR / "statistic.txt"
    with open(origin_stat_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(origin_lines))
    print(f"[main] {origin_stat_path}")

    # filter statistic.txt
    filter_lines = [
        "=" * 70,
        " LKP Synthesize Dataset Filter ",
        f" : atoms {FILTER_MIN_ATOMS}-{FILTER_MAX_ATOMS}, "
        f"stmts {FILTER_MIN_STMTS}-{FILTER_MAX_STMTS}",
        f" : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
    ]
    for s in strategy_order:
        if s in results:
            filter_lines.append(results[s]['filter_text'])
            filter_lines.append("")

    filter_lines.append("=" * 70)
    filter_lines.append(" ")
    filter_lines.append("=" * 70)
    filter_lines.append(f" {'':<20} {'':>8} {'':>8} {'':>8} {'':>8}")
    filter_lines.append("  " + "-" * 56)
    for s in strategy_order:
        if s in results:
            r = results[s]
            filter_path = FILTER_DIR / f"tangled_{s}_filtered.json"
            try:
                with open(filter_path, encoding='utf-8') as f:
                    ds = json.load(f)
                pos_n = sum(1 for d in ds if d['label'] == 'tangled')
                neg_n = sum(1 for d in ds if d['label'] == 'no_split')
            except Exception:
                pos_n = neg_n = -1
            retain = f"{r['filter_count']/r['origin_count']*100:.1f}%" if r['origin_count'] else "N/A"
            filter_lines.append(
                f"  {s:<20} {r['filter_count']:>8} {pos_n:>8} {neg_n:>8} {retain:>8}"
            )

    filter_stat_path = FILTER_DIR / "statistic.txt"
    with open(filter_stat_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(filter_lines))
    print(f"[main] {filter_stat_path}")


if __name__ == '__main__':
    main()
