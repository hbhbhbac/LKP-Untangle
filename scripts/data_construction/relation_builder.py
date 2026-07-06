"""
relation_builder.py

 SQLite
1. cover letter processors.version.link_based_analyzer
2. + series is_patch / is_cover / series_id
    series patch comment

- Re: comment is_patch=1
- series series patch patch_version comment None
- cover series.has_cover = 0 1/n is_cover=1
"""

from __future__ import annotations

import argparse
import re
from typing import List, Optional

from core import STRUCTURE_DIR
from lkml_preprocess import parse_epoch_spec
from relation_builder.series_builder import build_series
from relation_builder.versions_builder import build_version_relations


def run_relation_builder(
    epoch_spec: Optional[str] = None,
    clear_relations: bool = False,
    build_versions: bool = True,
    build_series_flag: bool = True,
    version_workers: int = 4,
    series_workers: int = 8,
    match_methods_spec: Optional[str] = None,
    similarity_threshold: float = 0.9,
    max_time_diff_days: Optional[float] = None,
) -> None:
    """

    1. / series series_id / is_patch / is_cover + series
    2. cover letter cover_*

    Args:
        epoch_spec: epoch
            - "14-18"
            - "14,15,18"
            - "14-14,18,20"
             epoch
        clear_relations: cover_* series
        build_versions: cover letter
        build_series_flag: series
        match_methods_spec: --versions
            - "0,1,2" 012
            - "0-2,4" 0124
            0=body_match, 1=subject_match, 2=thread_match,
                     3=subject_fuzz_match, 4=rep_body_match
        similarity_threshold: 0.9 (90%)
        max_time_diff_days: None
    """
    if epoch_spec:
        selected_epochs: Optional[List[int]] = sorted(parse_epoch_spec(epoch_spec))
        print(f" epoch: {selected_epochs}")
    else:
        selected_epochs = None
        print(" epoch SQLite epoch")

    if not (build_versions or build_series_flag):
        build_versions = True
        build_series_flag = True

    # series
    if build_series_flag:
        print("\n=== series ===")
        build_series(epoch_spec=epoch_spec, clear_relations=clear_relations, workers=series_workers)

    # series
    if build_versions:
        print("\n=== series ===")
        build_version_relations(
            epoch_spec=epoch_spec,
            workers=version_workers,
            match_methods_spec=match_methods_spec,
            similarity_threshold=similarity_threshold,
            max_time_diff_days=max_time_diff_days,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            " cover letter series "
            "Relation Builder"
        )
    )
    parser.add_argument(
        "--epochs",
        type=str,
        default="",
        help=(
            " epoch '14-18' '14,15,18' '14-14,18,20'"
            " SQLite epoch"
        ),
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help=" cover_* series ",
    )
    parser.add_argument(
        "--versions",
        action="store_true",
        help=" cover letter cover_* ",
    )
    parser.add_argument(
        "--series",
        action="store_true",
        help=" series series + series_id / is_patch / is_cover ",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="16",
    )
    parser.add_argument(
        "--series-workers",
        type=int,
        default=16,
        help="series 16",
    )
    parser.add_argument(
        "--match-methods",
        type=str,
        default="",
        help=(
            " --versions'0,1,2' '0-2,4'\n"
            "0=body_match, 1=subject_match, 2=thread_match, "
            "3=subject_fuzz_match, 4=rep_body_match。\n"
            ""
        ),
    )
    parser.add_argument(
        "--similarity",
        type=float,
        default=0.9,
        help="subject_fuzz_match 0.0-1.00.9 90%%",
    )
    parser.add_argument(
        "--max-days",
        type=float,
        default=None,
        help=(
            "subject_fuzz_match 90.12 90.12 "
            ""
        ),
    )

    args = parser.parse_args()
    epoch_spec = args.epochs.strip() or None

    # --versions / --series
    build_versions = args.versions
    build_series_flag = args.series
    if not (build_versions or build_series_flag):
        build_versions = True
        build_series_flag = True

    run_relation_builder(
        epoch_spec=epoch_spec,
        clear_relations=args.clear,
        build_versions=build_versions,
        build_series_flag=build_series_flag,
        version_workers=args.workers,
        series_workers=args.series_workers,
        match_methods_spec=args.match_methods.strip() or None,
        similarity_threshold=args.similarity,
        max_time_diff_days=args.max_days,
    )


if __name__ == "__main__":
    main()

