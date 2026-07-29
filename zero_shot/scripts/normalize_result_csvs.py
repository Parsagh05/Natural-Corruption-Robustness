#!/usr/bin/env python3
"""Normalize condition ordering in all exported SP/PX result CSV files."""

import argparse
import csv
import io
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPOSITORY_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from zero_shot.harness.result_order import (  # noqa: E402
    condition_sort_key,
    parse_condition,
)


RESULT_SUFFIXES = ("_PX.csv", "_SP.csv")


def discover_result_csvs(inputs: Sequence[Path]) -> List[Path]:
    """Return unique result CSVs found under files or directories."""
    discovered = set()
    for input_path in inputs:
        path = input_path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")
        candidates: Iterable[Path]
        if path.is_file():
            candidates = [path]
        else:
            candidates = path.rglob("*.csv")
        for candidate in candidates:
            if candidate.name.endswith(RESULT_SUFFIXES):
                discovered.add(candidate.resolve())
    return sorted(discovered, key=lambda path: str(path).casefold())


def _decode_csv(path: Path) -> Tuple[str, str, str]:
    raw = path.read_bytes()
    has_bom = raw.startswith(b"\xef\xbb\xbf")
    encoding = "utf-8-sig" if has_bom else "utf-8"
    line_ending = "\r\n" if b"\r\n" in raw else "\n"
    return raw.decode(encoding), encoding, line_ending


def normalized_rows(path: Path) -> Tuple[List[str], List[List[str]], int, bool, str, str]:
    """Read, validate, and order one result CSV without changing cell values."""
    text, encoding, line_ending = _decode_csv(path)
    parsed = list(csv.reader(io.StringIO(text, newline="")))
    if not parsed:
        raise ValueError("CSV is empty")

    header = parsed[0]
    if not header or header[0] != "transformation_level":
        raise ValueError(
            "First column must be 'transformation_level'; "
            f"got {header[0] if header else '<missing>'!r}"
        )

    rows = []
    removed_empty_rows = 0
    seen_conditions = set()
    for line_number, row in enumerate(parsed[1:], start=2):
        if not row or all(not value.strip() for value in row):
            removed_empty_rows += 1
            continue
        if len(row) != len(header):
            raise ValueError(
                f"Line {line_number} has {len(row)} columns; expected {len(header)}"
            )

        condition = row[0].strip()
        parse_condition(condition)
        if condition in seen_conditions:
            raise ValueError(
                f"Line {line_number} repeats condition {condition!r}"
            )
        seen_conditions.add(condition)
        rows.append(row)

    ordered = sorted(rows, key=lambda row: condition_sort_key(row[0]))
    changed = removed_empty_rows > 0 or ordered != rows
    return (
        header,
        ordered,
        removed_empty_rows,
        changed,
        encoding,
        line_ending,
    )


def normalize_csv(path: Path, check_only: bool = False) -> Tuple[bool, int]:
    """Normalize one CSV atomically and return ``(changed, removed_empty)``."""
    (
        header,
        rows,
        removed_empty_rows,
        changed,
        encoding,
        line_ending,
    ) = normalized_rows(path)
    if not changed or check_only:
        return changed, removed_empty_rows

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            writer = csv.writer(temporary_file, lineterminator=line_ending)
            writer.writerow(header)
            writer.writerows(rows)

        # Verify the serialized temporary file before it can replace the
        # original. This compares exact decoded cell strings, including metric
        # precision/trailing zeros, and also confirms the requested row order.
        (
            verified_header,
            verified_rows,
            verified_empty_rows,
            verified_changed,
            _,
            _,
        ) = normalized_rows(temporary_path)
        if (
            verified_header != header
            or verified_rows != rows
            or verified_empty_rows != 0
            or verified_changed
        ):
            raise ValueError(
                "Round-trip integrity verification failed; original CSV was "
                "left unchanged."
            )
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return True, removed_empty_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Order clean/corruption result rows consistently in every *_PX.csv "
            "and *_SP.csv file. Cell values are preserved verbatim."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Result file or directory to scan (default: repository root).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report files needing normalization without modifying them.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    inputs = args.paths or [REPOSITORY_ROOT]
    try:
        result_paths = discover_result_csvs(inputs)
    except (FileNotFoundError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not result_paths:
        print("No *_PX.csv or *_SP.csv result files found.")
        return 0

    changed_count = 0
    error_count = 0
    for path in result_paths:
        display_path = (
            path.relative_to(REPOSITORY_ROOT)
            if path.is_relative_to(REPOSITORY_ROOT)
            else path
        )
        try:
            changed, removed_empty_rows = normalize_csv(
                path, check_only=args.check
            )
        except (OSError, UnicodeError, ValueError, csv.Error) as exc:
            error_count += 1
            print(f"ERROR {display_path}: {exc}", file=sys.stderr)
            continue

        if changed:
            changed_count += 1
            action = "NEEDS_NORMALIZATION" if args.check else "NORMALIZED"
            suffix = (
                f"; removed {removed_empty_rows} empty row(s)"
                if removed_empty_rows
                else ""
            )
            print(f"{action} {display_path}{suffix}")
        else:
            print(f"OK {display_path}")

    print(
        f"Checked {len(result_paths)} file(s): "
        f"{changed_count} {'need changes' if args.check else 'changed'}, "
        f"{error_count} error(s)."
    )
    if error_count:
        return 2
    if args.check and changed_count:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
