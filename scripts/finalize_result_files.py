#!/usr/bin/env python3
"""Merge split per-image JSON exports and normalize result CSV row order.

The JSON merge is deliberately conservative:

* all non-record metadata must match across parts;
* logical duplicate records must be exactly equal;
* conflicting duplicate identities abort the merge;
* the merged file is validated before atomically replacing its destination;
* source part files are never modified or removed.

CSV normalization delegates to ``normalize_result_csvs.py``, which preserves
every decoded cell string verbatim and changes only row order.
"""

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from harness.result_order import condition_sort_key  # noqa: E402
from normalize_result_csvs import (  # noqa: E402
    discover_result_csvs,
    normalize_csv,
)


SPLIT_JSON_PATTERN = re.compile(
    r"^(?P<base>.+_FINE_GRAINED_PER_IMAGE)(?P<part>\d+)\.json$"
)
REQUIRED_DOCUMENT_FIELDS = (
    "model",
    "dataset",
    "protocol",
    "description",
    "records",
)
REQUIRED_RECORD_FIELDS = (
    "model",
    "dataset",
    "class_name",
    "transformation_level",
    "sample_id",
)


def _display_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.is_relative_to(REPOSITORY_ROOT):
        return resolved.relative_to(REPOSITORY_ROOT)
    return resolved


def discover_split_json_groups(
    inputs: Sequence[Path],
) -> Dict[Path, List[Path]]:
    """Group numbered per-image JSON parts by canonical output path."""
    grouped: Dict[Path, Dict[int, Path]] = {}
    for input_path in inputs:
        path = input_path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")

        candidates: Iterable[Path]
        if path.is_file():
            candidates = [path]
        else:
            candidates = path.rglob("*.json")

        for candidate in candidates:
            match = SPLIT_JSON_PATTERN.fullmatch(candidate.name)
            if match is None:
                continue
            part_number = int(match.group("part"))
            output_path = candidate.with_name(f"{match.group('base')}.json")
            parts = grouped.setdefault(output_path.resolve(), {})
            existing = parts.get(part_number)
            if existing is not None and existing.resolve() != candidate.resolve():
                raise ValueError(
                    f"Duplicate JSON part number {part_number} for {output_path}: "
                    f"{existing} and {candidate}"
                )
            parts[part_number] = candidate.resolve()

    return {
        output: [numbered[number] for number in sorted(numbered)]
        for output, numbered in sorted(
            grouped.items(), key=lambda item: str(item[0]).casefold()
        )
    }


def _load_json_document(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as json_file:
        document = json.load(json_file)
    if not isinstance(document, dict):
        raise ValueError(f"Top-level JSON value must be an object: {path}")

    missing = [field for field in REQUIRED_DOCUMENT_FIELDS if field not in document]
    if missing:
        raise ValueError(f"Missing top-level fields {missing} in {path}")
    if not isinstance(document["records"], list):
        raise ValueError(f"'records' must be a list in {path}")
    return document


def _record_identity(record: Mapping[str, Any], source: Path) -> Tuple[str, ...]:
    missing = [field for field in REQUIRED_RECORD_FIELDS if field not in record]
    if missing:
        raise ValueError(f"Record is missing fields {missing} in {source}")
    return tuple(str(record[field]) for field in REQUIRED_RECORD_FIELDS)


def _record_sort_key(record: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        condition_sort_key(str(record["transformation_level"])),
        str(record["class_name"]).casefold(),
        str(record["sample_id"]).casefold(),
    )


def build_merged_document(
    part_paths: Sequence[Path],
) -> Tuple[Dict[str, Any], int]:
    """Validate and merge parts, returning document and duplicate count."""
    if not part_paths:
        raise ValueError("At least one JSON part is required")

    first_document = _load_json_document(part_paths[0])
    metadata = {
        key: value for key, value in first_document.items() if key != "records"
    }
    records_by_identity: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    duplicate_count = 0

    for part_path in part_paths:
        document = _load_json_document(part_path)
        part_metadata = {
            key: value for key, value in document.items() if key != "records"
        }
        if part_metadata != metadata:
            raise ValueError(
                f"Top-level metadata differs between {part_paths[0]} and "
                f"{part_path}"
            )

        for index, record in enumerate(document["records"]):
            if not isinstance(record, dict):
                raise ValueError(
                    f"Record {index} in {part_path} must be a JSON object"
                )
            if record.get("model") != metadata["model"]:
                raise ValueError(
                    f"Record {index} in {part_path} has model "
                    f"{record.get('model')!r}; expected {metadata['model']!r}"
                )
            if record.get("dataset") != metadata["dataset"]:
                raise ValueError(
                    f"Record {index} in {part_path} has dataset "
                    f"{record.get('dataset')!r}; expected {metadata['dataset']!r}"
                )

            identity = _record_identity(record, part_path)
            existing = records_by_identity.get(identity)
            if existing is None:
                records_by_identity[identity] = record
            elif existing == record:
                duplicate_count += 1
            else:
                raise ValueError(
                    "Conflicting duplicate record identity found while merging "
                    f"{part_path}: {identity}"
                )

    records = sorted(records_by_identity.values(), key=_record_sort_key)
    return {**metadata, "records": records}, duplicate_count


def _serialize_json(document: Mapping[str, Any]) -> str:
    return json.dumps(
        document,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ) + "\n"


def merge_json_parts(
    part_paths: Sequence[Path], output_path: Path, check_only: bool = False
) -> Tuple[bool, int, int]:
    """Merge numbered JSON parts and return changed/record/duplicate counts."""
    document, duplicate_count = build_merged_document(part_paths)
    serialized = _serialize_json(document)
    changed = not output_path.exists() or output_path.read_text(
        encoding="utf-8"
    ) != serialized
    if check_only or not changed:
        return changed, len(document["records"]), duplicate_count

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(serialized)

        verified = _load_json_document(temporary_path)
        if verified != document:
            raise ValueError(
                "JSON round-trip integrity verification failed; destination "
                "was left unchanged."
            )
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return True, len(document["records"]), duplicate_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge numbered *_FINE_GRAINED_PER_IMAGE<N>.json files and "
            "normalize all result CSV row ordering. JSON source parts and "
            "all CSV cell values are preserved."
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
        help="Report required changes without writing files.",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Merge JSON parts without normalizing CSVs.",
    )
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="Normalize CSVs without merging JSON parts.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.json_only and args.csv_only:
        print("ERROR: --json-only and --csv-only cannot be used together.", file=sys.stderr)
        return 2

    inputs = args.paths or [REPOSITORY_ROOT]
    change_count = 0
    error_count = 0

    if not args.csv_only:
        try:
            json_groups = discover_split_json_groups(inputs)
        except (FileNotFoundError, OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

        for output_path, part_paths in json_groups.items():
            try:
                changed, record_count, duplicate_count = merge_json_parts(
                    part_paths, output_path, check_only=args.check
                )
            except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
                error_count += 1
                print(f"ERROR {_display_path(output_path)}: {exc}", file=sys.stderr)
                continue

            if changed:
                change_count += 1
                action = "NEEDS_MERGE" if args.check else "MERGED"
            else:
                action = "OK"
            print(
                f"{action} {_display_path(output_path)} from {len(part_paths)} "
                f"part(s): {record_count} unique record(s), "
                f"{duplicate_count} exact duplicate(s) removed"
            )

    if not args.json_only:
        try:
            csv_paths = discover_result_csvs(inputs)
        except (FileNotFoundError, OSError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

        for csv_path in csv_paths:
            try:
                changed, removed_empty_rows = normalize_csv(
                    csv_path, check_only=args.check
                )
            except (OSError, UnicodeError, ValueError) as exc:
                error_count += 1
                print(f"ERROR {_display_path(csv_path)}: {exc}", file=sys.stderr)
                continue

            if changed:
                change_count += 1
                action = "NEEDS_NORMALIZATION" if args.check else "NORMALIZED"
            else:
                action = "OK"
            suffix = (
                f"; removed {removed_empty_rows} empty row(s)"
                if removed_empty_rows
                else ""
            )
            print(f"{action} {_display_path(csv_path)}{suffix}")

    print(f"Summary: {change_count} change(s), {error_count} error(s).")
    if error_count:
        return 2
    if args.check and change_count:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
