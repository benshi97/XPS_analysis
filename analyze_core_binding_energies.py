#!/usr/bin/env python3
"""Analyze gas-phase core binding energies from the pickle database.

The database stores one Python list per field. Matching row indexes across
those lists describe one binding-energy entry, so this script loads the lists,
joins them into row dictionaries, and optionally filters by SMILES/core level.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATABASE = SCRIPT_DIR / "database_2023_05_30"

FIELD_FILES = {
    "binding_energy_ev": "BE_list.pkl",
    "core_level": "Core_level_list.pkl",
    "isomeric_smiles": "Isomeric_SMILES_list.pkl",
    "iupac_name": "IUPAC_Name_list.pkl",
    "chemical_formula": "CF_list.pkl",
    "molecular_formula": "MF_list.pkl",
    "inchi": "InChI_list.pkl",
    "reference_doi": "Reference_DOI_list.pkl",
    "reference_text": "Reference_text_list.pkl",
    "reference_comment": "Reference_comment_list.pkl",
    "comment": "Comment_list.pkl",
    "binding_energy_comment": "BE_comment_list.pkl",
}

CSV_COLUMNS = [
    "row_index",
    "binding_energy_ev",
    "core_level",
    "core_element",
    "core_orbital",
    "isomeric_smiles",
    "iupac_name",
    "chemical_formula",
    "molecular_formula",
    "inchi",
    "reference_doi",
    "binding_energy_comment",
    "comment",
    "reference_text",
    "reference_comment",
]


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_database(database_dir: Path) -> dict[str, list[Any]]:
    missing = [name for name in FIELD_FILES.values() if not (database_dir / name).exists()]
    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(f"Missing expected pickle files in {database_dir}: {missing_list}")

    data = {
        field: load_pickle(database_dir / filename)
        for field, filename in FIELD_FILES.items()
    }

    lengths = {field: len(values) for field, values in data.items()}
    if len(set(lengths.values())) != 1:
        formatted = ", ".join(f"{field}={length}" for field, length in sorted(lengths.items()))
        raise ValueError(f"Pickle lists do not all have the same length: {formatted}")

    return data


def split_core_level(core_level: Any) -> tuple[str | None, str | None]:
    if not isinstance(core_level, str) or not core_level.strip():
        return None, None

    parts = core_level.strip().split(maxsplit=1)
    element = parts[0]
    orbital = parts[1] if len(parts) > 1 else None
    return element, orbital


def formula_to_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return " ".join(
            f"{element}{count if count != 1 else ''}"
            for element, count in sorted(value.items())
        )
    return str(value)


def make_rows(data: dict[str, list[Any]], include_null_be: bool = False) -> list[dict[str, Any]]:
    length = len(next(iter(data.values())))
    rows: list[dict[str, Any]] = []

    for row_index in range(length):
        binding_energy = data["binding_energy_ev"][row_index]
        if binding_energy is None and not include_null_be:
            continue

        core_element, core_orbital = split_core_level(data["core_level"][row_index])
        row = {
            field: values[row_index]
            for field, values in data.items()
        }
        row["row_index"] = row_index
        row["core_element"] = core_element
        row["core_orbital"] = core_orbital
        row["molecular_formula"] = formula_to_string(row["molecular_formula"])
        rows.append(row)

    return rows


def canonicalize_smiles(smiles: str) -> str | None:
    try:
        from rdkit import Chem, RDLogger  # type: ignore
    except ImportError:
        return None

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def smiles_matches(candidate: Any, queries: list[str], mode: str) -> bool:
    if not queries:
        return True
    if not isinstance(candidate, str):
        return False

    if mode == "contains":
        return any(query in candidate for query in queries)

    if mode == "exact":
        return candidate in queries

    candidate_canonical = canonicalize_smiles(candidate)
    for query in queries:
        if candidate == query:
            return True
        query_canonical = canonicalize_smiles(query)
        if candidate_canonical is not None and query_canonical is not None:
            if candidate_canonical == query_canonical:
                return True

    return False


def filter_rows(
    rows: Iterable[dict[str, Any]],
    smiles: list[str],
    smiles_match: str,
    core_level: str | None,
    element: str | None,
    name: str | None,
    min_be: float | None,
    max_be: float | None,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []

    for row in rows:
        if not smiles_matches(row["isomeric_smiles"], smiles, smiles_match):
            continue
        if core_level and row["core_level"] != core_level:
            continue
        if element and row["core_element"] != element:
            continue
        if name:
            haystack = " ".join(
                str(row.get(field) or "")
                for field in ("iupac_name", "chemical_formula", "isomeric_smiles", "inchi")
            ).lower()
            if name.lower() not in haystack:
                continue
        binding_energy = row["binding_energy_ev"]
        if min_be is not None and (binding_energy is None or binding_energy < min_be):
            continue
        if max_be is not None and (binding_energy is None or binding_energy > max_be):
            continue
        filtered.append(row)

    return filtered


def summarize(rows: list[dict[str, Any]], total_rows: int) -> str:
    energies = [
        float(row["binding_energy_ev"])
        for row in rows
        if isinstance(row["binding_energy_ev"], (int, float))
    ]
    lines = [
        f"Rows in database: {total_rows}",
        f"Rows selected: {len(rows)}",
        f"Rows with numeric binding energy: {len(energies)}",
        f"Unique SMILES selected: {len({row['isomeric_smiles'] for row in rows if row['isomeric_smiles']})}",
        f"Unique core levels selected: {len({row['core_level'] for row in rows if row['core_level']})}",
    ]

    if energies:
        lines.extend(
            [
                f"Binding energy range: {min(energies):.6g} to {max(energies):.6g} eV",
                f"Binding energy mean: {statistics.fmean(energies):.6g} eV",
                f"Binding energy median: {statistics.median(energies):.6g} eV",
            ]
        )

    core_counts = Counter(row["core_level"] for row in rows if row["core_level"])
    if core_counts:
        top = ", ".join(f"{level} ({count})" for level, count in core_counts.most_common(10))
        lines.append(f"Most common core levels: {top}")

    return "\n".join(lines)


def print_preview(rows: list[dict[str, Any]], limit: int) -> None:
    if limit <= 0:
        return

    shown = rows[:limit]
    if not shown:
        print("No matching rows.")
        return

    print()
    print(f"First {len(shown)} matching rows:")
    for row in shown:
        print(
            "\t".join(
                "" if row.get(column) is None else str(row.get(column))
                for column in (
                    "row_index",
                    "binding_energy_ev",
                    "core_level",
                    "isomeric_smiles",
                    "iupac_name",
                    "reference_doi",
                )
            )
        )


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(rows: list[dict[str, Any]], output: Path) -> None:
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_grouped_by_smiles(rows: list[dict[str, Any]], output: Path) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = row["isomeric_smiles"] or ""
        grouped[key].append(row)

    columns = [
        "isomeric_smiles",
        "iupac_name",
        "chemical_formula",
        "n_entries",
        "binding_energies_by_core_level",
        "row_indexes",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for smiles, entries in sorted(grouped.items()):
            by_core = defaultdict(list)
            for entry in entries:
                by_core[entry["core_level"]].append(entry["binding_energy_ev"])
            compact = {
                "" if core is None else str(core): energies
                for core, energies in sorted(by_core.items(), key=lambda item: str(item[0]))
            }
            writer.writerow(
                {
                    "isomeric_smiles": smiles,
                    "iupac_name": entries[0]["iupac_name"],
                    "chemical_formula": entries[0]["chemical_formula"],
                    "n_entries": len(entries),
                    "binding_energies_by_core_level": json.dumps(compact, sort_keys=True),
                    "row_indexes": ",".join(str(entry["row_index"]) for entry in entries),
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load gas-phase-database pickle lists and report core binding energies.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE,
        help=f"Database directory containing *_list.pkl files. Default: {DEFAULT_DATABASE}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path. Writes selected rows as CSV by default.",
    )
    parser.add_argument(
        "--format",
        choices=("csv", "jsonl"),
        default="csv",
        help="Output format for --output.",
    )
    parser.add_argument(
        "--group-by-smiles",
        action="store_true",
        help="Write one CSV row per SMILES with binding energies grouped by core level.",
    )
    parser.add_argument(
        "--smiles",
        action="append",
        default=[],
        help="SMILES query. Repeat for multiple queries.",
    )
    parser.add_argument(
        "--smiles-match",
        choices=("auto", "exact", "contains"),
        default="auto",
        help="SMILES matching mode. 'auto' does exact plus canonical matching if RDKit is installed.",
    )
    parser.add_argument("--core-level", help="Exact core-level filter, e.g. 'C 1s'.")
    parser.add_argument("--element", help="Exact core element filter, e.g. C, O, N, Cl.")
    parser.add_argument("--name", help="Case-insensitive substring filter over name/formula/SMILES/InChI.")
    parser.add_argument("--min-be", type=float, help="Minimum binding energy in eV.")
    parser.add_argument("--max-be", type=float, help="Maximum binding energy in eV.")
    parser.add_argument(
        "--include-null-be",
        action="store_true",
        help="Include rows without a numeric binding energy.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=10,
        help="Number of selected rows to print after the summary. Use 0 to disable.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    database_dir = args.database.expanduser().resolve()
    try:
        data = load_database(database_dir)
        rows = make_rows(data, include_null_be=args.include_null_be)
        filtered = filter_rows(
            rows,
            smiles=args.smiles,
            smiles_match=args.smiles_match,
            core_level=args.core_level,
            element=args.element,
            name=args.name,
            min_be=args.min_be,
            max_be=args.max_be,
        )
    except (FileNotFoundError, ValueError, pickle.UnpicklingError) as exc:
        parser.exit(status=1, message=f"error: {exc}\n")

    print(summarize(filtered, total_rows=len(next(iter(data.values())))))
    print_preview(filtered, args.preview)

    if args.output:
        output = args.output.expanduser().resolve()
        if args.group_by_smiles:
            write_grouped_by_smiles(filtered, output)
        elif args.format == "jsonl":
            write_jsonl(filtered, output)
        else:
            write_csv(filtered, output)
        print(f"\nWrote {len(filtered)} selected rows to {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
