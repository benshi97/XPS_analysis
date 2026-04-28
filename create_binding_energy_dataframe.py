#!/usr/bin/env python3
"""Create a Pandas DataFrame of gas-phase core binding energies and ASE Atoms."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATABASE = SCRIPT_DIR / "database_2023_05_30"
DEFAULT_OUTPUT_STEM = SCRIPT_DIR / "binding_energy_dataframe"

FIELD_FILES = {
    "molecular_formula": "MF_list.pkl",
    "chemical_formula": "CF_list.pkl",
    "iupac_name": "IUPAC_Name_list.pkl",
    "isomeric_smiles": "Isomeric_SMILES_list.pkl",
    "inchi": "InChI_list.pkl",
    "core_level": "Core_level_list.pkl",
    "binding_energy_ev": "BE_list.pkl",
    "binding_energy_comment": "BE_comment_list.pkl",
    "reference_doi": "Reference_DOI_list.pkl",
    "reference_text": "Reference_text_list.pkl",
    "reference_comment": "Reference_comment_list.pkl",
    "comment": "Comment_list.pkl",
}


def load_pickle(path: Path) -> list[Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_database(database_dir: Path) -> dict[str, list[Any]]:
    database_dir = database_dir.expanduser().resolve()
    data = {
        field: load_pickle(database_dir / filename)
        for field, filename in FIELD_FILES.items()
    }

    lengths = {FIELD_FILES[field]: len(values) for field, values in data.items()}
    if len(set(lengths.values())) != 1:
        detail = ", ".join(f"{name}={length}" for name, length in lengths.items())
        raise ValueError(f"Input pickle lists have different lengths: {detail}")

    return data


def core_level_to_atom(core_level: Any) -> str | None:
    """Return the element/atom label from a core level such as 'C 1s'."""
    if not isinstance(core_level, str) or not core_level.strip():
        return None
    return core_level.split(maxsplit=1)[0]


def formula_dict_to_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return "".join(
            f"{element}{count if count != 1 else ''}"
            for element, count in sorted(value.items())
        )
    return str(value)


def elements_from_formula(value: Any) -> list[str] | None:
    if isinstance(value, dict):
        return sorted(str(element) for element, count in value.items() if count)
    return None


def elements_from_atoms_or_formula(row: pd.Series) -> list[str] | None:
    atoms = row.get("atoms")
    if atoms is not None:
        return sorted(set(atoms.get_chemical_symbols()))
    return elements_from_formula(row.get("molecular_formula"))


def smiles_to_atoms(smiles: str) -> tuple[Any | None, str]:
    """Generate one 3D ASE Atoms object from a SMILES string using RDKit."""
    try:
        from rdkit import Chem, RDLogger  # type: ignore
        from rdkit.Chem import AllChem  # type: ignore
    except ImportError:
        return None, "rdkit_not_installed"
    try:
        from ase import Atoms  # type: ignore
    except ImportError:
        return None, "ase_not_installed"

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "invalid_smiles"

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D
    embed_status = AllChem.EmbedMolecule(mol, params)
    if embed_status != 0:
        embed_status = AllChem.EmbedMolecule(mol, randomSeed=0xF00D, useRandomCoords=True)
    if embed_status != 0:
        return None, "embed_failed"

    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass

    conformer = mol.GetConformer()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    positions = [
        (
            conformer.GetAtomPosition(index).x,
            conformer.GetAtomPosition(index).y,
            conformer.GetAtomPosition(index).z,
        )
        for index in range(mol.GetNumAtoms())
    ]
    return Atoms(symbols=symbols, positions=positions), "ok"


def build_atoms_lookup(df: pd.DataFrame) -> dict[str, tuple[Any | None, str]]:
    lookup: dict[str, tuple[Any | None, str]] = {}
    unique = (
        df[["isomeric_smiles"]]
        .dropna(subset=["isomeric_smiles"])
        .drop_duplicates(subset=["isomeric_smiles"])
    )
    for row in unique.itertuples(index=False):
        smiles = row.isomeric_smiles
        lookup[smiles] = smiles_to_atoms(smiles)

    return lookup


def load_binding_energy_dataframe(
    database_dir: Path = DEFAULT_DATABASE,
    generate_atoms: bool = True,
) -> pd.DataFrame:
    """Load the database pickle lists into a DataFrame.

    The first four columns keep the requested compact view:
    formula_label, atom, core_binding_energy_ev, smiles.
    The remaining columns preserve the source database fields and add ASE Atoms.
    """
    data = load_database(database_dir)
    df = pd.DataFrame(data)
    df.insert(0, "row_index", df.index)
    df.insert(1, "formula_label", df["chemical_formula"])
    df.insert(2, "atom", [core_level_to_atom(value) for value in df["core_level"]])
    df.insert(3, "core_binding_energy_ev", df["binding_energy_ev"])
    df.insert(4, "smiles", df["isomeric_smiles"])
    df["molecular_formula_label"] = df["molecular_formula"].map(formula_dict_to_string)

    df = df.dropna(subset=["core_binding_energy_ev"]).reset_index(drop=True)

    if generate_atoms:
        atoms_lookup = build_atoms_lookup(df)
        df["atoms"] = df["isomeric_smiles"].map(
            lambda smiles: (
                atoms_lookup.get(smiles, (None, "missing_smiles"))[0].copy()
                if atoms_lookup.get(smiles, (None, "missing_smiles"))[0] is not None
                else None
            )
        )
        df["atoms_status"] = df["isomeric_smiles"].map(
            lambda smiles: atoms_lookup.get(smiles, (None, "missing_smiles"))[1]
        )
    else:
        df["atoms"] = None
        df["atoms_status"] = "not_requested"

    df["elements"] = df.apply(elements_from_atoms_or_formula, axis=1)

    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a Pandas DataFrame with metadata, binding energies, SMILES, and ASE Atoms.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE,
        help=f"Database directory containing the pickle lists. Default: {DEFAULT_DATABASE}",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=DEFAULT_OUTPUT_STEM,
        help="Output path without extension. Writes .csv and .pkl files.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=10,
        help="Number of DataFrame rows to print.",
    )
    parser.add_argument(
        "--no-atoms",
        action="store_true",
        help="Build the DataFrame without generating ASE Atoms objects.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    df = load_binding_energy_dataframe(
        args.database,
        generate_atoms=not args.no_atoms,
    )

    output_stem = args.output_stem.expanduser().resolve()
    csv_path = output_stem.with_suffix(".csv")
    pickle_path = output_stem.with_suffix(".pkl")

    df.to_csv(csv_path, index=False)
    df.to_pickle(pickle_path)

    print(f"Created DataFrame with shape {df.shape}")
    if args.preview > 0:
        print(df.head(args.preview).to_string(index=False))
    print("Atoms status counts:")
    print(df["atoms_status"].value_counts(dropna=False).to_string())
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote pickle: {pickle_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
