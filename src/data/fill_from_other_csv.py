from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _infer_id_column(df: pd.DataFrame) -> str:
    for candidate in ("id", "ID", "Id", "Unnamed: 0"):
        if candidate in df.columns:
            return candidate
    msg = "Could not determine an id column (expected one of id/Unnamed: 0)."
    raise ValueError(msg)


def fill_missing_columns(
    data2_path: Path,
    data_path: Path,
    output_path: Path,
) -> None:
    data2_df = pd.read_csv(data2_path)
    data_df = pd.read_csv(data_path)

    id_column = _infer_id_column(data2_df)
    if id_column not in data_df.columns:
        msg = f"Column '{id_column}' not found in {data_path}"
        raise ValueError(msg)

    # Drop pq before alignment as per requirement.
    data_df = data_df.drop(columns=["pq"], errors="ignore")

    target = data2_df.set_index(id_column)
    source = data_df.set_index(id_column)
    source_aligned = source.reindex(target.index)

    new_columns = [col for col in source_aligned.columns if col not in target.columns]
    if new_columns:
        target[new_columns] = source_aligned[new_columns]

    shared_columns = [col for col in source_aligned.columns if col in target.columns]
    if shared_columns:
        target[shared_columns] = target[shared_columns].fillna(source_aligned[shared_columns])

    target.reset_index().to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fill missing columns in data2.csv using values from data.csv while omitting the 'pq' column."
        )
    )
    parser.add_argument("--data2", type=Path, default=Path("data/data2.csv"), help="Path to data2.csv")
    parser.add_argument("--data", type=Path, default=Path("data/data.csv"), help="Path to data.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/data2_filled.csv"),
        help="Where to write the filled CSV",
    )
    args = parser.parse_args()

    fill_missing_columns(args.data2, args.data, args.output)


if __name__ == "__main__":
    main()
