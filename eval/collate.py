# Merge the per-cell csvs written by submit_eval.sh into one table.
#
# Each array task writes its own file so the parallel jobs never race on a shared
# csv, which leaves one row per (model, dataset, chunk size) scattered across the
# run directories.
#
# python eval/collate.py /shared/people/jatin.prakash/Results/fineweb-15b/2026-08-02/*/eval_4096

import sys
import argparse
from pathlib import Path

import pandas as pd


def load(directories):
    frames = []
    for directory in directories:
        directory = Path(directory)
        for path in sorted(directory.glob("*.csv")):
            frame = pd.read_csv(path)
            # the filename is the only place the run label survives
            frame["label"] = path.stem.rsplit("_", 2)[0]
            frames.append(frame)
    if not frames:
        raise SystemExit(f"No csvs found under: {', '.join(map(str, directories))}")
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collate eval csvs into one table")
    parser.add_argument("directories", nargs="+", help="Directories containing per-cell csvs")
    parser.add_argument("--out", type=str, default=None, help="Optional merged csv path")
    args = parser.parse_args()

    df = load(args.directories)
    df["chunk_size"] = 2 ** df["chunk_size_power"]
    df["dataset"] = df["dataset_name"].str.replace("hazyresearch/based-", "", regex=False)

    pivot = df.pivot_table(
        index=["dataset", "chunk_size"],
        columns="label",
        values="acc",
    )
    counts = df.pivot_table(
        index=["dataset", "chunk_size"],
        columns="label",
        values="num_samples",
    )

    pd.set_option("display.width", 160)
    print("\n=== accuracy ===")
    print(pivot.round(4).to_string())
    print("\n=== scored samples ===")
    print(counts.astype("Int64").to_string())

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nMerged rows written to {args.out}")
