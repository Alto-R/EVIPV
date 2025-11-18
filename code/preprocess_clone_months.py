import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clone preprocessed March trajectories to other months by changing only "
            "the month field in the datetime column."
        )
    )
    parser.add_argument(
        "--source-month",
        type=int,
        default=3,
        help="Source month in the original data (default: 3).",
    )
    parser.add_argument(
        "--target-months",
        type=str,
        default="1,2,4,5,6,7,8,9,10,11,12",
        help=(
            "Comma-separated list of target months, e.g. '1,4,7'. "
            "Default: all months except 3."
        ),
    )
    parser.add_argument(
        "--traj-dir",
        type=str,
        default=None,
        help=(
            "Directory containing '*_processed.csv'. "
            "Default: '<repo_root>/code/traj'."
        ),
    )
    parser.add_argument(
        "--datetime-column",
        type=str,
        default="datetime",
        help="Name of the datetime column in processed CSV files (default: datetime).",
    )
    return parser.parse_args()


def get_default_traj_dir() -> Path:
    # Fixed absolute path to trajectory directory on this machine.
    return Path(
        r"D:\1-PKU\PKU\1 Master\Master 1\Papers\EVIPV\code\traj"
    )


def parse_month_list(target_months_str: str, source_month: int) -> list[int]:
    months = []
    for part in target_months_str.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if not 1 <= value <= 12:
            raise ValueError(f"Invalid month value: {value}")
        months.append(value)
    months = sorted(set(months))
    if source_month in months:
        months.remove(source_month)
    return months


def clone_month_variants(
    traj_dir: Path,
    source_month: int,
    target_months: list[int],
    datetime_column: str,
) -> None:
    csv_files = sorted(traj_dir.glob("*_processed.csv"))
    if not csv_files:
        print(f"[WARN] No '*_processed.csv' found in {traj_dir}")
        return

    print(f"[INFO] Using trajectory directory: {traj_dir}")
    print(f"[INFO] Source month: {source_month}")
    print(f"[INFO] Target months: {target_months}")

    for csv_path in csv_files:
        stem = csv_path.stem
        if stem.endswith("_processed"):
            base_name = stem[: -len("_processed")]
        else:
            base_name = stem

        print(f"\n[INFO] Processing {csv_path.name}")
        df = pd.read_csv(csv_path)

        if datetime_column not in df.columns:
            raise KeyError(
                f"Column '{datetime_column}' not found in {csv_path}. "
                "Please ensure you are using preprocessed trajectory files."
            )

        df[datetime_column] = pd.to_datetime(df[datetime_column], errors="raise")
        month_counts = df[datetime_column].dt.month.value_counts().to_dict()
        if len(month_counts) != 1 or source_month not in month_counts:
            print(
                f"[WARN] Expected all rows to be in month {source_month}, "
                f"but found month distribution: {month_counts}. "
                "Rows in other months will still be transformed if possible."
            )

        for target_month in target_months:
            print(f"  [INFO] Creating variant for month {target_month:02d}")
            df_variant = df.copy()

            def replace_month(dt):
                try:
                    return dt.replace(month=target_month)
                except ValueError:
                    # Invalid calendar date for this month (e.g., Feb 30).
                    return pd.NaT

            df_variant[datetime_column] = df_variant[datetime_column].apply(
                replace_month
            )
            before = len(df_variant)
            df_variant = df_variant.dropna(subset=[datetime_column])
            dropped = before - len(df_variant)
            if dropped > 0:
                print(
                    f"    [WARN] Dropped {dropped} rows with invalid dates for "
                    f"month {target_month:02d} (e.g., 31st in 30-day months)."
                )

            month_dir = traj_dir / f"{target_month:02d}"
            month_dir.mkdir(parents=True, exist_ok=True)
            output_name = f"{base_name}_m{target_month:02d}_processed.csv"
            output_path = month_dir / output_name
            df_variant.to_csv(output_path, index=False)
            print(f"    [OK] Saved: {output_path.name} (rows: {len(df_variant)})")


def main() -> None:
    args = parse_args()
    traj_dir = Path(args.traj_dir) if args.traj_dir is not None else get_default_traj_dir()
    target_months = parse_month_list(args.target_months, args.source_month)
    if not target_months:
        print("[INFO] No target months specified after removing source month. Nothing to do.")
        return

    clone_month_variants(
        traj_dir=traj_dir,
        source_month=args.source_month,
        target_months=target_months,
        datetime_column=args.datetime_column,
    )  


if __name__ == "__main__":
    main()
