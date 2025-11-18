import argparse
import gc
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import pandas as pd

# 强制禁用输出缓冲，确保在超算环境中实时打印
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


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
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of processes to run concurrently (default: 4).",
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
    max_workers: int,
) -> None:
    csv_files = sorted(traj_dir.glob("*_processed.csv"))
    if not csv_files:
        print(f"[WARN] No '*_processed.csv' found in {traj_dir}", flush=True)
        return

    max_files = 50
    if len(csv_files) > max_files:
        print(
            f"[INFO] Found {len(csv_files)} files. Processing only the first {max_files}.",
            flush=True,
        )
        csv_files = csv_files[:max_files]

    print(f"[INFO] Using trajectory directory: {traj_dir}", flush=True)
    print(f"[INFO] Source month: {source_month}", flush=True)
    print(f"[INFO] Target months: {target_months}", flush=True)

    worker_count = max(1, min(len(csv_files), max_workers))
    print(f"[INFO] Using {worker_count} processes for parallel execution.", flush=True)
    print(f"[INFO] Total files to process: {len(csv_files)}", flush=True)

    def submit_args() -> Iterable[tuple[str, str]]:
        for csv_path in csv_files:
            yield str(csv_path), str(traj_dir)

    completed_count = 0

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                process_single_file,
                csv_path_str,
                traj_dir_str,
                source_month,
                target_months,
                datetime_column,
            ): Path(csv_path_str)
            for csv_path_str, traj_dir_str in submit_args()
        }
        for future in as_completed(futures):
            csv_path = futures[future]
            try:
                logs = future.result()
                # 实时打印每个文件的处理日志
                for line in logs:
                    print(line, flush=True)
                completed_count += 1
                print(f"[PROGRESS] Completed {completed_count}/{len(csv_files)} files", flush=True)
            except Exception as exc:
                print(
                    f"[ERROR] Failed while processing {csv_path.name}: {exc}",
                    flush=True,
                )
                raise

    print(f"[INFO] All {len(csv_files)} files processed successfully!", flush=True)


def process_single_file(
    csv_path_str: str,
    traj_dir_str: str,
    source_month: int,
    target_months: list[int],
    datetime_column: str,
) -> list[str]:
    csv_path = Path(csv_path_str)
    traj_dir = Path(traj_dir_str)
    logs: list[str] = []

    def log(message: str) -> None:
        logs.append(message)

    stem = csv_path.stem
    if stem.endswith("_processed"):
        base_name = stem[: -len("_processed")]
    else:
        base_name = stem

    log(f"\n[INFO] Processing {csv_path.name}")
    df = pd.read_csv(csv_path)

    if datetime_column not in df.columns:
        raise KeyError(
            f"Column '{datetime_column}' not found in {csv_path}. "
            "Please ensure you are using preprocessed trajectory files."
        )

    df[datetime_column] = pd.to_datetime(df[datetime_column], errors="raise")
    month_counts = df[datetime_column].dt.month.value_counts().to_dict()
    if len(month_counts) != 1 or source_month not in month_counts:
        log(
            f"[WARN] Expected all rows to be in month {source_month}, "
            f"but found month distribution: {month_counts}. "
            "Rows in other months will still be transformed if possible."
        )

    # 提取原始日期组件（只做一次，避免重复计算）
    original_dt = df[datetime_column]
    original_year = original_dt.dt.year
    original_day = original_dt.dt.day
    original_time = original_dt.dt.time

    # 保存非datetime列数据（避免每次都复制）
    other_columns = df.drop(columns=[datetime_column])

    for target_month in target_months:
        log(f"  [INFO] Creating variant for month {target_month:02d}")

        # 使用向量化操作构建新日期，避免完整复制DataFrame
        # 先创建新的datetime，无效日期会自动变为NaT
        try:
            new_dates = pd.to_datetime({
                'year': original_year,
                'month': target_month,
                'day': original_day,
                'hour': original_dt.dt.hour,
                'minute': original_dt.dt.minute,
                'second': original_dt.dt.second,
            }, errors='coerce')
        except Exception:
            # 如果向量化失败，回退到apply方法
            def replace_month(dt):
                try:
                    return dt.replace(month=target_month)
                except ValueError:
                    return pd.NaT
            new_dates = original_dt.apply(replace_month)

        # 找出有效日期的行索引
        valid_mask = new_dates.notna()
        dropped = (~valid_mask).sum()

        if dropped > 0:
            log(
                f"    [WARN] Dropped {dropped} rows with invalid dates for "
                f"month {target_month:02d} (e.g., 31st in 30-day months)."
            )

        # 只复制有效行，并组合数据
        df_variant = other_columns[valid_mask].copy()
        df_variant[datetime_column] = new_dates[valid_mask]

        month_dir = traj_dir / f"{target_month:02d}"
        month_dir.mkdir(parents=True, exist_ok=True)
        output_name = f"{base_name}_m{target_month:02d}_processed.csv"
        output_path = month_dir / output_name
        df_variant.to_csv(output_path, index=False)
        log(
            f"    [OK] Saved: {output_path.name} (rows: {len(df_variant)})"
        )

        # 立即释放内存
        del df_variant
        del new_dates
        del valid_mask
        gc.collect()

    # 清理所有中间变量
    del df, other_columns, original_dt, original_year, original_day, original_time
    gc.collect()
    return logs


def main() -> None:
    args = parse_args()
    traj_dir = Path(args.traj_dir) if args.traj_dir is not None else get_default_traj_dir()
    target_months = parse_month_list(args.target_months, args.source_month)
    if not target_months:
        print(
            "[INFO] No target months specified after removing source month. Nothing to do.",
            flush=True,
        )
        return

    clone_month_variants(
        traj_dir=traj_dir,
        source_month=args.source_month,
        target_months=target_months,
        datetime_column=args.datetime_column,
        max_workers=args.max_workers,
    )  


if __name__ == "__main__":
    main()
