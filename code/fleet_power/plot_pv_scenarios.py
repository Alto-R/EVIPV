#!/usr/bin/env python3
"""
EV Fleet Charging Scenario Comparison Visualization

Compare battery state changes over time between scenarios for multiple vehicles:
- Scenario A: Without rooftop PV (original charging data)
- Scenario C: With rooftop PV (including PV generation impact)

Aggregates data from all vehicles in a folder and plots sum curves.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Set Chinese font for labels
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False


def load_charging_data(file_path: str) -> pd.DataFrame:
    """Load charging data and parse timestamps"""
    df = pd.read_csv(file_path)
    df['stime'] = pd.to_datetime(df['stime'])
    df['etime'] = pd.to_datetime(df['etime'])
    return df.sort_values('stime').reset_index(drop=True)


def load_fleet_data(data_folder: str,
                   suffix_original: str = '_charging.csv',
                   suffix_pv: str = '_charging_with_pv.csv') -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """
    Load charging data for all vehicles in a folder

    Args:
        data_folder: Path to folder containing vehicle data files
        suffix_original: Suffix for original charging files (Scenario A)
        suffix_pv: Suffix for PV charging files (Scenario C)

    Returns:
        (list of original DataFrames, list of PV DataFrames)
    """
    folder = Path(data_folder)

    # Find all PV charging files
    pv_files = sorted(folder.glob(f'*{suffix_pv}'))

    original_dfs = []
    pv_dfs = []
    vehicle_ids = []

    for pv_file in pv_files:
        # Get vehicle ID by removing suffix
        vehicle_id = pv_file.stem.replace(suffix_pv.replace('.csv', ''), '')

        # Construct original file path
        original_file = folder / f"{vehicle_id}{suffix_original}"

        if original_file.exists():
            try:
                df_original = load_charging_data(str(original_file))
                df_pv = load_charging_data(str(pv_file))

                if len(df_original) > 0 and len(df_pv) > 0:
                    original_dfs.append(df_original)
                    pv_dfs.append(df_pv)
                    vehicle_ids.append(vehicle_id)
                    print(f"Loaded vehicle {vehicle_id}: {len(df_original)} original records, {len(df_pv)} PV records")
            except Exception as e:
                print(f"Error loading {vehicle_id}: {e}")

    print(f"\nTotal vehicles loaded: {len(vehicle_ids)}")
    return original_dfs, pv_dfs, vehicle_ids


def create_continuous_timeline(df: pd.DataFrame,
                               use_adjusted: bool = False,
                               time_start: Optional[pd.Timestamp] = None,
                               time_end: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Create continuous timeline for a single vehicle

    Args:
        df: Charging data DataFrame
        use_adjusted: Whether to use adjusted SOC (Scenario C)
        time_start: Start of time range filter
        time_end: End of time range filter

    Returns:
        DataFrame with columns ['timestamp', 'soc']
    """
    # Filter by time range
    if time_start is not None:
        df = df[df['etime'] >= time_start]
    if time_end is not None:
        df = df[df['stime'] <= time_end]

    if len(df) == 0:
        return pd.DataFrame(columns=['timestamp', 'soc'])

    records = []

    for idx, row in df.iterrows():
        stime = row['stime']
        etime = row['etime']

        if use_adjusted:
            ssoc = row['adjusted_ssoc']
            esoc = row['adjusted_esoc']
        else:
            ssoc = row['ssoc']
            esoc = row['esoc']

        # Add charging start point
        records.append({'timestamp': stime, 'soc': ssoc})

        # Add charging end point
        records.append({'timestamp': etime, 'soc': esoc})

        # Add connection to next charging session
        if idx < len(df) - 1:
            next_stime = df.iloc[idx + 1]['stime']
            if use_adjusted:
                next_ssoc = df.iloc[idx + 1]['adjusted_ssoc']
            else:
                next_ssoc = df.iloc[idx + 1]['ssoc']

            # Only add if there's a time gap
            if next_stime > etime:
                records.append({'timestamp': next_stime, 'soc': next_ssoc})

    timeline_df = pd.DataFrame(records)
    return timeline_df.sort_values('timestamp').reset_index(drop=True)


def aggregate_fleet_timelines(dfs_list: List[pd.DataFrame],
                              use_adjusted: bool = False,
                              time_start: Optional[pd.Timestamp] = None,
                              time_end: Optional[pd.Timestamp] = None,
                              freq: str = '15min') -> pd.DataFrame:
    """
    Aggregate timelines from multiple vehicles onto a common time grid

    Args:
        dfs_list: List of charging DataFrames for each vehicle
        use_adjusted: Whether to use adjusted SOC (Scenario C)
        time_start: Start of time range
        time_end: End of time range
        freq: Resampling frequency (e.g., '15min', '1h')

    Returns:
        DataFrame with columns ['timestamp', 'total_soc', 'num_vehicles']
    """
    # Create timelines for each vehicle
    timelines = []
    for df in dfs_list:
        timeline = create_continuous_timeline(df, use_adjusted, time_start, time_end)
        if len(timeline) > 0:
            timelines.append(timeline)

    if len(timelines) == 0:
        return pd.DataFrame(columns=['timestamp', 'total_soc', 'num_vehicles'])

    # Find overall time range
    all_times = pd.concat([tl['timestamp'] for tl in timelines])
    global_start = time_start if time_start is not None else all_times.min()
    global_end = time_end if time_end is not None else all_times.max()

    # Create common time grid
    time_grid = pd.date_range(start=global_start, end=global_end, freq=freq)

    # Interpolate each vehicle's SOC onto the common time grid
    aggregated_soc = np.zeros(len(time_grid))
    vehicle_count = np.zeros(len(time_grid))

    for timeline in timelines:
        # Set timestamp as index for interpolation
        timeline = timeline.set_index('timestamp')

        # Reindex to time grid and forward-fill (SOC remains constant until next event)
        reindexed = timeline.reindex(timeline.index.union(time_grid)).sort_index()
        reindexed['soc'] = reindexed['soc'].ffill()

        # Extract values at grid points
        soc_at_grid = reindexed.loc[time_grid, 'soc'].values

        # Add to aggregate (only where data exists)
        valid_mask = ~np.isnan(soc_at_grid)
        aggregated_soc[valid_mask] += soc_at_grid[valid_mask]
        vehicle_count[valid_mask] += 1

    # Create result DataFrame
    result = pd.DataFrame({
        'timestamp': time_grid,
        'total_soc': aggregated_soc,
        'num_vehicles': vehicle_count
    })

    return result


def plot_fleet_comparison(data_folder: str,
                         time_start: Optional[str] = None,
                         time_end: Optional[str] = None,
                         freq: str = '15min',
                         output_file: str = 'fleet_scenario_comparison.png'):
    """
    Plot fleet-level comparison between scenarios

    Args:
        data_folder: Path to folder containing vehicle CSV files
        time_start: Start date string (e.g., '2023-01-01') or None for auto
        time_end: End date string (e.g., '2023-12-31') or None for auto
        freq: Time resolution for aggregation (e.g., '15min', '1h')
        output_file: Output image path
    """
    print("=" * 60)
    print("EV Fleet Scenario Comparison Analysis")
    print("=" * 60)

    # Parse time range
    ts_start = pd.to_datetime(time_start) if time_start else None
    ts_end = pd.to_datetime(time_end) if time_end else None

    if ts_start:
        print(f"Time range start: {ts_start}")
    if ts_end:
        print(f"Time range end: {ts_end}")
    print(f"Aggregation frequency: {freq}\n")

    # Load fleet data
    print("Loading fleet data...")
    original_dfs, pv_dfs, vehicle_ids = load_fleet_data(data_folder)

    if len(original_dfs) == 0:
        print("No vehicle data found!")
        return

    # Aggregate timelines
    print("\nAggregating Scenario A (without PV)...")
    agg_original = aggregate_fleet_timelines(original_dfs,
                                            use_adjusted=False,
                                            time_start=ts_start,
                                            time_end=ts_end,
                                            freq=freq)

    print("Aggregating Scenario C (with PV)...")
    agg_pv = aggregate_fleet_timelines(pv_dfs,
                                       use_adjusted=True,
                                       time_start=ts_start,
                                       time_end=ts_end,
                                       freq=freq)

    # Calculate statistics
    print("\nCalculating statistics...")
    total_pv_energy = sum(df['total_pv_contribution_kwh'].sum() for df in pv_dfs)
    total_charging_events = sum(len(df) for df in pv_dfs)
    total_eliminated = sum(df['charging_eliminated'].sum() for df in pv_dfs)

    # Plot
    print("Creating plot...")
    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot Scenario A
    ax.plot(agg_original['timestamp'],
            agg_original['total_soc'],
            color='#1f77b4',
            linewidth=2.5,
            label='Scenario A: Without Rooftop PV',
            alpha=0.8)

    # Plot Scenario C
    ax.plot(agg_pv['timestamp'],
            agg_pv['total_soc'],
            color='#2ca02c',
            linewidth=2.5,
            label='Scenario C: With Rooftop PV',
            alpha=0.8)

    # Styling
    ax.set_title('EV Fleet Charging Scenario Comparison: Rooftop PV Impact Analysis',
                fontsize=16,
                fontweight='bold',
                pad=20)
    ax.set_xlabel('Date', fontsize=13, fontweight='bold')
    ax.set_ylabel('Fleet Total SOC (%)', fontsize=13, fontweight='bold')

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45, ha='right')

    # Legend
    ax.legend(loc='best', fontsize=12, framealpha=0.95, shadow=True)

    # Statistics text box
    stats_text = (
        f"Fleet Statistics:\n"
        f"• Number of vehicles: {len(vehicle_ids)}\n"
        f"• Total charging events: {total_charging_events}\n"
        f"• Total PV contribution: {total_pv_energy:.2f} kWh\n"
        f"• Average PV per vehicle: {total_pv_energy/len(vehicle_ids):.2f} kWh\n"
        f"• Charging events eliminated: {total_eliminated}\n"
        f"• Aggregation frequency: {freq}"
    )

    ax.text(0.02, 0.98, stats_text,
           transform=ax.transAxes,
           fontsize=10,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85),
           family='monospace')

    # Layout
    plt.tight_layout()

    # Save
    print(f"\nSaving plot to: {output_file}")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    # Show
    print("Displaying plot...")
    plt.show()

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == '__main__':
    # Configuration
    data_folder = '.'  # Folder containing vehicle CSV files
    time_start = None  # e.g., '2023-06-01' or None for auto
    time_end = None    # e.g., '2023-08-31' or None for auto
    freq = '1h'        # Aggregation frequency: '15min', '30min', '1h', etc.
    output_file = 'fleet_scenario_comparison.png'

    # Run analysis
    plot_fleet_comparison(
        data_folder=data_folder,
        time_start=time_start,
        time_end=time_end,
        freq=freq,
        output_file=output_file
    )