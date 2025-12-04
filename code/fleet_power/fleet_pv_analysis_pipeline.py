#!/usr/bin/env python3
"""
EV Fleet PV Analysis Pipeline

Complete pipeline for processing and visualizing rooftop PV impact on EV fleet charging:
1. Integrates PV generation data with charging data for all vehicles in a folder
2. Generates fleet-level comparison plot between scenarios

Scenarios:
- Scenario A: Without rooftop PV (original charging data)
- Scenario C: With rooftop PV (PV generation integrated)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Set font for plots
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================================
# Part 1: PV Integration Functions (from integrate_pv_charging.py)
# ============================================================================

def load_charging_data(file_path: str) -> pd.DataFrame:
    """Load charging data and parse timestamps"""
    df = pd.read_csv(file_path)
    df['stime'] = pd.to_datetime(df['stime'])
    df['etime'] = pd.to_datetime(df['etime'])
    return df.sort_values('stime').reset_index(drop=True)


def load_pv_data(file_path: str) -> pd.DataFrame:
    """Load PV generation data and parse timestamps"""
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df.sort_values('datetime').reset_index(drop=True)


def load_vehicle_info(file_path: str) -> pd.DataFrame:
    """Load vehicle information"""
    return pd.read_csv(file_path)


def get_battery_capacity(vin: str, vehicle_info: pd.DataFrame) -> float:
    """
    Query battery capacity by VIN

    Returns: Battery capacity (kWh)
    """
    match = vehicle_info[vehicle_info['vin'] == vin]
    if match.empty:
        raise ValueError(f"Vehicle info not found for VIN {vin}")
    return match['energy_device_capacity'].iloc[0]


def preprocess_pv_data(pv_data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess PV data: add minute-level time index"""
    pv_data = pv_data.copy()
    pv_data['datetime_minute'] = pv_data['datetime'].dt.floor('min').dt.tz_localize(None)
    return pv_data


def preprocess_charging_data(charging_data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess charging data: add minute-level time index"""
    charging_data = charging_data.copy()
    charging_data['stime_minute'] = charging_data['stime'].dt.floor('min')
    charging_data['etime_minute'] = charging_data['etime'].dt.floor('min')
    return charging_data


def calculate_pv_impact(
    charge: pd.Series,
    pv_data: pd.DataFrame,
    charge_index: int,
    all_charges: pd.DataFrame,
    battery_capacity: float,
    charging_efficiency: float
) -> dict:
    """
    Calculate PV impact on a single charging event

    Returns dict with all calculated fields
    """
    # 1. Determine time window before charging
    if charge_index == 0:
        time_window_start = pv_data['datetime_minute'].min()
    else:
        time_window_start = all_charges.loc[charge_index - 1, 'etime_minute']

    time_window_end = charge['stime_minute']

    # 2. Calculate accumulated PV energy before charging
    pv_before = pv_data[
        (pv_data['datetime_minute'] >= time_window_start) &
        (pv_data['datetime_minute'] < time_window_end)
    ]
    pv_energy_before_kwh = pv_before['energy_kwh'].sum() * charging_efficiency

    # 3. Calculate adjusted starting SOC
    soc_gain_before = (pv_energy_before_kwh / battery_capacity) * 100
    adjusted_ssoc = min(charge['ssoc'] + soc_gain_before, 100.0)

    # 4. Calculate PV energy during charging
    pv_during = pv_data[
        (pv_data['datetime_minute'] >= charge['stime_minute']) &
        (pv_data['datetime_minute'] <= charge['etime_minute'])
    ]
    pv_energy_during_kwh = pv_during['energy_kwh'].sum() * charging_efficiency

    # 5. Calculate original and actual charging demand
    # Ensure non-negative values (negative means discharge, not charging)
    original_charge_demand_kwh = max(((charge['esoc'] - charge['ssoc']) / 100) * battery_capacity, 0)
    actual_charge_demand_kwh = max(((charge['esoc'] - adjusted_ssoc) / 100) * battery_capacity, 0)

    # 6. Calculate net grid energy
    net_grid_energy_kwh = max(actual_charge_demand_kwh - pv_energy_during_kwh, 0)

    # 7. Calculate adjusted ending SOC
    if net_grid_energy_kwh == 0 and pv_energy_during_kwh > actual_charge_demand_kwh:
        surplus_pv_kwh = pv_energy_during_kwh - actual_charge_demand_kwh
        additional_soc = (surplus_pv_kwh / battery_capacity) * 100
        adjusted_esoc = min(charge['esoc'] + additional_soc, 100.0)
    else:
        adjusted_esoc = charge['esoc']

    # 8. Determine if charging is eliminated
    charging_eliminated = (adjusted_ssoc >= charge['esoc'])

    # 9. Return complete result
    return {
        # Original fields
        'vin': charge['vin'],
        'stime': charge['stime'],
        'slon': charge['slon'],
        'slat': charge['slat'],
        'ssoc': charge['ssoc'],
        'vehicledata_chargestatus': charge['vehicledata_chargestatus'],
        'etime': charge['etime'],
        'elon': charge['elon'],
        'elat': charge['elat'],
        'esoc': charge['esoc'],

        # New fields
        'battery_capacity_kwh': round(battery_capacity, 2),  # Add battery capacity for later use
        'pv_energy_before_kwh': round(pv_energy_before_kwh, 4),
        'pv_energy_during_kwh': round(pv_energy_during_kwh, 4),
        'total_pv_contribution_kwh': round(pv_energy_before_kwh + pv_energy_during_kwh, 4),
        'adjusted_ssoc': round(adjusted_ssoc, 2),
        'adjusted_esoc': round(adjusted_esoc, 2),
        'original_charge_demand_kwh': round(original_charge_demand_kwh, 4),
        'net_grid_energy_kwh': round(net_grid_energy_kwh, 4),
        'charging_eliminated': charging_eliminated
    }


def integrate_pv_with_charging(
    charging_file: str,
    pv_generation_file: str,
    vehicle_info_file: str,
    output_file: str,
    charging_efficiency: float = 0.95,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Integrate PV generation data with charging data

    Args:
        charging_file: Charging data CSV path
        pv_generation_file: PV generation data CSV path
        vehicle_info_file: Vehicle info CSV path
        output_file: Output file path
        charging_efficiency: Charging efficiency (default 0.95)
        verbose: Print progress messages

    Returns:
        Enhanced charging data DataFrame
    """
    # Step 1: Load data
    if verbose:
        print("  Loading data...")
    charging_data = load_charging_data(charging_file)
    pv_data = load_pv_data(pv_generation_file)
    vehicle_info = load_vehicle_info(vehicle_info_file)

    if verbose:
        print(f"    Charging records: {len(charging_data)}")
        print(f"    PV records: {len(pv_data)}")

    # Step 2: Get battery capacity
    battery_capacity = get_battery_capacity(charging_data['vin'].iloc[0], vehicle_info)
    if verbose:
        print(f"    Battery capacity: {battery_capacity} kWh")

    # Step 3: Preprocess for time alignment
    if verbose:
        print("  Preprocessing...")
    pv_data = preprocess_pv_data(pv_data)
    charging_data = preprocess_charging_data(charging_data)

    # Step 4: Process each charging record
    if verbose:
        print("  Calculating PV impact...")
    results = []
    for idx, charge in charging_data.iterrows():
        result = calculate_pv_impact(
            charge,
            pv_data,
            idx,
            charging_data,
            battery_capacity,
            charging_efficiency
        )
        results.append(result)

    # Step 5: Combine results
    enhanced_data = pd.DataFrame(results)

    # Step 6: Save results
    if verbose:
        print(f"  Saving to: {output_file}")
    enhanced_data.to_csv(output_file, index=False, encoding='utf-8-sig')

    return enhanced_data


# ============================================================================
# Part 2: Fleet Plotting Functions (from plot_pv_scenarios.py)
# ============================================================================

def load_fleet_data(charging_folder: str,
                   pv_folder: str,
                   suffix_original: str = '_charging.csv',
                   suffix_pv: str = '_charging_with_pv.csv') -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[str]]:
    """
    Load charging data for all vehicles from separate folders

    Args:
        charging_folder: Path to folder containing original charging files
        pv_folder: Path to folder containing PV-integrated charging files
        suffix_original: Suffix for original charging files (Scenario A)
        suffix_pv: Suffix for PV charging files (Scenario C)

    Returns:
        (list of original DataFrames, list of PV DataFrames, list of vehicle IDs)
    """
    charging_dir = Path(charging_folder)
    pv_dir = Path(pv_folder)

    # Find all PV charging files
    pv_files = sorted(pv_dir.glob(f'*{suffix_pv}'))

    original_dfs = []
    pv_dfs = []
    vehicle_ids = []

    for pv_file in pv_files:
        # Get vehicle ID by removing suffix
        vehicle_id = pv_file.stem.replace(suffix_pv.replace('.csv', ''), '')

        # Construct original file path in the charging folder
        original_file = charging_dir / f"{vehicle_id}{suffix_original}"

        if original_file.exists():
            try:
                df_original = load_charging_data(str(original_file))
                df_pv = load_charging_data(str(pv_file))

                if len(df_original) > 0 and len(df_pv) > 0:
                    # Add battery_capacity_kwh to original data from PV data
                    # (PV data already has this field from integration process)
                    if 'battery_capacity_kwh' in df_pv.columns and 'battery_capacity_kwh' not in df_original.columns:
                        battery_capacity = df_pv['battery_capacity_kwh'].iloc[0]
                        df_original['battery_capacity_kwh'] = battery_capacity

                    # Add original_charge_demand_kwh to original data if not present
                    if 'original_charge_demand_kwh' not in df_original.columns:
                        battery_capacity = df_original['battery_capacity_kwh'].iloc[0]
                        # Ensure non-negative values (negative means discharge, not charging)
                        df_original['original_charge_demand_kwh'] = ((df_original['esoc'] - df_original['ssoc']) / 100.0) * battery_capacity
                        df_original['original_charge_demand_kwh'] = df_original['original_charge_demand_kwh'].clip(lower=0)

                    original_dfs.append(df_original)
                    pv_dfs.append(df_pv)
                    vehicle_ids.append(vehicle_id)
                    print(f"  Loaded {vehicle_id}: {len(df_original)} original, {len(df_pv)} PV records")
            except Exception as e:
                print(f"  Error loading {vehicle_id}: {e}")
        else:
            print(f"  Warning: Original charging file not found for {vehicle_id}")

    print(f"\nTotal vehicles loaded: {len(vehicle_ids)}")
    return original_dfs, pv_dfs, vehicle_ids


def aggregate_charging_demand(df: pd.DataFrame,
                             use_net_grid: bool = False,
                             time_start: Optional[pd.Timestamp] = None,
                             time_end: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Extract charging demand events for a single vehicle

    Args:
        df: Charging data DataFrame
        use_net_grid: Whether to use net grid energy (Scenario C) or original demand (Scenario A)
        time_start: Start of time range filter
        time_end: End of time range filter

    Returns:
        DataFrame with columns ['timestamp', 'energy_kwh'] for each charging event
    """
    # Filter by time range
    if time_start is not None:
        df = df[df['etime'] >= time_start]
    if time_end is not None:
        df = df[df['stime'] <= time_end]

    if len(df) == 0:
        return pd.DataFrame(columns=['timestamp', 'energy_kwh'])

    records = []
    for _, row in df.iterrows():
        # Use charging end time as the timestamp for demand
        timestamp = row['etime']

        if use_net_grid:
            # Scenario C: Net grid energy (after PV contribution)
            energy = row['net_grid_energy_kwh']
        else:
            # Scenario A: Original charging demand
            energy = row['original_charge_demand_kwh']

        records.append({
            'timestamp': timestamp,
            'energy_kwh': energy
        })

    return pd.DataFrame(records)


def aggregate_fleet_charging_demand(dfs_list: List[pd.DataFrame],
                                   use_net_grid: bool = False,
                                   time_start: Optional[pd.Timestamp] = None,
                                   time_end: Optional[pd.Timestamp] = None,
                                   freq: str = '1h') -> pd.DataFrame:
    """
    Aggregate charging demand from multiple vehicles onto a common time grid

    Args:
        dfs_list: List of charging DataFrames for each vehicle
        use_net_grid: Whether to use net grid energy (Scenario C) or original demand (Scenario A)
        time_start: Start of time range
        time_end: End of time range
        freq: Time period for aggregation (e.g., '1h', '1D')

    Returns:
        DataFrame with columns ['timestamp', 'total_demand_kwh', 'num_charges']
    """
    # Collect all charging events from all vehicles
    all_events = []
    for df in dfs_list:
        events = aggregate_charging_demand(df, use_net_grid, time_start, time_end)
        if len(events) > 0:
            all_events.append(events)

    if len(all_events) == 0:
        return pd.DataFrame(columns=['timestamp', 'total_demand_kwh', 'num_charges'])

    # Combine all events
    combined = pd.concat(all_events, ignore_index=True)

    # Determine time range
    if time_start is None:
        time_start = combined['timestamp'].min()
    if time_end is None:
        time_end = combined['timestamp'].max()

    # Create time bins
    time_bins = pd.date_range(start=time_start, end=time_end, freq=freq)

    # Add one more bin to include the end
    if time_bins[-1] < time_end:
        time_bins = time_bins.union([time_end + pd.Timedelta(freq)])

    # Aggregate by time period
    combined['period'] = pd.cut(combined['timestamp'], bins=time_bins, labels=time_bins[:-1], include_lowest=True)

    result = combined.groupby('period').agg(
        total_demand_kwh=('energy_kwh', 'sum'),
        num_charges=('energy_kwh', 'count')
    ).reset_index()

    # Convert period (category) back to timestamp
    result['timestamp'] = pd.to_datetime(result['period'])
    result = result.drop(columns=['period'])

    # Fill missing periods with zeros
    full_range = pd.DataFrame({'timestamp': time_bins[:-1]})
    result = full_range.merge(result, on='timestamp', how='left').fillna(0)

    return result


def plot_daily_pattern(charging_folder: str,
                      pv_folder: str,
                      time_start: Optional[str] = None,
                      time_end: Optional[str] = None,
                      output_file: str = 'fleet_daily_pattern.png'):
    """
    Plot 24-hour daily pattern of charging demand (averaged across all days)

    Args:
        charging_folder: Path to folder containing original charging CSV files
        pv_folder: Path to folder containing PV-integrated charging CSV files
        time_start: Start date string (e.g., '2023-01-01') or None for auto
        time_end: End date string (e.g., '2023-12-31') or None for auto
        output_file: Output image path
    """
    print("\n" + "=" * 70)
    print("STEP 3: DAILY PATTERN VISUALIZATION (24-HOUR)")
    print("=" * 70)

    # Parse time range
    ts_start = pd.to_datetime(time_start) if time_start else None
    ts_end = pd.to_datetime(time_end) if time_end else None

    # Load fleet data
    print("Loading fleet data...")
    original_dfs, pv_dfs, vehicle_ids = load_fleet_data(charging_folder, pv_folder)

    if len(original_dfs) == 0:
        print("No vehicle data found!")
        return

    # Aggregate by hour of day (0-23)
    print("\nAggregating by hour of day...")

    # Collect all charging events with hour of day
    original_hourly = []
    pv_hourly = []

    for df in original_dfs:
        events = aggregate_charging_demand(df, use_net_grid=False, time_start=ts_start, time_end=ts_end)
        if len(events) > 0:
            events['hour'] = events['timestamp'].dt.hour
            original_hourly.append(events)

    for df in pv_dfs:
        events = aggregate_charging_demand(df, use_net_grid=True, time_start=ts_start, time_end=ts_end)
        if len(events) > 0:
            events['hour'] = events['timestamp'].dt.hour
            pv_hourly.append(events)

    # Combine and aggregate by hour
    if len(original_hourly) == 0 or len(pv_hourly) == 0:
        print("No data to plot!")
        return

    original_combined = pd.concat(original_hourly, ignore_index=True)
    pv_combined = pd.concat(pv_hourly, ignore_index=True)

    # Group by hour and calculate mean
    original_by_hour = original_combined.groupby('hour')['energy_kwh'].sum().reindex(range(24), fill_value=0)
    pv_by_hour = pv_combined.groupby('hour')['energy_kwh'].sum().reindex(range(24), fill_value=0)

    # Calculate reduction
    reduction_by_hour = original_by_hour - pv_by_hour

    # Plot
    print("Creating plot...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), height_ratios=[2, 1])

    hours = range(24)

    # ========== Upper Plot: Daily Pattern Comparison ==========
    ax1.plot(hours, original_by_hour.values,
            color='#1f77b4',
            linewidth=2.5,
            label='Scenario A: Without Rooftop PV',
            alpha=0.8,
            marker='o',
            markersize=6)

    ax1.plot(hours, pv_by_hour.values,
            color='#2ca02c',
            linewidth=2.5,
            label='Scenario C: With Rooftop PV',
            alpha=0.8,
            marker='s',
            markersize=6)

    # Styling for upper plot
    ax1.set_title('EV Fleet Daily Charging Pattern: Rooftop PV Impact (24-Hour Average)',
                fontsize=16,
                fontweight='bold',
                pad=20)
    ax1.set_xlabel('Hour of Day', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Total Charging Demand (kWh)', fontsize=13, fontweight='bold')
    ax1.set_xticks(hours)
    ax1.set_xlim(-0.5, 23.5)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(loc='best', fontsize=12, framealpha=0.95, shadow=True)

    # ========== Lower Plot: Demand Reduction by Hour ==========
    ax2.fill_between(hours, reduction_by_hour.values, 0,
                     color='#ff7f0e',
                     alpha=0.3,
                     label='Grid Demand Reduction')
    ax2.plot(hours, reduction_by_hour.values,
            color='#ff7f0e',
            linewidth=2.5,
            alpha=0.9,
            marker='D',
            markersize=5)

    # Add a zero reference line
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Styling for lower plot
    ax2.set_title('Grid Demand Reduction by Hour',
                fontsize=14,
                fontweight='bold',
                pad=15)
    ax2.set_xlabel('Hour of Day', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Demand Reduction (kWh)', fontsize=13, fontweight='bold')
    ax2.set_xticks(hours)
    ax2.set_xlim(-0.5, 23.5)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax2.legend(loc='best', fontsize=12, framealpha=0.95, shadow=True)

    # Calculate statistics
    total_pv_energy = sum(df['total_pv_contribution_kwh'].sum() for df in pv_dfs)
    total_demand_original = original_by_hour.sum()
    total_demand_pv = pv_by_hour.sum()
    demand_reduction = total_demand_original - total_demand_pv
    reduction_percentage = (demand_reduction / total_demand_original * 100) if total_demand_original > 0 else 0

    # Peak hours
    peak_hour_original = original_by_hour.idxmax()
    peak_hour_pv = pv_by_hour.idxmax()
    max_reduction_hour = reduction_by_hour.idxmax()

    # Statistics text box
    stats_text = (
        f"Daily Pattern Statistics:\n"
        f"• Number of vehicles: {len(vehicle_ids)}\n"
        f"• Total PV contribution: {total_pv_energy:.2f} kWh\n"
        f"\n"
        f"Hourly Aggregated Demand:\n"
        f"• Original demand: {total_demand_original:.2f} kWh\n"
        f"• Net grid demand (w/ PV): {total_demand_pv:.2f} kWh\n"
        f"• Demand reduction: {demand_reduction:.2f} kWh ({reduction_percentage:.1f}%)\n"
        f"\n"
        f"Peak Hours:\n"
        f"• Original peak: {peak_hour_original:02d}:00\n"
        f"• Max reduction at: {max_reduction_hour:02d}:00"
    )

    ax1.text(0.02, 0.98, stats_text,
           transform=ax1.transAxes,
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

    print("\n" + "=" * 70)
    print("DAILY PATTERN VISUALIZATION COMPLETE!")
    print("=" * 70)


def plot_fleet_comparison(charging_folder: str,
                         pv_folder: str,
                         time_start: Optional[str] = None,
                         time_end: Optional[str] = None,
                         freq: str = '1h',
                         output_file: str = 'fleet_scenario_comparison.png'):
    """
    Plot fleet-level comparison between scenarios

    Args:
        charging_folder: Path to folder containing original charging CSV files
        pv_folder: Path to folder containing PV-integrated charging CSV files
        time_start: Start date string (e.g., '2023-01-01') or None for auto
        time_end: End date string (e.g., '2023-12-31') or None for auto
        freq: Time resolution for aggregation (e.g., '15min', '1h')
        output_file: Output image path
    """
    print("\n" + "=" * 70)
    print("STEP 2: FLEET-LEVEL VISUALIZATION")
    print("=" * 70)

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
    original_dfs, pv_dfs, vehicle_ids = load_fleet_data(charging_folder, pv_folder)

    if len(original_dfs) == 0:
        print("No vehicle data found!")
        return

    # Aggregate charging demand
    print("\nAggregating Scenario A (without PV)...")
    agg_original = aggregate_fleet_charging_demand(original_dfs,
                                                   use_net_grid=False,
                                                   time_start=ts_start,
                                                   time_end=ts_end,
                                                   freq=freq)

    print("Aggregating Scenario C (with PV)...")
    agg_pv = aggregate_fleet_charging_demand(pv_dfs,
                                            use_net_grid=True,
                                            time_start=ts_start,
                                            time_end=ts_end,
                                            freq=freq)

    # Calculate statistics
    print("\nCalculating fleet statistics...")
    total_pv_energy = sum(df['total_pv_contribution_kwh'].sum() for df in pv_dfs)
    total_charging_events = sum(len(df) for df in pv_dfs)
    total_eliminated = sum(df['charging_eliminated'].sum() for df in pv_dfs)

    # Plot
    print("Creating plot...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), height_ratios=[2, 1])

    # ========== Upper Plot: Scenario Comparison ==========
    # Plot Scenario A
    ax1.plot(agg_original['timestamp'],
            agg_original['total_demand_kwh'],
            color='#1f77b4',
            linewidth=2.5,
            label='Scenario A: Without Rooftop PV',
            alpha=0.8,
            marker='o',
            markersize=4)

    # Plot Scenario C
    ax1.plot(agg_pv['timestamp'],
            agg_pv['total_demand_kwh'],
            color='#2ca02c',
            linewidth=2.5,
            label='Scenario C: With Rooftop PV',
            alpha=0.8,
            marker='s',
            markersize=4)

    # Styling for upper plot
    ax1.set_title('EV Fleet Charging Demand Comparison: Rooftop PV Impact Analysis',
                fontsize=16,
                fontweight='bold',
                pad=20)
    ax1.set_xlabel('Date', fontsize=13, fontweight='bold')
    ax1.set_ylabel(f'Total Charging Demand per {freq} (kWh)', fontsize=13, fontweight='bold')

    # Grid for upper plot
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.4)

    # Format x-axis for upper plot
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.tick_params(axis='x', rotation=45)

    # Legend for upper plot
    ax1.legend(loc='best', fontsize=12, framealpha=0.95, shadow=True)

    # ========== Lower Plot: Demand Reduction ==========
    # Calculate demand reduction for each time period
    demand_reduction_series = agg_original['total_demand_kwh'] - agg_pv['total_demand_kwh']

    # Plot demand reduction
    ax2.fill_between(agg_original['timestamp'],
                     demand_reduction_series,
                     0,
                     color='#ff7f0e',
                     alpha=0.3,
                     label='Grid Demand Reduction')
    ax2.plot(agg_original['timestamp'],
            demand_reduction_series,
            color='#ff7f0e',
            linewidth=2.5,
            alpha=0.9,
            marker='D',
            markersize=3)

    # Add a zero reference line
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Styling for lower plot
    ax2.set_title('Grid Demand Reduction by Rooftop PV',
                fontsize=14,
                fontweight='bold',
                pad=15)
    ax2.set_xlabel('Date', fontsize=13, fontweight='bold')
    ax2.set_ylabel(f'Demand Reduction per {freq} (kWh)', fontsize=13, fontweight='bold')

    # Grid for lower plot
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.4)

    # Format x-axis for lower plot
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.tick_params(axis='x', rotation=45)

    # Legend for lower plot
    ax2.legend(loc='best', fontsize=12, framealpha=0.95, shadow=True)

    # Calculate demand statistics
    total_demand_original = agg_original['total_demand_kwh'].sum()
    total_demand_pv = agg_pv['total_demand_kwh'].sum()
    demand_reduction = total_demand_original - total_demand_pv
    reduction_percentage = (demand_reduction / total_demand_original * 100) if total_demand_original > 0 else 0

    # Statistics text box
    stats_text = (
        f"Fleet Statistics:\n"
        f"• Number of vehicles: {len(vehicle_ids)}\n"
        f"• Total charging events: {total_charging_events}\n"
        f"• Total PV contribution: {total_pv_energy:.2f} kWh\n"
        f"• Charging events eliminated: {total_eliminated}\n"
        f"\n"
        f"Demand Comparison ({freq} periods):\n"
        f"• Original demand: {total_demand_original:.2f} kWh\n"
        f"• Net grid demand (w/ PV): {total_demand_pv:.2f} kWh\n"
        f"• Demand reduction: {demand_reduction:.2f} kWh ({reduction_percentage:.1f}%)"
    )

    ax1.text(0.02, 0.98, stats_text,
           transform=ax1.transAxes,
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

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE!")
    print("=" * 70)


# ============================================================================
# Part 3: Complete Pipeline
# ============================================================================

def find_vehicle_files(charging_folder: str,
                      pv_gen_folder: str,
                      suffix_charging: str = '_charging.csv',
                      suffix_pv_gen: str = '_pv_generation.csv') -> List[Tuple[str, str, str]]:
    """
    Find all vehicle file pairs from separate folders

    Args:
        charging_folder: Path to folder containing charging files
        pv_gen_folder: Path to folder containing PV generation files
        suffix_charging: Suffix for charging files
        suffix_pv_gen: Suffix for PV generation files

    Returns:
        List of (vehicle_id, charging_file_path, pv_gen_file_path) tuples
    """
    charging_dir = Path(charging_folder)
    pv_gen_dir = Path(pv_gen_folder)

    # Find all charging files
    charging_files = sorted(charging_dir.glob(f'*{suffix_charging}'))

    vehicle_files = []
    for charging_file in charging_files:
        # Get vehicle ID by removing suffix
        vehicle_id = charging_file.stem.replace(suffix_charging.replace('.csv', ''), '')

        # Construct PV generation file path in the separate folder
        pv_gen_file = pv_gen_dir / f"{vehicle_id}{suffix_pv_gen}"

        if pv_gen_file.exists():
            vehicle_files.append((
                vehicle_id,
                str(charging_file),
                str(pv_gen_file)
            ))
            print(f"  Found vehicle: {vehicle_id}")
        else:
            print(f"  Warning: PV generation file not found for {vehicle_id}")

    return vehicle_files


def process_fleet_and_plot(
    charging_folder: str,
    pv_gen_folder: str,
    output_folder: str,
    vehicle_info_file: str,
    charging_efficiency: float = 0.95,
    skip_existing: bool = True,
    time_start: Optional[str] = None,
    time_end: Optional[str] = None,
    freq: str = '1h',
    output_plot: str = 'fleet_scenario_comparison.png'
):
    """
    Complete pipeline: Process all vehicles and create fleet comparison plot

    Args:
        charging_folder: Path to folder containing charging CSV files
        pv_gen_folder: Path to folder containing PV generation CSV files
        output_folder: Path to folder for output files (integrated data and plot)
        vehicle_info_file: Path to vehicle info CSV file
        charging_efficiency: Charging efficiency (default 0.95)
        skip_existing: Skip processing if output file already exists (default True)
        time_start: Start date for plot (e.g., '2023-01-01') or None
        time_end: End date for plot (e.g., '2023-12-31') or None
        freq: Time resolution for aggregation (e.g., '15min', '1h')
        output_plot: Output plot filename (will be saved in output_folder)
    """
    print("\n" + "=" * 70)
    print("EV FLEET PV ANALYSIS PIPELINE")
    print("=" * 70)
    print(f"Charging data folder: {charging_folder}")
    print(f"PV generation folder: {pv_gen_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Vehicle info: {vehicle_info_file}")
    print(f"Charging efficiency: {charging_efficiency}")
    print("=" * 70)

    # Create output folder if it doesn't exist
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Find all vehicle files
    print("\n" + "=" * 70)
    print("STEP 1: PV DATA INTEGRATION FOR ALL VEHICLES")
    print("=" * 70)
    print("\nScanning for vehicle files...")
    vehicle_files = find_vehicle_files(charging_folder, pv_gen_folder)

    if len(vehicle_files) == 0:
        print("\nNo vehicle files found! Looking for pairs of:")
        print(f"  - Charging files in: {charging_folder}/*_charging.csv")
        print(f"  - PV generation files in: {pv_gen_folder}/*_pv_generation.csv")
        return

    print(f"\nFound {len(vehicle_files)} vehicles to process\n")

    # Step 2: Process each vehicle
    processed_count = 0
    skipped_count = 0
    failed_count = 0

    for vehicle_id, charging_file, pv_gen_file in vehicle_files:
        print(f"\nProcessing vehicle: {vehicle_id}")
        print("-" * 70)

        output_file = str(output_dir / f"{vehicle_id}_charging_with_pv.csv")

        # Check if output file already exists (if skip_existing is True)
        if skip_existing and Path(output_file).exists():
            print(f"  ⊙ Output file already exists, skipping...")
            skipped_count += 1
            processed_count += 1  # Count as processed for plotting
            continue

        try:
            integrate_pv_with_charging(
                charging_file=charging_file,
                pv_generation_file=pv_gen_file,
                vehicle_info_file=vehicle_info_file,
                output_file=output_file,
                charging_efficiency=charging_efficiency,
                verbose=True
            )
            processed_count += 1
            print(f"  ✓ Success!")

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed_count += 1

    # Summary
    print("\n" + "=" * 70)
    print("INTEGRATION SUMMARY")
    print("=" * 70)
    print(f"Total vehicles found: {len(vehicle_files)}")
    print(f"Already processed (skipped): {skipped_count}")
    print(f"Newly processed: {processed_count - skipped_count}")
    print(f"Failed: {failed_count}")
    print(f"Total available for plotting: {processed_count}")
    print("=" * 70)

    if processed_count == 0:
        print("\nNo vehicles were processed successfully. Cannot create plot.")
        return

    # Step 3: Create fleet comparison plot
    output_plot_path = str(output_plot)
    plot_fleet_comparison(
        charging_folder=charging_folder,  # Load original charging files
        pv_folder=str(output_dir),        # Load PV-integrated files from output folder
        time_start=time_start,
        time_end=time_end,
        freq=freq,
        output_file=output_plot_path
    )

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"Total vehicles: {len(vehicle_files)}")
    print(f"  - Skipped (already processed): {skipped_count}")
    print(f"  - Newly processed: {processed_count - skipped_count}")
    print(f"  - Failed: {failed_count}")
    print(f"Output plot: {output_plot_path}")
    print(f"All files saved in: {output_folder}")
    print("=" * 70)


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == '__main__':
    # Configuration
    config = {
        'charging_folder': '/data2/hcr/evipv/shanghaidata/chargedata',           # Folder containing *_charging.csv files
        'pv_gen_folder': '/data2/hcr/evipv/output_ev_charge',        # Folder containing *_pv_generation.csv files
        'output_folder': '/data2/hcr/evipv/shanghaidata/charge_with_pv',                    # Folder for output files
        'vehicle_info_file': '车型信息.csv',             # Vehicle information file
        'charging_efficiency': 0.95,                    # Charging efficiency
        'skip_existing': False,                          # Skip already processed vehicles (True/False)
        'time_start': '2020-11-01',                             # Plot start date (e.g., '2020-10-01')
        'time_end': '2020-11-07',                               # Plot end date (e.g., '2020-12-31')
        'freq': '1min',                                 # Aggregation frequency: '1min', '15min', '30min', '1h'
        'output_plot': 'fleet_scenario_comparison.png'  # Output plot filename
    }

    # Run complete pipeline
    process_fleet_and_plot(**config)