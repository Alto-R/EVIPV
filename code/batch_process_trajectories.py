"""
批量处理轨迹数据 - 单GPU串行计算（多文件夹版本）

功能：
1. 按顺序处理多个文件夹（01, 02, 04, 05-12）
2. 自动发现每个文件夹中的预处理轨迹文件
3. 为每个轨迹自动匹配对应日期的气象数据（缓存/API）
4. 串行处理每个轨迹（GPU内部高度并行）
5. 保存简化的输出文件到对应的output_XX文件夹
6. 为每个文件夹生成独立的批处理汇总报告

文件夹结构：
    traj/01/              → 输出到 output_01/
    traj/02/              → 输出到 output_02/
    traj/04/              → 输出到 output_04/
    traj/05-12/           → 输出到 output_05 - output_12/

使用示例：
    # 使用内部CONFIG配置（推荐）
    python batch_process_trajectories.py

    # 使用外部config.yaml（可选）
    python batch_process_trajectories.py --config config.yaml

    # 指定使用GPU 0
    python batch_process_trajectories.py --gpu 0

    # 指定使用GPU 1
    python batch_process_trajectories.py --gpu 1

    # 组合使用
    python batch_process_trajectories.py --config config.yaml --gpu 1
"""

import os
import sys
from pathlib import Path
import pandas as pd
import trimesh
import yaml
import argparse
from datetime import datetime
import time

# 导入自定义模块
from prepare_building_mesh_from_footprint import prepare_building_mesh_from_footprint
from fetch_irradiance_data import fetch_and_cache_irradiance_data, convert_to_pvlib_format
from pv_calculator_gpu import GPUAcceleratedSolarPVCalculator


# ============================================================================
# 配置参数 - 在此修改您的设置
# ============================================================================
CONFIG = {
    'location': {
        'name': '深圳市',
        'lat': 22.543099,
        'lon': 114.057868,
    },
    'data_sources': {
        'footprint_path': 'data/shenzhen_buildings.geojson',
        'trajectory_dir': 'traj',  # 轨迹文件目录
    },
    'pv_system': {
        'panel_area': 2.0,          # 光伏板面积(m²)
        'panel_efficiency': 0.22,   # 效率 22%
        'tilt': 5,                  # 倾角(度)
        'vehicle_height': 1.5,      # 车顶高度(m)
    },
    'computation': {
        'time_resolution_minutes': 1,  # 时间分辨率
        'use_gpu': True,               # 启用GPU
        'gpu_id': 0,                   # GPU编号 (0, 1, 2...), None=自动选择
        'batch_size': 100,             # 批处理大小
        'mesh_grid_size': None,        # mesh网格大小(m), None=不细分
    },
    'output': {
        'mesh_path': 'building_mesh.ply',
        'output_dir': 'output',
    },
}


def load_config(config_path='config.yaml'):
    """
    加载外部配置文件（可选）

    如果提供config.yaml，将覆盖内部CONFIG
    如果不存在，则使用内部CONFIG
    """
    if Path(config_path).exists():
        print(f"📄 加载外部配置文件: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    else:
        return None


def find_processed_trajectories(traj_dir='traj'):
    """
    查找所有预处理后的轨迹文件

    Parameters
    ----------
    traj_dir : str
        轨迹文件目录

    Returns
    -------
    list
        轨迹文件路径列表
    """
    traj_dir = Path(traj_dir)
    traj_files = list(traj_dir.glob('*_processed.csv'))

    return sorted(traj_files)


def calculate_stats(result_df):
    """
    计算轨迹的统计信息

    Parameters
    ----------
    result_df : pandas.DataFrame
        计算结果

    Returns
    -------
    dict
        统计信息
    """
    return {
        'total_records': len(result_df),
        'total_energy_kwh': result_df['energy_kwh'].sum(),
        'avg_power_w': result_df['ac_power'].mean(),
        'max_power_w': result_df['ac_power'].max(),
        'shaded_ratio': result_df['is_shaded'].mean(),
        'avg_cell_temp': result_df['cell_temp'].mean(),
        'time_range': (result_df['datetime'].min(), result_df['datetime'].max()),
    }


def save_summary(result_df, vehicle_id, output_path, elapsed_time, config):
    """
    保存统计摘要

    Parameters
    ----------
    result_df : pandas.DataFrame
        计算结果
    vehicle_id : str
        车辆ID
    output_path : str or Path
        输出文件路径
    elapsed_time : float
        计算耗时（秒）
    config : dict
        配置信息
    """
    stats = calculate_stats(result_df)

    # 按小时统计
    result_df_copy = result_df.copy()
    result_df_copy['hour'] = pd.to_datetime(result_df_copy['datetime']).dt.hour
    hourly_stats = result_df_copy.groupby('hour').agg({
        'ac_power': 'mean',
        'energy_kwh': 'sum'
    })

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write(f"Vehicle PV Generation Summary - {vehicle_id}\n")
        f.write("="*60 + "\n\n")
        f.write(f"Calculation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Location: {config['location']['name']} ")
        f.write(f"({config['location']['lat']:.4f}, {config['location']['lon']:.4f})\n")
        f.write(f"Time Range: {stats['time_range'][0]} to {stats['time_range'][1]}\n")
        f.write(f"Time Resolution: {config['computation']['time_resolution_minutes']} min\n")
        f.write(f"GPU Acceleration: {'Enabled' if config['computation']['use_gpu'] else 'Disabled'}\n\n")

        f.write("Overall Statistics:\n")
        f.write(f"  Total Energy: {stats['total_energy_kwh']:.2f} kWh\n")
        f.write(f"  Average Power: {stats['avg_power_w']:.2f} W\n")
        f.write(f"  Peak Power: {stats['max_power_w']:.2f} W\n")
        f.write(f"  Shaded Ratio: {stats['shaded_ratio']*100:.1f}%\n")
        f.write(f"  Average Cell Temperature: {stats['avg_cell_temp']:.1f}°C\n")
        f.write(f"  Calculation Time: {elapsed_time:.1f} seconds\n\n")

        f.write("Hourly Generation (kWh):\n")
        for hour, row in hourly_stats.iterrows():
            f.write(f"  {hour:02d}:00 - {row['energy_kwh']:.3f} kWh "
                   f"(Avg Power: {row['ac_power']:.1f} W)\n")


def save_batch_summary(all_stats, output_path):
    """
    保存批处理汇总报告

    Parameters
    ----------
    all_stats : dict
        所有车辆的统计信息
    output_path : str or Path
        输出文件路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("Batch Processing Summary - All Vehicles\n")
        f.write("="*60 + "\n\n")
        f.write(f"Processing Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Vehicles: {len(all_stats)}\n\n")

        total_energy = sum(s['stats']['total_energy_kwh'] for s in all_stats.values())
        total_time = sum(s['elapsed_time'] for s in all_stats.values())

        f.write("Overall Summary:\n")
        f.write(f"  Total Energy (All Vehicles): {total_energy:.2f} kWh\n")
        f.write(f"  Total Calculation Time: {total_time:.1f} seconds ({total_time/60:.1f} min)\n\n")

        f.write("Per-Vehicle Statistics:\n")
        f.write("-"*60 + "\n")

        for vehicle_id, data in all_stats.items():
            stats = data['stats']
            f.write(f"\n{vehicle_id}:\n")
            f.write(f"  Records: {stats['total_records']:,}\n")
            f.write(f"  Energy: {stats['total_energy_kwh']:.2f} kWh\n")
            f.write(f"  Avg Power: {stats['avg_power_w']:.2f} W\n")
            f.write(f"  Peak Power: {stats['max_power_w']:.2f} W\n")
            f.write(f"  Shaded Ratio: {stats['shaded_ratio']*100:.1f}%\n")
            f.write(f"  Calculation Time: {data['elapsed_time']:.1f}s\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='批量处理轨迹数据计算光伏发电量（单GPU串行）'
    )
    parser.add_argument(
        '--config', '-c',
        default=None,
        help='外部配置文件路径（可选，不指定则使用脚本内部CONFIG）'
    )
    parser.add_argument(
        '--gpu', '-g',
        type=int,
        default=None,
        help='指定GPU编号 (0, 1, 2...), 不指定则使用配置文件中的设置'
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print(" "*15 + "🚀 Batch Vehicle PV Generation Calculation (GPU)")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 加载配置：优先使用外部配置，否则使用内部CONFIG
    if args.config:
        external_config = load_config(args.config)
        if external_config:
            config = external_config
        else:
            print(f"⚠️  外部配置文件 {args.config} 不存在，使用内部CONFIG")
            config = CONFIG
    else:
        print("📋 使用脚本内部CONFIG配置")
        config = CONFIG

    # GPU设置：命令行参数优先于配置文件
    gpu_id = args.gpu if args.gpu is not None else config['computation'].get('gpu_id', 0)

    if config['computation']['use_gpu'] and gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        print(f"🎮 GPU设置: 使用GPU {gpu_id}")
        print(f"   环境变量: CUDA_VISIBLE_DEVICES={gpu_id}")
    elif config['computation']['use_gpu']:
        print(f"🎮 GPU设置: 使用默认GPU（自动选择）")
    else:
        print(f"💻 GPU设置: GPU已禁用，使用CPU")

    print("="*80)

    # 定义要处理的文件夹列表
    folder_ids = ['01', '02', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    base_traj_dir = Path(config['data_sources']['trajectory_dir'])  # 从配置中读取轨迹基础目录

    print(f"\n📁 将按顺序处理以下文件夹: {', '.join([str(base_traj_dir / fid) for fid in folder_ids])}")
    print("="*80)

    # 准备建筑mesh（所有文件夹共享）
    print("\n" + "="*80)
    print("Preparing Building Mesh (Shared by All Folders)")
    print("="*80)

    mesh_path = Path(config['output']['mesh_path'])

    if mesh_path.exists():
        print(f"✅ Loading existing mesh: {mesh_path}")
        building_mesh = trimesh.load(mesh_path)
        print(f"   Vertices: {len(building_mesh.vertices):,}")
        print(f"   Faces: {len(building_mesh.faces):,}")
    else:
        print(f"🔄 Converting footprint to mesh...")
        building_mesh = prepare_building_mesh_from_footprint(
            footprint_path=config['data_sources']['footprint_path'],
            output_mesh_path=str(mesh_path),
            grid_size=config['computation']['mesh_grid_size']
        )

    # 初始化GPU计算器（所有文件夹共享）
    print("\n" + "="*80)
    print("Initialize GPU Calculator (Shared by All Folders)")
    print("="*80)

    calculator = GPUAcceleratedSolarPVCalculator(
        lon_center=config['location']['lon'],
        lat_center=config['location']['lat'],
        building_mesh=building_mesh,
        panel_area=config['pv_system']['panel_area'],
        panel_efficiency=config['pv_system']['panel_efficiency'],
        time_resolution_minutes=config['computation']['time_resolution_minutes'],
        use_gpu=config['computation']['use_gpu'],
        batch_size=config['computation']['batch_size']
    )

    # 外层循环：遍历每个文件夹
    for folder_idx, folder_id in enumerate(folder_ids, 1):
        print("\n\n" + "="*80)
        print(f"{'='*80}")
        print(f"  Processing Folder {folder_idx}/{len(folder_ids)}: {folder_id}")
        print(f"{'='*80}")
        print("="*80 + "\n")

        # 设置当前文件夹的输入输出路径
        traj_dir = base_traj_dir / folder_id  # traj/01, traj/02, ...
        output_dir = Path(f'output_{folder_id}')  # output_01, output_02, ...
        output_dir.mkdir(parents=True, exist_ok=True)

        # 查找轨迹文件
        print("="*80)
        print(f"Step 1: Discover Trajectory Files in Folder {folder_id}")
        print("="*80)

        traj_files = find_processed_trajectories(traj_dir)

        if not traj_files:
            print(f"⚠️  No processed trajectory files found in {traj_dir}/")
            print(f"   Skipping folder {folder_id}...")
            continue

        print(f"✅ Found {len(traj_files)} processed trajectory files:")
        for f in traj_files:
            vehicle_id = f.stem.replace('_processed', '')
            print(f"  - {f.name} → Vehicle ID: {vehicle_id}")

        # 批量处理当前文件夹的轨迹
        print("\n" + "="*80)
        print(f"Step 2: Process Trajectories in Folder {folder_id}")
        print("="*80)

        all_stats = {}

        for idx, traj_file in enumerate(traj_files, 1):
            vehicle_id = traj_file.stem.replace('_processed', '')

            print(f"\n{'='*80}")
            print(f"Processing Vehicle {idx}/{len(traj_files)}: {vehicle_id}")
            print('='*80)

            try:
                # 读取轨迹
                print(f"\n📂 Loading trajectory: {traj_file.name}")
                trajectory_df = pd.read_csv(traj_file)
                trajectory_df['datetime'] = pd.to_datetime(trajectory_df['datetime'])
                print(f"   Records: {len(trajectory_df):,}")

                # 推断日期范围
                start_date = trajectory_df['datetime'].min().strftime('%Y-%m-%d')
                end_date = trajectory_df['datetime'].max().strftime('%Y-%m-%d')
                print(f"   Date Range: {start_date} to {end_date}")

                # 获取气象数据（自动缓存/API）
                print(f"\n☀️  Fetching Irradiance Data (Auto Cache/API)...")
                irradiance_data = fetch_and_cache_irradiance_data(
                    lat=config['location']['lat'],
                    lon=config['location']['lon'],
                    start_date=start_date,
                    end_date=end_date,
                    granularity='1min' if config['computation']['time_resolution_minutes'] == 1 else '1hour',
                    save_csv=False,  # 不额外保存CSV（已有缓存）
                    output_dir='irradiance_data'
                )

                weather_data = convert_to_pvlib_format(irradiance_data)

                # GPU计算
                print(f"\n⚡ Calculating PV Generation (GPU Accelerated)...")
                start_time = time.time()

                result_df = calculator.process_trajectory(
                    trajectory_df,
                    weather_data=weather_data
                )

                elapsed_time = time.time() - start_time

                # 保存结果（简化文件结构）
                print(f"\n💾 Saving Results...")

                # 详细结果
                result_csv = output_dir / f"{vehicle_id}_pv_generation.csv"
                result_df.to_csv(result_csv, index=False)
                file_size_mb = result_csv.stat().st_size / (1024 * 1024)
                print(f"   ✅ PV Generation: {result_csv}")
                print(f"      Size: {file_size_mb:.2f} MB, Records: {len(result_df):,}")

                # 收集统计
                stats = calculate_stats(result_df)
                all_stats[vehicle_id] = {
                    'stats': stats,
                    'elapsed_time': elapsed_time
                }

                print(f"\n📊 Quick Stats:")
                print(f"   Total Energy: {stats['total_energy_kwh']:.2f} kWh")
                print(f"   Avg Power: {stats['avg_power_w']:.2f} W")
                print(f"   Peak Power: {stats['max_power_w']:.2f} W")
                print(f"   Calculation Time: {elapsed_time:.1f}s")

            except Exception as e:
                print(f"\n❌ Error processing {vehicle_id}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # 保存当前文件夹的批处理汇总
        print("\n" + "="*80)
        print(f"Step 3: Generate Summary for Folder {folder_id}")
        print("="*80)

        if all_stats:
            batch_summary_path = output_dir / "batch_summary.txt"
            save_batch_summary(all_stats, batch_summary_path)
            print(f"✅ Batch Summary: {batch_summary_path}")

            print(f"\n📊 Folder {folder_id} Processing Summary:")
            print(f"   Successfully Processed: {len(all_stats)} vehicles")
            total_energy = sum(s['stats']['total_energy_kwh'] for s in all_stats.values())
            total_time = sum(s['elapsed_time'] for s in all_stats.values())
            print(f"   Total Energy (All Vehicles): {total_energy:.2f} kWh")
            print(f"   Total Calculation Time: {total_time:.1f}s ({total_time/60:.1f} min)")
        else:
            print(f"❌ No vehicles processed successfully in folder {folder_id}")

        print(f"\n✅ Folder {folder_id} Processing Complete!")

    # 完成所有处理
    print("\n\n" + "="*80)
    print("="*80)
    print("✅ All Folders Processing Complete!")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print("="*80 + "\n")

    return 0


if __name__ == "__main__":
    exit(main())
