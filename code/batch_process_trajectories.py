"""
批量处理轨迹数据 - 单GPU串行计算

功能：
1. 自动发现轨迹目录中的预处理轨迹文件
2. 自动识别源月份，克隆轨迹到全年12个月
3. 为每个月份获取对应的气象数据
4. 计算全年12个月的光伏发电量
5. 保存合并的全年结果到单个文件
6. 生成批处理汇总报告

使用示例：
    # 使用内部CONFIG配置
    python batch_process_trajectories.py

    # 使用外部config.yaml（可选）
    python batch_process_trajectories.py --config config.yaml

    # 指定使用GPU 0
    python batch_process_trajectories.py --gpu 0

    # 计算车辆范围 [101:200]（1-based索引）
    python batch_process_trajectories.py --vehicle-range 101:200

    # 计算从第501辆到末尾
    python batch_process_trajectories.py --vehicle-range 501:

    # 计算前100辆（等同于 --vehicle-range 1:100）
    python batch_process_trajectories.py --vehicle-range :100

    # 组合使用
    python batch_process_trajectories.py --config config.yaml --gpu 1 --vehicle-range 1:50
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import trimesh
import yaml
import argparse
from datetime import datetime
import time
import gc  # 垃圾回收

print("\n📦 正在导入模块...", flush=True)

# 导入自定义模块
print("   - 导入 prepare_building_mesh_from_footprint...", flush=True)
from prepare_building_mesh_from_footprint import prepare_building_mesh_from_footprint

print("   - 导入 fetch_irradiance_data...", flush=True)
from fetch_irradiance_data import fetch_and_cache_irradiance_data, convert_to_pvlib_format

print("   - 导入 pv_calculator_gpu...", flush=True)
from pv_calculator_gpu import GPUAcceleratedSolarPVCalculator

print("   ✅ 所有模块导入完成！\n", flush=True)


# ============================================================================
# 辅助函数
# ============================================================================
def detect_source_month(df, datetime_column='datetime'):
    """
    自动检测轨迹数据的源月份

    Parameters
    ----------
    df : pandas.DataFrame
        轨迹数据
    datetime_column : str
        日期时间列名

    Returns
    -------
    int
        源月份 (1-12)
    """
    month_counts = df[datetime_column].dt.month.value_counts()
    source_month = month_counts.idxmax()
    return int(source_month)


def clone_trajectory_to_month(df, target_month, datetime_column='datetime'):
    """
    将轨迹时间戳克隆到目标月份

    Parameters
    ----------
    df : pandas.DataFrame
        原始轨迹数据
    target_month : int
        目标月份 (1-12)
    datetime_column : str
        日期时间列名

    Returns
    -------
    pandas.DataFrame
        时间戳已转换的轨迹数据（过滤掉无效日期）
    """
    original_dt = df[datetime_column]

    # 保存原始时区信息
    original_tz = original_dt.dt.tz

    # 使用向量化操作构建新日期
    try:
        new_dates = pd.to_datetime({
            'year': original_dt.dt.year,
            'month': target_month,
            'day': original_dt.dt.day,
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

    # 恢复时区信息
    if original_tz is not None:
        new_dates = new_dates.dt.tz_localize(original_tz)

    # 找出有效日期的行
    valid_mask = new_dates.notna()

    # 复制有效行并更新时间戳
    df_cloned = df[valid_mask].copy()
    df_cloned[datetime_column] = new_dates[valid_mask]

    return df_cloned, (~valid_mask).sum()


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
        'trajectory_dir': '../../../../data2/hcr/evipv/shenzhendata/taxi/taxi/processed',  # 轨迹文件目录
    },
    'pv_system': {
        'panel_area': 2.2,          # 光伏板面积(m²)
        'panel_efficiency': 0.20,   # 效率 20%
        'tilt': 0,                  # 倾角(度)
        'vehicle_height': 1.5,      # 车顶高度(m)
    },
    'computation': {
        'time_resolution_minutes': 1,  # 时间分辨率
        'use_gpu': True,               # 启用GPU
        'gpu_id': 1,                   # GPU编号 (0, 1, 2...), None=自动选择
        'mesh_grid_size': None,        # mesh网格大小(m), None=不细分
        'clone_to_all_months': True,   # 是否克隆到全年12个月
        'max_vehicles': 1000,          # 最大处理车辆数, None=不限制（若使用vehicle_range将忽略此参数）
        'vehicle_range': None,         # 车辆索引区间（1-based），格式: "起始:结束" 或 [起始, 结束]
                                       # 示例: "101:200" 表示处理第101到200辆车
                                       # "501:" 表示从第501辆到末尾, ":100" 表示前100辆
                                       # None 表示使用 max_vehicles 参数
        'vehicles_per_batch': 200,     # 每批GPU同时处理的车辆数（充分利用显存）
    },
    'output': {
        'mesh_path': 'data/shenzhen_building_mesh.ply',
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


def parse_vehicle_range(range_value):
    """
    解析车辆区间配置，支持字符串 "a:b" 或长度为2的列表/元组
    返回 (start, end)，1-based索引，end为None表示到末尾
    """
    if range_value is None:
        return None

    start, end = None, None

    if isinstance(range_value, str):
        if ':' not in range_value:
            raise ValueError("vehicle_range 字符串格式应为 'a:b'")
        start_str, end_str = range_value.split(':', 1)
        start = int(start_str) if start_str.strip() else None
        end = int(end_str) if end_str.strip() else None
    elif isinstance(range_value, (list, tuple)):
        if len(range_value) != 2:
            raise ValueError("vehicle_range 列表/元组长度必须为2，例如 [1, 100]")
        start = int(range_value[0]) if range_value[0] is not None else None
        end = int(range_value[1]) if range_value[1] is not None else None
    else:
        raise ValueError("vehicle_range 仅支持字符串 'a:b' 或长度为2的列表/元组")

    if start is not None and start < 1:
        raise ValueError("vehicle_range 起始索引必须>=1")
    if end is not None and end < 1:
        raise ValueError("vehicle_range 结束索引必须>=1")

    if end is not None and start is not None and end < start:
        raise ValueError("vehicle_range 结束索引必须大于等于起始索引")

    return start, end


def calculate_stats(result_df):
    """
    计算轨迹的统计信息（支持全年数据）

    Parameters
    ----------
    result_df : pandas.DataFrame
        计算结果

    Returns
    -------
    dict
        统计信息
    """
    stats = {
        'total_records': len(result_df),
        'total_energy_kwh': result_df['energy_kwh'].sum(),
        'avg_power_w': result_df['ac_power'].mean(),
        'max_power_w': result_df['ac_power'].max(),
        'shaded_ratio': result_df['is_shaded'].mean(),
        'avg_cell_temp': result_df['cell_temp'].mean(),
        'time_range': (result_df['datetime'].min(), result_df['datetime'].max()),
    }

    # 按月统计（全年模式下month列必然存在）
    monthly_energy = result_df.groupby('month')['energy_kwh'].sum().to_dict()
    stats['monthly_energy_kwh'] = monthly_energy

    return stats


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
        f.write("Batch Processing Summary - All Vehicles (Full Year)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Processing Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Vehicles: {len(all_stats)}\n\n")

        total_energy = sum(s['stats']['total_energy_kwh'] for s in all_stats.values())
        total_time = sum(s['elapsed_time'] for s in all_stats.values())

        f.write("Overall Summary:\n")
        f.write(f"  Total Energy (All Vehicles, Full Year): {total_energy:.2f} kWh\n")
        f.write(f"  Total Calculation Time: {total_time:.1f} seconds ({total_time/60:.1f} min)\n\n")

        # 按月汇总（全年模式下monthly_energy_kwh必然存在）
        monthly_totals = {}
        for vehicle_id, data in all_stats.items():
            for month, energy in data['stats']['monthly_energy_kwh'].items():
                monthly_totals[month] = monthly_totals.get(month, 0) + energy

        f.write("Monthly Energy Summary (All Vehicles):\n")
        for month in sorted(monthly_totals.keys()):
            f.write(f"  Month {month:02d}: {monthly_totals[month]:.2f} kWh\n")
        f.write("\n")

        f.write("Per-Vehicle Statistics:\n")
        f.write("-"*60 + "\n")

        for vehicle_id, data in all_stats.items():
            stats = data['stats']
            f.write(f"\n{vehicle_id}:\n")
            f.write(f"  Records: {stats['total_records']:,}\n")
            f.write(f"  Total Energy (Full Year): {stats['total_energy_kwh']:.2f} kWh\n")
            f.write(f"  Avg Power: {stats['avg_power_w']:.2f} W\n")
            f.write(f"  Peak Power: {stats['max_power_w']:.2f} W\n")
            f.write(f"  Shaded Ratio: {stats['shaded_ratio']*100:.1f}%\n")
            f.write(f"  Calculation Time: {data['elapsed_time']:.1f}s\n")

            # 显示每月发电量（全年模式下必然存在）
            f.write(f"  Monthly Breakdown:\n")
            for month in sorted(stats['monthly_energy_kwh'].keys()):
                f.write(f"    Month {month:02d}: {stats['monthly_energy_kwh'][month]:.2f} kWh\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='批量处理轨迹数据计算光伏发电量（单GPU串行，全年版本）'
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
    parser.add_argument(
        '--vehicle-range', '-r',
        type=str,
        default=None,
        help="车辆索引区间，格式 'a:b'（1-based，b可省略表示到末尾，示例：--vehicle-range 101:200）"
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print(" "*15 + "🚀 Batch Vehicle PV Generation Calculation (GPU)")
    print(" "*20 + "Full Year Mode - 12 Months")
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

    # 全年模式提示
    clone_to_all_months = config['computation'].get('clone_to_all_months', True)
    if clone_to_all_months:
        print(f"📅 全年模式: 启用 (将克隆轨迹到12个月)")
    else:
        print(f"📅 全年模式: 禁用 (仅计算原始月份)")

    # GPU可用性检查
    if config['computation']['use_gpu']:
        try:
            import torch
            print(f"\n🔍 GPU可用性检查:")
            print(f"   PyTorch版本: {torch.__version__}")
            print(f"   CUDA可用: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"   CUDA版本: {torch.version.cuda}")
                print(f"   可用GPU数量: {torch.cuda.device_count()}")
                print(f"   当前GPU: {torch.cuda.current_device()}")
                print(f"   GPU名称: {torch.cuda.get_device_name(0)}")
                print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

                # 测试GPU是否真的可用
                print(f"\n🧪 GPU性能测试...")
                test_start = time.time()
                test_tensor = torch.randn(1000, 1000).cuda()
                test_result = torch.matmul(test_tensor, test_tensor)
                torch.cuda.synchronize()
                test_time = time.time() - test_start
                print(f"   ✅ GPU测试成功! 耗时: {test_time:.3f}s")
            else:
                print(f"   ⚠️  警告: CUDA不可用，将使用CPU计算（会很慢）")
        except Exception as e:
            print(f"   ❌ GPU检查失败: {e}")

    print("="*80)

    # 设置输入输出路径
    traj_dir = Path(config['data_sources']['trajectory_dir'])
    output_dir = Path(config['output']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 轨迹目录: {traj_dir}")
    print(f"📁 输出目录: {output_dir}")
    print("="*80)

    # 准备建筑mesh
    print("\n" + "="*80)
    print("Preparing Building Mesh")
    print("="*80)

    mesh_start_time = time.time()
    mesh_path = Path(config['output']['mesh_path'])

    if mesh_path.exists():
        print(f"⏱️  [{datetime.now().strftime('%H:%M:%S')}] Loading existing mesh: {mesh_path}")
        building_mesh = trimesh.load(mesh_path)
        print(f"   Vertices: {len(building_mesh.vertices):,}")
        print(f"   Faces: {len(building_mesh.faces):,}")
        print(f"   ✅ Mesh加载完成，耗时: {time.time() - mesh_start_time:.2f}s")
    else:
        print(f"⏱️  [{datetime.now().strftime('%H:%M:%S')}] Converting footprint to mesh...")
        print(f"   ⚠️  这是首次运行，生成mesh可能需要较长时间（数分钟）...")
        building_mesh = prepare_building_mesh_from_footprint(
            footprint_path=config['data_sources']['footprint_path'],
            output_mesh_path=str(mesh_path),
            grid_size=config['computation']['mesh_grid_size']
        )
        print(f"   ✅ Mesh生成完成，耗时: {time.time() - mesh_start_time:.2f}s")

    # 初始化GPU计算器
    print("\n" + "="*80, flush=True)
    print("Initialize GPU Calculator", flush=True)
    print("="*80, flush=True)

    calc_init_start = time.time()
    print(f"⏱️  [{datetime.now().strftime('%H:%M:%S')}] 正在初始化GPU计算器...", flush=True)
    print(f"   这一步可能需要10-30秒，请耐心等待...", flush=True)

    calculator = GPUAcceleratedSolarPVCalculator(
        lon_center=config['location']['lon'],
        lat_center=config['location']['lat'],
        building_mesh=building_mesh,
        panel_area=config['pv_system']['panel_area'],
        panel_efficiency=config['pv_system']['panel_efficiency'],
        time_resolution_minutes=config['computation']['time_resolution_minutes'],
        use_gpu=config['computation']['use_gpu']
    )

    print(f"   ✅ GPU计算器初始化完成，耗时: {time.time() - calc_init_start:.2f}s", flush=True)

    # 查找轨迹文件
    print("\n" + "="*80)
    print("Step 1: Discover Trajectory Files")
    print("="*80)

    traj_files = find_processed_trajectories(traj_dir)

    if not traj_files:
        print(f"⚠️  No processed trajectory files found in {traj_dir}/")
        print(f"   请确保轨迹文件以 '_processed.csv' 结尾")
        return 1

    print(f"✅ Found {len(traj_files)} processed trajectory files in total")

    # 车辆筛选：优先使用 vehicle_range，否则使用 max_vehicles
    # 命令行参数 > 配置文件
    vehicle_range_arg = args.vehicle_range if args.vehicle_range else config['computation'].get('vehicle_range', None)

    if vehicle_range_arg:
        # 使用范围筛选
        try:
            start_idx, end_idx = parse_vehicle_range(vehicle_range_arg)

            # 保存原始文件总数（用于显示）
            total_files = len(traj_files)

            # 转换为0-based索引
            start_0based = (start_idx - 1) if start_idx else 0
            end_0based = end_idx if end_idx else total_files

            # 验证范围有效性
            if start_0based < 0:
                start_0based = 0
            if end_0based > total_files:
                end_0based = total_files
            if start_0based >= total_files:
                print(f"❌ 错误: 起始索引 {start_idx} 超出范围（共 {total_files} 个文件）")
                return 1

            traj_files = traj_files[start_0based:end_0based]
            print(f"📌 使用车辆范围: [{start_idx if start_idx else 1}:{end_idx if end_idx else total_files}]")
            print(f"   选择了 {len(traj_files)} 个车辆（索引 {start_0based+1} 到 {end_0based}）")

        except ValueError as e:
            print(f"❌ 错误: vehicle_range 参数格式错误 - {e}")
            return 1
    else:
        # 使用 max_vehicles 限制
        max_vehicles = config['computation'].get('max_vehicles', None)
        if max_vehicles and len(traj_files) > max_vehicles:
            print(f"⚠️  限制为前 {max_vehicles} 个车辆")
            traj_files = traj_files[:max_vehicles]
        else:
            print(f"📌 处理所有 {len(traj_files)} 个车辆")

    # 批量处理轨迹
    print("\n" + "="*80)
    print("Step 2: Process Trajectories (Full Year - Batch Mode)")
    print("="*80)
          
    all_stats = {}
    vehicles_per_batch = config['computation'].get('vehicles_per_batch', 1)

    print(f"\n⚡ Batch Configuration:")
    print(f"   Total vehicles: {len(traj_files)}")
    print(f"   Vehicles per batch: {vehicles_per_batch}")
    print(f"   Total batches: {(len(traj_files) + vehicles_per_batch - 1) // vehicles_per_batch}")
    print(f"   Expected GPU memory saving: ~{vehicles_per_batch}x speedup\n")

    # 分批处理车辆
    for batch_idx in range(0, len(traj_files), vehicles_per_batch):
        batch_files = traj_files[batch_idx:batch_idx + vehicles_per_batch]
        batch_num = batch_idx // vehicles_per_batch + 1
        total_batches = (len(traj_files) + vehicles_per_batch - 1) // vehicles_per_batch

        print(f"\n{'='*80}")
        print(f"📦 Processing Batch {batch_num}/{total_batches} ({len(batch_files)} vehicles)")
        print('='*80)

        batch_start_time = time.time()

        # 存储批次中所有车辆的数据
        batch_trajectories = {}  # {vehicle_id: full_year_traj}

        # 1️⃣ 读取并准备批次中所有车辆的轨迹
        for idx, traj_file in enumerate(batch_files, 1):
            vehicle_id = traj_file.stem.replace('_processed', '')

            print(f"\n--- Vehicle {batch_idx + idx}/{len(traj_files)}: {vehicle_id} ---")

            try:
                # 读取轨迹
                print(f"📂 Loading trajectory: {traj_file.name}", flush=True)
                trajectory_df = pd.read_csv(traj_file)
                trajectory_df['datetime'] = pd.to_datetime(trajectory_df['datetime'])

                # 确保时区统一为 Asia/Shanghai
                if trajectory_df['datetime'].dt.tz is None:
                    trajectory_df['datetime'] = trajectory_df['datetime'].dt.tz_localize('Asia/Shanghai')
                else:
                    trajectory_df['datetime'] = trajectory_df['datetime'].dt.tz_convert('Asia/Shanghai')

                print(f"   Records: {len(trajectory_df):,}", flush=True)

                # 检测源月份
                source_month = detect_source_month(trajectory_df)
                print(f"   Source Month: {source_month}", flush=True)

                # 确定要处理的月份
                clone_to_all_months = config['computation'].get('clone_to_all_months', True)
                if clone_to_all_months:
                    months_to_process = list(range(1, 13))
                else:
                    months_to_process = [source_month]

                # 克隆轨迹到所有月份
                print(f"   Cloning to {len(months_to_process)} months...", flush=True)
                all_monthly_trajs = []
                total_dropped = 0

                for target_month in months_to_process:
                    if target_month == source_month:
                        month_traj_df = trajectory_df.copy()
                        dropped_rows = 0
                    else:
                        month_traj_df, dropped_rows = clone_trajectory_to_month(
                            trajectory_df, target_month
                        )
                        total_dropped += dropped_rows

                    if len(month_traj_df) > 0:
                        month_traj_df['month'] = target_month
                        all_monthly_trajs.append(month_traj_df)

                if total_dropped > 0:
                    print(f"   ⚠️  Dropped {total_dropped} invalid dates", flush=True)

                if not all_monthly_trajs:
                    print(f"   ⚠️  No valid data, skipping", flush=True)
                    continue

                # 合并全年数据
                full_year_traj = pd.concat(all_monthly_trajs, ignore_index=True)

                # 🔄 重要：在合并前先重采样每个车辆的轨迹
                print(f"   🔄 Resampling trajectory ({len(full_year_traj):,} → resampled)...", flush=True)
                resampled_traj = calculator.resample_trajectory(full_year_traj)

                # 🔧 修复：resample_trajectory 现在返回 DatetimeIndex，需要重置为列以便后续操作
                resampled_traj.reset_index(inplace=True)

                # 添加车辆ID和月份标识
                resampled_traj['vehicle_id'] = vehicle_id
                # 直接从datetime提取月份
                resampled_traj['month'] = resampled_traj['datetime'].dt.month

                batch_trajectories[vehicle_id] = resampled_traj

                print(f"   ✅ Prepared: {len(resampled_traj):,} records (resampled)", flush=True)

                # 清理
                del trajectory_df, all_monthly_trajs

            except Exception as e:
                print(f"   ❌ Error preparing {vehicle_id}: {e}", flush=True)
                continue

        if not batch_trajectories:
            print(f"\n⚠️  No valid vehicles in this batch, skipping")
            continue

        # 2️⃣ 合并批次中所有车辆的轨迹
        print(f"\n🔗 Merging {len(batch_trajectories)} vehicles for batch GPU processing...")
        merged_batch_traj = pd.concat(batch_trajectories.values(), ignore_index=True)
        print(f"   Total records (all vehicles): {len(merged_batch_traj):,}")

        # 推断日期范围（全批次）
        start_date = merged_batch_traj['datetime'].min().strftime('%Y-%m-%d')
        end_date = merged_batch_traj['datetime'].max().strftime('%Y-%m-%d')
        print(f"   Date range: {start_date} to {end_date}")

        # 3️⃣ 获取全年气象数据（批次共享）
        print(f"\n☀️  Fetching full-year irradiance data...", flush=True)
        irrad_start = time.time()
        irradiance_data = fetch_and_cache_irradiance_data(
            lat=config['location']['lat'],
            lon=config['location']['lon'],
            start_date=start_date,
            end_date=end_date,
            granularity='1min' if config['computation']['time_resolution_minutes'] == 1 else '1hour',
            save_csv=False,
            output_dir='irradiance_data'
        )
        weather_data = convert_to_pvlib_format(irradiance_data)
        print(f"   ✅ Weather data ready ({time.time() - irrad_start:.1f}s)", flush=True)

        # 4️⃣ 一次性GPU计算整个批次
        print(f"\n⚡ GPU Batch Calculation ({len(batch_trajectories)} vehicles simultaneously)...", flush=True)
        print(f"   合并数据大小: {len(merged_batch_traj):,} 行", flush=True)
        print(f"   内存估算: {merged_batch_traj.memory_usage(deep=True).sum() / 1024**2:.1f} MB", flush=True)
        calc_start = time.time()

        print(f"   开始调用 calculator.process_trajectory()...", flush=True)
        batch_result_df = calculator.process_trajectory(
            merged_batch_traj,
            weather_data=weather_data,
            skip_resample=True  # 已在外层对每个车辆单独重采样
        )
        print(f"   ✅ process_trajectory 返回成功", flush=True)

        calc_time = time.time() - calc_start
        print(f"   ✅ Batch GPU calculation complete ({calc_time:.1f}s)", flush=True)
        print(f"   Average time per vehicle: {calc_time/len(batch_trajectories):.1f}s", flush=True)

        # 5️⃣ 验证vehicle_id和month信息已在结果中
        # （skip_resample=True 应该保证这些列存在）
        assert 'vehicle_id' in batch_result_df.columns, "vehicle_id列丢失，请检查process_trajectory逻辑"
        assert 'month' in batch_result_df.columns, "month列丢失，请检查process_trajectory逻辑"
        print(f"   ✅ vehicle_id和month信息已保留在结果中")

        # 6️⃣ 拆分结果并保存每个车辆（使用groupby优化）
        print(f"\n💾 Splitting and saving results...")
        for vehicle_id, vehicle_result in batch_result_df.groupby('vehicle_id'):
            # 移除临时列
            vehicle_result = vehicle_result.drop(columns=['vehicle_id'])

            # 显示月度统计（全年模式下month列必然存在）
            month_stats = []
            for month in sorted(vehicle_result['month'].unique()):
                month_energy = vehicle_result[vehicle_result['month'] == month]['energy_kwh'].sum()
                month_stats.append(f"{month:02d}:{month_energy:.1f}kWh")
            print(f"   {vehicle_id}: {', '.join(month_stats)}", flush=True)

            # 保存结果
            result_csv = output_dir / f"{vehicle_id}_pv_generation.csv"
            vehicle_result.to_csv(result_csv, index=False)
            file_size_mb = result_csv.stat().st_size / (1024 * 1024)

            # 收集统计
            stats = calculate_stats(vehicle_result)
            all_stats[vehicle_id] = {
                'stats': stats,
                'elapsed_time': calc_time / len(batch_trajectories)  # 均摊时间
            }

            print(f"   ✅ {vehicle_id}: {file_size_mb:.1f}MB, {len(vehicle_result):,} records, {stats['total_energy_kwh']:.1f}kWh", flush=True)

        # 清理批次数据
        batch_elapsed = time.time() - batch_start_time
        print(f"\n✅ Batch {batch_num} complete: {batch_elapsed:.1f}s ({batch_elapsed/len(batch_trajectories):.1f}s per vehicle)")

        del merged_batch_traj, batch_result_df, batch_trajectories, irradiance_data, weather_data
        gc.collect()

        if config['computation']['use_gpu']:
            try:
                import torch
                torch.cuda.empty_cache()
            except:
                pass


    # 保存批处理汇总
    print("\n" + "="*80)
    print("Step 3: Generate Summary")
    print("="*80)

    if all_stats:
        # batch_summary_path = output_dir / "batch_summary.txt"
        # save_batch_summary(all_stats, batch_summary_path)
        # print(f"✅ Batch Summary: {batch_summary_path}")

        print(f"\n📊 Processing Summary:")
        print(f"   Successfully Processed: {len(all_stats)} vehicles")
        total_energy = sum(s['stats']['total_energy_kwh'] for s in all_stats.values())
        total_time = sum(s['elapsed_time'] for s in all_stats.values())
        print(f"   Total Energy (All Vehicles, Full Year): {total_energy:.2f} kWh")
        print(f"   Total Calculation Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    else:
        print(f"❌ No vehicles processed successfully")

    # 完成所有处理
    print("\n\n" + "="*80)
    print("="*80)
    print("✅ All Processing Complete!")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print("="*80 + "\n")

    return 0


if __name__ == "__main__":
    exit(main())
