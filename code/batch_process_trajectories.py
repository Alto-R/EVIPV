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

    # 组合使用
    python batch_process_trajectories.py --config config.yaml --gpu 1
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
        'gpu_id': 0,                   # GPU编号 (0, 1, 2...), None=自动选择
        'batch_size': 10000,             # 批处理大小
        'mesh_grid_size': None,        # mesh网格大小(m), None=不细分
        'clone_to_all_months': True,   # 是否克隆到全年12个月
        'max_vehicles': None,          # 最大处理车辆数, None=不限制
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

    # 按月统计
    if 'month' in result_df.columns:
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

        # 按月汇总
        monthly_totals = {}
        for vehicle_id, data in all_stats.items():
            if 'monthly_energy_kwh' in data['stats']:
                for month, energy in data['stats']['monthly_energy_kwh'].items():
                    monthly_totals[month] = monthly_totals.get(month, 0) + energy

        if monthly_totals:
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

            # 显示每月发电量
            if 'monthly_energy_kwh' in stats:
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
        use_gpu=config['computation']['use_gpu'],
        batch_size=config['computation']['batch_size']
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

    # 限制最大车辆数
    max_vehicles = config['computation'].get('max_vehicles', None)
    if max_vehicles and len(traj_files) > max_vehicles:
        print(f"⚠️  Found {len(traj_files)} files, limiting to first {max_vehicles}")
        traj_files = traj_files[:max_vehicles]

    print(f"✅ Found {len(traj_files)} processed trajectory files:")
    for f in traj_files:
        vehicle_id = f.stem.replace('_processed', '')
        print(f"  - {f.name} → Vehicle ID: {vehicle_id}")

    # 批量处理轨迹
    print("\n" + "="*80)
    print("Step 2: Process Trajectories (Full Year)")
    print("="*80)

    all_stats = {}

    for idx, traj_file in enumerate(traj_files, 1):
        vehicle_id = traj_file.stem.replace('_processed', '')

        print(f"\n{'='*80}")
        print(f"Processing Vehicle {idx}/{len(traj_files)}: {vehicle_id}")
        print('='*80)

        try:
            # 读取轨迹
            print(f"\n📂 Loading trajectory: {traj_file.name}", flush=True)

            trajectory_df = pd.read_csv(traj_file)
            trajectory_df['datetime'] = pd.to_datetime(trajectory_df['datetime'])

            # 确保时间戳有时区信息（与气象数据匹配）
            if trajectory_df['datetime'].dt.tz is None:
                trajectory_df['datetime'] = trajectory_df['datetime'].dt.tz_localize('Asia/Shanghai')

            print(f"   Records: {len(trajectory_df):,}", flush=True)

            # 检测源月份
            source_month = detect_source_month(trajectory_df)
            print(f"   Source Month: {source_month}", flush=True)

            # 确定要处理的月份
            if clone_to_all_months:
                months_to_process = list(range(1, 13))
                print(f"   Months to Process: 1-12 (Full Year)", flush=True)
            else:
                months_to_process = [source_month]
                print(f"   Months to Process: {source_month} only", flush=True)

            # 获取年份用于气象数据
            base_year = trajectory_df['datetime'].dt.year.mode()[0]

            # 存储所有月份的结果
            all_monthly_results = []
            vehicle_start_time = time.time()

            for target_month in months_to_process:
                print(f"\n📅 Processing Month {target_month:02d}/12...", flush=True)

                # 克隆轨迹到目标月份
                if target_month == source_month:
                    month_traj_df = trajectory_df.copy()
                    dropped_rows = 0
                else:
                    month_traj_df, dropped_rows = clone_trajectory_to_month(
                        trajectory_df, target_month
                    )

                if dropped_rows > 0:
                    print(f"   ⚠️  Dropped {dropped_rows} rows (invalid dates)", flush=True)

                if len(month_traj_df) == 0:
                    print(f"   ⚠️  No valid records for month {target_month}, skipping", flush=True)
                    continue

                # 推断日期范围
                start_date = month_traj_df['datetime'].min().strftime('%Y-%m-%d')
                end_date = month_traj_df['datetime'].max().strftime('%Y-%m-%d')

                # 获取气象数据
                print(f"   ☀️  Fetching irradiance data: {start_date} to {end_date}", flush=True)

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

                # GPU计算
                print(f"   ⚡ Calculating PV generation...", flush=True)
                calc_start = time.time()

                result_df = calculator.process_trajectory(
                    month_traj_df,
                    weather_data=weather_data
                )

                # 添加月份列
                result_df['month'] = target_month

                all_monthly_results.append(result_df)

                month_energy = result_df['energy_kwh'].sum()
                print(f"   ✅ Month {target_month:02d}: {month_energy:.2f} kWh ({time.time() - calc_start:.1f}s)", flush=True)

                # 清理中间变量
                del month_traj_df, irradiance_data, weather_data, result_df
                gc.collect()

            # 合并所有月份结果
            if all_monthly_results:
                combined_result = pd.concat(all_monthly_results, ignore_index=True)

                elapsed_time = time.time() - vehicle_start_time

                # 保存结果
                print(f"\n💾 Saving Results...", flush=True)
                result_csv = output_dir / f"{vehicle_id}_pv_generation.csv"
                combined_result.to_csv(result_csv, index=False)
                file_size_mb = result_csv.stat().st_size / (1024 * 1024)
                print(f"   ✅ Saved: {result_csv}", flush=True)
                print(f"      Size: {file_size_mb:.2f} MB, Records: {len(combined_result):,}", flush=True)

                # 收集统计
                stats = calculate_stats(combined_result)
                all_stats[vehicle_id] = {
                    'stats': stats,
                    'elapsed_time': elapsed_time
                }

                print(f"\n📊 Vehicle Summary:", flush=True)
                print(f"   Total Energy (Full Year): {stats['total_energy_kwh']:.2f} kWh", flush=True)
                print(f"   Avg Power: {stats['avg_power_w']:.2f} W", flush=True)
                print(f"   Peak Power: {stats['max_power_w']:.2f} W", flush=True)
                print(f"   Calculation Time: {elapsed_time:.1f}s", flush=True)

                # 清理
                del combined_result, all_monthly_results
            else:
                print(f"\n⚠️  No results generated for {vehicle_id}", flush=True)

            # 清理内存
            del trajectory_df
            gc.collect()

            if config['computation']['use_gpu']:
                try:
                    import torch
                    torch.cuda.empty_cache()
                except:
                    pass

        except Exception as e:
            print(f"\n❌ Error processing {vehicle_id}: {e}", flush=True)
            import traceback
            traceback.print_exc()

            # 清理内存
            gc.collect()
            if config['computation']['use_gpu']:
                try:
                    import torch
                    torch.cuda.empty_cache()
                except:
                    pass

            continue

    # 保存批处理汇总
    print("\n" + "="*80)
    print("Step 3: Generate Summary")
    print("="*80)

    if all_stats:
        batch_summary_path = output_dir / "batch_summary.txt"
        save_batch_summary(all_stats, batch_summary_path)
        print(f"✅ Batch Summary: {batch_summary_path}")

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
