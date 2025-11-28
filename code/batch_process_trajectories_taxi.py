"""
批量处理轨迹数据 - 生产者-消费者架构（多进程并行）

架构设计：
┌─────────────────────────────────────────────────────────────┐
│  主进程 (Coordinator)                                        │
│                                                             │
│  ┌──────────────────┐    ┌──────────────┐    ┌───────────┐  │
│  │ 数据准备进程池     │───>│ GPU计算进程   │───>│保存进程池    │  │
│  │ (n个CPU workers) │    │ (单GPU批量)   │    │(n个workers)│  │
│  └──────────────────┘    └──────────────┘    └───────────┘  │
│         生产者               消费者/生产者         消费者       │
└─────────────────────────────────────────────────────────────┘

优势：
1. 并行读取CSV + 数据准备（充分利用多核CPU）
2. GPU批量计算（充分利用48GB显存）
3. 并行保存结果（I/O不阻塞）
4. 流水线处理，减少等待时间
5. ⚡ 优化：准备进程不加载mesh（节省80%内存和启动时间）
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
import gc
from multiprocessing import Process, Queue, Manager
import queue
import traceback

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
    """自动检测轨迹数据的源月份"""
    month_counts = df[datetime_column].dt.month.value_counts()
    source_month = month_counts.idxmax()
    return int(source_month)


def clone_trajectory_to_month(df, target_month, datetime_column='datetime'):
    """将轨迹时间戳克隆到目标月份"""
    original_dt = df[datetime_column]
    original_tz = original_dt.dt.tz

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
        def replace_month(dt):
            try:
                return dt.replace(month=target_month)
            except ValueError:
                return pd.NaT
        new_dates = original_dt.apply(replace_month)

    if original_tz is not None:
        new_dates = new_dates.dt.tz_localize(original_tz)

    valid_mask = new_dates.notna()
    df_cloned = df[valid_mask].copy()
    df_cloned[datetime_column] = new_dates[valid_mask]

    return df_cloned, (~valid_mask).sum()


def calculate_stats(result_df):
    """计算轨迹的统计信息"""
    stats = {
        'total_records': len(result_df),
        'total_energy_kwh': result_df['energy_kwh'].sum(),
        'avg_power_w': result_df['ac_power'].mean(),
        'max_power_w': result_df['ac_power'].max(),
        'shaded_ratio': result_df['is_shaded'].mean(),
        'avg_cell_temp': result_df['cell_temp'].mean(),
        'time_range': (result_df['datetime'].min(), result_df['datetime'].max()),
    }

    monthly_energy = result_df.groupby('month')['energy_kwh'].sum().to_dict()
    stats['monthly_energy_kwh'] = monthly_energy

    return stats


def parse_vehicle_range(range_value):
    """解析车辆区间配置"""
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
            raise ValueError("vehicle_range 列表/元组长度必须为2")
        start = int(range_value[0]) if range_value[0] is not None else None
        end = int(range_value[1]) if range_value[1] is not None else None
    else:
        raise ValueError("vehicle_range 仅支持字符串或列表/元组")

    if start is not None and start < 1:
        raise ValueError("vehicle_range 起始索引必须>=1")
    if end is not None and end < 1:
        raise ValueError("vehicle_range 结束索引必须>=1")
    if end is not None and start is not None and end < start:
        raise ValueError("vehicle_range 结束索引必须大于等于起始索引")

    return start, end


def interpolate_parking_periods(df, threshold_minutes=2, resolution_minutes=1):
    """
    在GPS间隔过大时（停车期间）插入插值点以计算发电量

    背景：
    - 车辆停车时GPS可能不记录新点，但太阳能板仍在发电
    - 太阳位置随时间变化，遮挡情况也会变化
    - 需要插值以准确计算每个时间段的发电功率

    参数
    ----
    df : pd.DataFrame
        轨迹数据（必须已按时间排序，包含datetime列）
    threshold_minutes : float
        GPS间隔超过此值（分钟）认为是停车，需要插值
    resolution_minutes : float
        插值时间步长（分钟）

    返回
    ----
    pd.DataFrame
        插值后的轨迹数据

    示例
    ----
    原始数据：10:00 → 10:30（停车30分钟）
    插值后：10:00, 10:01, 10:02, ..., 10:29, 10:30
    """
    if len(df) < 2:
        return df

    # 找出需要插值的间隔
    threshold_seconds = threshold_minutes * 60
    resolution_seconds = resolution_minutes * 60

    interpolated_rows = []

    for i in range(len(df)):
        # 添加当前点
        interpolated_rows.append(df.iloc[i])

        # 检查是否需要插值（不是最后一个点，且间隔超过阈值）
        if i < len(df) - 1:
            time_gap = (df.iloc[i + 1]['datetime'] - df.iloc[i]['datetime']).total_seconds()

            if time_gap > threshold_seconds:
                # 计算需要插入的点数
                num_points = int(time_gap / resolution_seconds) - 1

                if num_points > 0:
                    # 创建插值点（保持位置、速度、角度不变）
                    base_row = df.iloc[i].copy()

                    for j in range(1, num_points + 1):
                        new_row = base_row.copy()
                        # 更新时间戳
                        new_row['datetime'] = base_row['datetime'] + pd.Timedelta(seconds=j * resolution_seconds)
                        interpolated_rows.append(new_row)

    # 合并所有行
    result_df = pd.DataFrame(interpolated_rows).reset_index(drop=True)

    return result_df


# ============================================================================
# Worker Functions - 生产者-消费者模式
# ============================================================================

def prepare_worker(task_queue, data_queue, config, mesh_path, worker_id):
    """
    数据准备Worker（生产者）

    职责：
    1. 从task_queue获取轨迹文件路径
    2. 读取CSV、重采样、克隆到12个月
    3. 将准备好的数据放入data_queue

    Parameters
    ----------
    task_queue : Queue
        任务队列（文件路径）
    data_queue : Queue
        数据队列（准备好的轨迹数据）
    config : dict
        配置字典
    mesh_path : str
        建筑mesh路径（此worker不使用，仅保留接口兼容性）
    worker_id : int
        Worker编号（用于日志）
    """
    try:
        # ✅ 优化：准备阶段不需要加载mesh（节省80%内存和初始化时间）
        # ✅ 优化2：不重采样，直接使用原始GPS点计算实际时间间隔

        while True:
            try:
                # 获取任务（阻塞等待）
                traj_file = task_queue.get(timeout=1)

                # 毒丸信号：结束worker
                if traj_file is None:
                    print(f"[准备Worker-{worker_id}] 收到结束信号，退出", flush=True)
                    break

                vehicle_id = traj_file.stem.replace('_processed', '')

                # 读取轨迹
                trajectory_df = pd.read_csv(traj_file)
                trajectory_df['datetime'] = pd.to_datetime(trajectory_df['datetime'])

                # 确保时区统一
                if trajectory_df['datetime'].dt.tz is None:
                    trajectory_df['datetime'] = trajectory_df['datetime'].dt.tz_localize('Asia/Shanghai')
                else:
                    trajectory_df['datetime'] = trajectory_df['datetime'].dt.tz_convert('Asia/Shanghai')

                # 检测源月份并过滤
                source_month = detect_source_month(trajectory_df)
                trajectory_df = trajectory_df[trajectory_df['datetime'].dt.month == source_month].copy()

                # ✅ FIX: 清理NaN角度值（停车点常缺失角度）
                # 原因：停车时speed=0，GPS不计算heading angle
                # 解决：前向填充最后已知方向，后向填充起始点，最后用默认值0°
                trajectory_df['angle'] = trajectory_df['angle'].ffill().bfill()

                # 如果整列都是NaN（极少见），使用默认北向
                if trajectory_df['angle'].isna().any():
                    print(f"[准备Worker-{worker_id}]    {vehicle_id}: ⚠️  缺少角度数据，使用默认值0°", flush=True)
                    trajectory_df['angle'] = trajectory_df['angle'].fillna(0.0)

                # ✅ 不重采样，直接使用原始GPS点
                # ✅ 停车期间插值（计算停车时的发电量）
                # 按时间排序
                trajectory_df = trajectory_df.sort_values('datetime').reset_index(drop=True)

                # 停车期间插值：GPS间隔>阈值时，插入额外点以计算发电量
                parking_threshold = config['computation'].get('parking_threshold_minutes', 2)
                interpolation_resolution = config['computation'].get('interpolation_resolution_minutes', 1)

                original_count = len(trajectory_df)
                trajectory_df = interpolate_parking_periods(
                    trajectory_df,
                    threshold_minutes=parking_threshold,
                    resolution_minutes=interpolation_resolution
                )
                interpolated_count = len(trajectory_df) - original_count

                if interpolated_count > 0:
                    print(f"[准备Worker-{worker_id}]    {vehicle_id}: 插值 {interpolated_count} 个停车点", flush=True)

                # ✅ 验证：检查角度清理是否成功
                if trajectory_df['angle'].isna().any():
                    nan_count = trajectory_df['angle'].isna().sum()
                    print(f"[准备Worker-{worker_id}]    {vehicle_id}: ⚠️  仍有{nan_count}个NaN角度", flush=True)

                # 计算实际时间间隔（到下一个点的时间）
                # ⚠️ 关键：能量计算为 E = P(t) × Δt，其中Δt是从t到t+1的时间
                trajectory_df['delta_t_seconds'] = (
                    trajectory_df['datetime'].shift(-1) - trajectory_df['datetime']
                ).dt.total_seconds()

                # 处理最后一个点：使用前面点的中位数间隔（稳健估计）
                median_interval = trajectory_df['delta_t_seconds'].median()
                trajectory_df.loc[len(trajectory_df)-1, 'delta_t_seconds'] = median_interval

                # 克隆到12个月
                clone_to_all_months = config['computation'].get('clone_to_all_months', True)
                months_to_process = list(range(1, 13)) if clone_to_all_months else [source_month]

                all_monthly_trajs = []
                for target_month in months_to_process:
                    if target_month == source_month:
                        month_traj_df = trajectory_df.copy()
                    else:
                        month_traj_df, _ = clone_trajectory_to_month(trajectory_df, target_month)

                    if len(month_traj_df) > 0:
                        month_traj_df['month'] = target_month
                        all_monthly_trajs.append(month_traj_df)

                if not all_monthly_trajs:
                    print(f"[准备Worker-{worker_id}] ⚠️  {vehicle_id}: 无有效数据", flush=True)
                    continue

                # 合并全年数据
                full_year_traj = pd.concat(all_monthly_trajs, ignore_index=True)

                # ✅ FIX: 排序保证时间单调递增（pandas reindex with ffill/bfill需要单调索引）
                full_year_traj = full_year_traj.sort_values('datetime').reset_index(drop=True)

                full_year_traj['vehicle_id'] = vehicle_id

                # 计算日期范围（用于获取气象数据）
                start_date = full_year_traj['datetime'].min().strftime('%Y-%m-%d')
                end_date = full_year_traj['datetime'].max().strftime('%Y-%m-%d')

                # 放入数据队列
                data_queue.put({
                    'vehicle_id': vehicle_id,
                    'trajectory': full_year_traj,
                    'start_date': start_date,
                    'end_date': end_date,
                })

                print(f"[准备Worker-{worker_id}] ✅ {vehicle_id}: {len(full_year_traj):,} records", flush=True)

                # 清理内存
                del trajectory_df, all_monthly_trajs, full_year_traj

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[准备Worker-{worker_id}] ❌ 处理失败: {e}", flush=True)
                traceback.print_exc()
                continue

    except Exception as e:
        print(f"[准备Worker-{worker_id}] ❌ Worker初始化失败: {e}", flush=True)
        traceback.print_exc()


def gpu_worker(data_queue, result_queue, config, mesh_path):
    """
    GPU计算Worker（消费者/生产者）

    职责：
    1. 从data_queue消费准备好的轨迹数据
    2. 累积到指定批次大小后，批量GPU计算
    3. 将计算结果放入result_queue

    Parameters
    ----------
    data_queue : Queue
        数据队列（准备好的轨迹）
    result_queue : Queue
        结果队列（计算完成的结果）
    config : dict
        配置字典
    mesh_path : str
        建筑mesh路径
    """
    try:
        # 在子进程中初始化GPU（避免fork问题）
        print(f"[GPU Worker] 正在初始化...", flush=True)

        building_mesh = trimesh.load(mesh_path)

        calculator = GPUAcceleratedSolarPVCalculator(
            lon_center=config['location']['lon'],
            lat_center=config['location']['lat'],
            building_mesh=building_mesh,
            panel_area=config['pv_system']['panel_area'],
            panel_efficiency=config['pv_system']['panel_efficiency'],
            time_resolution_minutes=config['computation']['time_resolution_minutes'],
            use_gpu=config['computation']['use_gpu']
        )

        print(f"[GPU Worker] ✅ 初始化完成", flush=True)

        # 批次累积
        batch_data = []
        gpu_batch_size = config['computation'].get('gpu_batch_size', 50)
        batch_timeout = config['computation'].get('batch_timeout', 10)

        # 气象数据缓存（全年数据可复用）
        weather_cache = {}

        while True:
            try:
                # 尝试获取数据（带超时）
                data = data_queue.get(timeout=batch_timeout)

                # 毒丸信号：处理剩余批次后退出
                if data is None:
                    print(f"[GPU Worker] 收到结束信号，处理剩余批次...", flush=True)
                    if batch_data:
                        process_batch(batch_data, calculator, weather_cache, result_queue, config)
                    break

                batch_data.append(data)

                # 达到批次大小，立即处理
                if len(batch_data) >= gpu_batch_size:
                    print(f"[GPU Worker] 批次已满({len(batch_data)}辆)，开始GPU计算...", flush=True)
                    process_batch(batch_data, calculator, weather_cache, result_queue, config)
                    batch_data = []

            except queue.Empty:
                # 超时：如果有积累的数据，也进行处理
                if batch_data:
                    print(f"[GPU Worker] 超时触发，处理当前批次({len(batch_data)}辆)...", flush=True)
                    process_batch(batch_data, calculator, weather_cache, result_queue, config)
                    batch_data = []
                continue

    except Exception as e:
        print(f"[GPU Worker] ❌ Worker失败: {e}", flush=True)
        traceback.print_exc()


def process_batch(batch_data, calculator, weather_cache, result_queue, config):
    """
    处理一个批次的车辆数据

    Parameters
    ----------
    batch_data : list
        批次数据列表
    calculator : GPUAcceleratedSolarPVCalculator
        GPU计算器
    weather_cache : dict
        气象数据缓存
    result_queue : Queue
        结果队列
    config : dict
        配置字典
    """
    try:
        calc_start = time.time()

        # 合并批次中所有车辆的轨迹
        merged_traj = pd.concat([d['trajectory'] for d in batch_data], ignore_index=True)

        # ✅ FIX: 排序保证datetime单调递增（reindex要求）
        # 说明：合并多车后时间会倒退（车A的12月 → 车B的1月）
        # 排序后：车辆数据会交错但各自时间顺序不变，最后用vehicle_id分组还原
        merged_traj = merged_traj.sort_values('datetime').reset_index(drop=True)

        # 推断日期范围（取所有车辆的最小/最大）
        all_start_dates = [pd.to_datetime(d['start_date']) for d in batch_data]
        all_end_dates = [pd.to_datetime(d['end_date']) for d in batch_data]
        start_date = min(all_start_dates).strftime('%Y-%m-%d')
        end_date = max(all_end_dates).strftime('%Y-%m-%d')

        # 获取气象数据（带缓存）
        cache_key = f"{start_date}_{end_date}"
        if cache_key not in weather_cache:
            irradiance_data = fetch_and_cache_irradiance_data(
                lat=config['location']['lat'],
                lon=config['location']['lon'],
                start_date=start_date,
                end_date=end_date,
                granularity='1min' if config['computation']['time_resolution_minutes'] == 1 else '1hour',
                save_csv=False,
                output_dir='irradiance_data'
            )
            weather_cache[cache_key] = convert_to_pvlib_format(irradiance_data)

        weather_data = weather_cache[cache_key]

        # 批量GPU计算
        batch_result = calculator.process_trajectory(
            merged_traj,
            weather_data=weather_data,
            skip_resample=True,  # 已在准备阶段重采样
            vehicle_height=config['pv_system']['vehicle_height']  # ✅ 传入车辆高度
        )

        calc_time = time.time() - calc_start

        # 拆分结果并放入结果队列
        for vehicle_id, vehicle_result in batch_result.groupby('vehicle_id'):
            vehicle_result = vehicle_result.drop(columns=['vehicle_id'])

            result_queue.put({
                'vehicle_id': vehicle_id,
                'result': vehicle_result,
                'calc_time': calc_time / len(batch_data)  # 均摊时间
            })

        print(f"[GPU Worker] ✅ 批次完成: {len(batch_data)}辆, {calc_time:.1f}s "
              f"(avg {calc_time/len(batch_data):.1f}s/vehicle)", flush=True)

        # 清理
        del merged_traj, batch_result
        gc.collect()

    except Exception as e:
        print(f"[GPU Worker] ❌ 批次处理失败: {e}", flush=True)
        traceback.print_exc()


def save_worker(result_queue, config, stats_dict, worker_id):
    """
    结果保存Worker（消费者）

    职责：
    1. 从result_queue获取计算结果
    2. 保存为CSV文件
    3. 收集统计信息

    Parameters
    ----------
    result_queue : Queue
        结果队列
    config : dict
        配置字典
    stats_dict : Manager.dict
        共享统计字典
    worker_id : int
        Worker编号
    """
    try:
        output_dir = Path(config['output']['output_dir'])

        while True:
            try:
                # 获取结果（阻塞等待）
                result_data = result_queue.get(timeout=1)

                # 毒丸信号：结束worker
                if result_data is None:
                    print(f"[保存Worker-{worker_id}] 收到结束信号，退出", flush=True)
                    break

                vehicle_id = result_data['vehicle_id']
                vehicle_result = result_data['result']
                calc_time = result_data['calc_time']

                # 保存CSV
                result_csv = output_dir / f"{vehicle_id}_pv_generation.csv"
                vehicle_result.to_csv(result_csv, index=False)

                # 计算统计
                stats = calculate_stats(vehicle_result)

                # 存入共享字典
                stats_dict[vehicle_id] = {
                    'stats': stats,
                    'elapsed_time': calc_time
                }

                # 显示月度统计
                month_stats = []
                for month in sorted(stats['monthly_energy_kwh'].keys()):
                    month_energy = stats['monthly_energy_kwh'][month]
                    month_stats.append(f"{month:02d}:{month_energy:.1f}kWh")

                print(f"[保存Worker-{worker_id}] ✅ {vehicle_id}: "
                      f"{stats['total_energy_kwh']:.1f}kWh ({', '.join(month_stats[:3])}...)",
                      flush=True)

                # 清理
                del vehicle_result

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[保存Worker-{worker_id}] ❌ 保存失败: {e}", flush=True)
                traceback.print_exc()
                continue

    except Exception as e:
        print(f"[保存Worker-{worker_id}] ❌ Worker失败: {e}", flush=True)
        traceback.print_exc()


# ============================================================================
# 配置参数
# ============================================================================
CONFIG = {
    'location': {
        'name': '深圳市',
        'lat': 22.543099,
        'lon': 114.057868,
    },
    'data_sources': {
        'footprint_path': 'data/shenzhen_buildings.geojson',
        'trajectory_dir': '../../../../data2/hcr/evipv/shenzhendata/taxi/taxi/processed',
    },
    'pv_system': {
        'panel_area': 2.2,
        'panel_efficiency': 0.20,
        'tilt': 0,
        'vehicle_height': 1.5,
    },
    'computation': {
        'time_resolution_minutes': 1,
        'use_gpu': True,
        'gpu_id': 1,
        'mesh_grid_size': None,
        'clone_to_all_months': True,  # ✅ 克隆到12个月以计算全年发电量
        'max_vehicles': 1050,
        'vehicle_range': None,

        # 停车期间插值参数
        'parking_threshold_minutes': 2,        # GPS间隔超过2分钟认为停车，需插值
        'interpolation_resolution_minutes': 1, # 停车期间每1分钟插值一个点

        # 生产者-消费者参数
        'num_prepare_workers': 5,    # 数据准备并行进程数
        'num_save_workers': 5,       # 结果保存并行进程数
        'gpu_batch_size': 100,       # GPU批量处理车辆数
        'queue_maxsize': 100,        # 队列最大长度
        'batch_timeout': 10,         # 批次超时(秒)
    },
    'output': {
        'mesh_path': 'data/shenzhen_building_mesh.ply',
        'output_dir': 'output',
    },
}

# ============================================================================
# 主函数
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='生产者-消费者架构批量处理轨迹数据（并行I/O + CPU + 批量GPU）'
    )
    parser.add_argument('--config', '-c', default=None)
    parser.add_argument('--gpu', '-g', type=int, default=None)
    parser.add_argument('--vehicle-range', '-r', type=str, default=None)
    parser.add_argument('--prepare-workers', '-p', type=int, default=None,
                        help='数据准备并行进程数（默认4）')
    parser.add_argument('--save-workers', '-s', type=int, default=None,
                        help='结果保存并行进程数（默认2）')
    parser.add_argument('--gpu-batch-size', '-b', type=int, default=None,
                        help='GPU批量处理车辆数（默认50）')

    args = parser.parse_args()

    print("\n" + "="*80)
    print(" "*8 + "🚀 Producer-Consumer Architecture PV Calculation")
    print(" "*15 + "Parallel I/O + CPU + Batch GPU")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 加载配置
    if args.config:
        if Path(args.config).exists():
            with open(args.config, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        else:
            print(f"⚠️  配置文件不存在，使用内部CONFIG")
            config = CONFIG
    else:
        config = CONFIG

    # GPU设置
    gpu_id = args.gpu if args.gpu is not None else config['computation'].get('gpu_id', 0)
    if config['computation']['use_gpu'] and gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        print(f"🎮 GPU: {gpu_id}")

    # 进程数设置
    num_prepare = args.prepare_workers or config['computation'].get('num_prepare_workers', 4)
    num_save = args.save_workers or config['computation'].get('num_save_workers', 2)
    gpu_batch = args.gpu_batch_size or config['computation'].get('gpu_batch_size', 50)

    print(f"⚙️  Architecture Configuration:")
    print(f"   Prepare Workers: {num_prepare} (parallel CPU)")
    print(f"   GPU Batch Size: {gpu_batch} vehicles")
    print(f"   Save Workers: {num_save} (parallel I/O)")

    # 准备输出目录
    output_dir = Path(config['output']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 准备建筑mesh
    print("\n" + "="*80)
    print("Preparing Building Mesh")
    print("="*80)

    mesh_path = Path(config['output']['mesh_path'])
    if mesh_path.exists():
        print(f"⏱️  Loading mesh: {mesh_path}")
        building_mesh = trimesh.load(mesh_path)
        print(f"   ✅ Mesh loaded")
    else:
        print(f"⏱️  Generating mesh...")
        building_mesh = prepare_building_mesh_from_footprint(
            footprint_path=config['data_sources']['footprint_path'],
            output_mesh_path=str(mesh_path),
            grid_size=config['computation']['mesh_grid_size']
        )
        print(f"   ✅ Mesh generated")

    # 查找轨迹文件
    traj_dir = Path(config['data_sources']['trajectory_dir'])
    traj_files = sorted(traj_dir.glob('*_processed.csv'))

    # 车辆筛选
    if args.vehicle_range:
        vehicle_range = parse_vehicle_range(args.vehicle_range)
        if vehicle_range:
            start_idx, end_idx = vehicle_range
            start_0based = (start_idx - 1) if start_idx else 0
            end_0based = end_idx if end_idx else len(traj_files)
            traj_files = traj_files[start_0based:end_0based]
            print(f"\n📌 Vehicle Range: [{start_idx if start_idx else 1}:{end_idx if end_idx else len(traj_files)}]")
    elif config['computation'].get('max_vehicles'):
        max_v = config['computation']['max_vehicles']
        traj_files = traj_files[:max_v]

    print(f"✅ Found {len(traj_files)} vehicles to process")

    # ========== 启动生产者-消费者架构 ==========
    print("\n" + "="*80)
    print(f"Starting Producer-Consumer Pipeline")
    print("="*80 + "\n")

    start_time = time.time()

    # 创建队列
    queue_maxsize = config['computation'].get('queue_maxsize', 100)
    task_queue = Queue()  # 任务队列（文件路径）
    data_queue = Queue(maxsize=queue_maxsize)  # 数据队列（准备好的轨迹）
    result_queue = Queue(maxsize=queue_maxsize)  # 结果队列（计算结果）

    # 创建共享统计字典
    manager = Manager()
    stats_dict = manager.dict()

    # 填充任务队列
    for traj_file in traj_files:
        task_queue.put(traj_file)

    # 添加毒丸信号（让prepare workers知道何时停止）
    for _ in range(num_prepare):
        task_queue.put(None)

    print(f"📋 Task queue filled: {len(traj_files)} vehicles")

    # 启动数据准备进程池（生产者）
    prepare_processes = []
    for i in range(num_prepare):
        p = Process(
            target=prepare_worker,
            args=(task_queue, data_queue, config, str(mesh_path), i)
        )
        p.start()
        prepare_processes.append(p)
    print(f"🏭 Started {num_prepare} prepare workers")

    # 启动GPU计算进程（消费者/生产者）
    gpu_process = Process(
        target=gpu_worker,
        args=(data_queue, result_queue, config, str(mesh_path))
    )
    gpu_process.start()
    print(f"⚡ Started GPU worker")

    # 启动结果保存进程池（消费者）
    save_processes = []
    for i in range(num_save):
        p = Process(
            target=save_worker,
            args=(result_queue, config, stats_dict, i)
        )
        p.start()
        save_processes.append(p)
    print(f"💾 Started {num_save} save workers\n")

    print("="*80)
    print("Pipeline Running... (Ctrl+C to stop)")
    print("="*80 + "\n")

    # 等待所有准备进程完成
    for p in prepare_processes:
        p.join()
    print(f"\n✅ All prepare workers finished")

    # 发送毒丸给GPU进程
    data_queue.put(None)
    gpu_process.join()
    print(f"✅ GPU worker finished")

    # 发送毒丸给保存进程
    for _ in range(num_save):
        result_queue.put(None)

    for p in save_processes:
        p.join()
    print(f"✅ All save workers finished")

    total_time = time.time() - start_time

    # ========== 输出汇总 ==========
    print("\n" + "="*80)
    print("Processing Summary")
    print("="*80)

    if stats_dict:
        all_stats = dict(stats_dict)
        total_energy = sum(s['stats']['total_energy_kwh'] for s in all_stats.values())

        print(f"\n✅ Successfully Processed: {len(all_stats)} vehicles")
        print(f"   Total Energy (Full Year): {total_energy:.2f} kWh")
        print(f"   Wall Time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"   Avg Time/Vehicle: {total_time/len(all_stats):.2f}s")

        # 按月汇总
        monthly_totals = {}
        for vehicle_id, data in all_stats.items():
            for month, energy in data['stats']['monthly_energy_kwh'].items():
                monthly_totals[month] = monthly_totals.get(month, 0) + energy

        print(f"\n📊 Monthly Energy Summary (All Vehicles):")
        for month in sorted(monthly_totals.keys()):
            print(f"   Month {month:02d}: {monthly_totals[month]:.2f} kWh")
    else:
        print("❌ No vehicles processed successfully")

    print("\n" + "="*80)
    print("✅ All Processing Complete!")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")

    return 0


if __name__ == "__main__":
    exit(main())
