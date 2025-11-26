"""
轨迹数据预处理脚本（批量模式）

功能：
1. 批量读取原始轨迹CSV文件（无表头格式）
2. 添加标准列名，解析datetime格式
3. 自动检测并分离多车辆文件（按车牌号拆分）
4. 使用完整车牌号作为车辆ID
5. 使用 transbigdata 进行数据清洗：
   - 清理边界外数据（深圳区域）
   - 清理冗余重复记录
   - 清理漂移异常点（速度/距离/角度）
6. 支持并行处理多个文件
7. 保存为标准CSV格式（每个车辆一个文件）

使用方法：
    在脚本内部修改CONFIG配置后直接运行：
    python preprocess_trajectories.py

配置说明：
    - input_dir: 输入目录
    - output_dir: 输出目录
    - parallel: True/False 是否启用并行处理
    - n_workers: 并行工作进程数（None=自动检测CPU核心数）
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import transbigdata as tbd
from concurrent.futures import ProcessPoolExecutor, as_completed
import os


# ==================== 配置参数 ====================
CONFIG = {
    # 输入目录
    'input_dir': '../traj',

    # 输出目录
    'output_dir': '../traj',

    # 并行处理配置
    'parallel': True,           # 是否启用并行处理
    'n_workers': 10             # 并行工作进程数（None = 自动使用 CPU 核心数）
}
# =================================================


def preprocess_trajectory(input_path, output_dir='traj'):
    """
    预处理单个轨迹文件，自动检测并分离多车辆数据

    Parameters
    ----------
    input_path : str or Path
        输入CSV文件路径
    output_dir : str
        输出目录

    Returns
    -------
    list of str
        处理成功的车辆ID列表
    """
    input_path = Path(input_path)

    print(f"\n{'='*60}")
    print(f"预处理轨迹文件: {input_path.name}")
    print('='*60)

    # 读取原始CSV（无表头）
    print("📂 读取原始CSV...")
    df = pd.read_csv(input_path, header=None, names=[
        'datetime', 'vehicle_id', 'lng', 'lat',
        'speed', 'angle', 'operation_status'
    ])

    print(f"   原始记录数: {len(df):,}")

    # 解析datetime
    print("🕐 解析datetime格式...")
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%d%H%M%S')
    df['datetime'] = df['datetime'].dt.tz_localize('Asia/Shanghai')  # 标记为深圳/中国时区

    # 检查车辆ID并按车辆分组处理
    print("🚗 检查车辆ID...")
    unique_raw_ids = df['vehicle_id'].unique()

    if len(unique_raw_ids) > 1:
        print(f"⚠️  发现多个车辆ID: {len(unique_raw_ids)} 个，将分别处理")
    else:
        print(f"   车辆ID: {unique_raw_ids[0]}")

    # 统一处理：按车辆ID分组处理（无论单车辆还是多车辆）
    all_results = []
    for idx, raw_vehicle_id in enumerate(unique_raw_ids, 1):
        vehicle_id = str(raw_vehicle_id)

        if len(unique_raw_ids) > 1:
            print(f"\n   --- 处理车辆 {idx}/{len(unique_raw_ids)}: {vehicle_id} ---")

        # 过滤当前车辆的数据
        vehicle_df = df[df['vehicle_id'] == raw_vehicle_id].copy()

        if len(unique_raw_ids) > 1:
            print(f"   记录数: {len(vehicle_df):,}")

        # 数据清洗
        print("   ✅ 数据验证...")

        # 使用 transbigdata 清理边界外数据（深圳区域）
        records_before = len(vehicle_df)
        vehicle_df = tbd.clean_outofbounds(
            vehicle_df,
            bounds=[113, 22, 115, 23],
            col=['lng', 'lat']
        )
        removed_coords = records_before - len(vehicle_df)
        if removed_coords > 0:
            print(f"      ⚠️  [transbigdata] 移除 {removed_coords} 条边界外记录")

        # 检查角度范围
        invalid_angle = (vehicle_df['angle'] < 0) | (vehicle_df['angle'] > 359)
        if invalid_angle.sum() > 0:
            print(f"      ⚠️  发现 {invalid_angle.sum()} 条角度异常记录")
            vehicle_df = vehicle_df[~invalid_angle]

        # 使用 transbigdata 清理重复记录
        records_before = len(vehicle_df)
        vehicle_df = tbd.traj_clean_redundant(
            vehicle_df,
            col=['vehicle_id', 'datetime', 'lng', 'lat']
        )
        removed_duplicates = records_before - len(vehicle_df)
        if removed_duplicates > 0:
            print(f"      ⚠️  [transbigdata] 移除 {removed_duplicates} 条冗余记录")

        # 使用 transbigdata 清理漂移异常点
        records_before = len(vehicle_df)
        vehicle_df = tbd.traj_clean_drift(
            vehicle_df,
            col=['vehicle_id', 'datetime', 'lng', 'lat'],
            speedlimit=100,
            dislimit=1000,
            anglelimit=30
        )
        removed_drift = records_before - len(vehicle_df)
        if removed_drift > 0:
            print(f"      ⚠️  [transbigdata] 移除 {removed_drift} 条漂移异常点")

        # 统计信息
        stats = {
            'vehicle_id': vehicle_id,
            'raw_vehicle_id': raw_vehicle_id,
            'total_records': len(vehicle_df),
            'time_range': (vehicle_df['datetime'].min(), vehicle_df['datetime'].max()),
            'duration_hours': (vehicle_df['datetime'].max() - vehicle_df['datetime'].min()).total_seconds() / 3600,
            'avg_speed': vehicle_df['speed'].mean(),
            'coord_bounds': {
                'lng_min': vehicle_df['lng'].min(),
                'lng_max': vehicle_df['lng'].max(),
                'lat_min': vehicle_df['lat'].min(),
                'lat_max': vehicle_df['lat'].max()
            }
        }

        print(f"\n   📊 统计信息:")
        print(f"      有效记录数: {stats['total_records']:,}")
        print(f"      时间范围: {stats['time_range'][0]} 至 {stats['time_range'][1]}")
        print(f"      持续时间: {stats['duration_hours']:.2f} 小时")
        print(f"      平均速度: {stats['avg_speed']:.1f} km/h")
        print(f"      经度范围: {stats['coord_bounds']['lng_min']:.4f} ~ {stats['coord_bounds']['lng_max']:.4f}")
        print(f"      纬度范围: {stats['coord_bounds']['lat_min']:.4f} ~ {stats['coord_bounds']['lat_max']:.4f}")

        # 保存处理后的CSV
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        output_filename = f"{vehicle_id}_processed.csv"
        output_path = output_dir_path / output_filename

        vehicle_df.to_csv(output_path, index=False)
        file_size_mb = output_path.stat().st_size / (1024 * 1024)

        print(f"\n   💾 保存到: {output_path}")
        print(f"      文件大小: {file_size_mb:.2f} MB")

        if len(unique_raw_ids) > 1:
            print('   ' + '='*60)

        all_results.append((vehicle_id, output_path, stats))

    if len(unique_raw_ids) > 1:
        print(f"\n✅ 多车辆文件拆分完成: {len(all_results)} 个车辆")

    print('='*60)

    # 返回所有处理成功的车辆ID列表
    processed_vehicle_ids = [vid for vid, _, _ in all_results]
    print(f"\n✅ 处理完成，生成车辆: {', '.join(processed_vehicle_ids)}")

    return processed_vehicle_ids


def _process_single_file(args):
    """
    单个文件处理的包装函数（用于并行处理）

    Parameters
    ----------
    args : tuple
        (csv_file, output_dir)

    Returns
    -------
    list of str or None
        成功时返回车辆ID列表，失败时返回 None
    """
    csv_file, output_dir = args
    try:
        vehicle_ids = preprocess_trajectory(csv_file, output_dir)
        return vehicle_ids
    except Exception as e:
        print(f"\n❌ 处理 {csv_file.name} 失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def preprocess_all_trajectories(input_dir='traj', output_dir='traj', parallel=False, n_workers=None):
    """
    批量预处理所有轨迹文件

    Parameters
    ----------
    input_dir : str
        输入目录
    output_dir : str
        输出目录
    parallel : bool
        是否启用并行处理
    n_workers : int or None
        并行工作进程数（None = 自动使用 CPU 核心数）

    Returns
    -------
    list of str
        所有处理成功的车辆ID列表
    """
    input_dir = Path(input_dir)

    # 查找所有原始CSV文件（排除已处理的）
    csv_files = list(input_dir.glob('*.csv'))
    csv_files = [f for f in csv_files if '_processed' not in f.name]

    if not csv_files:
        print("未找到待处理的CSV文件")
        return []

    print(f"\n{'='*60}")
    print(f"批量预处理轨迹数据")
    print('='*60)
    print(f"发现 {len(csv_files)} 个待处理文件:")
    for f in csv_files:
        print(f"  - {f.name}")

    all_vehicle_ids = []

    if parallel and len(csv_files) > 1:
        # 并行处理模式
        if n_workers is None:
            n_workers = os.cpu_count()
        print(f"\n🚀 启用并行处理模式（{n_workers} 个工作进程）")

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # 提交所有任务
            future_to_file = {
                executor.submit(_process_single_file, (csv_file, output_dir)): csv_file
                for csv_file in csv_files
            }

            # 收集结果
            completed = 0
            for future in as_completed(future_to_file):
                completed += 1
                result = future.result()
                if result is not None:
                    all_vehicle_ids.extend(result)  # 添加所有车辆ID
                print(f"   进度: {completed}/{len(csv_files)}")
    else:
        # 串行处理模式
        if parallel:
            print(f"\n⚠️  文件数量少于2个，使用串行处理")
        else:
            print(f"\n📝 使用串行处理模式")

        for csv_file in csv_files:
            try:
                vehicle_ids = preprocess_trajectory(csv_file, output_dir)
                all_vehicle_ids.extend(vehicle_ids)  # 添加所有车辆ID
            except Exception as e:
                print(f"\n❌ 处理 {csv_file.name} 失败: {e}")
                import traceback
                traceback.print_exc()
                continue

    # 汇总报告
    print(f"\n{'='*60}")
    print("预处理汇总")
    print('='*60)
    print(f"成功处理: {len(all_vehicle_ids)} 个车辆")
    print(f"车辆列表:")
    for vid in all_vehicle_ids:
        print(f"  - {vid}")
    print('='*60)

    return all_vehicle_ids


def main():
    """
    主函数 - 批量预处理模式
    """
    print("\n" + "="*60)
    print("轨迹数据预处理（批量模式）")
    print("="*60)
    print(f"输入目录: {CONFIG['input_dir']}")
    print(f"输出目录: {CONFIG['output_dir']}")
    print(f"并行处理: {'启用' if CONFIG['parallel'] else '禁用'}")
    if CONFIG['parallel']:
        workers = CONFIG['n_workers'] or os.cpu_count()
        print(f"工作进程数: {workers}")
    print("="*60)

    try:
        all_vehicle_ids = preprocess_all_trajectories(
            CONFIG['input_dir'],
            CONFIG['output_dir'],
            parallel=CONFIG['parallel'],
            n_workers=CONFIG['n_workers']
        )
        print(f"\n✅ 批量预处理完成，共生成 {len(all_vehicle_ids)} 个车辆文件")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
