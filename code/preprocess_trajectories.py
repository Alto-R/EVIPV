"""
轨迹数据预处理脚本

功能：
1. 读取原始轨迹CSV文件（无表头格式）
2. 添加标准列名
3. 解析datetime格式
4. 提取车辆ID（英文标识）
5. 保存为标准CSV格式

使用方法：
    在脚本内部修改CONFIG配置后直接运行：
    python preprocess_trajectories.py
"""

import pandas as pd
from pathlib import Path
from datetime import datetime


# ==================== 配置参数 ====================
CONFIG = {
    # 处理模式：'single' 处理单个文件，'batch' 批量处理目录下所有文件
    'mode': 'batch',

    # 单文件模式：指定输入文件路径
    'input_file': 'traj/Z8.csv',

    # 批量模式：指定输入目录
    'input_dir': 'traj',

    # 输出目录
    'output_dir': 'traj'
}
# =================================================


def extract_vehicle_id(raw_id):
    """
    从原始车牌提取英文车辆ID

    Parameters
    ----------
    raw_id : str
        原始车牌号，如 "粤B7J7Z8"

    Returns
    -------
    str
        英文车辆ID，如 "Z8"

    Examples
    --------
    >>> extract_vehicle_id("粤B7J7Z8")
    'Z8'
    >>> extract_vehicle_id("粤B2L90G")
    '0G'
    """
    # 提取最后两位作为车辆ID
    return raw_id[-2:]


def preprocess_trajectory(input_path, output_dir='traj'):
    """
    预处理单个轨迹文件

    Parameters
    ----------
    input_path : str or Path
        输入CSV文件路径
    output_dir : str
        输出目录

    Returns
    -------
    tuple
        (vehicle_id, output_path, stats)
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

    # 提取车辆ID
    raw_vehicle_id = df['vehicle_id'].iloc[0]
    vehicle_id = extract_vehicle_id(raw_vehicle_id)
    print(f"🚗 车辆ID: {raw_vehicle_id} → {vehicle_id}")

    # 数据验证
    print("✅ 数据验证...")

    # 检查坐标范围（深圳区域）
    invalid_coords = (
        (df['lng'] < 113) | (df['lng'] > 115) |
        (df['lat'] < 22) | (df['lat'] > 23.5)
    )
    if invalid_coords.sum() > 0:
        print(f"   ⚠️  发现 {invalid_coords.sum()} 条坐标异常记录")
        df = df[~invalid_coords]

    # 检查角度范围
    invalid_angle = (df['angle'] < 0) | (df['angle'] > 359)
    if invalid_angle.sum() > 0:
        print(f"   ⚠️  发现 {invalid_angle.sum()} 条角度异常记录")
        df = df[~invalid_angle]

    # 移除重复时间戳
    duplicates = df.duplicated(subset=['datetime'], keep='first')
    if duplicates.sum() > 0:
        print(f"   ⚠️  移除 {duplicates.sum()} 条重复时间戳")
        df = df[~duplicates]

    # 统计信息
    stats = {
        'vehicle_id': vehicle_id,
        'raw_vehicle_id': raw_vehicle_id,
        'total_records': len(df),
        'time_range': (df['datetime'].min(), df['datetime'].max()),
        'duration_hours': (df['datetime'].max() - df['datetime'].min()).total_seconds() / 3600,
        'avg_speed': df['speed'].mean(),
        'coord_bounds': {
            'lng_min': df['lng'].min(),
            'lng_max': df['lng'].max(),
            'lat_min': df['lat'].min(),
            'lat_max': df['lat'].max()
        }
    }

    print(f"\n📊 统计信息:")
    print(f"   有效记录数: {stats['total_records']:,}")
    print(f"   时间范围: {stats['time_range'][0]} 至 {stats['time_range'][1]}")
    print(f"   持续时间: {stats['duration_hours']:.2f} 小时")
    print(f"   平均速度: {stats['avg_speed']:.1f} km/h")
    print(f"   经度范围: {stats['coord_bounds']['lng_min']:.4f} ~ {stats['coord_bounds']['lng_max']:.4f}")
    print(f"   纬度范围: {stats['coord_bounds']['lat_min']:.4f} ~ {stats['coord_bounds']['lat_max']:.4f}")

    # 保存处理后的CSV
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = f"{vehicle_id}_processed.csv"
    output_path = output_dir / output_filename

    df.to_csv(output_path, index=False)
    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    print(f"\n💾 保存到: {output_path}")
    print(f"   文件大小: {file_size_mb:.2f} MB")
    print('='*60)

    return vehicle_id, output_path, stats


def preprocess_all_trajectories(input_dir='traj', output_dir='traj'):
    """
    批量预处理所有轨迹文件

    Parameters
    ----------
    input_dir : str
        输入目录
    output_dir : str
        输出目录

    Returns
    -------
    dict
        所有车辆的统计信息
    """
    input_dir = Path(input_dir)

    # 查找所有原始CSV文件（排除已处理的）
    csv_files = list(input_dir.glob('*.csv'))
    csv_files = [f for f in csv_files if '_processed' not in f.name]

    if not csv_files:
        print("未找到待处理的CSV文件")
        return {}

    print(f"\n{'='*60}")
    print(f"批量预处理轨迹数据")
    print('='*60)
    print(f"发现 {len(csv_files)} 个待处理文件:")
    for f in csv_files:
        print(f"  - {f.name}")

    all_stats = {}

    for csv_file in csv_files:
        try:
            vehicle_id, output_path, stats = preprocess_trajectory(
                csv_file, output_dir
            )
            all_stats[vehicle_id] = stats
        except Exception as e:
            print(f"\n❌ 处理 {csv_file.name} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 汇总报告
    print(f"\n{'='*60}")
    print("预处理汇总")
    print('='*60)
    print(f"成功处理: {len(all_stats)} 个轨迹文件")

    total_records = sum(s['total_records'] for s in all_stats.values())
    total_duration = sum(s['duration_hours'] for s in all_stats.values())

    print(f"总记录数: {total_records:,}")
    print(f"总时长: {total_duration:.2f} 小时")
    print()

    for vid, stats in all_stats.items():
        print(f"  {vid}: {stats['total_records']:,} 条记录, "
              f"{stats['duration_hours']:.2f}h, "
              f"平均速度 {stats['avg_speed']:.1f} km/h")

    print('='*60)

    return all_stats


def main():
    """
    主函数

    根据CONFIG配置运行预处理任务
    """
    print("\n" + "="*60)
    print("轨迹数据预处理")
    print("="*60)
    print(f"处理模式: {CONFIG['mode']}")

    if CONFIG['mode'] == 'single':
        print(f"输入文件: {CONFIG['input_file']}")
    else:
        print(f"输入目录: {CONFIG['input_dir']}")

    print(f"输出目录: {CONFIG['output_dir']}")
    print("="*60)

    try:
        if CONFIG['mode'] == 'single':
            # 处理单个文件
            vehicle_id, output_path, stats = preprocess_trajectory(
                CONFIG['input_file'],
                CONFIG['output_dir']
            )
            print(f"\n✅ 预处理完成: {output_path}")
        elif CONFIG['mode'] == 'batch':
            # 批量处理
            all_stats = preprocess_all_trajectories(
                CONFIG['input_dir'],
                CONFIG['output_dir']
            )
            print(f"\n✅ 批量预处理完成")
        else:
            raise ValueError(f"无效的处理模式: {CONFIG['mode']}，请使用 'single' 或 'batch'")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
