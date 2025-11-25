"""
公交车GPS数据预处理脚本

功能：
1. 读取公交车原始GPS数据（F.csv格式）
2. 转换为出租车标准格式（与preprocess_trajectories.py输出一致）
3. 使后续PV计算流程可以无缝处理公交车数据

数据转换：
    公交车原始格式（10列）：
        fdate, ftime, busline_name, vehicle_id, lng, lat, speed, angle, operation_status, company_code

    转换为出租车标准格式（7列）：
        datetime, vehicle_id, lng, lat, speed, angle, operation_status

关键转换步骤：
1. fdate + ftime → datetime（ftime需补齐为6位，如 51923 → 051923）
2. 删除 busline_name 和 company_code 列
3. 重新排序列以匹配出租车格式
4. 按车牌号分别保存为CSV（包含header），文件名：车牌号_processed.csv
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import sys


# ============================================================================
# 配置参数
# ============================================================================
CONFIG = {
    'input': {
        'data_dir': '../traj',           # 公交车数据目录
        'file_pattern': '*.csv',         # 文件匹配模式
    },
    'output': {
        'output_dir': '../traj/bus_processed',  # 输出目录
        'suffix': '_processed',          # 输出文件后缀
    },
    'validation': {
        'lon_range': (113.5, 114.8),    # 深圳市经度范围
        'lat_range': (22.4, 22.9),      # 深圳市纬度范围
        'speed_range': (0, 120),        # 速度范围 (km/h)
        'angle_range': (0, 360),        # 方向角范围 (度)
    },
    'verbose': True,                     # 是否显示详细信息
}


def process_bus_file(input_path, output_dir, config):
    """
    处理公交车GPS数据文件，按车牌号分别保存

    Parameters
    ----------
    input_path : Path
        输入的公交车CSV文件路径
    output_dir : Path
        输出目录
    config : dict
        配置字典

    Returns
    -------
    list
        每辆车的处理统计信息列表
    """
    verbose = config.get('verbose', True)
    suffix = config['output']['suffix']

    try:
        if verbose:
            print(f"  Reading: {input_path.name}")

        # 读取公交车数据（无header，10列）
        df = pd.read_csv(
            input_path,
            header=None,
            names=['fdate', 'ftime', 'busline_name', 'vehicle_id', 'lng', 'lat',
                   'speed', 'angle', 'operation_status', 'company_code']
        )

        original_count = len(df)
        unique_vehicles = df['vehicle_id'].nunique()

        if verbose:
            print(f"    Original records: {original_count:,}")
            print(f"    Unique vehicles: {unique_vehicles}")
            print(f"    Unique bus lines: {df['busline_name'].nunique()}")

        # 合并日期和时间为datetime
        # fdate: 20190301
        # ftime: 51923 → 需补齐为 051923 (表示 05:19:23)
        df['datetime'] = (
            df['fdate'].astype(str) +
            df['ftime'].astype(str).str.zfill(6)
        )

        # 验证datetime格式（应为14位：YYYYMMDDHHmmss）
        df['datetime_len'] = df['datetime'].str.len()
        invalid_datetime = df[df['datetime_len'] != 14]
        if len(invalid_datetime) > 0:
            if verbose:
                print(f"    Warning: {len(invalid_datetime)} records with invalid datetime format")
            # 过滤掉格式错误的记录
            df = df[df['datetime_len'] == 14].copy()

        # 删除临时列
        df = df.drop(columns=['datetime_len'])

        # 选择标准列并重新排序（匹配出租车格式）
        standard_df = df[[
            'datetime',
            'vehicle_id',
            'lng',
            'lat',
            'speed',
            'angle',
            'operation_status'
        ]].copy()

        # 数据验证
        validation = config.get('validation', {})

        # 1. 检查坐标范围
        lon_min, lon_max = validation.get('lon_range', (113.5, 114.8))
        lat_min, lat_max = validation.get('lat_range', (22.4, 22.9))
        valid_lng = (standard_df['lng'] >= lon_min) & (standard_df['lng'] <= lon_max)
        valid_lat = (standard_df['lat'] >= lat_min) & (standard_df['lat'] <= lat_max)
        valid_coords = valid_lng & valid_lat

        if (~valid_coords).sum() > 0:
            if verbose:
                print(f"    Warning: {(~valid_coords).sum()} records with out-of-bounds coordinates")
            standard_df = standard_df[valid_coords].copy()

        # 2. 检查速度范围
        speed_min, speed_max = validation.get('speed_range', (0, 120))
        valid_speed = (standard_df['speed'] >= speed_min) & (standard_df['speed'] <= speed_max)
        if (~valid_speed).sum() > 0:
            if verbose:
                print(f"    Warning: {(~valid_speed).sum()} records with invalid speed")
            standard_df = standard_df[valid_speed].copy()

        # 3. 检查方向角范围
        angle_min, angle_max = validation.get('angle_range', (0, 360))
        valid_angle = (standard_df['angle'] >= angle_min) & (standard_df['angle'] < angle_max)
        if (~valid_angle).sum() > 0:
            if verbose:
                print(f"    Warning: {(~valid_angle).sum()} records with invalid angle")
            standard_df = standard_df[valid_angle].copy()

        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)

        # 按车牌号分组并分别保存
        if verbose:
            print(f"    Splitting by vehicle_id...")

        results = []
        for vehicle_id, vehicle_df in standard_df.groupby('vehicle_id'):
            # 生成输出文件名：车牌号_processed.csv
            output_file = output_dir / f"{vehicle_id}{suffix}.csv"

            # 保存为CSV（包含header和vehicle_id列，与出租车格式一致）
            vehicle_df.to_csv(output_file, index=False)

            results.append({
                'input_file': str(input_path),
                'output_file': str(output_file),
                'vehicle_id': vehicle_id,
                'record_count': len(vehicle_df),
                'success': True
            })

            if verbose:
                print(f"      {vehicle_id}: {len(vehicle_df):,} records → {output_file.name}")

        if verbose:
            total_records = sum(r['record_count'] for r in results)
            print(f"    Total final records: {total_records:,} ({total_records/original_count*100:.1f}%)")
            print(f"    Saved {len(results)} vehicle file(s)\n")

        return results

    except Exception as e:
        if verbose:
            print(f"    Error: {e}\n")
        return [{
            'input_file': str(input_path),
            'output_file': '',
            'success': False,
            'error': str(e)
        }]


def main():
    """主函数"""
    config = CONFIG

    print("\n" + "="*80)
    print("  公交车GPS数据格式转换工具")
    print("  Bus GPS Data → Taxi Standard Format Converter")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 获取配置
    input_dir = Path(config['input']['data_dir'])
    output_dir = Path(config['output']['output_dir'])
    file_pattern = config['input']['file_pattern']

    # 检查输入目录
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return 1

    # 查找所有匹配的CSV文件
    csv_files = sorted(input_dir.glob(file_pattern))

    if len(csv_files) == 0:
        print(f"Error: No files found matching pattern '{file_pattern}' in {input_dir}")
        return 1

    print(f"Found {len(csv_files)} file(s) to process")
    print(f"Output directory: {output_dir}\n")
    print("="*80 + "\n")

    # 处理所有文件
    all_results = []

    for i, csv_file in enumerate(csv_files, 1):
        print(f"[{i}/{len(csv_files)}] Processing: {csv_file.name}")

        # 按车牌号分别保存
        file_results = process_bus_file(csv_file, output_dir, config)
        all_results.extend(file_results)

    # 打印汇总信息
    successful = [r for r in all_results if r.get('success', False)]
    failed = [r for r in all_results if not r.get('success', False)]

    print("="*80)
    print("  Processing Summary")
    print("="*80)
    print(f"  Total vehicles processed: {len(all_results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")

    if successful:
        total_records = sum(r['record_count'] for r in successful)
        unique_vehicles = len(set(r['vehicle_id'] for r in successful))

        print(f"\n  Total records saved: {total_records:,}")
        print(f"  Unique vehicles: {unique_vehicles}")

    if failed:
        print(f"\n  Failed vehicles:")
        for r in failed:
            print(f"    - {Path(r['input_file']).name}: {r.get('error', 'Unknown error')}")

    print("\n" + "="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")

    return 0 if len(failed) == 0 else 1


if __name__ == '__main__':
    sys.exit(main())