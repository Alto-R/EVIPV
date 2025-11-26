"""
公交车GPS数据预处理脚本

功能：
1. 读取公交车原始GPS数据（F.csv格式）
2. 转换为出租车标准格式（与preprocess_taxi_trajectories.py输出一致）
3. 使后续PV计算流程可以无缝处理公交车数据

数据转换：
    公交车原始格式（10列）：
        fdate, ftime, busline_name, vehicle_id, lng, lat, speed, angle, operation_status, company_code

    转换为标准格式（8列，保留公交线路信息）：
        datetime, vehicle_id, lng, lat, speed, angle, operation_status, busline_name

关键转换步骤：
1. fdate + ftime → datetime（ftime需补齐为6位，如 51923 → 051923）
2. 转换为pandas datetime对象并添加 'Asia/Shanghai' 时区（与出租车一致）
3. 保留 busline_name 列（公交线路名称）
4. 删除 company_code 列
5. 重新排序列
6. 按车牌号分别保存为CSV（包含header），文件名：车牌号_processed.csv

输出格式示例：
    datetime,vehicle_id,lng,lat,speed,angle,operation_status,busline_name
    2019-03-01 05:19:23+08:00,粤B12345,114.12345,22.54321,35.0,120.5,1,M191
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
        'data_dir': '../../../../data2/hcr/evipv/shenzhendata/bus',           # 公交车数据目录
        'file_pattern': '*.csv',         # 文件匹配模式
    },
    'output': {
        'output_dir': '../../../../data2/hcr/evipv/shenzhendata/bus/processed',  # 输出目录
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

        # 合并日期和时间为datetime（与出租车格式一致）
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

        # 转换为pandas datetime对象并添加时区（与出租车处理一致）
        if verbose:
            print(f"    Converting datetime format...")
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%d%H%M%S')
        df['datetime'] = df['datetime'].dt.tz_localize('Asia/Shanghai')  # 标记为深圳/中国时区

        # 选择标准列并重新排序（保留公交线路信息）
        standard_df = df[[
            'datetime',
            'vehicle_id',
            'lng',
            'lat',
            'speed',
            'angle',
            'operation_status',
            'busline_name'
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

            # 保存为CSV（包含header和busline_name列）
            vehicle_df.to_csv(output_file, index=False)

            # 获取该车辆的公交线路（一辆车可能跑多条线路）
            # 过滤掉NaN值，避免后续处理出错
            buslines = [
                str(bl) for bl in vehicle_df['busline_name'].unique()
                if pd.notna(bl)
            ]

            results.append({
                'input_file': str(input_path),
                'output_file': str(output_file),
                'vehicle_id': vehicle_id,
                'record_count': len(vehicle_df),
                'buslines': buslines,  # 该车辆运行的公交线路列表
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


def generate_busline_summary(all_results, output_dir, config):
    """
    生成公交线路统计汇总CSV

    Parameters
    ----------
    all_results : list
        所有车辆的处理结果列表
    output_dir : Path
        输出目录
    config : dict
        配置字典

    Returns
    -------
    Path or None
        统计文件路径，如果失败则返回None
    """
    verbose = config.get('verbose', True)

    try:
        if verbose:
            print("\n" + "="*80)
            print("  Generating Bus Line Summary")
            print("="*80 + "\n")

        # 收集所有成功处理的结果
        successful = [r for r in all_results if r.get('success', False)]

        if not successful:
            if verbose:
                print("  No successful results to summarize\n")
            return None

        # 统计每条公交线路的信息
        busline_stats = {}

        for result in successful:
            vehicle_id = result['vehicle_id']
            record_count = result['record_count']
            buslines = result.get('buslines', [])

            for busline in buslines:
                # 跳过空值或NaN（额外保护，理论上已在前面过滤）
                if not busline or pd.isna(busline):
                    continue

                if busline not in busline_stats:
                    busline_stats[busline] = {
                        'busline_name': busline,
                        'vehicle_count': 0,
                        'vehicle_ids': set(),
                        'total_records': 0
                    }

                busline_stats[busline]['vehicle_ids'].add(vehicle_id)
                busline_stats[busline]['vehicle_count'] = len(busline_stats[busline]['vehicle_ids'])
                busline_stats[busline]['total_records'] += record_count

        # 转换为DataFrame
        summary_data = []
        for busline, stats in busline_stats.items():
            summary_data.append({
                'busline_name': stats['busline_name'],
                'vehicle_count': stats['vehicle_count'],
                'total_records': stats['total_records'],
                'avg_records_per_vehicle': stats['total_records'] / stats['vehicle_count']
            })

        summary_df = pd.DataFrame(summary_data)

        # 按公交线路名称排序
        summary_df = summary_df.sort_values('busline_name').reset_index(drop=True)

        # 保存统计文件
        summary_file = output_dir / 'busline_summary.csv'
        summary_df.to_csv(summary_file, index=False)

        if verbose:
            print(f"  Total unique bus lines: {len(summary_df)}")
            print(f"  Statistics saved to: {summary_file.name}")
            print(f"\n  Top 10 bus lines by vehicle count:")
            print("  " + "-"*76)

            top_10 = summary_df.nlargest(10, 'vehicle_count')
            for _, row in top_10.iterrows():
                print(f"    {row['busline_name']:15s} | "
                      f"Vehicles: {int(row['vehicle_count']):4d} | "
                      f"Records: {int(row['total_records']):10,d} | "
                      f"Avg: {int(row['avg_records_per_vehicle']):7,d}")

            print()

        return summary_file

    except Exception as e:
        if verbose:
            print(f"  Error generating summary: {e}\n")
        return None


def select_representative_trajectories(all_results, output_dir, config):
    """
    为每条公交线路选择一条代表性轨迹并保存到独立文件夹

    选择策略：对于每条线路，选择GPS记录数最多的车辆作为代表

    Parameters
    ----------
    all_results : list
        所有车辆的处理结果列表
    output_dir : Path
        输出目录
    config : dict
        配置字典

    Returns
    -------
    dict
        统计信息字典
    """
    verbose = config.get('verbose', True)

    try:
        if verbose:
            print("\n" + "="*80)
            print("  Selecting Representative Trajectories for Each Bus Line")
            print("="*80 + "\n")

        # 收集所有成功处理的结果
        successful = [r for r in all_results if r.get('success', False)]

        if not successful:
            if verbose:
                print("  No successful results to process\n")
            return None

        # 为每条公交线路收集所有车辆
        busline_vehicles = {}

        for result in successful:
            vehicle_id = result['vehicle_id']
            record_count = result['record_count']
            output_file = Path(result['output_file'])
            buslines = result.get('buslines', [])

            for busline in buslines:
                # 跳过空值或NaN（额外保护，理论上已在前面过滤）
                if not busline or pd.isna(busline):
                    continue

                if busline not in busline_vehicles:
                    busline_vehicles[busline] = []

                busline_vehicles[busline].append({
                    'vehicle_id': vehicle_id,
                    'record_count': record_count,
                    'file_path': output_file
                })

        # 创建代表轨迹输出目录
        repr_dir = output_dir / 'representative_trajectories'
        repr_dir.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"  Output directory: {repr_dir}")
            print(f"  Total bus lines (raw): {len(busline_vehicles)}")

            # 过滤有效线路（排除NaN）
            valid_busline_count = sum(1 for k in busline_vehicles.keys() if pd.notna(k))
            invalid_count = len(busline_vehicles) - valid_busline_count

            if invalid_count > 0:
                print(f"  ⚠️  Skipping {invalid_count} lines with missing names")
            print(f"  Valid bus lines: {valid_busline_count}")
            print(f"\n  Selecting representative vehicle for each line...\n")

        # 为每条线路选择代表车辆（记录数最多的）
        selected_count = 0
        skipped_count = 0

        # 过滤并排序公交线路（排除NaN，按线路名排序）
        valid_buslines = {
            str(k): v for k, v in busline_vehicles.items()
            if pd.notna(k)
        }

        for busline, vehicles in sorted(valid_buslines.items()):
            # 按记录数排序，选择最多的
            vehicles_sorted = sorted(vehicles, key=lambda x: x['record_count'], reverse=True)
            representative = vehicles_sorted[0]

            # 复制文件
            src_file = representative['file_path']
            dst_file = repr_dir / f"{busline}_representative_processed.csv"

            try:
                # 读取源文件并添加busline标识
                df = pd.read_csv(src_file)

                # 保存到新位置
                df.to_csv(dst_file, index=False)

                selected_count += 1

                if verbose and selected_count <= 20:  # 只显示前20条
                    print(f"    {busline:20s} → {representative['vehicle_id']} "
                          f"({representative['record_count']:,} records)")
                elif verbose and selected_count == 21:
                    print(f"    ... ({len(valid_buslines) - 20} more lines)")

            except Exception as e:
                if verbose:
                    print(f"    ⚠️  {busline}: Failed to copy - {e}")
                skipped_count += 1

        if verbose:
            print(f"\n  ✅ Successfully selected {selected_count} representative trajectories")
            if skipped_count > 0:
                print(f"  ⚠️  Skipped {skipped_count} lines due to errors")
            print(f"  📁 Saved to: {repr_dir}\n")

        return {
            'total_lines': len(busline_vehicles),
            'selected': selected_count,
            'skipped': skipped_count,
            'output_dir': str(repr_dir)
        }

    except Exception as e:
        if verbose:
            print(f"  Error selecting representative trajectories: {e}\n")
        return None


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

    # 生成公交线路统计汇总
    _ = generate_busline_summary(all_results, output_dir, config)

    # 为每条公交线路选择代表性轨迹
    _ = select_representative_trajectories(all_results, output_dir, config)

    print("\n" + "="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")

    return 0 if len(failed) == 0 else 1


if __name__ == '__main__':
    sys.exit(main())