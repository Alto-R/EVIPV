"""
电动车GPS数据预处理脚本

功能：
1. 读取EV原始GPS数据（code/traj/*.csv格式）
2. 转换为标准格式（与preprocess_taxi_trajectories.py输出一致）
3. 使后续PV计算流程可以无缝处理EV数据

数据转换：
    EV原始格式（22列）：
        vehicle_id, datetime, ..., speed, ..., lng, lat, ...

    转换为标准格式（6列）：
        datetime, vehicle_id, lng, lat, speed, angle

关键转换步骤：
1. 提取车辆ID（第1列，索引0）
2. 解析datetime（第2列，索引1）并添加 Asia/Shanghai 时区
3. 提取速度（第6列，索引5）
4. 提取经纬度（第14列和第15列，索引13和14）
5. 计算移动角度（根据连续GPS点计算）
6. 数据清洗：去重、坐标验证、漂移点过滤
7. 按车辆ID分别保存为CSV（包含header），文件名：车辆ID_processed.csv

输出格式示例：
    datetime,vehicle_id,lng,lat,speed,angle
    2020-11-01 07:12:49+08:00,50418749edd6374ffbbd3263d7992549,121.27634,31.35390,35.3,120.5
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import transbigdata as tbd
from concurrent.futures import ProcessPoolExecutor, as_completed


# ============================================================================
# 配置参数
# ============================================================================
CONFIG = {
    'input': {
        'data_dir': '/data2/hcr/evipv/shanghaidata/evdata/车辆分组',               # EV数据目录
        'file_pattern': '*.csv',             # 文件匹配模式
    },
    'output': {
        'output_dir': '/data2/hcr/evipv/shanghaidata/evdata/车辆分组/processed',   # 输出目录
        'suffix': '_processed',              # 输出文件后缀
    },
    'validation': {
        'lon_range': (120.84, 122.0),        # 上海市经度范围
        'lat_range': (30.67, 31.87),          # 上海市纬度范围
        'speed_range': (0, 120),            # 速度范围 (km/h)
        'angle_range': (0, 360),            # 方向角范围 (度)
    },
    'processing': {
        'parallel': False,                  # 是否并行处理
        'n_jobs': 1,                        # 并行进程数
        'skip_files': ['0.csv', '1.csv', '2.csv', '3.csv', '4.csv'
                       , '5.csv', '6.csv', '7.csv'],  # 跳过文件
    },
    'verbose': True,                        # 是否显示详细信息
}


def calculate_angle(df):
    """
    根据连续GPS点计算移动角度（方位角）
    
    Parameters
    ----------
    df : pd.DataFrame
        包含 lng, lat 列的轨迹数据（已按时间排序）
    
    Returns
    -------
    np.ndarray
        角度数组（0-360度，正北为0度，顺时针）
    """
    # 获取经纬度
    lons = df['lng'].values
    lats = df['lat'].values
    
    # 计算相邻点之间的角度
    angles = np.zeros(len(df))
    
    for i in range(len(df) - 1):
        lon1, lat1 = lons[i], lats[i]
        lon2, lat2 = lons[i + 1], lats[i + 1]
        
        # 计算方位角（使用简化公式）
        # 将经纬度转换为弧度
        lon1_rad = np.radians(lon1)
        lat1_rad = np.radians(lat1)
        lon2_rad = np.radians(lon2)
        lat2_rad = np.radians(lat2)
        
        # 计算方位角
        dlon = lon2_rad - lon1_rad
        y = np.sin(dlon) * np.cos(lat2_rad)
        x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
        bearing = np.arctan2(y, x)
        
        # 转换为0-360度（正北为0度，顺时针）
        angle = (np.degrees(bearing) + 360) % 360
        angles[i] = angle
    
    # 最后一个点使用前一个点的角度
    if len(df) > 1:
        angles[-1] = angles[-2]
    
    return angles


def process_ev_file(input_path, output_dir, config):
    """
    处理EV GPS数据文件，按车辆ID分别保存
    
    Parameters
    ----------
    input_path : Path
        输入的EV CSV文件路径
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
            print(f"\n{'='*80}")
            print(f"处理文件: {input_path.name}")
            print('='*80)
        
        # 读取EV数据（无header，22列）
        # 只读取需要的列：0(vehicle_id), 1(datetime), 5(speed), 13(lng), 14(lat)
        if verbose:
            print("  读取原始CSV...")
        
        df = pd.read_csv(
            input_path,
            header=None,
            usecols=[0, 1, 5, 13, 14],
            names=['vehicle_id', 'datetime', 'speed', 'lng', 'lat']
        )
        
        original_count = len(df)
        unique_vehicles = df['vehicle_id'].nunique()
        
        if verbose:
            print(f"    原始记录数: {original_count:,}")
            print(f"    唯一车辆数: {unique_vehicles}")
        
        # 解析datetime（自动推断格式，支持混合格式）
        if verbose:
            print("  转换datetime格式...")

        df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
        df['datetime'] = df['datetime'].dt.tz_localize('Asia/Shanghai')  # 标记为中国时区
        
        # 数据验证和清洗
        if verbose:
            print("  数据验证和清洗...")
        
        validation = config.get('validation', {})
        
        # 1. 检查坐标范围（上海市边界）
        lon_min, lon_max = validation.get('lon_range', (120.84, 122.0))
        lat_min, lat_max = validation.get('lat_range', (30.67, 31.87))
        valid_lng = (df['lng'] >= lon_min) & (df['lng'] <= lon_max)
        valid_lat = (df['lat'] >= lat_min) & (df['lat'] <= lat_max)
        valid_coords = valid_lng & valid_lat
        
        removed_coords = (~valid_coords).sum()
        if removed_coords > 0:
            if verbose:
                print(f"    警告：移除 {removed_coords} 条边界外记录")
            df = df[valid_coords].copy()
        
        # 2. 检查速度范围
        speed_min, speed_max = validation.get('speed_range', (0, 150))
        valid_speed = (df['speed'] >= speed_min) & (df['speed'] <= speed_max)
        removed_speed = (~valid_speed).sum()
        if removed_speed > 0:
            if verbose:
                print(f"    警告：移除 {removed_speed} 条速度异常记录")
            df = df[valid_speed].copy()
        
        # 3. 去除NaN值
        df = df.dropna()
        
        # 按车辆ID分组处理
        if verbose:
            print(f"\n  按车辆ID分组处理...")
        
        vehicle_stats = []
        
        for vehicle_id in df['vehicle_id'].unique():
            if verbose:
                print(f"\n    --- 处理车辆: {vehicle_id} ---")
            
            # 过滤当前车辆的数据
            vehicle_df = df[df['vehicle_id'] == vehicle_id].copy()
            
            # 按时间排序
            vehicle_df = vehicle_df.sort_values('datetime').reset_index(drop=True)
            
            original_vehicle_count = len(vehicle_df)
            if verbose:
                print(f"        原始记录数: {original_vehicle_count:,}")
            
            # 使用 transbigdata 清理重复记录
            try:
                vehicle_df = tbd.traj_clean_redundant(
                    vehicle_df,
                    col=['vehicle_id', 'datetime', 'lng', 'lat']
                )
                removed_duplicates = original_vehicle_count - len(vehicle_df)
                if removed_duplicates > 0 and verbose:
                    print(f"        警告：移除 {removed_duplicates} 条冗余记录")
            except Exception as e:
                if verbose:
                    print(f"        警告：清理冗余记录失败，跳过此步骤: {e}")

            # 使用 transbigdata 清理漂移异常点
            before_drift = len(vehicle_df)
            try:
                vehicle_df = tbd.traj_clean_drift(
                    vehicle_df,
                    col=['vehicle_id', 'datetime', 'lng', 'lat'],
                    speedlimit=120,
                    dislimit=1000,
                    anglelimit=60
                )
                removed_drift = before_drift - len(vehicle_df)
                if removed_drift > 0 and verbose:
                    print(f"        警告：移除 {removed_drift} 条漂移异常点")
            except Exception as e:
                if verbose:
                    print(f"        警告：清理漂移点失败，跳过此步骤: {e}")
            
            if len(vehicle_df) < 2:
                if verbose:
                    print(f"        警告：记录数不足，跳过")
                continue
            
            # 计算移动角度
            if verbose:
                print(f"        计算移动角度...")
            vehicle_df['angle'] = calculate_angle(vehicle_df)

            # 清理NaN角度值（停车点GPS重合可能产生NaN）
            # 前向填充 → 后向填充 → 最后用0°填充
            vehicle_df['angle'] = vehicle_df['angle'].ffill().bfill().fillna(0.0)

            # 选择标准列并重新排序
            standard_df = vehicle_df[[
                'datetime',
                'vehicle_id',
                'lng',
                'lat',
                'speed',
                'angle'
            ]].copy()
            
            # 统计信息
            stats = {
                'vehicle_id': vehicle_id,
                'total_records': len(standard_df),
                'time_range': (standard_df['datetime'].min(), standard_df['datetime'].max()),
                'duration_hours': (standard_df['datetime'].max() - standard_df['datetime'].min()).total_seconds() / 3600,
                'avg_speed': standard_df['speed'].mean(),
                'coord_bounds': {
                    'lng_min': standard_df['lng'].min(),
                    'lng_max': standard_df['lng'].max(),
                    'lat_min': standard_df['lat'].min(),
                    'lat_max': standard_df['lat'].max()
                }
            }
            
            if verbose:
                print(f"\n        统计信息:")
                print(f"            有效记录数: {stats['total_records']:,}")
                print(f"            时间范围: {stats['time_range'][0]} 至 {stats['time_range'][1]}")
                print(f"            持续时间: {stats['duration_hours']:.2f} 小时")
                print(f"            平均速度: {stats['avg_speed']:.1f} km/h")
                print(f"            经度范围: {stats['coord_bounds']['lng_min']:.4f} ~ {stats['coord_bounds']['lng_max']:.4f}")
                print(f"            纬度范围: {stats['coord_bounds']['lat_min']:.4f} ~ {stats['coord_bounds']['lat_max']:.4f}")
            
            # 保存处理后的CSV
            output_filename = f"{vehicle_id}{suffix}.csv"
            output_path = output_dir / output_filename
            
            standard_df.to_csv(output_path, index=False)
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            
            if verbose:
                print(f"\n        保存到: {output_path.name}")
                print(f"            文件大小: {file_size_mb:.2f} MB")
            
            vehicle_stats.append(stats)
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"文件处理完成: {input_path.name}")
            print(f"   成功处理车辆数: {len(vehicle_stats)}")
            print('='*80)
        
        return vehicle_stats
    
    except Exception as e:
        print(f"错误：处理文件失败: {input_path.name}")
        print(f"   错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


def main():
    """主函数：批量处理所有EV轨迹文件"""
    
    print("\n" + "="*80)
    print(" "*20 + "EV轨迹数据预处理")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 获取配置
    input_dir = Path(CONFIG['input']['data_dir'])
    output_dir = Path(CONFIG['output']['output_dir'])
    file_pattern = CONFIG['input']['file_pattern']
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有CSV文件
    csv_files = sorted(input_dir.glob(file_pattern))

    # 过滤掉需要跳过的文件
    skip_files = CONFIG['processing'].get('skip_files', [])
    if skip_files:
        original_count = len(csv_files)
        csv_files = [f for f in csv_files if f.name not in skip_files]
        skipped_count = original_count - len(csv_files)
        if skipped_count > 0:
            print(f"跳过 {skipped_count} 个文件: {', '.join(skip_files)}")

    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"找到 {len(csv_files)} 个CSV文件待处理\n")
    
    if len(csv_files) == 0:
        print("错误：未找到CSV文件")
        return
    
    # 处理所有文件
    all_stats = []
    
    if CONFIG['processing'].get('parallel', False):
        # 并行处理
        n_jobs = CONFIG['processing'].get('n_jobs', 10)
        print(f"使用并行处理: {n_jobs} 个进程\n")
        
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(process_ev_file, csv_file, output_dir, CONFIG): csv_file
                for csv_file in csv_files
            }
            
            for future in as_completed(futures):
                csv_file = futures[future]
                try:
                    stats = future.result()
                    all_stats.extend(stats)
                except Exception as e:
                    print(f"错误：处理失败: {csv_file.name}")
                    print(f"   错误: {str(e)}")
    else:
        # 串行处理
        print(f"使用串行处理\n")
        for csv_file in csv_files:
            stats = process_ev_file(csv_file, output_dir, CONFIG)
            all_stats.extend(stats)
    
    # 输出汇总
    print("\n" + "="*80)
    print(" "*20 + "处理汇总")
    print("="*80)
    print(f"成功处理车辆数: {len(all_stats)}")
    
    if len(all_stats) > 0:
        total_records = sum(s['total_records'] for s in all_stats)
        total_hours = sum(s['duration_hours'] for s in all_stats)
        avg_speed = np.mean([s['avg_speed'] for s in all_stats])
        
        print(f"总记录数: {total_records:,}")
        print(f"总时长: {total_hours:.2f} 小时")
        print(f"平均速度: {avg_speed:.1f} km/h")
    
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
