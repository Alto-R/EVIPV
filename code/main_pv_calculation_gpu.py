"""
车顶光伏发电量计算 - 主执行脚本 (GPU加速版)

完整流程:
1. 从建筑footprint生成建筑mesh
2. 获取太阳辐射数据并缓存
3. GPU加速计算车顶光伏发电量 (1分钟分辨率)

使用示例:
    python main_pv_calculation_gpu.py --config config.yaml
    python main_pv_calculation_gpu.py --lat 22.543099 --lon 114.057868 --date 2019-03-12
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import pyvista as pv
import yaml
from datetime import datetime

# 导入自定义模块
from prepare_building_mesh_from_footprint import prepare_building_mesh_from_footprint
from fetch_irradiance_data import fetch_and_cache_irradiance_data, convert_to_pvlib_format
from pv_calculator_gpu import GPUAcceleratedSolarPVCalculator


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_default_config(output_path='config.yaml'):
    """创建默认配置文件"""
    default_config = {
        'location': {
            'name': '深圳市',
            'lat': 22.543099,
            'lon': 114.057868,
        },
        'data_sources': {
            'footprint_path': 'data/shenzhen_buildings.geojson',
            'trajectory_path': 'traj/onetra_0312_1.csv',
        },
        'pv_system': {
            'panel_area': 2.0,  # 光伏板面积(平方米)
            'panel_efficiency': 0.22,  # 光伏板效率
            'tilt': 5,  # 倾角(度)
            'vehicle_height': 1.5,  # 车顶高度(米)
        },
        'computation': {
            'time_resolution_minutes': 1,  # 时间分辨率(分钟)
            'use_gpu': True,  # 是否使用GPU加速
            'batch_size': 100,  # 批处理大小
            'mesh_grid_size': None,  # mesh网格细分精度(米)，None表示不细分
        },
        'output': {
            'mesh_path': 'building_mesh.vtk',
            'result_path': 'output/pv_generation_1min_gpu.csv',
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(default_config, f, allow_unicode=True, default_flow_style=False)

    print(f"✅ 默认配置文件已创建: {output_path}")
    return default_config


def main(config=None, args=None):
    """
    主函数

    Parameters
    ----------
    config : dict, optional
        配置字典
    args : argparse.Namespace, optional
        命令行参数
    """
    print("\n" + "="*80)
    print(" "*20 + "🚀 车顶光伏发电量计算系统 (GPU加速)")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 合并配置
    if config is None:
        config = create_default_config()

    if args is not None:
        # 命令行参数覆盖配置文件
        if args.lat is not None:
            config['location']['lat'] = args.lat
        if args.lon is not None:
            config['location']['lon'] = args.lon
        if args.footprint is not None:
            config['data_sources']['footprint_path'] = args.footprint
        if args.trajectory is not None:
            config['data_sources']['trajectory_path'] = args.trajectory
        if args.date is not None:
            config['date'] = args.date
        if args.no_gpu:
            config['computation']['use_gpu'] = False

    # 提取配置
    lat = config['location']['lat']
    lon = config['location']['lon']
    footprint_path = config['data_sources']['footprint_path']
    trajectory_path = config['data_sources']['trajectory_path']
    mesh_path = config['output']['mesh_path']
    result_path = config['output']['result_path']
    time_resolution = config['computation']['time_resolution_minutes']
    use_gpu = config['computation']['use_gpu']
    batch_size = config['computation']['batch_size']
    mesh_grid_size = config['computation'].get('mesh_grid_size', None)

    # 确定日期范围
    if 'date' in config:
        start_date = end_date = config['date']
    else:
        # 从轨迹文件推断日期
        print("📅 从轨迹文件推断日期...")
        traj_df = pd.read_csv(trajectory_path, nrows=10)
        traj_df['datetime'] = pd.to_datetime(traj_df['datetime'])
        start_date = traj_df['datetime'].min().strftime('%Y-%m-%d')
        end_date = traj_df['datetime'].max().strftime('%Y-%m-%d')
        print(f"   推断日期范围: {start_date} 至 {end_date}")

    print("\n📋 配置信息:")
    print(f"   位置: {config['location']['name']} ({lat:.4f}, {lon:.4f})")
    print(f"   日期: {start_date} 至 {end_date}")
    print(f"   Footprint数据: {footprint_path}")
    print(f"   轨迹数据: {trajectory_path}")
    print(f"   时间分辨率: {time_resolution} 分钟")
    print(f"   Mesh网格精度: {mesh_grid_size if mesh_grid_size else '不细分'}")
    print(f"   GPU加速: {'启用' if use_gpu else '禁用'}")
    print(f"   输出路径: {result_path}")

    # ===== 步骤1: 准备建筑mesh =====
    print("\n" + "="*80)
    print("步骤 1/4: 准备建筑mesh")
    print("="*80)

    if Path(mesh_path).exists():
        print(f"✅ 发现已有mesh文件: {mesh_path}")
        building_mesh = pv.read(mesh_path)
        print(f"   顶点数: {building_mesh.n_points:,}")
        print(f"   三角形数: {building_mesh.n_faces:,}")
    else:
        print(f"🔄 从建筑footprint转换mesh...")
        building_mesh = prepare_building_mesh_from_footprint(
            footprint_path=footprint_path,
            output_mesh_path=mesh_path,
            grid_size=mesh_grid_size
        )

    # ===== 步骤2: 获取太阳辐射数据 =====
    print("\n" + "="*80)
    print("步骤 2/4: 获取太阳辐射数据")
    print("="*80)

    irradiance_data = fetch_and_cache_irradiance_data(
        lat=lat,
        lon=lon,
        start_date=start_date,
        end_date=end_date,
        granularity='1min' if time_resolution == 1 else '1hour',
        save_csv=True,
        output_dir='irradiance_data'
    )

    # 转换为pvlib格式
    weather_data = convert_to_pvlib_format(irradiance_data)

    # ===== 步骤3: 创建GPU加速计算器 =====
    print("\n" + "="*80)
    print("步骤 3/4: 初始化GPU加速计算器")
    print("="*80)

    calculator = GPUAcceleratedSolarPVCalculator(
        lon_center=lon,
        lat_center=lat,
        building_mesh=building_mesh,
        panel_area=config['pv_system']['panel_area'],
        panel_efficiency=config['pv_system']['panel_efficiency'],
        time_resolution_minutes=time_resolution,
        use_gpu=use_gpu,
        batch_size=batch_size
    )

    # ===== 步骤4: 计算发电量 =====
    print("\n" + "="*80)
    print("步骤 4/4: 计算车顶光伏发电量")
    print("="*80)

    # 读取轨迹数据
    print(f"📂 读取轨迹数据: {trajectory_path}")
    trajectory_df = pd.read_csv(trajectory_path)
    print(f"   轨迹点数: {len(trajectory_df):,}")

    # 检查必需列
    required_cols = ['datetime', 'lng', 'lat', 'angle']
    missing_cols = [col for col in required_cols if col not in trajectory_df.columns]
    if missing_cols:
        raise ValueError(f"轨迹数据缺少必需列: {missing_cols}")

    # 执行计算
    import time
    start_time = time.time()

    result_df = calculator.process_trajectory(
        trajectory_df,
        weather_data=weather_data
    )

    elapsed_time = time.time() - start_time

    # ===== 保存结果 =====
    print("\n" + "="*80)
    print("💾 保存结果")
    print("="*80)

    output_dir = Path(result_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    result_df.to_csv(result_path, index=False)
    file_size_mb = Path(result_path).stat().st_size / (1024 * 1024)
    print(f"✅ 结果已保存到: {result_path}")
    print(f"   文件大小: {file_size_mb:.2f} MB")
    print(f"   记录数: {len(result_df):,}")

    # ===== 统计分析 =====
    print("\n" + "="*80)
    print("📊 计算结果统计")
    print("="*80)

    total_energy_kwh = result_df['energy_kwh'].sum()
    avg_power_w = result_df['ac_power'].mean()
    max_power_w = result_df['ac_power'].max()
    shaded_ratio = result_df['is_shaded'].mean()
    avg_cell_temp = result_df['cell_temp'].mean()

    # 按小时统计
    result_df['hour'] = pd.to_datetime(result_df['datetime']).dt.hour
    hourly_stats = result_df.groupby('hour').agg({
        'ac_power': 'mean',
        'energy_kwh': 'sum'
    })

    print(f"\n总体统计:")
    print(f"  总发电量: {total_energy_kwh:.2f} kWh")
    print(f"  平均功率: {avg_power_w:.2f} W")
    print(f"  峰值功率: {max_power_w:.2f} W")
    print(f"  遮阴时长占比: {shaded_ratio*100:.1f}%")
    print(f"  平均电池温度: {avg_cell_temp:.1f}°C")
    print(f"  计算耗时: {elapsed_time:.1f} 秒")

    print(f"\n逐小时发电量 (kWh):")
    for hour, stats in hourly_stats.iterrows():
        print(f"  {hour:02d}:00 - {stats['energy_kwh']:.3f} kWh (平均功率: {stats['ac_power']:.1f} W)")

    # 保存统计摘要
    summary_path = result_path.replace('.csv', '_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("车顶光伏发电量计算 - 结果摘要\n")
        f.write("="*60 + "\n\n")
        f.write(f"计算时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"位置: {config['location']['name']} ({lat:.4f}, {lon:.4f})\n")
        f.write(f"日期: {start_date} 至 {end_date}\n")
        f.write(f"时间分辨率: {time_resolution} 分钟\n")
        f.write(f"GPU加速: {'启用' if use_gpu else '禁用'}\n\n")
        f.write("总体统计:\n")
        f.write(f"  总发电量: {total_energy_kwh:.2f} kWh\n")
        f.write(f"  平均功率: {avg_power_w:.2f} W\n")
        f.write(f"  峰值功率: {max_power_w:.2f} W\n")
        f.write(f"  遮阴时长占比: {shaded_ratio*100:.1f}%\n")
        f.write(f"  平均电池温度: {avg_cell_temp:.1f}°C\n")
        f.write(f"  计算耗时: {elapsed_time:.1f} 秒\n\n")
        f.write("逐小时发电量 (kWh):\n")
        for hour, stats in hourly_stats.iterrows():
            f.write(f"  {hour:02d}:00 - {stats['energy_kwh']:.3f} kWh\n")

    print(f"\n💾 摘要已保存到: {summary_path}")

    # 完成
    print("\n" + "="*80)
    print("✅ 所有计算完成!")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")

    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='车顶光伏发电量计算 (GPU加速)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 使用配置文件
  python main_pv_calculation_gpu.py --config config.yaml

  # 使用命令行参数
  python main_pv_calculation_gpu.py --lat 22.543099 --lon 114.057868 --date 2019-03-12

  # 生成默认配置文件
  python main_pv_calculation_gpu.py --create-config
        """
    )

    parser.add_argument('--config', type=str, help='配置文件路径(.yaml)')
    parser.add_argument('--create-config', action='store_true', help='创建默认配置文件')
    parser.add_argument('--lat', type=float, help='纬度')
    parser.add_argument('--lon', type=float, help='经度')
    parser.add_argument('--date', type=str, help='日期 YYYY-MM-DD')
    parser.add_argument('--footprint', type=str, help='建筑footprint数据路径 (GeoJSON/Shapefile)')
    parser.add_argument('--trajectory', type=str, help='轨迹数据CSV路径')
    parser.add_argument('--no-gpu', action='store_true', help='禁用GPU加速')

    args = parser.parse_args()

    try:
        # 创建默认配置文件
        if args.create_config:
            create_default_config()
            sys.exit(0)

        # 加载配置
        if args.config:
            print(f"📂 加载配置文件: {args.config}")
            config = load_config(args.config)
        else:
            config = None

        # 执行主流程
        result_df = main(config=config, args=args)

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
