"""
获取并缓存太阳辐射数据
使用Open-Meteo API获取GHI, DHI, DNI等气象数据
"""

import pandas as pd
import sys
from pathlib import Path

# 添加RealSceneDL库路径
REALSCENEDL_PATH = r"D:\1-PKU\PKU\1 Master\Projects\RealSceneDL\src"
if REALSCENEDL_PATH not in sys.path:
    sys.path.insert(0, REALSCENEDL_PATH)

from RealSceneDL.analysis.meteo import get_openmeteo_irradiance_data

# ============================================================================
# 配置区域 - 在此修改参数
# ============================================================================
CONFIG = {
    # 位置坐标
    'LAT': 22.543099,  # 纬度
    'LON': 114.057868,  # 经度

    # 日期范围 (格式: YYYY-MM-DD)
    'START_DATE': '2019-01-01',  # 开始日期
    'END_DATE': '2020-01-01',    # 结束日期

    # 时间粒度
    'GRANULARITY': '1hour',  # 时间粒度: '1min' 或 '1hour'

    # 输出设置
    'SAVE_CSV': True,  # 是否额外保存为CSV格式（已自动缓存parquet格式）
    'OUTPUT_DIR': 'irradiance_data',  # CSV输出目录
}
# ============================================================================


def fetch_and_cache_irradiance_data(lat, lon, start_date, end_date,
                                     granularity='1min',
                                     save_csv=True,
                                     output_dir='irradiance_data'):
    """
    获取太阳辐射数据并自动缓存

    Parameters
    ----------
    lat, lon : float
        位置坐标
    start_date, end_date : str
        日期范围 'YYYY-MM-DD'
    granularity : str
        时间粒度 '1min' 或 '1hour'
    save_csv : bool
        是否额外保存为CSV格式
    output_dir : str
        CSV输出目录

    Returns
    -------
    pandas.DataFrame
        辐射数据，包含以下列：
        - GHI (W/m^2): 全球水平辐照度
        - DHI (W/m^2): 散射水平辐照度
        - DNI (W/m^2): 直射法向辐照度
        - Temperature: 温度(°C)
        - WindSpeed: 风速(m/s)
    """
    print("="*60)
    print("☀️  获取太阳辐射数据")
    print("="*60)
    print(f"\n📍 位置: ({lat:.4f}, {lon:.4f})")
    print(f"📅 日期: {start_date} 至 {end_date}")
    print(f"⏱️  分辨率: {granularity}")
    print()

    # 使用RealSceneDL内置函数获取数据（自动缓存到openmeteo_cache/）
    irradiance_df = get_openmeteo_irradiance_data(
        latitude=lat,
        longitude=lon,
        start_date=start_date,
        end_date=end_date,
        target_timezone='Asia/Shanghai',
        granularity=granularity,
        print_progress=True
    )

    if irradiance_df.empty:
        print("❌ 未获取到数据")
        return irradiance_df

    print("\n" + "="*60)
    print("📊 数据统计")
    print("="*60)
    print(f"记录数: {len(irradiance_df):,}")
    print(f"时间范围: {irradiance_df.index.min()} 至 {irradiance_df.index.max()}")
    print(f"时区: {irradiance_df.index.tz}")
    print()

    # 数据质量检查
    print("数据完整性:")
    for col in irradiance_df.columns:
        missing = irradiance_df[col].isna().sum()
        missing_pct = missing / len(irradiance_df) * 100
        print(f"  {col}: {missing:,} 缺失 ({missing_pct:.2f}%)")

    # 统计信息
    print("\n数值统计:")
    print(irradiance_df.describe())

    # 额外保存为CSV（可选）
    if save_csv:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        csv_filename = f'irradiance_{lat:.4f}_{lon:.4f}_{start_date}_{end_date}_{granularity}.csv'
        csv_path = output_path / csv_filename

        irradiance_df.to_csv(csv_path)
        file_size_mb = csv_path.stat().st_size / (1024 * 1024)
        print(f"\n💾 额外保存CSV到: {csv_path}")
        print(f"   文件大小: {file_size_mb:.2f} MB")

    print("="*60)
    print("✅ 数据获取完成")
    print("="*60)

    return irradiance_df


def convert_to_pvlib_format(irradiance_df):
    """
    将数据转换为pvlib兼容格式

    Parameters
    ----------
    irradiance_df : pandas.DataFrame
        从get_openmeteo_irradiance_data获取的数据

    Returns
    -------
    pandas.DataFrame
        重命名后的DataFrame，列名符合pvlib习惯
    """
    return irradiance_df.rename(columns={
        'GHI (W/m^2)': 'ghi',
        'DNI (W/m^2)': 'dni',
        'DHI (W/m^2)': 'dhi',
        'Temperature': 'temp_air',
        'WindSpeed': 'wind_speed'
    })


def load_cached_irradiance_data(lat, lon, start_date, end_date,
                                 granularity='1min',
                                 source='parquet'):
    """
    加载已缓存的辐射数据

    Parameters
    ----------
    lat, lon : float
        位置坐标
    start_date, end_date : str
        日期范围
    granularity : str
        时间粒度
    source : str
        数据源 'parquet'(默认缓存) 或 'csv'

    Returns
    -------
    pandas.DataFrame
        缓存的辐射数据
    """
    if source == 'parquet':
        # 从openmeteo_cache目录加载
        cache_filename = f"openmeteo_{lat:.4f}_{lon:.4f}_{start_date}_{end_date}_Asia-Shanghai_{granularity}.parquet"
        cache_path = Path('openmeteo_cache') / cache_filename

        if not cache_path.exists():
            raise FileNotFoundError(f"缓存文件不存在: {cache_path}")

        df = pd.read_parquet(cache_path)
        print(f"✅ 从缓存加载: {cache_path}")

    elif source == 'csv':
        # 从irradiance_data目录加载
        csv_filename = f'irradiance_{lat:.4f}_{lon:.4f}_{start_date}_{end_date}_{granularity}.csv'
        csv_path = Path('irradiance_data') / csv_filename

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV文件不存在: {csv_path}")

        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        print(f"✅ 从CSV加载: {csv_path}")

    else:
        raise ValueError(f"未知的数据源: {source}")

    return df


if __name__ == "__main__":
    """
    获取太阳辐射数据

    使用方法:
    1. 在文件顶部的CONFIG区域修改参数
    2. 运行: python fetch_irradiance_data.py

    数据将自动缓存到 openmeteo_cache/ 目录
    """

    try:
        df = fetch_and_cache_irradiance_data(
            lat=CONFIG['LAT'],
            lon=CONFIG['LON'],
            start_date=CONFIG['START_DATE'],
            end_date=CONFIG['END_DATE'],
            granularity=CONFIG['GRANULARITY'],
            save_csv=CONFIG['SAVE_CSV'],
            output_dir=CONFIG['OUTPUT_DIR']
        )

        print("\n数据预览:")
        print(df.head())

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
