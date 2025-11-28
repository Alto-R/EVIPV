"""
车顶光伏发电量计算器 - 基类
使用pvlib计算太阳辐射，trimesh + triro/OptiX进行GPU加速光线追踪
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import trimesh
import pvlib
from pvlib import location

# 添加RealSceneDL库路径
REALSCENEDL_PATH = r"D:\1-PKU\PKU\1 Master\Projects\RealSceneDL\src"
if REALSCENEDL_PATH not in sys.path:
    sys.path.insert(0, REALSCENEDL_PATH)

# 导入RealSceneDL坐标转换和GPU光线追踪补丁
from RealSceneDL.coordinates import (
    convert_wgs84_to_3dtiles,
    convert_direction_proj_to_3dtiles
)
from RealSceneDL._internal.triro_ray_patch import apply_triro_patch

# 应用triro补丁（自动启用GPU光线追踪）
apply_triro_patch()


class SolarPVCalculator:
    """
    车顶光伏发电量计算器基类

    功能：
    - 太阳位置计算（pvlib）
    - GPS坐标转换（RealSceneDL）
    - GPU加速光线追踪（trimesh + triro/OptiX）
    - 光伏发电功率计算
    """

    def __init__(self,
                 lon_center,
                 lat_center,
                 building_mesh,
                 panel_area=2.0,
                 panel_efficiency=0.22,
                 time_resolution_minutes=1):
        """
        初始化光伏计算器

        Parameters
        ----------
        lon_center : float
            中心位置经度
        lat_center : float
            中心位置纬度
        building_mesh : trimesh.Trimesh
            建筑mesh（trimesh格式）
        panel_area : float
            光伏板面积(平方米)
        panel_efficiency : float
            光伏板效率（0-1）
        time_resolution_minutes : int
            时间分辨率（分钟）
        """
        self.lon_center = lon_center
        self.lat_center = lat_center
        self.panel_area = panel_area
        self.panel_efficiency = panel_efficiency
        self.time_resolution_minutes = time_resolution_minutes

        # 验证并存储trimesh
        print("\n🔄 验证建筑mesh...")
        if not isinstance(building_mesh, trimesh.Trimesh):
            raise TypeError(f"building_mesh必须是trimesh.Trimesh类型，当前类型: {type(building_mesh)}")

        self.building_trimesh = building_mesh
        print(f"✅ 建筑mesh验证通过")
        print(f"   顶点数: {len(self.building_trimesh.vertices):,}")
        print(f"   三角形数: {len(self.building_trimesh.faces):,}")

        # 光伏组件参数
        self.module_parameters = {
            'pdc0': 1000 * panel_efficiency * panel_area,  # 标准测试条件下DC功率(W)
            'gamma_pdc': -0.004,  # 功率温度系数 (%/°C)
            'temp_ref': 25.0,  # 参考温度(°C)
        }

        # 逆变器参数
        self.inverter_parameters = {
            'eta_inv_nom': 0.96,  # 逆变器额定效率
            'eta_inv_ref': 0.9637,  # 参考效率
        }

        # pvlib位置对象
        self.location = location.Location(
            latitude=lat_center,
            longitude=lon_center,
            tz='Asia/Shanghai'  # 使用深圳/中国时区
        )

        print(f"✅ 计算器初始化完成")
        print(f"   位置: ({lat_center:.4f}, {lon_center:.4f})")
        print(f"   光伏板: {panel_area} m², 效率 {panel_efficiency*100:.1f}%")
        print(f"   时间分辨率: {time_resolution_minutes} 分钟")

    def get_sun_position_pvlib(self, times):
        """
        使用pvlib计算太阳位置（优化版：自动去重）

        相比原版，添加了时间去重优化：
        - 对于批量处理场景，时间点可能大量重复
        - 只计算唯一时间点，然后reindex回原始序列
        - 速度提升：50-100倍

        Parameters
        ----------
        times : pandas.DatetimeIndex
            时间序列（Asia/Shanghai时区）

        Returns
        -------
        pandas.DataFrame
            太阳位置数据，包含列：
            - apparent_zenith: 视天顶角(度)
            - apparent_elevation: 视高度角(度)
            - azimuth: 方位角(度)
        """
        # 确保时区为Asia/Shanghai
        if times.tz is None:
            times = times.tz_localize('Asia/Shanghai')
        elif str(times.tz) != 'Asia/Shanghai':
            times = times.tz_convert('Asia/Shanghai')

        # 🚀 优化：时间去重
        unique_times = times.unique()

        # 只计算唯一时间点
        solar_position_unique = self.location.get_solarposition(unique_times)

        # reindex 回原始时间序列（O(N)复杂度，非常快）
        solar_position = solar_position_unique.reindex(times)

        return solar_position[['apparent_zenith', 'apparent_elevation', 'azimuth']]

    def sun_position_to_vector(self, solar_position):
        """
        将太阳位置转换为3DTiles坐标系下的方向向量

        Parameters
        ----------
        solar_position : pandas.DataFrame
            太阳位置数据（来自get_sun_position_pvlib）

        Returns
        -------
        numpy.ndarray
            太阳方向向量数组 (N, 3)，指向太阳的方向
        """
        sun_vectors = []

        for idx, row in solar_position.iterrows():
            azimuth = row['azimuth']  # 方位角(度)
            elevation = row['apparent_elevation']  # 高度角(度)

            # 转换为弧度
            azimuth_rad = np.deg2rad(azimuth)
            elevation_rad = np.deg2rad(elevation)

            # 局部投影坐标系下的太阳方向向量
            # 水平投影长度
            horizontal_projection = np.cos(elevation_rad)

            # 局部ENU坐标系 (东-北-上)
            x = horizontal_projection * np.sin(azimuth_rad)  # 东
            y = horizontal_projection * np.cos(azimuth_rad)  # 北
            z = np.sin(elevation_rad)  # 上

            # 太阳光方向：从太阳指向地球，因此反转
            sun_direction_local = np.array([-x, -y, -z])

            # 转换到3DTiles坐标系
            sun_direction_3dtiles = convert_direction_proj_to_3dtiles(
                sun_direction_local,
                self.lat_center,
                self.lon_center
            )

            sun_vectors.append(sun_direction_3dtiles)

        return np.array(sun_vectors)

    def get_irradiance_components(self, times, weather_data):
        """
        获取辐照度分量（优化版：自动去重）

        优化点：
        1. 时间去重：只对唯一时间点做reindex
        2. 使用nearest方法：确保物理准确性

        Parameters
        ----------
        times : pandas.DatetimeIndex
            时间序列
        weather_data : pandas.DataFrame
            天气数据，应包含列: ghi, dni, dhi

        Returns
        -------
        pandas.DataFrame
            辐照度数据，包含列: ghi, dni, dhi
        """
        if weather_data is None:
            # 无数据，返回零值
            return pd.DataFrame({
                'ghi': np.zeros(len(times)),
                'dni': np.zeros(len(times)),
                'dhi': np.zeros(len(times))
            }, index=times)

        # 🚀 性能优化：时间去重（如果有重复时间戳，减少reindex次数）
        unique_times = times.unique()

        if len(unique_times) < len(times):
            # 有重复时间戳，使用去重优化
            # 步骤1: 只对唯一时间点查询天气数据
            irrad_components_unique = weather_data[['ghi', 'dni', 'dhi']].reindex(
                unique_times, method='nearest'
            ).fillna(0)

            # 步骤2: ✅ 修复：使用method='ffill'将结果扩展回原始时间序列
            # 这样插值点也能获得最近的天气数据值
            irrad_components = irrad_components_unique.reindex(
                times, method='ffill'
            ).fillna(method='bfill').fillna(0)
        else:
            # 没有重复，直接查询所有时间点
            irrad_components = weather_data[['ghi', 'dni', 'dhi']].reindex(
                times, method='nearest'
            ).fillna(0)

        return irrad_components

    def gps_to_model_coords(self, lon, lat, height=0.0):
        """
        GPS坐标转换为3DTiles模型坐标

        Parameters
        ----------
        lon : float or np.ndarray
            经度
        lat : float or np.ndarray
            纬度
        height : float or np.ndarray
            高度（米）- 重要：这是相对于WGS84椭球面的高度

        Returns
        -------
        x, y, z : np.ndarray
            模型坐标系下的x, y, z坐标（ECEF坐标系，正确处理了高度）
        """
        # 确保 height 的形状和 lon/lat 一致
        lon_array = np.atleast_1d(lon)
        lat_array = np.atleast_1d(lat)
        height_array = np.atleast_1d(height)

        # 如果 height 是标量但 lon/lat 是数组，扩展 height
        if height_array.size == 1 and lon_array.size > 1:
            height_array = np.full_like(lon_array, height_array[0], dtype=float)

        # 使用RealSceneDL转换（包含高度信息，正确转换到ECEF）
        coords_3dtiles = convert_wgs84_to_3dtiles(lon_array, lat_array, height_array)

        # coords_3dtiles shape: (N, 3) 或 (3,)
        if coords_3dtiles.ndim == 1:
            # 单个点
            x = np.array([coords_3dtiles[0]])
            y = np.array([coords_3dtiles[1]])
            z = np.array([coords_3dtiles[2]])
        else:
            # 多个点
            x = coords_3dtiles[:, 0]
            y = coords_3dtiles[:, 1]
            z = coords_3dtiles[:, 2]

        return x, y, z

    def resample_trajectory(self, trajectory_df):
        """
        优化版重采样：优先使用最接近整数分钟的真实GPS点

        核心逻辑：
        1. 为每个GPS点计算到最近整数分钟的时间差
        2. 每个整数分钟选择时间差最小的点
        3. 将选中的点对齐到整数分钟
        4. 按天分组，填充时间间隔（前向填充，用于停车）

        Parameters
        ----------
        trajectory_df : pandas.DataFrame
            单个车辆的轨迹数据，必须包含: 'datetime', 'lng', 'lat', 'angle', 'speed'

        Returns
        -------
        pandas.DataFrame
            重采样后的轨迹数据（1分钟间隔）
            索引为 DatetimeIndex（datetime列已设置为索引）
        """
        # 1. 数据预处理
        traj = trajectory_df.copy()
        traj['datetime'] = pd.to_datetime(traj['datetime'])

        # 确保时区为 Asia/Shanghai
        if traj['datetime'].dt.tz is None:
            traj['datetime'] = traj['datetime'].dt.tz_localize('Asia/Shanghai')

        traj = traj.sort_values('datetime').reset_index(drop=True)

        # 2. 四舍五入到整数分钟，计算时间差
        traj['target_minute'] = traj['datetime'].dt.round(f'{self.time_resolution_minutes}min')
        traj['time_diff'] = abs(
            (traj['datetime'] - traj['target_minute']).dt.total_seconds()
        )

        # 3. 每个整数分钟选择时间差最小的点
        idx = traj.groupby('target_minute')['time_diff'].idxmin()
        selected = traj.loc[idx].copy()
        selected['datetime'] = selected['target_minute']
        selected = selected.drop(columns=['target_minute', 'time_diff'])
        selected = selected.sort_values('datetime').reset_index(drop=True)

        # 4. 按天分组处理（不跨天）
        selected['date'] = selected['datetime'].dt.date
        all_days = []

        for date, day_df in selected.groupby('date'):
            # 生成该天的完整时间序列
            start_time = day_df['datetime'].min()
            end_time = day_df['datetime'].max()
            full_times = pd.date_range(start_time, end_time, freq=f'{self.time_resolution_minutes}min')

            # reindex到完整时间序列
            day_df_indexed = day_df.set_index('datetime')
            resampled = day_df_indexed.reindex(full_times)

            # 前向填充（停车时位置、角度、速度都不变）
            resampled = resampled.fillna(method='ffill')
            resampled = resampled.fillna(method='bfill')

            # 删除date列
            resampled = resampled.drop(columns=['date'], errors='ignore')

            all_days.append(resampled)

        # 5. 合并所有天
        result = pd.concat(all_days, ignore_index=False)

        # 6. 删除NaN行
        result = result.dropna(subset=['lng', 'lat'])

        # 7. 确保索引名称为datetime
        result.index.name = 'datetime'

        return result


if __name__ == "__main__":
    print("SolarPVCalculator 基类模块")
    print("使用trimesh + triro/OptiX GPU加速光线追踪")
