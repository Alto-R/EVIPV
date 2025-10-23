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
        使用pvlib计算太阳位置

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

        # 计算太阳位置
        solar_position = self.location.get_solarposition(times)

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
        获取辐照度分量（GHI, DNI, DHI）

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

        # 重新索引到目标时间
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
            高度（米）

        Returns
        -------
        x, y : np.ndarray
            模型坐标系下的x, y坐标（z由height参数指定）
        """
        # 确保 height 的形状和 lon/lat 一致
        lon_array = np.atleast_1d(lon)
        lat_array = np.atleast_1d(lat)
        height_array = np.atleast_1d(height)

        # 如果 height 是标量但 lon/lat 是数组，扩展 height
        if height_array.size == 1 and lon_array.size > 1:
            height_array = np.full_like(lon_array, height_array[0], dtype=float)

        # 使用RealSceneDL转换
        coords_3dtiles = convert_wgs84_to_3dtiles(lon_array, lat_array, height_array)

        # coords_3dtiles shape: (N, 3) 或 (3,)
        if coords_3dtiles.ndim == 1:
            # 单个点
            x = np.array([coords_3dtiles[0]])
            y = np.array([coords_3dtiles[1]])
        else:
            # 多个点
            x = coords_3dtiles[:, 0]
            y = coords_3dtiles[:, 1]

        return x, y


if __name__ == "__main__":
    print("SolarPVCalculator 基类模块")
    print("使用trimesh + triro/OptiX GPU加速光线追踪")
