"""
GPU加速的车顶光伏发电量计算器
继承自pv_generation_pvlib.py的SolarPVCalculator
使用PyTorch进行GPU加速批量计算
"""

import os
import pandas as pd
import numpy as np
import pyvista as pv
from tqdm import tqdm
import pvlib
from pvlib import temperature

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  警告: PyTorch未安装，将使用CPU模式")

from pv_generation_pvlib import SolarPVCalculator


class GPUAcceleratedSolarPVCalculator(SolarPVCalculator):
    """
    GPU加速版本的光伏计算器

    使用PyTorch进行批量矩阵运算加速:
    - GPU批量生成光线
    - 批量POA计算
    - 批量功率计算
    """

    def __init__(self, *args, use_gpu=True, batch_size=100, **kwargs):
        """
        Parameters
        ----------
        *args, **kwargs :
            传递给SolarPVCalculator的参数
        use_gpu : bool
            是否使用GPU加速(需要安装PyTorch和CUDA)
        batch_size : int
            批处理大小，用于控制显存占用
        """
        super().__init__(*args, **kwargs)

        # GPU配置
        self.batch_size = batch_size

        if TORCH_AVAILABLE and use_gpu:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')

        # 输出GPU信息
        print("\n" + "="*60)
        print("🚀 GPU加速配置")
        print("="*60)
        if self.device.type == 'cuda':
            print(f"✅ GPU加速: 启用")
            print(f"   GPU设备: {torch.cuda.get_device_name(0)}")
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   总显存: {gpu_memory:.1f} GB")
            print(f"   批处理大小: {batch_size}")
        else:
            reason = "PyTorch未安装" if not TORCH_AVAILABLE else "CUDA不可用"
            print(f"⚠️  GPU加速: 禁用 ({reason})")
            print(f"   使用CPU计算")
        print("="*60 + "\n")

    def calculate_shadows_batch_gpu(self, points_xy, height, times):
        """
        GPU加速的批量阴影计算

        核心优化:
        1. 将坐标数据转移到GPU
        2. 批量矩阵运算替代循环生成光线
        3. PyVista批量光线追踪

        Parameters
        ----------
        points_xy : numpy.ndarray
            点的xy坐标 (N, 2)
        height : float
            点的高度(车顶高度，约1.5米)
        times : pandas.DatetimeIndex
            时间序列

        Returns
        -------
        pandas.DataFrame
            阴影矩阵，行=位置，列=时间
        """
        print(f"\n🌓 GPU加速阴影计算...")
        print(f"   位置数: {len(points_xy):,}")
        print(f"   时间点数: {len(times):,}")
        print(f"   总光线数: {len(points_xy) * len(times):,}")

        # 获取太阳位置
        solar_position = self.get_sun_position_pvlib(times)

        # 过滤白天时间
        mask_daytime = solar_position['apparent_elevation'] > 0
        daytime_indices = np.where(mask_daytime)[0]

        if len(daytime_indices) == 0:
            print("   ⚠️  所有时间都是夜晚")
            shadow_matrix = np.ones((len(points_xy), len(times)), dtype=int)
            return pd.DataFrame(shadow_matrix, columns=times)

        # 白天数据
        daytime_times = times[daytime_indices]
        daytime_solar_position = solar_position.iloc[daytime_indices]
        sun_vectors = self.sun_position_to_vector(daytime_solar_position)

        print(f"   白天时间点数: {len(daytime_times):,}")

        # 构建查询点
        query_points = np.column_stack([
            points_xy[:, 0],
            points_xy[:, 1],
            np.full(len(points_xy), height)
        ])

        n_points = len(query_points)
        n_times = len(daytime_times)

        # 🔥 GPU加速部分: 批量构建光线
        if self.device.type == 'cuda' and TORCH_AVAILABLE:
            print(f"   ⚡ GPU批量生成光线...")

            with torch.no_grad():
                # 转移到GPU
                query_points_gpu = torch.from_numpy(query_points).float().to(self.device)
                sun_vectors_gpu = torch.from_numpy(sun_vectors).float().to(self.device)

                # 批量计算光线起点
                # shape: (n_times, n_points, 3)
                ray_origins_batch = query_points_gpu.unsqueeze(0) - sun_vectors_gpu.unsqueeze(1) * 5e5
                ray_directions_batch = sun_vectors_gpu.unsqueeze(1).expand(n_times, n_points, 3)

                # 转回CPU用于PyVista光线追踪
                all_ray_origins = ray_origins_batch.reshape(-1, 3).cpu().numpy()
                all_ray_directions = ray_directions_batch.reshape(-1, 3).cpu().numpy()

            print(f"   ✅ GPU光线生成完成")

        else:
            # CPU模式
            print(f"   🖥️  CPU批量生成光线...")
            all_ray_origins = []
            all_ray_directions = []

            for sun_vec in tqdm(sun_vectors, desc="   生成光线"):
                ray_origins = query_points - sun_vec * 5e5
                ray_directions = np.tile(sun_vec, (n_points, 1))
                all_ray_origins.append(ray_origins)
                all_ray_directions.append(ray_directions)

            all_ray_origins = np.vstack(all_ray_origins)
            all_ray_directions = np.vstack(all_ray_directions)

        # 批量光线追踪（PyVista内部已优化）
        print(f"   🎯 批量光线追踪...")
        _, intersection_rays, _ = self.building_mesh.multi_ray_trace(
            all_ray_origins, all_ray_directions, first_point=True
        )

        print(f"   ✅ 追踪完成，检测到 {len(intersection_rays):,} 个遮挡")

        # 解析结果
        shadow_matrix = np.zeros((n_points, len(times)), dtype=int)

        # 夜晚时间默认为阴影
        shadow_matrix[:, ~mask_daytime] = 1

        # 处理白天的阴影
        if len(intersection_rays) > 0:
            time_indices = intersection_rays // n_points
            point_indices = intersection_rays % n_points
            full_time_indices = daytime_indices[time_indices]
            shadow_matrix[point_indices, full_time_indices] = 1

        shaded_ratio = shadow_matrix.sum() / shadow_matrix.size
        print(f"   遮阴比例: {shaded_ratio*100:.2f}%")

        return pd.DataFrame(shadow_matrix, columns=times)

    def calculate_pv_power_gpu(self, times, points_xy, vehicle_azimuths,
                               weather_data=None, tilt=5, height=1.5):
        """
        GPU加速的光伏功率计算

        Parameters
        ----------
        times : pandas.DatetimeIndex
            时间序列
        points_xy : numpy.ndarray
            车辆位置坐标 (N, 2)
        vehicle_azimuths : numpy.ndarray
            车辆朝向角度(度) (N,)
        weather_data : pandas.DataFrame, optional
            天气数据
        tilt : float
            光伏板倾角(度)
        height : float
            车顶高度(米)

        Returns
        -------
        pandas.DataFrame
            发电功率和相关参数
        """
        n_points = len(points_xy)

        print("\n" + "="*60)
        print("💡 GPU加速光伏功率计算")
        print("="*60)

        # 获取太阳位置
        solar_position = self.get_sun_position_pvlib(times)

        # GPU加速阴影计算
        shadow_matrix = self.calculate_shadows_batch_gpu(points_xy, height, times)

        # 获取辐照度数据
        print("\n☀️  处理辐照度数据...")
        irrad_components = self.get_irradiance_components(times, weather_data)

        # 温度和风速
        if weather_data is not None and 'temp_air' in weather_data.columns:
            temp_air = weather_data['temp_air'].reindex(times, method='nearest').values
            wind_speed = weather_data['wind_speed'].reindex(times, method='nearest').values
        else:
            temp_air = np.full(len(times), 25.0)
            wind_speed = np.full(len(times), 2.0)

        # 批量计算发电功率
        print(f"\n⚡ 批量计算发电功率...")
        print(f"   位置数: {n_points:,}")
        print(f"   批处理大小: {self.batch_size}")

        results = []

        # 批处理循环
        for batch_start in tqdm(range(0, n_points, self.batch_size), desc="   计算进度"):
            batch_end = min(batch_start + self.batch_size, n_points)

            for i in range(batch_start, batch_end):
                surface_azimuth = vehicle_azimuths[i]

                # 使用pvlib计算POA
                poa_irradiance = pvlib.irradiance.get_total_irradiance(
                    surface_tilt=tilt,
                    surface_azimuth=surface_azimuth,
                    solar_zenith=solar_position['apparent_zenith'],
                    solar_azimuth=solar_position['azimuth'],
                    dni=irrad_components['dni'],
                    ghi=irrad_components['ghi'],
                    dhi=irrad_components['dhi'],
                    model='isotropic'
                )

                # 应用阴影
                is_shaded = shadow_matrix.iloc[i].values
                poa_global = poa_irradiance['poa_global'].values * (1 - is_shaded)
                poa_direct = poa_irradiance['poa_direct'].values * (1 - is_shaded)
                poa_diffuse = poa_irradiance['poa_diffuse'].values

                # 计算电池温度
                cell_temp = temperature.sapm_cell(
                    poa_global=poa_global,
                    temp_air=temp_air,
                    wind_speed=wind_speed,
                    a=-3.56,
                    b=-0.075,
                    deltaT=3
                )

                # GPU加速功率计算（如果可用）
                if self.device.type == 'cuda' and TORCH_AVAILABLE:
                    with torch.no_grad():
                        poa_global_gpu = torch.from_numpy(poa_global).float().to(self.device)
                        cell_temp_gpu = torch.from_numpy(cell_temp).float().to(self.device)

                        # DC功率
                        gamma_pdc = self.module_parameters['gamma_pdc']
                        temp_ref = self.module_parameters['temp_ref']
                        pdc0_per_m2 = 1000 * self.panel_efficiency

                        temp_correction = 1 + gamma_pdc * (cell_temp_gpu - temp_ref)
                        dc_power_gpu = (poa_global_gpu / 1000) * pdc0_per_m2 * self.panel_area * temp_correction
                        dc_power_gpu = torch.clamp(dc_power_gpu, min=0)

                        # AC功率
                        eta_inv = self.inverter_parameters['eta_inv_nom']
                        ac_power_gpu = dc_power_gpu * eta_inv

                        # 转回CPU
                        dc_power = dc_power_gpu.cpu().numpy()
                        ac_power = ac_power_gpu.cpu().numpy()
                else:
                    # CPU计算
                    gamma_pdc = self.module_parameters['gamma_pdc']
                    temp_ref = self.module_parameters['temp_ref']
                    pdc0_per_m2 = 1000 * self.panel_efficiency

                    temp_correction = 1 + gamma_pdc * (cell_temp - temp_ref)
                    dc_power = (poa_global / 1000) * pdc0_per_m2 * self.panel_area * temp_correction
                    dc_power = np.clip(dc_power, 0, None)

                    eta_inv = self.inverter_parameters['eta_inv_nom']
                    ac_power = dc_power * eta_inv

                # 保存结果
                result_df = pd.DataFrame({
                    'time': times,
                    'location_idx': i,
                    'vehicle_azimuth': surface_azimuth,
                    'is_shaded': is_shaded,
                    'poa_global': poa_global,
                    'poa_direct': poa_direct,
                    'poa_diffuse': poa_diffuse,
                    'cell_temp': cell_temp,
                    'dc_power': dc_power,
                    'ac_power': ac_power,
                })
                results.append(result_df)

        print(f"   ✅ 计算完成")
        return pd.concat(results, ignore_index=True)

    def process_trajectory(self, trajectory_df, weather_data=None):
        """
        处理完整的轨迹数据（重写以使用GPU加速）

        Parameters
        ----------
        trajectory_df : pandas.DataFrame
            轨迹数据，必须包含: 'datetime', 'lng', 'lat', 'angle'
        weather_data : pandas.DataFrame, optional
            天气数据

        Returns
        -------
        pandas.DataFrame
            包含发电量的轨迹数据
        """
        print("\n" + "="*60)
        print("🚗 处理轨迹数据")
        print("="*60)
        print(f"轨迹点数: {len(trajectory_df):,}")

        # 数据预处理
        trajectory_df = trajectory_df.copy()
        trajectory_df['datetime'] = pd.to_datetime(trajectory_df['datetime'])
        trajectory_df = trajectory_df.sort_values('datetime').reset_index(drop=True)

        # 生成完整时间序列
        start_time = trajectory_df['datetime'].min().floor(f'{self.time_resolution_minutes}min')
        end_time = trajectory_df['datetime'].max().ceil(f'{self.time_resolution_minutes}min')
        full_times = pd.date_range(start_time, end_time,
                                   freq=f'{self.time_resolution_minutes}min')

        print(f"时间范围: {start_time} 至 {end_time}")
        print(f"时间分辨率: {self.time_resolution_minutes} 分钟")
        print(f"总时间点数: {len(full_times):,}")

        # 重采样轨迹
        trajectory_df.set_index('datetime', inplace=True)
        resampled = trajectory_df.resample(f'{self.time_resolution_minutes}min').first()
        resampled = resampled.reindex(full_times, method='ffill')
        resampled = resampled.dropna(subset=['lng', 'lat', 'angle'])

        if len(resampled) == 0:
            print("⚠️  警告: 重采样后没有有效数据")
            return pd.DataFrame()

        # 转换坐标
        x, y = self.gps_to_model_coords(resampled['lng'].values, resampled['lat'].values)
        points_xy = np.column_stack([x, y])
        vehicle_azimuths = resampled['angle'].values

        # GPU加速计算发电功率
        power_results = self.calculate_pv_power_gpu(
            times=resampled.index,
            points_xy=points_xy,
            vehicle_azimuths=vehicle_azimuths,
            weather_data=weather_data
        )

        # 合并结果
        power_results['datetime'] = power_results['time']
        power_results.set_index('datetime', inplace=True)

        merged = resampled.join(power_results[['is_shaded', 'poa_global', 'cell_temp',
                                              'dc_power', 'ac_power']])

        # 计算能量
        merged['time_delta_hours'] = self.time_resolution_minutes / 60.0
        merged['energy_kwh'] = merged['ac_power'] / 1000 * merged['time_delta_hours']

        merged.reset_index(inplace=True)

        print("\n✅ 轨迹处理完成")
        return merged


if __name__ == "__main__":
    print("GPU加速光伏计算器模块")
    print("请使用 main_pv_calculation_gpu.py 运行完整流程")
