"""
GPU加速的车顶光伏发电量计算器
继承自pv_generation_pvlib.py的SolarPVCalculator
使用PyTorch + triro/OptiX进行完整GPU加速流水线
"""

import os
import pandas as pd
import numpy as np
import trimesh
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

    使用完整GPU流水线加速:
    - GPU批量生成光线（torch）
    - GPU光线追踪（triro/OptiX）
    - GPU批量功率计算（torch）
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
        GPU加速的批量阴影计算（完整GPU流水线）

        核心优化:
        1. torch在GPU上批量生成光线
        2. trimesh + triro/OptiX GPU光线追踪（无CPU传输）
        3. GPU上解析结果

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
            print(f"   ⚡ GPU完整流水线加速...")

            with torch.no_grad():
                # 转移到GPU
                query_points_gpu = torch.from_numpy(query_points).float().to(self.device)
                sun_vectors_gpu = torch.from_numpy(sun_vectors).float().to(self.device)

                # 批量计算光线起点和方向
                # shape: (n_times, n_points, 3)
                ray_origins_batch = query_points_gpu.unsqueeze(0) - sun_vectors_gpu.unsqueeze(1) * 5e5
                ray_directions_batch = sun_vectors_gpu.unsqueeze(1).expand(n_times, n_points, 3)

                # 重塑为 (n_total_rays, 3)
                all_ray_origins_gpu = ray_origins_batch.reshape(-1, 3)
                all_ray_directions_gpu = ray_directions_batch.reshape(-1, 3)

            print(f"   ✅ GPU光线生成完成")
            print(f"   🎯 GPU光线追踪（triro/OptiX）...")

            # GPU光线追踪（triro补丁自动使用OptiX）
            # 注意：triro可以直接处理torch tensor
            try:
                location_np, ray_idx_np, tri_idx_np = self.building_trimesh.ray.intersects_location(
                    all_ray_origins_gpu,  # torch.Tensor on CUDA
                    all_ray_directions_gpu,  # torch.Tensor on CUDA
                    multiple_hits=False
                )

                # ray_idx_np是相交光线的索引数组
                intersection_rays = ray_idx_np

                print(f"   ✅ GPU追踪完成，检测到 {len(intersection_rays):,} 个遮挡")

            except Exception as e:
                print(f"   ⚠️  GPU光线追踪失败，回退到CPU: {e}")
                # 回退到CPU
                all_ray_origins = all_ray_origins_gpu.cpu().numpy()
                all_ray_directions = all_ray_directions_gpu.cpu().numpy()

                location_np, ray_idx_np, tri_idx_np = self.building_trimesh.ray.intersects_location(
                    all_ray_origins,
                    all_ray_directions,
                    multiple_hits=False
                )
                intersection_rays = ray_idx_np

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

            # CPU光线追踪
            print(f"   🎯 CPU光线追踪...")
            location_np, ray_idx_np, tri_idx_np = self.building_trimesh.ray.intersects_location(
                all_ray_origins,
                all_ray_directions,
                multiple_hits=False
            )
            intersection_rays = ray_idx_np
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

    def _calculate_poa_vectorized(self, surface_tilt, surface_azimuth,
                                   solar_zenith, solar_azimuth,
                                   dni, dhi, ghi, albedo=0.2):
        """
        手动实现POA计算（与RealSceneDL源码一致的各向同性天空模型）

        支持GPU加速的向量化计算

        Parameters
        ----------
        surface_tilt : float
            光伏板倾角(度)
        surface_azimuth : float
            光伏板方位角(度)
        solar_zenith : numpy.ndarray
            太阳天顶角(度) shape: (n_times,)
        solar_azimuth : numpy.ndarray
            太阳方位角(度) shape: (n_times,)
        dni : numpy.ndarray
            直射辐照度(W/m²) shape: (n_times,)
        dhi : numpy.ndarray
            散射辐照度(W/m²) shape: (n_times,)
        ghi : numpy.ndarray
            总水平辐照度(W/m²) shape: (n_times,)
        albedo : float
            地面反射率，默认0.2

        Returns
        -------
        dict
            包含 'poa_global', 'poa_direct', 'poa_diffuse' 的字典
            每个值的shape: (n_times,)
        """
        # 转换为弧度
        surface_tilt_rad = np.deg2rad(surface_tilt)
        surface_azimuth_rad = np.deg2rad(surface_azimuth)
        solar_zenith_rad = np.deg2rad(solar_zenith)
        solar_azimuth_rad = np.deg2rad(solar_azimuth)

        # 🔥 GPU加速计算（如果可用）
        if self.device.type == 'cuda' and TORCH_AVAILABLE:
            import torch
            with torch.no_grad():
                # 转换为torch张量
                solar_zenith_t = torch.from_numpy(solar_zenith_rad).float().to(self.device)
                solar_azimuth_t = torch.from_numpy(solar_azimuth_rad).float().to(self.device)
                dni_t = torch.from_numpy(dni).float().to(self.device)
                dhi_t = torch.from_numpy(dhi).float().to(self.device)
                ghi_t = torch.from_numpy(ghi).float().to(self.device)

                # 计算入射角余弦值（AOI: Angle of Incidence）
                # cos(AOI) = cos(zenith) * cos(tilt) + sin(zenith) * sin(tilt) * cos(azimuth_sun - azimuth_surf)
                cos_aoi = (torch.cos(solar_zenith_t) * np.cos(surface_tilt_rad) +
                          torch.sin(solar_zenith_t) * np.sin(surface_tilt_rad) *
                          torch.cos(solar_azimuth_t - surface_azimuth_rad))

                # 直射分量：POA_direct = DNI × cos(AOI)，但AOI必须 <= 90度
                poa_direct_t = torch.clamp(dni_t * cos_aoi, min=0.0)

                # 散射分量（各向同性天空模型）：POA_diffuse = DHI × (1 + cos(tilt)) / 2
                poa_diffuse_t = dhi_t * (1 + np.cos(surface_tilt_rad)) / 2

                # 反射分量：POA_reflected = GHI × albedo × (1 - cos(tilt)) / 2
                poa_reflected_t = ghi_t * albedo * (1 - np.cos(surface_tilt_rad)) / 2

                # 总POA
                poa_global_t = poa_direct_t + poa_diffuse_t + poa_reflected_t

                # 转回CPU numpy
                poa_global = poa_global_t.cpu().numpy()
                poa_direct = poa_direct_t.cpu().numpy()
                poa_diffuse = poa_diffuse_t.cpu().numpy()
        else:
            # CPU计算
            # 计算入射角余弦值
            cos_aoi = (np.cos(solar_zenith_rad) * np.cos(surface_tilt_rad) +
                      np.sin(solar_zenith_rad) * np.sin(surface_tilt_rad) *
                      np.cos(solar_azimuth_rad - surface_azimuth_rad))

            # 直射分量
            poa_direct = np.maximum(dni * cos_aoi, 0.0)

            # 散射分量
            poa_diffuse = dhi * (1 + np.cos(surface_tilt_rad)) / 2

            # 反射分量
            poa_reflected = ghi * albedo * (1 - np.cos(surface_tilt_rad)) / 2

            # 总POA
            poa_global = poa_direct + poa_diffuse + poa_reflected

        return {
            'poa_global': poa_global,
            'poa_direct': poa_direct,
            'poa_diffuse': poa_diffuse
        }

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
        print(f"\n⚡ 批量计算发电功率（向量化POA）...")
        print(f"   位置数: {n_points:,}")
        print(f"   批处理大小: {self.batch_size}")

        # 预提取数组（避免重复访问DataFrame）
        solar_zenith = solar_position['apparent_zenith'].values
        solar_azimuth = solar_position['azimuth'].values
        dni = irrad_components['dni'].values
        dhi = irrad_components['dhi'].values
        ghi = irrad_components['ghi'].values

        results = []

        # 批处理循环
        for batch_start in tqdm(range(0, n_points, self.batch_size), desc="   计算进度"):
            batch_end = min(batch_start + self.batch_size, n_points)

            for i in range(batch_start, batch_end):
                surface_azimuth = vehicle_azimuths[i]

                # 🔥 使用向量化POA计算（与RealSceneDL源码一致）
                poa_result = self._calculate_poa_vectorized(
                    surface_tilt=tilt,
                    surface_azimuth=surface_azimuth,
                    solar_zenith=solar_zenith,
                    solar_azimuth=solar_azimuth,
                    dni=dni,
                    dhi=dhi,
                    ghi=ghi,
                    albedo=0.2
                )

                # 应用阴影
                is_shaded = shadow_matrix.iloc[i].values
                poa_global = poa_result['poa_global'] * (1 - is_shaded)
                poa_direct = poa_result['poa_direct'] * (1 - is_shaded)
                poa_diffuse = poa_result['poa_diffuse']  # 散射分量不受直射阴影影响

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
