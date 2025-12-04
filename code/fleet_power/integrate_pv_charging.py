#!/usr/bin/env python3
"""
车顶光伏发电与充电数据集成

实现场景C：综合考虑充电前和充电期间的光伏发电
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple


def load_charging_data(file_path: str) -> pd.DataFrame:
    """加载充电数据并解析时间"""
    df = pd.read_csv(file_path)
    df['stime'] = pd.to_datetime(df['stime'])
    df['etime'] = pd.to_datetime(df['etime'])
    return df.sort_values('stime').reset_index(drop=True)


def load_pv_data(file_path: str) -> pd.DataFrame:
    """加载光伏发电数据并解析时间"""
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df.sort_values('datetime').reset_index(drop=True)


def load_vehicle_info(file_path: str) -> pd.DataFrame:
    """加载车型信息"""
    return pd.read_csv(file_path)


def get_battery_capacity(vin: str, vehicle_info: pd.DataFrame) -> float:
    """
    根据VIN查询电池容量

    返回: 电池容量 (kWh)
    """
    match = vehicle_info[vehicle_info['vin'] == vin]
    if match.empty:
        raise ValueError(f"找不到VIN {vin} 的车型信息")
    return match['energy_device_capacity'].iloc[0]


def preprocess_pv_data(pv_data: pd.DataFrame) -> pd.DataFrame:
    """预处理光伏数据：添加分钟级时间索引"""
    pv_data = pv_data.copy()
    pv_data['datetime_minute'] = pv_data['datetime'].dt.floor('min').dt.tz_localize(None)
    return pv_data


def preprocess_charging_data(charging_data: pd.DataFrame) -> pd.DataFrame:
    """预处理充电数据：添加分钟级时间索引"""
    charging_data = charging_data.copy()
    charging_data['stime_minute'] = charging_data['stime'].dt.floor('min')
    charging_data['etime_minute'] = charging_data['etime'].dt.floor('min')
    return charging_data


def calculate_pv_impact(
    charge: pd.Series,
    pv_data: pd.DataFrame,
    charge_index: int,
    all_charges: pd.DataFrame,
    battery_capacity: float,
    charging_efficiency: float
) -> dict:
    """
    计算单次充电的光伏影响

    返回包含所有计算字段的字典
    """
    # 1. 确定充电前的时间窗口
    if charge_index == 0:
        time_window_start = pv_data['datetime_minute'].min()
    else:
        time_window_start = all_charges.loc[charge_index - 1, 'etime_minute']

    time_window_end = charge['stime_minute']

    # 2. 计算充电前累积的光伏能量
    pv_before = pv_data[
        (pv_data['datetime_minute'] >= time_window_start) &
        (pv_data['datetime_minute'] < time_window_end)
    ]
    pv_energy_before_kwh = pv_before['energy_kwh'].sum() * charging_efficiency

    # 3. 计算调整后的起始SOC
    soc_gain_before = (pv_energy_before_kwh / battery_capacity) * 100
    adjusted_ssoc = min(charge['ssoc'] + soc_gain_before, 100.0)

    # 4. 计算充电期间的光伏能量
    pv_during = pv_data[
        (pv_data['datetime_minute'] >= charge['stime_minute']) &
        (pv_data['datetime_minute'] <= charge['etime_minute'])
    ]
    pv_energy_during_kwh = pv_during['energy_kwh'].sum() * charging_efficiency

    # 5. 计算原始和实际充电需求
    original_charge_demand_kwh = ((charge['esoc'] - charge['ssoc']) / 100) * battery_capacity
    actual_charge_demand_kwh = ((charge['esoc'] - adjusted_ssoc) / 100) * battery_capacity

    # 6. 计算净电网能量
    net_grid_energy_kwh = max(actual_charge_demand_kwh - pv_energy_during_kwh, 0)

    # 7. 计算调整后的结束SOC
    if net_grid_energy_kwh == 0 and pv_energy_during_kwh > actual_charge_demand_kwh:
        surplus_pv_kwh = pv_energy_during_kwh - actual_charge_demand_kwh
        additional_soc = (surplus_pv_kwh / battery_capacity) * 100
        adjusted_esoc = min(charge['esoc'] + additional_soc, 100.0)
    else:
        adjusted_esoc = charge['esoc']

    # 8. 判断充电是否被消除
    charging_eliminated = (adjusted_ssoc >= charge['esoc'])

    # 9. 返回完整结果
    return {
        # 原始字段
        'vin': charge['vin'],
        'stime': charge['stime'],
        'slon': charge['slon'],
        'slat': charge['slat'],
        'ssoc': charge['ssoc'],
        'vehicledata_chargestatus': charge['vehicledata_chargestatus'],
        'etime': charge['etime'],
        'elon': charge['elon'],
        'elat': charge['elat'],
        'esoc': charge['esoc'],

        # 新增字段
        'pv_energy_before_kwh': round(pv_energy_before_kwh, 4),
        'pv_energy_during_kwh': round(pv_energy_during_kwh, 4),
        'total_pv_contribution_kwh': round(pv_energy_before_kwh + pv_energy_during_kwh, 4),
        'adjusted_ssoc': round(adjusted_ssoc, 2),
        'adjusted_esoc': round(adjusted_esoc, 2),
        'original_charge_demand_kwh': round(original_charge_demand_kwh, 4),
        'net_grid_energy_kwh': round(net_grid_energy_kwh, 4),
        'charging_eliminated': charging_eliminated
    }


def integrate_pv_with_charging(
    charging_file: str,
    pv_generation_file: str,
    vehicle_info_file: str,
    output_file: str,
    charging_efficiency: float = 0.95
) -> pd.DataFrame:
    """
    整合光伏发电数据与充电数据

    参数:
        charging_file: 充电数据CSV路径
        pv_generation_file: 光伏发电数据CSV路径
        vehicle_info_file: 车型信息CSV路径
        output_file: 输出文件路径
        charging_efficiency: 充电效率 (默认0.95)

    返回:
        增强后的充电数据DataFrame
    """
    # 步骤1: 加载数据
    print("加载数据...")
    charging_data = load_charging_data(charging_file)
    pv_data = load_pv_data(pv_generation_file)
    vehicle_info = load_vehicle_info(vehicle_info_file)

    print(f"  充电记录: {len(charging_data)} 条")
    print(f"  光伏记录: {len(pv_data)} 条")
    print(f"  车型信息: {len(vehicle_info)} 条")

    # 步骤2: 获取电池容量
    battery_capacity = get_battery_capacity(charging_data['vin'].iloc[0], vehicle_info)
    print(f"  电池容量: {battery_capacity} kWh")

    # 步骤3: 时间对齐预处理
    print("预处理数据...")
    pv_data = preprocess_pv_data(pv_data)
    charging_data = preprocess_charging_data(charging_data)

    # 步骤4: 逐条处理充电记录
    print("计算光伏影响...")
    results = []
    for idx, charge in charging_data.iterrows():
        result = calculate_pv_impact(
            charge,
            pv_data,
            idx,
            charging_data,
            battery_capacity,
            charging_efficiency
        )
        results.append(result)

        # 进度指示
        if (idx + 1) % 10 == 0:
            print(f"  已处理 {idx + 1}/{len(charging_data)} 条记录")

    # 步骤5: 合并结果
    enhanced_data = pd.DataFrame(results)

    # 步骤6: 保存结果
    print(f"保存结果到: {output_file}")
    enhanced_data.to_csv(output_file, index=False, encoding='utf-8-sig')

    return enhanced_data


if __name__ == '__main__':
    # 示例用法
    result = integrate_pv_with_charging(
        charging_file='8e686785d827d59821b63a4348e1893e_charging.csv',
        pv_generation_file='8e686785d827d59821b63a4348e1893e_pv_generation.csv',
        vehicle_info_file='车型信息.csv',
        output_file='8e686785d827d59821b63a4348e1893e_charging_with_pv.csv',
        charging_efficiency=0.95
    )

    print(f"\n{'='*60}")
    print(f"处理完成! 共处理 {len(result)} 条充电记录")
    print(f"{'='*60}")
    print(f"\n统计信息:")
    print(f"  平均充电前光伏能量: {result['pv_energy_before_kwh'].mean():.4f} kWh")
    print(f"  平均充电期间光伏能量: {result['pv_energy_during_kwh'].mean():.4f} kWh")
    print(f"  平均总光伏贡献: {result['total_pv_contribution_kwh'].mean():.4f} kWh")
    print(f"  平均净电网能量: {result['net_grid_energy_kwh'].mean():.4f} kWh")
    print(f"  充电被消除的次数: {result['charging_eliminated'].sum()}")
    print(f"\n数据验证:")
    print(f"  所有adjusted_ssoc <= 100: {(result['adjusted_ssoc'] <= 100).all()}")
    print(f"  所有adjusted_esoc <= 100: {(result['adjusted_esoc'] <= 100).all()}")
    print(f"  所有net_grid_energy >= 0: {(result['net_grid_energy_kwh'] >= 0).all()}")
