import pandas as pd
import os

def filter_charging_data(input_file='整理完充放电信息.csv', output_dir='charging_data_by_vehicle'):
    """
    从充放电信息CSV中提取chargestatus=1的数据，并按车辆ID分别输出

    参数:
        input_file: 输入CSV文件路径
        output_dir: 输出目录路径
    """
    # 读取CSV文件
    print(f"正在读取文件: {input_file}")
    df = pd.read_csv(input_file)

    # 打印数据基本信息
    print(f"总数据行数: {len(df)}")
    print(f"列名: {df.columns.tolist()}")

    # 筛选chargestatus=1的数据
    charging_df = df[df['vehicledata_chargestatus'] == 1]
    print(f"充电状态为1的数据行数: {len(charging_df)}")

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    # 按车辆ID分组并输出
    vehicle_ids = charging_df['vin'].unique()
    print(f"共有 {len(vehicle_ids)} 辆车有充电记录")

    for vehicle_id in vehicle_ids:
        # 提取该车辆的所有充电数据
        vehicle_data = charging_df[charging_df['vin'] == vehicle_id]

        # 生成输出文件名
        output_file = os.path.join(output_dir, f'{vehicle_id}_charging.csv')

        # 保存到CSV
        vehicle_data.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"车辆 {vehicle_id}: {len(vehicle_data)} 条充电记录 -> {output_file}")

    print(f"\n处理完成！所有文件已保存到 {output_dir} 目录")

    # 返回统计信息
    return {
        'total_records': len(df),
        'charging_records': len(charging_df),
        'vehicle_count': len(vehicle_ids)
    }


if __name__ == '__main__':
    # 运行过滤函数
    stats = filter_charging_data()

    # 打印统计信息
    print("\n=== 统计信息 ===")
    print(f"原始数据总行数: {stats['total_records']}")
    print(f"充电状态为1的行数: {stats['charging_records']}")
    print(f"涉及车辆数量: {stats['vehicle_count']}")
