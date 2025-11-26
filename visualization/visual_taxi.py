import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from tqdm import tqdm

# ================== Nature期刊绘图风格配置 ==================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Nature风格配色
NATURE_BLUE = '#4472C4'
NATURE_ORANGE = '#ED7D31'
ALPHA_FILL = 0.2

# ================== 数据读取函数 ==================
def get_all_pv_files(output_dir, max_files=None):
    """
    获取所有PV发电数据文件

    参数:
        output_dir: 输出目录路径
        max_files: 最大文件数量限制（None表示不限制）
    """
    output_path = Path(output_dir)
    pv_files = list(output_path.glob('*_pv_generation.csv'))
    print(f"找到 {len(pv_files)} 个PV发电数据文件")

    if max_files is not None and len(pv_files) > max_files:
        pv_files = pv_files[:max_files]
        print(f"限制处理前 {max_files} 个文件")

    return pv_files

def load_all_data(pv_files, target_date):
    """
    一次性加载所有数据，同时返回日视图和年视图数据（内存优化版本）

    参数:
        pv_files: CSV文件列表
        target_date: 目标日期字符串，如 '2019-07-15'

    返回:
        daily_df: 指定日期的按时间聚合数据
        annual_df: 全年的按日期聚合数据
    """
    target_date_obj = pd.to_datetime(target_date).date()

    # 存储日视图数据
    daily_power_data = []
    # 存储年视图数据
    annual_energy_data = {}

    print(f"\n正在加载所有数据（日期：{target_date}）...")
    for file in tqdm(pv_files, desc="读取文件"):
        try:
            # 读取所需列
            df = pd.read_csv(file, usecols=['datetime', 'dc_power', 'energy_kwh'])

            # 转换时间列（数据已包含时区信息）
            df['datetime'] = pd.to_datetime(df['datetime'])
            # 移除时区信息以便正确处理
            df['datetime'] = df['datetime'].dt.tz_localize(None)

            # ===== 处理日视图数据 =====
            df_day = df[df['datetime'].dt.date == target_date_obj].copy()
            if len(df_day) > 0:
                daily_power_data.append(df_day[['datetime', 'dc_power']])

            # ===== 处理年视图数据 =====
            df['date'] = df['datetime'].dt.date
            daily_sum = df.groupby('date')['energy_kwh'].sum()
            vehicle_id = file.stem.replace('_pv_generation', '')
            annual_energy_data[vehicle_id] = daily_sum

        except Exception as e:
            print(f"读取文件 {file.name} 时出错: {e}")
            continue

    # ===== 生成日视图结果 =====
    if len(daily_power_data) == 0:
        print(f"警告: 未找到 {target_date} 的有效数据")
        daily_df = None
    else:
        combined_daily = pd.concat(daily_power_data, ignore_index=True)
        daily_df = combined_daily.groupby('datetime')['dc_power'].agg([
            ('mean', 'mean'),
            ('std', 'std'),
            ('min', 'min'),
            ('max', 'max'),
            ('count', 'count')
        ]).reset_index()
        print(f"日视图: 成功加载 {len(daily_power_data)} 个车辆的数据，时间点数量: {len(daily_df)}")

    # ===== 生成年视图结果 =====
    if len(annual_energy_data) == 0:
        print("警告: 未找到有效的年度数据")
        annual_df = None
    else:
        energy_df = pd.DataFrame(annual_energy_data)
        stats_df = pd.DataFrame({
            'mean': energy_df.mean(axis=1),
            'std': energy_df.std(axis=1),
            'min': energy_df.min(axis=1),
            'max': energy_df.max(axis=1),
            'sum': energy_df.sum(axis=1),  # 添加总和列用于累计图
            'count': energy_df.count(axis=1)
        })
        annual_df = stats_df.reset_index()
        annual_df.columns = ['date', 'mean', 'std', 'min', 'max', 'sum', 'count']
        annual_df['date'] = pd.to_datetime(annual_df['date'])
        annual_df = annual_df.sort_values('date').reset_index(drop=True)
        print(f"年视图: 成功加载 {len(annual_energy_data)} 个车辆的数据")
        print(f"        日期范围: {annual_df['date'].min().date()} 到 {annual_df['date'].max().date()}")

    return daily_df, annual_df

# ================== 绘图函数 ==================
def plot_daily_generation(daily_df, target_date, save_path=None):
    """
    绘制日发电功率图（Nature期刊风格）

    参数:
        daily_df: 包含均值和标准差的DataFrame
        target_date: 日期字符串
        save_path: 保存路径（可选）
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # 绘制均值线
    ax.plot(daily_df['datetime'], daily_df['mean'],
            color=NATURE_BLUE, linewidth=2, label='Mean', zorder=3)

    # 绘制标准差范围
    ax.fill_between(daily_df['datetime'],
                     daily_df['mean'] - daily_df['std'],
                     daily_df['mean'] + daily_df['std'],
                     color=NATURE_BLUE, alpha=ALPHA_FILL,
                     label='±1 Std Dev', zorder=1)

    # 绘制最大最小值范围（更浅）
    ax.fill_between(daily_df['datetime'],
                     daily_df['min'],
                     daily_df['max'],
                     color=NATURE_BLUE, alpha=ALPHA_FILL*0.5,
                     label='Min-Max Range', zorder=0)

    # 设置标题和标签
    ax.set_title(f'Daily PV Power Generation - {target_date}',
                 fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel('Time of Day', fontsize=11, labelpad=10)
    ax.set_ylabel('DC Power (W)', fontsize=11, labelpad=10)

    # 设置图例
    ax.legend(loc='upper left', frameon=True, framealpha=0.9,
              edgecolor='gray', fontsize=9)

    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # 格式化x轴
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')

    # 设置y轴从0开始
    ax.set_ylim(bottom=0)

    # 设置边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")

    plt.show()

    # 打印统计信息
    print(f"\n{target_date} 统计信息:")
    print(f"平均峰值功率: {daily_df['mean'].max():.2f} W")
    print(f"最大记录功率: {daily_df['max'].max():.2f} W")
    print(f"数据车辆数: {int(daily_df['count'].max())}")

def plot_annual_generation(annual_df, save_path=None):
    """
    绘制年度发电量图（Nature期刊风格）

    参数:
        annual_df: 包含每日发电量均值和标准差的DataFrame
        save_path: 保存路径（可选）
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # 绘制均值线
    ax.plot(annual_df['date'], annual_df['mean'],
            color=NATURE_ORANGE, linewidth=2, label='Mean', zorder=3)

    # 绘制标准差范围
    ax.fill_between(annual_df['date'],
                     annual_df['mean'] - annual_df['std'],
                     annual_df['mean'] + annual_df['std'],
                     color=NATURE_ORANGE, alpha=ALPHA_FILL,
                     label='±1 Std Dev', zorder=1)

    # 绘制最大最小值范围
    ax.fill_between(annual_df['date'],
                     annual_df['min'],
                     annual_df['max'],
                     color=NATURE_ORANGE, alpha=ALPHA_FILL*0.5,
                     label='Min-Max Range', zorder=0)

    # 设置标题和标签
    year_start = annual_df['date'].min().year
    year_end = annual_df['date'].max().year
    year_label = f'{year_start}' if year_start == year_end else f'{year_start}-{year_end}'

    ax.set_title(f'Annual Daily PV Energy Generation',
                 fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel('Date', fontsize=11, labelpad=10)
    ax.set_ylabel('Daily Energy Generation (kWh)', fontsize=11, labelpad=10)

    # 设置图例
    ax.legend(loc='upper left', frameon=True, framealpha=0.9,
              edgecolor='gray', fontsize=9)

    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # 格式化x轴
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 设置x轴范围（手动指定）
    ax.set_xlim(pd.Timestamp('2018-12-01'), pd.Timestamp('2020-01-31'))

    # 设置y轴从0开始
    ax.set_ylim(bottom=0)

    # 设置边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")

    plt.show()

    # 打印统计信息
    print(f"\n年度统计信息:")
    print(f"平均每日发电量: {annual_df['mean'].mean():.3f} kWh")
    print(f"最大日发电量(均值): {annual_df['mean'].max():.3f} kWh")
    print(f"最大日发电量(记录): {annual_df['max'].max():.3f} kWh")
    print(f"总天数: {len(annual_df)}")
    print(f"平均车辆数: {annual_df['count'].mean():.0f}")

def plot_cumulative_generation(annual_df, save_path=None):
    """
    绘制年度累计发电总量图（Nature期刊风格）

    参数:
        annual_df: 包含每日发电量总和的DataFrame
        save_path: 保存路径（可选）
    """
    # 计算累计总量
    cumulative_df = annual_df.copy()
    cumulative_df['cumsum_total'] = cumulative_df['sum'].cumsum()

    fig, ax = plt.subplots(figsize=(10, 5))

    # 绘制累计总量曲线
    ax.plot(cumulative_df['date'], cumulative_df['cumsum_total'],
            color=NATURE_BLUE, linewidth=2.5, label='Total Fleet', zorder=3)

    # 设置标题和标签
    ax.set_title(f'Annual Cumulative PV Energy Generation',
                 fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel('Date', fontsize=11, labelpad=10)
    ax.set_ylabel('Cumulative Energy Generation (kWh)', fontsize=11, labelpad=10)

    # 设置图例
    ax.legend(loc='upper left', frameon=True, framealpha=0.9,
              edgecolor='gray', fontsize=9)

    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # 格式化x轴
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 设置x轴范围（手动指定）
    ax.set_xlim(pd.Timestamp('2018-12-01'), pd.Timestamp('2020-02-01'))

    # 设置y轴从0开始
    ax.set_ylim(bottom=0)

    # 设置边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")

    plt.show()

    # 打印统计信息
    print(f"\n年度累计统计信息:")
    print(f"全年累计发电总量: {cumulative_df['cumsum_total'].iloc[-1]:.2f} kWh")
    print(f"车队规模: {int(annual_df['count'].max())} 辆")
    print(f"平均每辆车年发电量: {cumulative_df['cumsum_total'].iloc[-1] / annual_df['count'].max():.2f} kWh")

# ================== 主程序 ==================
if __name__ == '__main__':
    # 配置参数
    OUTPUT_DIR = '../output_taxi'
    TARGET_DATE = '2019-08-15'  # 可修改日期
    MAX_FILES = None  # 限制处理的文件数量（设置为None则处理全部）

    # 获取所有PV文件
    pv_files = get_all_pv_files(OUTPUT_DIR, max_files=MAX_FILES)

    if len(pv_files) == 0:
        print("错误: 未找到任何PV发电数据文件")
        exit(1)

    # ========== 加载所有数据（只读取一次文件）==========
    try:
        daily_data, annual_data = load_all_data(pv_files, TARGET_DATE)
    except Exception as e:
        print(f"加载数据时出错: {e}")
        exit(1)

    # ========== 保存年度数据 ==========
    if annual_data is not None:
        annual_csv_path = 'annual_statistics_taxi.csv'
        annual_data.to_csv(annual_csv_path, index=False, encoding='utf-8')
        print(f"\n✅ 年度统计数据已保存至: {annual_csv_path}")
        print(f"   包含列: {', '.join(annual_data.columns.tolist())}")
    else:
        print("\n⚠️ 年度数据为空，跳过保存")

    # ========== 1. 绘制日发电功率图 ==========
    if daily_data is not None:
        try:
            plot_daily_generation(daily_data, TARGET_DATE,
                                save_path=f'daily_pv_generation_{TARGET_DATE}_taxi.png')
        except Exception as e:
            print(f"绘制日发电图时出错: {e}")
    else:
        print(f"跳过日发电图绘制（无数据）")

    # ========== 2. 绘制年度发电量图 ==========
    if annual_data is not None:
        try:
            plot_annual_generation(annual_data,
                                 save_path='annual_pv_generation_taxi.png')
        except Exception as e:
            print(f"绘制年度发电图时出错: {e}")
    else:
        print(f"跳过年度发电图绘制（无数据）")

    # ========== 3. 绘制年度累计发电量图 ==========
    if annual_data is not None:
        try:
            plot_cumulative_generation(annual_data,
                                     save_path='cumulative_pv_generation_taxi.png')
        except Exception as e:
            print(f"绘制累计发电图时出错: {e}")
    else:
        print(f"跳过累计发电图绘制（无数据）")

    print("\n所有可视化完成!")