import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_original_data(df, save_path='original_data.png'):
    """
    绘制填充后的原始农业数据
    df: 已填充缺失值的 DataFrame（含 date 索引）
    """
    # 确保 date 是索引
    if 'date' in df.columns:
        df = df.set_index('date')

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
    plt.rcParams['axes.unicode_minus'] = False

    # 创建子图 (4行2列)
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    fig.suptitle('糖料作物多源时序数据 (填充后)', fontsize=16, fontweight='bold')

    # 1. 气象数据
    axes[0, 0].plot(df.index, df['temperature'], 'b-', linewidth=1.2)
    axes[0, 0].set_title('日均气温 (°C)')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].bar(df.index, df['precipitation'], color='c', width=1.0, alpha=0.7)
    axes[0, 1].set_title('日降水量 (mm)')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(df.index, df['sunshine_hours'], 'y-', linewidth=1.2)
    axes[1, 0].set_title('日照时数 (h)')
    axes[1, 0].grid(True, alpha=0.3)

    # 2. 土壤数据
    axes[1, 1].plot(df.index, df['soil_moisture'], 'g-', linewidth=1.5)
    axes[1, 1].set_title('土壤墒情 (%)')
    axes[1, 1].grid(True, alpha=0.3)

    # 3. 作物生长指标（核心！）
    axes[2, 0].plot(df.index, df['plant_height'], 'r-o', markersize=3, linewidth=2)
    axes[2, 0].set_title('株高 (cm)', color='red', fontweight='bold')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_ylabel('高度 (cm)')

    axes[2, 1].plot(df.index, df['leaf_area_index'], 'm-s', markersize=3, linewidth=1.5)
    axes[2, 1].set_title('叶面积指数 (LAI)')
    axes[2, 1].grid(True, alpha=0.3)

    axes[3, 0].plot(df.index, df['stem_diameter'], 'k-^', markersize=3, linewidth=1.5)
    axes[3, 0].set_title('茎径 (mm)')
    axes[3, 0].grid(True, alpha=0.3)
    axes[3, 0].set_xlabel('日期')

    # 4. 株高 + 气象叠加（关键洞察！）
    ax4 = axes[3, 1]
    ax4.plot(df.index, df['plant_height'], 'r-', linewidth=2.5, label='株高 (cm)')
    ax4.set_ylabel('株高 (cm)', color='red')
    ax4.tick_params(axis='y', labelcolor='red')

    # 叠加降水量（次坐标轴）
    ax4_twin = ax4.twinx()
    ax4_twin.bar(df.index, df['precipitation'], color='c', alpha=0.3, width=1.0, label='降水 (mm)')
    ax4_twin.set_ylabel('降水量 (mm)', color='c')
    ax4_twin.tick_params(axis='y', labelcolor='c')

    ax4.set_title('株高与降水关系')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    ax4.set_xlabel('日期')

    # 优化日期显示
    for ax in axes.flat:
        if hasattr(ax, 'xaxis'):
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为标题留空间
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()