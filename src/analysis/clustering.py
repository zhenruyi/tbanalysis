# -*- coding: utf-8 -*-
"""
用户分群分析模块 - 基于RFM模型的K-means聚类分析
版本: 2.0
核心功能:
1. 数据标准化处理
2. 无监督学习聚类
3. 可视化呈现
4. 结果持久化存储
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from src.utils.config import load_config
from src.utils.database import create_db_connection, ensure_database_exists, write_to_database

config = load_config()
db_config = config['database']
ensure_database_exists(db_config)
engine = create_db_connection(db_config, include_dbname=True)

rfm_df = pd.read_sql("""
    SELECT user_id, recency, frequency, monetary
    FROM (
        SELECT
            user_id,
            DATEDIFF('2017-12-04', MAX(date)) AS recency,
            COUNT(DISTINCT date) AS frequency,
            COUNT(date) AS monetary
        FROM sampling
        WHERE behavior = 'buy'
        GROUP BY user_id
    ) AS base
""", con=engine)


# ------------------------- 数据标准化 -------------------------
# 创建标准化器实例
# 功能：将不同量纲的特征转化为标准正态分布（μ=0, σ=1）
# 原理：z = (x - μ) / σ
scaler = StandardScaler()

# 执行标准化转换（核心操作）
# 输入：原始RFM数据（n_samples=用户数, n_features=3）
# 输出：标准化后的数据矩阵（shape保持相同）
# 注意：只选择数值型特征，排除user_id等标识列
rfm_scaled = scaler.fit_transform(
    rfm_df[['recency','frequency','monetary']]  # 输入特征矩阵
)

"""
标准化效果示例：
原始数据 → 标准化后
Recency: 30天 → 1.2（比平均值晚1.2个标准差）
Frequency: 5次 → 0.8（比平均值高0.8个标准差）
Monetary: 1000元 → -0.5（比平均值低0.5个标准差）
"""


# ------------------------- K-means聚类 -------------------------
# 创建K-means模型实例
# 参数说明：
# n_clusters=4    → 预设分群数量（可通过肘部法则优化）
# random_state=42  → 随机种子（保证结果可复现）
kmeans = KMeans(
    n_clusters=4,       # 最终生成的用户群体数量
    random_state=42,    # 控制初始质心随机性
    max_iter=300,       # 最大迭代次数（确保收敛）
    n_init=10           # 不同质心初始化次数（避免局部最优）
)

# 执行聚类预测（核心操作）
# 流程：
# 1. 随机初始化4个质心
# 2. 计算每个用户到质心的距离
# 3. 分配用户到最近质心
# 4. 重新计算质心位置
# 5. 重复2-4直到质心稳定
rfm_df['cluster'] = kmeans.fit_predict(rfm_scaled)  # 返回每个用户的群体编号

"""
聚类结果解析：
cluster 0 → 高价值活跃用户（高频次、低间隔）
cluster 1 → 流失风险用户（高间隔、低频次）
cluster 2 → 潜力用户（中等频次）
cluster 3 → 新用户（低消费总额）
"""

# ------------------------- 可视化与存储 -------------------------
# 创建画布（设置尺寸为10x6英寸）
plt.figure(figsize=(10,6))

# 绘制散点图（核心可视化）
# 参数说明：
# x=购买频次，y=最近购买间隔，c=群体编号，cmap=颜色映射
plt.scatter(
    rfm_df['frequency'],      # X轴：购买频次（标准化前原始值）
    rfm_df['recency'],        # Y轴：最近购买间隔（天）
    c=rfm_df['cluster'],      # 颜色编码：用户群体划分
    cmap='viridis',           # 色图方案（黄-蓝渐变）
    alpha=0.6,                # 透明度（避免重叠点遮挡）
    edgecolors='w',           # 点边界颜色（增强区分度）
    s=50                      # 点大小
)

# 添加标签与标题
plt.xlabel('Purchase Frequency', fontsize=12)  # X轴标签（字号12）
plt.ylabel('Days Since Last Purchase', fontsize=12)  # Y轴标签
plt.title('Customer Segmentation by RFM', fontsize=14, pad=20)  # 标题（下边距20）

# 添加颜色条（群体编号解释）
plt.colorbar(label='Cluster Group')  # 右侧颜色条标注

# 设置网格线（提升可读性）
plt.grid(True, linestyle='--', alpha=0.5)

# 保存可视化结果
# 参数说明：
# dpi=300 → 图像分辨率（适合印刷）
# bbox_inches='tight' → 自动裁剪白边
plt.savefig(
    '../../outputs/figures/cluster_plot.png',  # 存储路径
    dpi=300,
    bbox_inches='tight'
)

# 关闭当前图像（避免内存泄漏）
plt.close()

# 保存聚类结果到CSV
rfm_df.to_csv(
    '../../data/analytics/clusters.csv',  # 文件路径
    index=False,           # 不保存行索引
    encoding='utf-8-sig'   # 支持中文编码
)

# 写入数据库（长期存储）
write_to_database(
    rfm_df,                # 数据集
    db_config,             # 数据库配置
    'clusters',            # 目标表名
)