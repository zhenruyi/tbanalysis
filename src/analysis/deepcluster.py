# -*- coding: utf-8 -*-
"""
淘宝用户深度聚类分析
技术栈：Python(特征工程+聚类建模) -> MySQL(结果存储) -> Tableau(可视化)
分析维度：结合用户活跃度、消费能力和行为偏好三大维度构建聚类特征

核心教学要点：
1. 时间序列特征工程构建方法
2. 聚类分析中的标准化处理与K值选择
3. 业务导向的特征设计（RFM扩展模型）
4. 生产级数据分析流程（数据处理->建模->存储->可视化）
"""

# 导入必要的库
import pandas as pd  # 数据处理分析库
import numpy as np  # 数值计算库
from sklearn.preprocessing import StandardScaler  # 数据标准化
from sklearn.cluster import KMeans  # K均值聚类算法
from sklearn.metrics import silhouette_score  # 聚类效果评估指标
import mysql.connector  # MySQL数据库连接
from datetime import datetime  # 时间处理


# ---------------------------- 数据加载与预处理 ----------------------------
def load_data(file_path):
    """
    功能：加载并预处理原始用户行为数据
    输入：csv文件路径
    输出：预处理后的DataFrame

    关键处理步骤：
    1. 指定数据类型优化内存使用
    2. 时间字段格式转换
    3. 分类字段类型优化
    """
    # 定义列数据类型（优化内存的关键步骤）
    dtype = {
        'user_id': 'int32',  # 用户ID用32位整型足够
        'item_id': 'int32',  # 商品ID
        'category_id': 'int32',  # 类目ID
        'behavior': 'category',  # 行为类型转为分类类型（优化内存与查询速度）
        'timestamp': 'str'  # 原始时间格式为字符串
    }

    # 读取CSV文件（注意：大数据量时可分块读取）
    df = pd.read_csv(file_path, dtype=dtype)

    # 时间格式转换（原始格式示例：2017-11-25 23:00:00）
    # 使用矢量化的pd.to_datetime替代循环，提升处理速度
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y/%m/%d %H:%M')

    return df


# ---------------------------- 特征工程模块 ----------------------------
def feature_engineering(df):
    """
    功能：构建用户行为特征矩阵
    输入：原始数据DataFrame
    输出：特征矩阵DataFrame

    特征体系设计：
    ┌───────────┬───────────────┬───────────────┐
    │ 活跃特征   │  消费特征     │  偏好特征     │
    ├───────────┼───────────────┼───────────────┤
    │ 最近活跃  │ 购买次数      │ 浏览深度      │
    │ 活跃天数  │ 加购转化率    │ 品类集中度    │
    │ 日均行为  │ 收藏转化率    │               │
    └───────────┴───────────────┴───────────────┘
    """
    # ========== 活跃特征 ==========
    # 计算当前时间（模拟分析时的时间点）
    current_time = df['timestamp'].max()  # 取数据中最新的时间戳作为"当前时间"

    # 按用户分组计算特征
    active_features = df.groupby('user_id').agg(
        last_active=('timestamp', lambda x: (current_time - x.max()).days),
        # 计算最近活跃间隔：用户最后一次行为距今的天数
        # x.max()获取用户最后行为时间，current_time - x.max()得到时间差
        # .days提取天数属性

        active_days=('timestamp', lambda x: x.dt.date.nunique()),
        # 计算活跃天数：用户有行为的不同日期数量
        # x.dt.date提取日期部分，nunique()统计唯一值数量

        total_actions=('timestamp', 'count')
        # 总行为次数：用户所有行为的总计数
    )

    # 计算日均行为次数（总行为次数 / 活跃天数）
    active_features['daily_actions'] = active_features['total_actions'] / active_features['active_days']

    # ========== 消费特征 ==========
    # 使用透视表统计各类行为次数
    behavior_counts = df.pivot_table(
        index='user_id',  # 行索引为用户ID
        columns='behavior',  # 列索引为行为类型
        values='timestamp',  # 统计值为出现次数（任意列均可）
        aggfunc='count',  # 计数统计
        fill_value=0  # 缺失值填充为0
    ).add_prefix('count_')  # 列名添加前缀（例如：count_pv）

    # 处理可能的缺失列（如某些行为类型在数据中未出现）
    if 'count_buy' not in behavior_counts:
        behavior_counts['count_buy'] = 0  # 如果无购买记录，创建全零列

    # 计算转化率指标（防止除以零错误）
    behavior_counts['cart_conversion'] = behavior_counts['count_cart'] / (behavior_counts['count_pv'] + 1e-6)
    # 加购转化率 = 加购次数 / 浏览次数（分母+1e-6避免除零）

    behavior_counts['fav_conversion'] = behavior_counts['count_fav'] / (behavior_counts['count_pv'] + 1e-6)
    # 收藏转化率同理

    # ========== 偏好特征 ==========
    # 统计用户在各品类的行为次数
    category_counts = df.groupby(['user_id', 'category_id']).size().unstack(fill_value=0)
    # groupby两列生成多级索引，unstack将category_id转为列
    # 结果矩阵示例：
    # user_id | category_1 | category_2 | ...
    # 1001    | 5          | 0          |
    # 1002    | 2          | 3          |

    # 初始化偏好特征DataFrame
    preference_features = pd.DataFrame(index=category_counts.index)

    # 浏览深度：用户在所有品类上的总行为次数
    preference_features['browse_depth'] = category_counts.sum(axis=1)

    # 品类集中度（赫芬达尔指数计算）
    preference_features['category_concentration'] = (category_counts ** 2).sum(axis=1) / (
                category_counts.sum(axis=1) ** 2 + 1e-6)
    """
    赫芬达尔指数解释：
    计算公式：Σ(每个品类占比²) 
    取值范围：0-1，值越大说明用户行为越集中
    示例：
    用户A在3个品类的行为分布为 [90, 5, 5] → 指数 ≈ 0.81
    用户B在3个品类的行为分布为 [33, 33, 34] → 指数 ≈ 0.33
    """

    # ========== 特征合并与清洗 ==========
    # 横向合并所有特征（注意索引对齐）
    features = active_features.join(behavior_counts).join(preference_features)

    # 删除原始计数特征（保留转化率等衍生特征）
    features = features.drop(columns=['count_pv', 'count_fav', 'count_cart'])

    # 缺失值填充（可能出现除零产生的NaN）
    return features.fillna(0)


# ---------------------------- 聚类建模模块 ----------------------------
def clustering_model(features):
    """
    功能：执行聚类分析流程
    输入：特征矩阵DataFrame
    输出：带聚类标签的特征矩阵

    技术要点：
    1. 标准化处理消除量纲影响
    2. 肘部法则与轮廓系数结合选择K值
    3. K-means算法应用
    """
    # ========== 数据标准化 ==========
    # 为什么要标准化？
    # 不同特征量纲差异大（如次数是百级，转化率是小数）
    # 使用Z-score标准化：将数据缩放为均值为0，方差为1
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features)  # 输入必须是二维数组

    # ========== 确定最佳K值 ==========
    sse = []  # 保存每个K值的SSE（Sum of Squared Errors）
    silhouette_scores = []  # 保存轮廓系数
    K_range = range(3, 8)  # 尝试K=3到7

    for k in K_range:
        # 创建K-means模型
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,  # 固定随机种子保证结果可复现
            n_init=10  # 多次初始化取最佳结果
        )
        kmeans.fit(scaled_data)

        # 记录SSE（模型惯性）
        sse.append(kmeans.inertia_)  # SSE越小表示聚类效果越好

        # 计算轮廓系数（不需要SSE，需要k>=2）
        if k >= 2:
            score = silhouette_score(scaled_data, kmeans.labels_)
            silhouette_scores.append(score)

    """
    肘部法则图示：
    plt.plot(K_range, sse, 'bx-')
    plt.xlabel('K')
    plt.ylabel('SSE')
    选择拐点位置对应的K值

    轮廓系数说明：
    取值范围[-1,1]，值越大表示聚类效果越好
    反映样本与自身簇和其他簇的相似度差异
    """

    # ========== 模型训练 ==========
    # 根据评估结果选择K值（此处示例固定为4）
    optimal_k = 4  # 实际项目应根据评估曲线选择

    # 使用最优参数训练最终模型
    final_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    final_model.fit(scaled_data)

    # 将聚类标签添加到特征矩阵
    features['cluster'] = final_model.labels_

    return features


# ---------------------------- 结果分析与存储 ----------------------------
def result_analysis(features, db_config):
    """
    功能：分析聚类结果并存储到数据库
    输入：带聚类标签的特征矩阵，数据库配置
    输出：群体特征报告

    关键步骤：
    1. 生成群体特征画像
    2. 数据持久化存储
    3. 可视化建议输出
    """
    # ========== 群体特征分析 ==========
    # 按聚类分组计算特征均值
    cluster_profile = features.groupby('cluster').agg({
        'last_active': 'mean',  # 平均最近活跃天数
        'active_days': 'mean',  # 平均活跃天数
        'count_buy': 'mean',  # 平均购买次数
        'cart_conversion': 'mean',  # 平均加购转化率
        'browse_depth': 'mean',  # 平均浏览深度
        'category_concentration': 'mean'  # 平均品类集中度
    }).reset_index()

    # ========== 数据库存储 ==========
    # 建立数据库连接
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    # 创建结果表（如果不存在）
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_clusters (
            user_id INT PRIMARY KEY,        -- 用户ID
            cluster_group TINYINT,           -- 所属群组（0-3）
            last_active FLOAT,               -- 最近活跃天数
            active_days FLOAT,               -- 活跃天数
            purchase_count INT,              -- 购买次数
            cart_conversion FLOAT            -- 加购转化率
        )
    """)

    # 逐行插入数据（生产环境建议使用批量插入）
    for user_id, row in features.iterrows():
        insert_query = """
            INSERT INTO user_clusters 
            (user_id, cluster_group, last_active, active_days, purchase_count, cart_conversion)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        # 参数化查询防止SQL注入
        cursor.execute(insert_query, (
            user_id,
            row['cluster'],
            row['last_active'],
            row['active_days'],
            row['count_buy'],
            row['cart_conversion']
        ))

    # 提交事务并关闭连接
    conn.commit()
    conn.close()

    # ========== 可视化建议 ==========
    print("Tableau可视化字段建议：")
    print("- 颜色编码：cluster_group → 区分不同群体")
    print("- 行：active_days, 列：purchase_count → 观察活跃与购买关系")
    print("- 筛选器：last_active → 分析不同活跃度群体")
    print("- 建议图表：雷达图对比群体特征，散点矩阵观察特征分布")

    return cluster_profile


# ---------------------------- 主程序 ----------------------------
if __name__ == "__main__":
    # 配置参数（根据实际环境修改）
    file_path = "user_behavior.csv"  # 原始数据路径
    db_config = {  # 数据库配置
        'host': 'localhost',
        'user': 'root',
        'password': 'root',  # 替换为实际密码
        'database': 'taobao'  # 确保数据库已存在
    }

    # 执行完整分析流程
    print("【1/4】数据加载中...")
    df = load_data(file_path)

    print("【2/4】特征工程处理中...")
    features = feature_engineering(df)

    print("【3/4】进行聚类分析...")
    clustered_data = clustering_model(features)

    print("【4/4】结果分析与存储...")
    profile_report = result_analysis(clustered_data, db_config)

    # 输出群体特征报告
    print("\n聚类群体特征报告：")
    print(profile_report.to_markdown(index=False))
    """
    预期输出示例：
    | cluster | last_active | active_days | count_buy | cart_conversion | browse_depth | category_concentration |
    |---------|-------------|-------------|-----------|-----------------|--------------|-------------------------|
    | 0       | 2.1         | 15.3        | 0.2       | 0.08            | 132.5        | 0.71                    |
    | 1       | 25.6        | 3.2         | 0.0       | 0.02            | 28.4         | 0.35                    |
    | 2       | 5.3         | 21.8        | 4.7       | 0.21            | 245.6        | 0.68                    |
    | 3       | 1.5         | 30.2        | 8.9       | 0.32            | 387.1        | 0.85                    |

    业务解读示例：
    - 群体3：高价值用户（最近活跃、购买力强、浏览深度高）
    - 群体1：流失风险用户（长期未活跃、低转化）
    - 群体0：普通活跃用户
    - 群体2：犹豫型用户（活跃但转化率一般）
    """