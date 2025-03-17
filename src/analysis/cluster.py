import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler    # 数据标准化工具
from sklearn.cluster import KMeans  #自动分类工具
import matplotlib.pyplot as plt

# 从MySQL加载RFM数据
engine = create_engine('mysql+pymysql://root:aa13516658919@localhost:3306/taobao')
rfm_df = pd.read_sql("""
    SELECT user_id, recency, frequency, monetary
    FROM (
        SELECT
            user_id,
            DATEDIFF('2017-12-04', MAX(date)) AS recency,
            COUNT(DISTINCT date) AS frequency,
            SUM(price) AS monetary
        FROM user_behavior
        WHERE behavior = 'buy'
        GROUP BY user_id
    ) AS base
""", con=engine)

# 数据标准化：统一量纲
# 不同纬度的单位不同，直接对比没有意义，需要把所有数据转为“跟平均水平相差多少倍标准差”
# 标准化之后，数据变为均值为0，标准差为1，每个特征对聚类的影响相同
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_df[['recency','frequency','monetary']])

# K-means聚类：自动分类
# 随机选择K个中心点，将每个顾客分配到最近的中心点，重新计算中心点的位置，直到中心点不再移动
# n_clusters=4的意义是讲顾客分成4类，即K=4
kmeans = KMeans(n_clusters=4, random_state=42)
rfm_df['cluster'] = kmeans.fit_predict(rfm_scaled)

# 可视化
plt.figure(figsize=(10,6))
# 颜色按照 cluster 进行划分；cmap指定颜色映射（colormap），viridis 是一种渐变色，从蓝色到黄色，适合区分不同的聚类。
plt.scatter(rfm_df['frequency'], rfm_df['recency'],
            c=rfm_df['cluster'], cmap='viridis')
plt.xlabel('Frequency')
plt.ylabel('Recency')
plt.title('User Clustering')
plt.savefig('cluster_plot.png')

# 保存聚类结果
rfm_df.to_csv('user_clusters.csv', index=False)