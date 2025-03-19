import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

# 数据加载
df = pd.read_csv(
    '../../data/raw/UserBehavior.csv',
    names=['user_id', 'item_id', 'category_id', 'behavior', 'timestamp']
)

# 打印前10行数据
print("原始数据前10行：")
print(df.head(10))


# 时间戳转换
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
df['date'] = df['timestamp'].dt.date
df['hour'] = df['timestamp'].dt.hour

# 数据抽样（10%的数据）
df_sampled = df.sample(frac=0.1, random_state=314)

# 插入随机生成的price字段，金额范围在10到10000之间
df_sampled['price'] = np.random.randint(10, 10001, size=len(df_sampled))

# 保存清洗后的数据到CSV文件
df_sampled.to_csv('../../data/processed/UserBehavior_sampled.csv', index=False)
print("清洗后的数据已保存到 UserBehavior_sampled.csv")

# 数据库连接信息
db_url = 'mysql+pymysql://root:aa13516658919@localhost:3306/taobao'

# 创建数据库（如果不存在）
try:
    engine = create_engine('mysql+pymysql://root:aa13516658919@localhost:3306/')
    with engine.connect() as connection:
        # 创建数据库
        connection.execute(text('CREATE DATABASE IF NOT EXISTS taobao'))
        print("数据库 'taobao' 已创建或已存在")
except Exception as e:
    print(f"创建数据库时出错: {e}")

# 将数据导入MySQL数据库
try:
    engine = create_engine(db_url)
    with engine.connect() as connection:
        # 将数据写入表
        df_sampled.to_sql('user_behavior', con=engine, if_exists='replace', index=False)
        print("数据已成功导入到 MySQL 数据库的 user_behavior 表中")
except Exception as e:
    print(f"导入数据到数据库时出错: {e}")