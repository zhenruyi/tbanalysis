import pandas as pd
import numpy as np
from datetime import timedelta
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import mysql.connector
from mysql.connector import Error

# 读取原始数据
# 数据列：user_id, item_id, category_id, behavior, timestamp
df = pd.read_csv('../../data/processed/UserBehavior_sampled.csv')
df = df[['user_id', 'item_id', 'category_id', 'behavior', 'timestamp']]

# ==============================================
# 第一步：数据预处理与行为序列构建
# ==============================================

# 转换时间格式
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 按用户分组处理
df = df.sort_values(['user_id', 'timestamp'])  # 修改排序字段

# 定义会话超时时间（30分钟）
SESSION_TIMEOUT = timedelta(minutes=30)

# 生成会话ID
def generate_session_id(group):
    # 计算时间差（原 datetime 字段改为 timestamp）
    time_diff = group['timestamp'].diff()
    # 标记新会话开始的位置（首次行为 或 间隔超过阈值）
    new_session = (time_diff > SESSION_TIMEOUT).astype(int)
    # 生成会话ID（累加新会话标记）
    group['session_id'] = new_session.cumsum()
    return group

# 应用会话分割
df = df.groupby('user_id', group_keys=False).apply(generate_session_id)

# 生成行为路径（按会话聚合）
def get_behavior_sequence(group):
    return list(group['behavior'])  # behavior 列保持不变

user_paths = df.groupby(['user_id', 'session_id']).apply(get_behavior_sequence).reset_index()
user_paths.columns = ['user_id', 'session_id', 'behavior_sequence']

# ==============================================
# 第二步：高频路径挖掘（使用Apriori算法）
# ==============================================

# 准备事务数据（无需修改）
transactions = user_paths['behavior_sequence'].tolist()

# 转换事务数据为矩阵格式
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_trans = pd.DataFrame(te_ary, columns=te.columns_)

# 运行Apriori算法（最小支持度设置为2%）
frequent_itemsets = apriori(df_trans, min_support=0.02, use_colnames=True)

# 过滤出长度≥2的频繁项集
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(len)
frequent_paths = frequent_itemsets[frequent_itemsets['length'] >= 2]

# 展示前10个高频路径
print("Top 10高频行为路径：")
print(frequent_paths.sort_values(by='support', ascending=False).head(10))

# ==============================================
# 第三步：转化漏斗分析（示例：浏览->加购->购买）
# ==============================================

# 定义核心转化路径（无需修改）
TARGET_PATH = ['pv', 'cart', 'buy']

# 计算各步骤用户数（无需修改）
def check_step(sequence, step):
    try:
        # 查找路径中的连续步骤
        for i in range(len(sequence)-len(step)+1):
            if sequence[i:i+len(step)] == step:
                return True
        return False
    except:
        return False

# 计算各步骤人数
step_counts = {
    'step1_pv': user_paths['behavior_sequence'].apply(lambda x: 'pv' in x).sum(),
    'step2_pv_cart': user_paths['behavior_sequence'].apply(lambda x: check_step(x, ['pv','cart'])).sum(),
    'step3_pv_cart_buy': user_paths['behavior_sequence'].apply(lambda x: check_step(x, TARGET_PATH)).sum()
}

# 计算转化率
conversion_rates = {
    'pv_to_cart': step_counts['step2_pv_cart'] / step_counts['step1_pv'],
    'cart_to_buy': step_counts['step3_pv_cart_buy'] / step_counts['step2_pv_cart']
}

print("\n转化漏斗分析结果：")
print(f"浏览用户数: {step_counts['step1_pv']}")
print(f"浏览->加购转化率: {conversion_rates['pv_to_cart']:.2%}")
print(f"加购->购买转化率: {conversion_rates['cart_to_buy']:.2%}")


# ==============================================
# 第四步：存储分析结果到MySQL
# ==============================================

def mysql_connect():
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='your_username',
            password='your_password',
            database='your_database'
        )
        return conn
    except Error as e:
        print(f"数据库连接错误: {e}")
        return None


# 存储高频路径
conn = mysql_connect()
if conn:
    # 创建高频路径表
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS frequent_paths (
        id INT AUTO_INCREMENT PRIMARY KEY,
        itemsets VARCHAR(255),
        support FLOAT,
        length INT
    )
    """
    cursor = conn.cursor()
    cursor.execute(create_table_sql)

    # 插入数据
    for _, row in frequent_paths.iterrows():
        insert_sql = """
        INSERT INTO frequent_paths (itemsets, support, length)
        VALUES (%s, %s, %s)
        """
        cursor.execute(insert_sql, (
            '->'.join(list(row['itemsets'])),
            row['support'],
            row['length']
        ))
    conn.commit()
    print("高频路径数据已存储到MySQL")


# ==============================================
# 第五步：生成Tableau可视化数据
# ==============================================

# 生成桑基图所需数据（行为转移矩阵）
def generate_sankey_data():
    # 获取所有行为转移对
    transfer_pairs = []
    for seq in user_paths['behavior_sequence']:
        for i in range(len(seq) - 1):
            transfer_pairs.append((seq[i], seq[i + 1]))

    # 统计转移频率
    transfer_df = pd.DataFrame(transfer_pairs, columns=['source', 'target'])
    sankey_data = transfer_df.groupby(['source', 'target']).size().reset_index(name='value')
    sankey_data.to_csv('sankey_data.csv', index=False)
    print("桑基图数据已保存到 sankey_data.csv")


generate_sankey_data()

# 生成漏斗图数据
funnel_data = pd.DataFrame({
    'step': ['浏览', '加购', '购买'],
    'count': [step_counts['step1_pv'],
              step_counts['step2_pv_cart'],
              step_counts['step3_pv_cart_buy']]
})
funnel_data.to_csv('funnel_data.csv', index=False)
print("漏斗图数据已保存到 funnel_data.csv")