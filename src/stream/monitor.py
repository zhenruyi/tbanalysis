import time
import csv
import mysql.connector
from mysql.connector import Error
from datetime import datetime
from collections import deque, defaultdict

# 配置项
CONFIG = {
    "csv_path": "../../data/processed/UserBehavior_sampled.csv",
    "window_size": 300,  # 滑动窗口容量（每秒1条）
    "rules": {
        "high_frequency": {"threshold": 30, "window_seconds": 100},
        "offpeak_activity": {"start": 2, "end": 5, "min_actions": 50}
    },
    "mysql": {
        "host": "localhost",
        "user": "root",
        "password": "aa13516658919",
        "database": "taobao"
    }
}


# --------------------------
# 数据库模块
# --------------------------
def get_db_connection():
    """获取数据库连接（自动重试）"""
    for _ in range(3):
        try:
            return mysql.connector.connect(**CONFIG['mysql'])
        except Error as e:
            print(f"数据库连接失败: {e}")
            time.sleep(2)
    return None


def save_alert(db_conn, record, alert_type):
    """保存告警（使用上下文管理器自动提交）"""
    if not db_conn: return

    insert_query = """
    INSERT INTO anomaly_log 
    (user_id, ip, event_type, trigger_time, rule_description)
    VALUES (%(user_id)s, %(ip)s, %(event_type)s, %(now)s, %(rule)s)
    """

    params = {
        'user_id': record['user_id'],
        'ip': record.get('ip', ''),
        'event_type': alert_type,
        'now': datetime.now(),
        'rule': str(CONFIG['rules'].get(alert_type, ""))
    }

    try:
        with db_conn.cursor() as cursor:
            cursor.execute(insert_query, params)
            db_conn.commit()
    except Error as e:
        print(f"保存失败: {e}")


# --------------------------
# 检测规则模块
# --------------------------
def is_high_frequency(user_logs, user_id, rule):
    """优化后的高频检测（使用自动清理的deque）"""
    now = time.time()
    time_window = now - rule['window_seconds']

    # 自动维护时间窗口
    logs = user_logs[user_id]
    while logs and logs[0] < time_window:
        logs.popleft()

    logs.append(now)
    return len(logs) > rule['threshold']


def is_offpeak(record, rule):
    """优化后的时段检测（使用字典解包）"""
    try:
        hour = datetime.strptime(record['timestamp'], "%Y/%m/%d %H:%M").hour
        return rule['start'] <= hour <= rule['end']
    except KeyError:
        print("缺少时间戳字段")
        return False


# --------------------------
# 主处理流程
# --------------------------
def process_stream():
    """主处理函数"""
    window = deque(maxlen=CONFIG['window_size'])
    user_logs = defaultdict(lambda: deque(maxlen=1000))  # 自动初始化用户日志

    with get_db_connection() as db_conn, open(CONFIG['csv_path']) as f:

        reader = csv.DictReader(f)
        for row in reader:
            # 更新滑动窗口
            window.append(row)

            # 并行执行规则检测
            alerts = []
            user_id = row['user_id']

            # 高频检测
            if is_high_frequency(user_logs, user_id, CONFIG['rules']['high_frequency']):
                alerts.append('high_frequency')

            # 时段检测
            if is_offpeak(row, CONFIG['rules']['offpeak_activity']) and len(window) >= CONFIG['rules']['offpeak_activity']['min_actions']:
                alerts.append('offpeak_activity')

            # 批量保存告警
            for alert in alerts:
                save_alert(db_conn, row, alert)

            time.sleep(0.1)  # 模拟实时处理


if __name__ == "__main__":
    process_stream()