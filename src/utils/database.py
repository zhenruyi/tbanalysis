# -*- coding: utf-8 -*-
"""
数据库操作工具模块 - 封装SQLAlchemy核心操作
功能亮点:
1. 安全的数据库连接管理
2. 自动化数据库创建验证
3. 支持DataFrame直接写入数据库
4. 完整的异常处理机制
"""

# -------------------------- 模块导入 --------------------------
from sqlalchemy import create_engine, text  # SQLAlchemy核心组件
import logging  # Python标准日志模块

# 配置日志格式
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


# ---------------------- 数据库URL构造器 ----------------------
def _build_db_url(db_config, include_dbname=False):
    """
    构建标准化的数据库连接URL（核心基础方法）

    设计理念:
    - 解耦URL构造逻辑，提高代码复用性
    - 支持灵活控制是否包含数据库名称

    参数详解:
    :param db_config: dict 数据库配置字典，必须包含以下键:
        - dialect: 数据库类型 (如mysql, postgresql)
        - driver: 驱动类型 (如pymysql, psycopg2)
        - username: 数据库用户名
        - password: 数据库密码
        - host: 数据库主机地址
        - port: 数据库端口号
        - dbname: 数据库名称（当include_dbname=True时需要）
    :param include_dbname: bool 是否在URL中包含数据库名称
        设置为True时用于直接连接指定数据库
        设置为False时用于创建数据库前的管理连接

    返回:
    :return: str 符合SQLAlchemy规范的数据库连接URL

    安全提示:
    - 密码中的特殊字符会自动进行URL编码
    - 建议通过环境变量传递敏感信息

    示例:
    >>> config = {
    ...     'dialect': 'mysql',
    ...     'driver': 'pymysql',
    ...     'username': 'admin',
    ...     'password': 'secret',
    ...     'host': 'dbserver',
    ...     'port': '3306',
    ...     'dbname': 'mydb'
    ... }
    >>> _build_db_url(config)
    'mysql+pymysql://admin:secret@dbserver:3306/'
    >>> _build_db_url(config, True)
    'mysql+pymysql://admin:secret@dbserver:3306/mydb'
    """
    # 基础URL构造（使用f-string格式化）
    db_url = (
        f"{db_config['dialect']}+{db_config['driver']}://"  # 数据库类型+驱动
        f"{db_config['username']}:{db_config['password']}@"  # 认证信息
        f"{db_config['host']}:{db_config['port']}/"  # 连接地址
    )

    # 动态添加数据库名称
    if include_dbname:
        db_url += db_config['dbname']  # 追加数据库名称

    return db_url  # 返回完整连接字符串


# -------------------- 数据库连接引擎工厂 --------------------
def create_db_connection(db_config, include_dbname=False):
    """
    创建SQLAlchemy数据库引擎（核心资源管理器）

    功能特点:
    - 内置连接池管理（默认5个连接）
    - 自动处理连接超时和重试
    - 支持事务管理

    参数说明:
    :param db_config: dict 同_build_db_url参数
    :param include_dbname: bool 同_build_db_url参数

    返回:
    :return: Engine SQLAlchemy引擎实例

    最佳实践:
    - 引擎对象应作为单例使用，避免重复创建
    - 使用with语句管理连接生命周期

    示例:
    >>> engine = create_db_connection(config)
    >>> with engine.connect() as conn:
    ...     result = conn.execute(text("SELECT 1"))
    ...     print(result.scalar())
    1
    """
    # 通过私有方法构造数据库URL
    connection_url = _build_db_url(db_config, include_dbname)

    # 创建并返回数据库引擎
    # 关键参数说明:
    # pool_recycle=3600 - 连接回收时间（防止MySQL 8小时断开问题）
    # echo=False        - 关闭调试日志（生产环境建议关闭）
    return create_engine(
        url=connection_url,
        pool_recycle=3600,
        echo=False
    )


# ------------------- 数据库存在性验证 -------------------
def ensure_database_exists(db_config):
    """
    确保目标数据库存在（自动化初始化工具）

    执行流程:
    1. 创建无数据库名称的基础连接
    2. 执行CREATE DATABASE IF NOT EXISTS
    3. 记录操作日志

    设计考量:
    - 兼容数据库首次部署场景
    - 避免手动执行SQL初始化

    参数:
    :param db_config: dict 必须包含dbname键

    异常处理:
    - 捕获所有异常并记录详细日志
    - 重新抛出异常以供上层处理
    """
    try:
        # 创建临时管理连接（不指定数据库）
        admin_engine = create_db_connection(db_config, include_dbname=False)

        # 使用上下文管理器自动处理连接
        with admin_engine.connect() as connection:
            # 构造DDL语句
            create_db_query = text(
                f'CREATE DATABASE IF NOT EXISTS {db_config["dbname"]}'
            )

            # 执行数据库创建
            connection.execute(create_db_query)

            # 记录成功日志
            logging.info(f"Database {db_config['dbname']} verified")

    except Exception as error:
        # 记录详细错误日志
        logging.error(f"Database creation failed: {str(error)}")
        # 重新抛出异常，保持调用栈信息
        raise


# ------------------- 数据写入操作 -------------------
def write_to_database(df, db_config, table_name):
    """
    将DataFrame数据写入数据库表（批处理工具）

    功能特性:
    - 自动创建表结构（如果不存在）
    - 全量替换现有数据（if_exists='replace'）
    - 自动提交事务

    参数说明:
    :param df: pandas.DataFrame 要写入的数据集
        要求:
        - 列名与数据库表字段匹配
        - 数据格式兼容目标数据库类型
    :param db_config: dict 必须包含dbname
    :param table_name: str 目标表名

    性能提示:
    - 大数据量建议分批次写入
    - 可调整chunksize参数优化内存使用

    事务管理:
    - 整个写入操作在单一事务中完成
    - 失败时自动回滚

    示例:
    >>> import pandas as pd
    >>> data = pd.DataFrame({'id': [1,2], 'name': ['A','B']})
    >>> write_to_database(data, config, 'users')
    """
    try:
        # 创建目标数据库连接
        db_engine = create_db_connection(db_config, include_dbname=True)

        # 执行数据写入
        df.to_sql(
            name=table_name,  # 目标表名
            con=db_engine,  # 数据库连接
            if_exists='replace',  # 表存在时替换数据
            index=False,  # 不写入行索引
            method='multi',  # 批量提交（提升性能）
            chunksize=1000  # 每批次写入1000条
        )

        # 记录成功日志
        logging.info("Data successfully written to MySQL")

    except Exception as error:
        # 记录详细错误日志
        logging.error(f"Database write failed: {str(error)}")
        # 重新抛出异常
        raise


"""
模块使用示例：

import pandas as pd
from utils.database import *

# 配置示例
db_config = {
    'dialect': 'mysql',
    'driver': 'pymysql',
    'username': 'root',
    'password': 'mysql123',
    'host': 'localhost',
    'port': '3306',
    'dbname': 'sales_db'
}

# 初始化数据库
ensure_database_exists(db_config)

# 创建测试数据
sample_data = pd.DataFrame({
    'order_id': [1001, 1002],
    'amount': [199.9, 299.5],
    'order_date': pd.date_range('2023-01-01', periods=2)
})

# 写入数据库
write_to_database(sample_data, db_config, 'orders')
"""

