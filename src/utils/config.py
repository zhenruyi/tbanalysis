# -*- coding: utf-8 -*-
"""
配置文件加载模块 - 安全读取YAML配置并支持环境变量注入
"""

import os
import yaml
from dotenv import load_dotenv
from yaml import SafeLoader


# ------------------------- 环境变量加载函数 -------------------------
def _load_env():
    """
    加载项目根目录下的.env文件到系统环境变量

    功能：
    - 定位项目根目录下的.env文件
    - 将.env文件中的键值对加载到Python环境变量中
    - 后续可以通过os.getenv()获取这些值

    路径说明：
    - os.path.dirname(__file__) 获取当前文件所在目录
    - 通过两次parent目录跳转定位到项目根目录
    - 最终路径结构：项目根目录/.env
    """
    # 构建.env文件的完整路径
    # os.path.join() 用于安全拼接路径（避免不同操作系统的路径分隔符问题）
    # ../../ 表示向上跳转两级目录（从当前文件所在位置到项目根目录）
    env_path = os.path.join(
        os.path.dirname(__file__),  # 当前文件所在目录
        "../../.env"  # 相对路径到项目根目录
    )

    # 加载.env文件到环境变量
    # load_dotenv() 会读取.env文件中的键值对
    # 后续可以通过os.getenv('KEY')获取这些值
    load_dotenv(env_path)


# ---------------------- 自定义YAML标签解析器 ----------------------
def _env_var_constructor(loader, node):
    """
    自定义YAML标签解析器 !env
    功能：将YAML中的!env标签替换为环境变量的实际值

    参数：
    - loader：YAML解析器的加载器对象
    - node：当前正在解析的YAML节点

    工作流程：
    1. 从YAML节点提取原始字符串（例如 "${DB_USER}"）
    2. 去除字符串中的${}包裹符号，得到环境变量名（例如 "DB_USER"）
    3. 从系统环境变量中获取对应的值
    4. 如果找不到则抛出明确错误

    示例：
    YAML文件内容：username: !env ${DB_USER}
    解析结果：username的值会被替换为环境变量DB_USER的实际值
    """
    # 从YAML节点提取原始字符串值
    # construct_scalar() 方法将节点转换为Python字符串
    value = loader.construct_scalar(node)

    # 去除字符串中的特殊符号
    # strip("${}") 会移除字符串首尾的 $ { } 符号
    # 示例输入 "${DB_USER}" → 输出 "DB_USER"
    var_name = value.strip("${}")

    # 从环境变量获取值
    # os.getenv() 是Python标准库方法
    # 如果环境变量不存在则返回None
    env_value = os.getenv(var_name)

    # 环境变量验证
    if env_value is None:
        # 抛出包含明确错误信息的异常
        raise ValueError(
            f"关键环境变量 {var_name} 未设置，请检查：\n"
            f"1. .env文件是否存在\n"
            f"2. 是否在系统环境变量中设置\n"
            f"3. 变量名是否拼写正确"
        )

    return env_value  # 返回环境变量的实际值


# 将自定义解析器注册到YAML安全加载器
# 参数说明：
# - "!env"：在YAML中使用的标签名称
# - _env_var_constructor：对应的处理函数
SafeLoader.add_constructor("!env", _env_var_constructor)


# ---------------------- 主配置加载函数 ----------------------
def load_config(rel_path="../../config/settings.yaml"):
    """
    加载并解析项目配置文件

    参数：
    rel_path -- 配置文件的相对路径（相对于当前文件）

    返回值：
    包含配置信息的字典对象

    工作流程：
    1. 加载环境变量
    2. 构建配置文件完整路径
    3. 安全解析YAML文件
    4. 自动处理!env标签替换

    安全特性：
    - 使用SafeLoader防止YAML注入攻击
    - 敏感信息不直接存储在配置文件中
    - 自动验证必需环境变量
    """
    # 第一步：加载环境变量
    _load_env()  # 确保后续操作能访问到.env中的变量

    # 构建配置文件路径
    # 使用与_load_env()相同的路径计算逻辑
    config_path = os.path.join(
        os.path.dirname(__file__),  # 当前文件所在目录
        rel_path  # 传入的相对路径
    )

    # 安全读取YAML文件
    # 使用with语句确保文件正确关闭
    # yaml.load() 的Loader参数指定使用安全加载器
    with open(config_path) as f:
        # yaml.load() 解析YAML内容为Python字典
        # Loader=SafeLoader 使用安全模式加载，防止恶意代码执行
        config = yaml.load(f, Loader=SafeLoader)

    return config  # 返回可以直接使用的配置字典


"""
使用示例：
假设配置文件内容：
database:
  host: !env ${DB_HOST}
  user: !env ${DB_USER}

调用 load_config() 会返回：
{
    "database": {
        "host": "127.0.0.1",  # 来自环境变量DB_HOST
        "user": "admin"        # 来自环境变量DB_USER
    }
}
"""

# if __name__ == "__main__":
#     # 测试代码
#     config = load_config()
#     print(config)