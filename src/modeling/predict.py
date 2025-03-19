import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

# 时间序列分解是将一个时间序列数据拆解为三个核心部分
# 趋势（Trend）：数据在较长时间内呈现的上升或下降的方向性变化。
# 季节性（Seasonality）：数据在固定周期内呈现相似的趋势，如每周、每月等。
# 残差（Residual）：数据在趋势和季节性方面已经去除，剩下的部分是随机的。


# 加载清洗后的数据
df = pd.read_csv('../../data/processed/UserBehavior_sampled.csv', parse_dates=['timestamp'])

# 按日统计PV（页面访问量）和UV（独立用户数）
daily_data = df[df['behavior'] == 'pv'].resample('D', on='timestamp').agg(
    pv = ('user_id', 'count'),  # PV
    uv = ('user_id', 'nunique')  # UV
)

# 保存预处理数据
daily_data.to_csv('../../data/processed/daily_pv_uv.csv')

# 可视化时间序列
plt.figure(figsize=(12,6))
plt.plot(daily_data.index, daily_data['pv'], label='Page Views')
plt.plot(daily_data.index, daily_data['uv'], label='Unique Users')
plt.xlabel('Date')
plt.ylabel('Count')
plt.title('Daily Traffic Trend')
plt.legend()
plt.grid(True)
plt.savefig('../../outputs/figures/daily_trend.png')


# 用加法模型分解趋势、季节性和残差（Time Series=Trend+Seasonality+Residual）
# 乘法模型的数据不能为0或者负数，因为这样会导致结果无意义（Time Series=Trend×Seasonality×Residual）
result = seasonal_decompose(daily_data['pv'], model='additive', period=7)

# 可视化分解结果
plt.figure(figsize=(12,8))
result.plot()
plt.savefig('../../outputs/figures/decomposition.png')


# 划分训练集（前28天）和测试集（最后7天）
train = daily_data['pv'][:-7]
test = daily_data['pv'][-7:]

# 模型训练
# ARIMA = AutoRegressive + Integrated + Moving Average
# AR（自回归）：用过去的数据预测未来
# I（差分）：让数据变平稳（消除趋势）
# MA（移动平均）：考虑历史预测误差的影响

# order=(p,d,q)：
# p=2：认为最近2天的流量影响今天，用过去两天的数据预测今天
# d=1：数据有明显上升趋势，需差分一次，即相邻两天相减
# q=1：考虑前1天的预测误差
model = ARIMA(train, order=(2,1,1))  # 非季节性部分
model_fit = model.fit()

# 预测未来7天
forecast = model_fit.forecast(steps=7)

# 可视化对比
plt.figure(figsize=(12,6))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='True')
plt.plot(test.index, forecast, label='Predicted')
plt.legend()
plt.savefig('../../outputs/figures/arima_forecast.png')


# 案例：通过分析发现每周五PV增长15%，建议：
#
# 周五增加秒杀活动
#
# 提前扩容服务器
#
# 在周四晚推送活动预告