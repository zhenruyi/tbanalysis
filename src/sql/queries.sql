-- 行为类型分布
SELECT behavior,
       COUNT(*)                                                        AS count,
       ROUND(COUNT(*) / (SELECT COUNT(*) FROM user_behavior) * 100, 2) AS ratio
FROM user_behavior
GROUP BY behavior;


-- 每小时PV量分析
SELECT hour,
       count(*) as pv_count
FROM user_behavior
WHERE behavior = 'pv'
GROUP BY hour
ORDER BY hour;


/* RFM分析 */
WITH rfm_base AS (
    SELECT user_id,
           -- 计算Recency：用户最近一次购买距离2017-12-04的天数
           DATEDIFF('2017-12-04', MAX(date)) AS recency,
           -- 计算Frequency：用户购买的总次数（加权计算）
           COUNT(DISTINCT date) * 0.7 + COUNT(*) * 0.3 AS frequency,
           -- 计算Monetary：用户购买的总金额（基于price字段）
           SUM(price) AS monetary
    FROM user_behavior
    WHERE behavior = 'buy' -- 仅统计购买行为
    GROUP BY user_id
),
rfm_score AS (
    SELECT user_id,
           recency,
           frequency,
           monetary,
           -- 计算R_Score：根据Recency分组打分
           CASE
               WHEN recency <= 7 THEN 4
               WHEN recency <= 14 THEN 3
               WHEN recency <= 21 THEN 2
               ELSE 1
           END AS R_Score,
           -- 计算F_Score：根据Frequency分组打分
           CASE
               WHEN frequency >= 15 THEN 4
               WHEN frequency >= 10 THEN 3
               WHEN frequency >= 5 THEN 2
               ELSE 1
           END AS F_Score,
           -- 计算M_Score：根据Monetary分组打分
           CASE
               WHEN monetary >= 5000 THEN 4
               WHEN monetary >= 3000 THEN 3
               WHEN monetary >= 1000 THEN 2
               ELSE 1
           END AS M_Score
    FROM rfm_base
)
SELECT user_id,
       recency,
       frequency,
       monetary,
       R_Score,
       F_Score,
       M_Score,
       -- 计算RFM总分
       (R_Score + F_Score + M_Score) AS RFM_Total_Score
FROM rfm_score
ORDER BY R_Score DESC, F_Score DESC, M_Score DESC;