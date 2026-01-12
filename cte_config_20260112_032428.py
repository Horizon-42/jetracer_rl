# CTE 估算器配置
# 生成时间: 2026-01-12 03:24:28
# 方法: centerline_tracking

# HSV 参数:
# H: [0, 180]
# S: [91, 255]
# V: [94, 250]

CTE_CONFIG = {
    "method": "centerline_tracking",
    "max_cte": 3.0,
    # 边缘检测阈值 (HSV)
    "track_lower": (0, 100, 120),
    "track_upper": (180, 255, 255),
    # 中心线检测阈值 (HSV)
    "centerline_lower": (0, 91, 94),
    "centerline_upper": (180, 255, 250),
}

# 用于 real_car_env.py:
# from real_car_env import VisualCTEEstimator
# cte_estimator = VisualCTEEstimator(**CTE_CONFIG)
