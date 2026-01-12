# CTE 估算器配置
# 生成时间: 2026-01-12 10:46:09
# 方法: centerline_tracking

# HSV 参数:
# H: [10, 25]
# S: [100, 255]
# V: [100, 255]

CTE_CONFIG = {
    "method": "centerline_tracking",
    "max_cte": 3.0,
    # 边缘检测阈值 (HSV)
    "track_lower": (0, 0, 200),
    "track_upper": (180, 30, 255),
    # 中心线检测阈值 (HSV)
    "centerline_lower": (10, 100, 100),
    "centerline_upper": (25, 255, 255),
}

# 用于 real_car_env.py:
# from real_car_env import VisualCTEEstimator
# cte_estimator = VisualCTEEstimator(**CTE_CONFIG)
