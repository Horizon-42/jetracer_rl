# CTE 估算器配置
# 生成时间: 2026-01-12 03:21:06
# 方法: edge_detection

# HSV 参数:
# H: [0, 180]
# S: [100, 255]
# V: [120, 255]

CTE_CONFIG = {
    "method": "edge_detection",
    "max_cte": 3.0,
    # 边缘检测阈值 (HSV)
    "track_lower": (0, 100, 120),
    "track_upper": (180, 255, 255),
    # 中心线检测阈值 (HSV)
    "centerline_lower": (10, 100, 100),
    "centerline_upper": (25, 255, 255),
}

# 用于 real_car_env.py:
# from real_car_env import VisualCTEEstimator
# cte_estimator = VisualCTEEstimator(**CTE_CONFIG)
