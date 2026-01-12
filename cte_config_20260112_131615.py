# CTE 估算器配置
# 生成时间: 2026-01-12 13:16:15
# 方法: color_edge_detection

# HSV 参数:
# H: [0, 180]
# S: [100, 255]
# V: [100, 255]

CTE_CONFIG = {
    "method": "color_edge_detection",
    "max_cte": 3.0,
    # HSV 阈值 - 用于 color_edge_detection 和 centerline_tracking
    # (centerline_tracking 现在使用和 color_edge_detection 相同的阈值)
    "track_lower": (0, 100, 100),
    "track_upper": (180, 255, 255),
}

# 用于 real_car_env.py:
# from real_car_env import VisualCTEEstimator
# cte_estimator = VisualCTEEstimator(**CTE_CONFIG)
