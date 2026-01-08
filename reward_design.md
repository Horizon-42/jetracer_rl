```python
def calc_reward(params):
    # 这是 AWS DeepRacer 的风格，可以改编用于 DonkeyCar
    
    # 1. 获取参数
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    all_wheels_on_track = params['all_wheels_on_track']
    speed = params['speed']
    
    # 2. 基础惩罚：如果出界，给极低分（虽然 Simulator 会自动 done，但给负分能加速收敛）
    if not all_wheels_on_track:
        return 1e-3
    
    # 3. 距离奖励：越靠近中心线，分数越高 (高斯分布)
    # 0.5 是赛道宽度的一半
    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    marker_3 = 0.5 * track_width
    
    reward = 1e-3
    if distance_from_center <= marker_1:
        reward = 1.0
    elif distance_from_center <= marker_2:
        reward = 0.5
    elif distance_from_center <= marker_3:
        reward = 0.1
        
    # 4. 速度激励：在安全的基础上，速度越快分越高
    reward += (speed * 0.5) 
    
    return float(reward)