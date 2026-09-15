# =====================
# Nick's Code
# Setting System Parameters
# param.name
# =====================

import numpy as np

# ==========================================
# Test
# ==========================================
th1_begin_deg = 30
th1_pos_a_deg = 30
th1_pos_b_deg = 60
position_hold_time = 2.0

# ==========================================
# Settings
# m1 大腿, m2 小腿
# m1 逆时针为正方向, m2 顺时针为正方向
# 所有 读状态 & 给指令 的地方, 乘这个 dir
# ==========================================

dir_1 = 1          # 1 代表逆时针为正
dir_2 = -1         # -1 代表顺时针为正

max_speed_pos = 600   # 60 deg/s: *0.1 gear ratio

# 位置反馈: /pos_gain; 位置控制: *pos_gain
pos_gain = 1000       # 1000 -> 1 deg: *0.01 LSB & *0.1 gear ratio

# 温度无需转换
temperature_gain = 1

# 速度控制
vel_gain = 1000

# 速度反馈
vel_state_gain = 10

# 编码器分辨率 
encoder_gain = 360/65535   # 编码器输出 * gain = 输出轴角度
### 1. 哪个是输出编码器?  multi_pos 与编码器有微小区别  2. 单圈输出, 无法多圈计数  

# 转矩控制 & 转矩反馈
torque_gain = 206.04  # (iq_command) to (output_torque)

