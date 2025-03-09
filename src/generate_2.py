import numpy as np
import pandas as pd

# 常量定义
NUM_SAMPLES = 1000  # 样本数量
MU = 0.3  # 摩擦系数

# 数据生成函数
def generate_data(num_samples):
    np.random.seed(42)
    weight = np.random.uniform(1, 100, num_samples)  # 重量 (N)
    angle_deg = np.random.uniform(0, 90, num_samples)  # 角度 (°)
    angle_rad = np.deg2rad(angle_deg)  # 转换为弧度

    # 物理计算
    normal_force = weight * np.cos(angle_rad)  # 法向力 (N)
    friction = MU * normal_force  # 摩擦力 (N)

    # 输入特征矩阵: [重量, 法向力, 角度(°)]
    X = np.column_stack((weight, normal_force, angle_deg))
    y = friction.reshape(-1, 1)

    # 将 X 和 y 合并为一个 DataFrame
    data = np.hstack((X, y))
    columns = ['Weight (N)', 'Normal Force (N)', 'Angle (deg)', 'Friction (N)']
    df = pd.DataFrame(data, columns=columns)

    return df

# 生成数据
df = generate_data(NUM_SAMPLES)

# 保存到 Excel 文件
df.to_excel('data/friction_data.xlsx', index=False)
print("数据已保存到 friction_data.xlsx")