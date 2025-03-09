import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import TensorDataset, DataLoader

# 参数设置
MU = 0.3  # 摩擦系数
NUM_SAMPLES = 10000
BATCH_SIZE = 64
NUM_EPOCHS = 200
LEARNING_RATE = 0.001


# # 数据生成函数
# def generate_data(num_samples):
#     np.random.seed(42)
#     weight = np.random.uniform(1, 100, num_samples)  # 重量 (N)
#     angle_deg = np.random.uniform(0, 90, num_samples)  # 角度 (°)
#     angle_rad = np.deg2rad(angle_deg)  # 转换为弧度
#
#     # 物理计算
#     normal_force = weight * np.cos(angle_rad)  # 法向力 (N)
#     friction = MU * normal_force  # 摩擦力 (N)
#
#     # 输入特征矩阵: [重量, 法向力, 角度(°)]
#     X = np.column_stack((weight, normal_force, angle_deg))
#     y = friction.reshape(-1, 1)
#     print(X,y)
#     return X, y

import pandas as pd

# 从 Excel 文件中读取数据
df = pd.read_excel('../data/friction_data_cat.xlsx')


# 提取特征 (X) 和目标值 (y)
X = df[['Weight (N)', 'Normal Force (N)', 'Angle (deg)']].values
y = df['Friction (N)'].values.reshape(-1, 1)

# 打印数据
print("特征矩阵 X:")
print(X[:5])  # 打印前 5 行
print("\n目标值 y:")
print(y[:5])  # 打印前 5 行





# 数据预处理
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 创建DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


# 定义神经网络模型
class FrictionModel(nn.Module):
    def __init__(self, input_size=3):
        super(FrictionModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FrictionModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 训练循环
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)

    # 验证步骤
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, targets).item() * inputs.size(0)

    # 打印训练进度
    train_loss = train_loss / len(train_loader.dataset)
    test_loss = test_loss / len(test_loader.dataset)
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}] | '
              f'Train Loss: {train_loss:.4f} | '
              f'Test Loss: {test_loss:.4f}')

# 模型评估
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor.to(device)).cpu().numpy()

print("\n最终评估结果:")
print(f"平均绝对误差 (MAE): {mean_absolute_error(y_test, y_pred):.4f}")
print(f"R²分数: {r2_score(y_test, y_pred):.4f}")


# 示例预测
def predict_friction(weight, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    normal_force = weight * np.cos(angle_rad)
    raw_input = np.array([[weight, normal_force, angle_deg]])
    scaled_input = scaler.transform(raw_input)
    input_tensor = torch.tensor(scaled_input, dtype=torch.float32).to(device)

    with torch.no_grad():
        prediction = model(input_tensor).cpu().item()

    true_value = MU * normal_force
    return prediction, true_value


# 测试示例
test_weight = 50.0  # N
test_angle = 30.0  # °
prediction, true_value = predict_friction(test_weight, test_angle)

print("\n示例预测:")
print(f"输入参数: 重量={test_weight}N, 角度={test_angle}°")
print(f"预测摩擦力: {prediction:.4f}N")
print(f"理论计算值: {true_value:.4f}N")
print(f"绝对误差: {abs(prediction - true_value):.4f}N")
