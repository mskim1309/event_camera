import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# CSV 읽기
pose_df = pd.read_csv("~/event_mocap/3d_pose/pose/pose_log_1119_id_fix.csv")

# 이상치 제거: x, y, z 중 하나라도 절댓값이 10000 이상인 행 제외
mask = (pose_df['x'].abs() < 5000) & (pose_df['y'].abs() < 5000) & (pose_df['z'].abs() < 5000)
filtered_df = pose_df[mask]

print(f"Original data: {len(pose_df)} points")
print(f"Filtered data: {len(filtered_df)} points (removed {len(pose_df) - len(filtered_df)} outliers)")

x = filtered_df['x'].values
y = filtered_df['y'].values
z = filtered_df['z'].values
t = filtered_df['timestamp'].values

# 시간은 색상으로 표현
t_norm = (t - t.min()) / (t.max() - t.min())

# 시각화
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(x, y, z, c=t_norm, cmap='viridis', marker='o', s=5)
ax.set_box_aspect([1, 1, 1])  # 축 비율 동일하게 유지

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Normalized Time')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Active Marker 3D Trajectory (Outliers Removed)')

plt.show()
