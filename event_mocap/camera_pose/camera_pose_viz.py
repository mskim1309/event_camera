import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# CSV 읽기
pose_df = pd.read_csv("~/event_mocap/camera_pose/pose/camera_pose_1119_idfix_v2.csv")

start_idx = 0
end_idx = 800
#pose_df = pose_df.iloc[start_idx:end_idx]

mask = (pose_df['x'].abs() < 5000) & (pose_df['y'].abs() < 5000) & (pose_df['z'].abs() < 5000)
filtered_df = pose_df[mask]

x = filtered_df['x'].values
y = filtered_df['y'].values
z = filtered_df['z'].values
t = filtered_df['timestamp_us'].values

# 정규화된 timestamp를 색상으로 사용
t_norm = (t - t.min()) / (t.max() - t.min())

start_coord = np.array([x[0], y[0], z[0]])
end_coord = np.array([x[-1], y[-1], z[-1]])
diff_coord = end_coord - start_coord

print("Start coordinate:", start_coord)
print("End coordinate:", end_coord)
print("Coordinate difference:", diff_coord)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with color mapping
sc = ax.scatter(x, y, z, c=t_norm, cmap='viridis', marker='o')

# 컬러바 추가
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Normalized Time')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Active Marker 3D Trajectory')

plt.show()
