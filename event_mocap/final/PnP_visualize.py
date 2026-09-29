import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# CSV 읽기
pose_df = pd.read_csv("~/event_mocap/final/pose/pose_log_1119_idfix_camera_bundle_sync.csv")

# 이상치 제거
mask = (pose_df['x'].abs() < 5000) & (pose_df['y'].abs() < 5000) & (pose_df['z'].abs() < 5000)
filtered_df = pose_df[mask]

print(f"Original data: {len(pose_df)} points")
print(f"Filtered data: {len(filtered_df)} points (removed {len(pose_df) - len(filtered_df)} outliers)")

x = filtered_df['x'].values
y = filtered_df['y'].values
z = filtered_df['z'].values
t = filtered_df['timestamp'].values

# 시간 정규화
t_norm = (t - t.min()) / (t.max() - t.min())

# 기준 marker 4개
marker_positions = [
    (75, 62, 0),
    (-75, 62, 0),
    (75, -62, 0),
    (-75, -62, 0),
]

sphere_radius = 20

# 구 생성
u = np.linspace(0, 2 * np.pi, 20)
v = np.linspace(0, np.pi, 20)
sphere_x = np.outer(np.cos(u), np.sin(v))
sphere_y = np.outer(np.sin(u), np.sin(v))
sphere_z = np.outer(np.ones_like(u), np.cos(v))

# ===== 축 스케일 동일하게 맞추는 함수 =====
def set_equal_aspect(ax):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()

    x_range = abs(xlim[1] - xlim[0])
    y_range = abs(ylim[1] - ylim[0])
    z_range = abs(zlim[1] - zlim[0])

    max_range = max(x_range, y_range, z_range)

    x_middle = np.mean(xlim)
    y_middle = np.mean(ylim)
    z_middle = np.mean(zlim)

    ax.set_xlim(x_middle - max_range/2, x_middle + max_range/2)
    ax.set_ylim(y_middle - max_range/2, y_middle + max_range/2)
    ax.set_zlim(z_middle - max_range/2, z_middle + max_range/2)
# ========================================

# 시각화
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Trajectory
sc = ax.scatter(x, y, z, c=t_norm, cmap='viridis', marker='o', s=5)

# Colorbar
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Normalized Time')

# Reference spheres
for (cx, cy, cz) in marker_positions:
    ax.plot_surface(
        sphere_x * sphere_radius + cx,
        sphere_y * sphere_radius + cy,
        sphere_z * sphere_radius + cz,
        rstride=1,
        cstride=1,
        color='red',
        alpha=0.6,
        linewidth=0
    )

# 축 스케일 동일하게!
set_equal_aspect(ax)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Active Marker 3D Trajectory')

plt.show()
