import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# CSV 읽기
pose_df = pd.read_csv("~/event_mocap/final/pose/pose_log_1119_idfix_camera_bundle_sync.csv")

# 이상치 제거: x, y, z 중 하나라도 절댓값이 10000 이상인 행 제외
mask = (pose_df['x'].abs() < 5000) & (pose_df['y'].abs() < 5000) & (pose_df['z'].abs() < 5000)
filtered_df = pose_df[mask]

plt.figure(figsize=(12,8))

plt.subplot(3,2,1)
plt.plot(filtered_df['x'])
plt.ylabel('x')
plt.grid(True)
plt.title("Position (x, y, z) over time")

plt.subplot(3,2,3)
plt.plot(filtered_df['y'])
plt.ylabel('y')
plt.grid(True)

plt.subplot(3,2,5)
plt.plot(filtered_df['z'])
plt.ylabel('z')
plt.grid(True)
plt.xlabel("Frame index")

plt.subplot(3,2,2)
plt.plot(filtered_df['rvec_x'])
plt.ylabel('rvec_x')
plt.grid(True)
plt.title("Rvec (x, y, z) over time")

plt.subplot(3,2,4)
plt.plot(filtered_df['rvec_y'])
plt.ylabel('rvec_y')
plt.grid(True)

plt.subplot(3,2,6)
plt.plot(filtered_df['rvec_z'])
plt.ylabel('rvec_z')
plt.grid(True)
plt.xlabel("Frame index")

plt.tight_layout()
plt.show()
