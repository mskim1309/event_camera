import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ================== 사용자 설정 ==================
csv_file = "~/data/0828/data.csv"     # CSV 파일 경로
img_size = (720, 1280)

start_time_us = 0        # 시작 시간 (μs)
end_time_us   = 400000        # 끝 시간 (μs)
# ================================================

print(f"Loading entire CSV to filter time range {start_time_us} - {end_time_us} us ...")

# 전체 파일 로드 (컬럼명 수동 지정)
df = pd.read_csv(csv_file, header=None, names=['x', 'y', 'p', 't'])

# 시간 조건으로 필터링
df_filtered = df[(df["t"] >= start_time_us) & (df["t"] < end_time_us)]

# 데이터 추출
xs = df_filtered["x"].to_numpy(dtype=np.int32)
ys = df_filtered["y"].to_numpy(dtype=np.int32)
pols = df_filtered["p"].to_numpy(dtype=np.int32)
timestamps = df_filtered["t"].to_numpy(dtype=np.int64) / 1e6  # μs → s

print(f"Number of events in selected time range: {len(xs)}")

# 색상 매핑
colors = np.where(pols > 0, 'b', 'r')

# 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs, timestamps, ys, c=colors, marker='.', s=2)

ax.set_xlabel('x [pix]')
ax.set_ylabel('time [s]')
ax.set_zlabel('y [pix]')
ax.view_init(elev=-180, azim=-90)

plt.title(f"3D Events from {start_time_us} to {end_time_us} μs")
plt.show()
