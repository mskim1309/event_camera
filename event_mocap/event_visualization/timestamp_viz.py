import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ================== 사용자 설정 ==================
csv_file = "~/data/0812/test2.csv"     # CSV 파일 경로
img_size = (720, 1280)

row_start = 0       # 읽을 시작 행 (0부터)
row_end = 10_000         # 읽을 끝 행 (이 행 전까지)
# ================================================

print(f"Loading CSV rows {row_start}:{row_end} ...")

# 특정 구간만 읽기 (header=None, skiprows, nrows 사용)
nrows = row_end - row_start
df = pd.read_csv(
    csv_file,
    header=None,
    names=['x', 'y', 'p', 't'],
    skiprows=row_start,
    nrows=nrows
)

# numpy 변환
xs = df["x"].to_numpy(dtype=np.int32)
ys = df["y"].to_numpy(dtype=np.int32)
pols = df["p"].to_numpy(dtype=np.int32)
timestamps = df["t"].to_numpy(dtype=np.int64) / 1e6  # μs → s

# 색상 (ON=blue, OFF=red)
colors = np.where(pols > 0, 'b', 'r')

# 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs, timestamps, ys, c=colors, marker='.', s=2)

ax.set_xlabel('x [pix]')
ax.set_ylabel('time [s]')
ax.set_zlabel('y [pix]')
ax.view_init(elev=-180, azim=-90)

plt.title(f"3D Events rows {row_start} to {row_end}")
plt.show()
