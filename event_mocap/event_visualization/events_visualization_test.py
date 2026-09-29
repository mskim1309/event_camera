import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ================== 사용자 설정 ==================
csv_file = "~/data/1017/data.csv"     # CSV 파일 경로
img_size = (720, 1280)
m = 100000                   # 표시할 이벤트 개수
# ================================================

print("Loading CSV...")
# CSV 헤더 없음 → 컬럼 이름 직접 지정
df = pd.read_csv(csv_file, header=None, names=['x', 'y', 'p', 't'])

# numpy 변환
xs = df["x"].to_numpy(dtype=np.int32)[:m]
ys = df["y"].to_numpy(dtype=np.int32)[:m]
pols = df["p"].to_numpy(dtype=np.int32)[:m]
timestamps = df["t"].to_numpy(dtype=np.int64)[:m] / 1e6  # μs → s

# 폴라리티에 따른 색상
colors = np.where(pols > 0, 'b', 'r')  # ON = blue, OFF = red

# 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# scatter(x, y축, 시간축) : y축과 시간축 순서 주의!
ax.scatter(xs, timestamps, ys, c=colors, marker='.', s=2)

ax.set_xlabel('x [pix]')
ax.set_ylabel('time [s]')
ax.set_zlabel('y [pix]')
# 보기 좋은 뷰포인트 (x 수평, time 수평, y 수직)
ax.view_init(elev=-180, azim=-90)

plt.title(f"3D Event Visualization (first {m} events)")
plt.show()
