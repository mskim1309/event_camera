import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

csv_file = "~/data/multiLED.csv"     # CSV 파일 경로

# 시간 구간 (마이크로초 단위)
start_time_us = 1_000_000
end_time_us   = 10_000_000
max_points    = 100000  # 샘플링 최대 표시 이벤트 수

chunksize = 100000  # 한 번에 10만 행씩 읽음

colnames = ['x', 'y', 'p', 't']
filtered_chunks = []

print("Reading CSV in chunks...")
for chunk in pd.read_csv(csv_file, header=None, names=colnames, chunksize=chunksize):
    # 1. 시간 조건으로 필터링
    mask = (chunk['t'] >= start_time_us) & (chunk['t'] < end_time_us)
    selected = chunk[mask]
    if not selected.empty:
        filtered_chunks.append(selected)

    # 2. chunk의 최대 timestamp가 end_time_us 넘으면 중단
    if chunk['t'].iloc[-1] > end_time_us:
        break

# 구간에 맞는 chunk들을 합침
if filtered_chunks:
    df_filtered = pd.concat(filtered_chunks, ignore_index=True)
else:
    print("No events in selected time range.")
    exit()

total_events = len(df_filtered)
print(f"Total events in selected range: {total_events}")

# 샘플링
if total_events > max_points:
    step = total_events // max_points
    df_filtered = df_filtered.iloc[::step]
    print(f"Downsampled to {len(df_filtered)} events (step={step})")

# numpy 변환
xs = df_filtered["x"].to_numpy(np.int32)
ys = df_filtered["y"].to_numpy(np.int32)
pols = df_filtered["p"].to_numpy(np.int32)
timestamps = df_filtered["t"].to_numpy(np.int64) / 1e6  # μs → s
colors = np.where(pols > 0, 'b', 'r')

# 3D 시각화
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs, timestamps, ys, c=colors, marker='.', s=2)

ax.set_xlabel('x [pix]')
ax.set_ylabel('time [s]')
ax.set_zlabel('y [pix]')
ax.view_init(elev=-180, azim=-90)

plt.title(f"3D Events {start_time_us}-{end_time_us} us (sampled)")
plt.show()
