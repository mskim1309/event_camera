import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# ================== 사용자 설정 ==================
csv_file = "~/data/out_test.csv"     # CSV 파일 경로
img_size = (720, 1280)
accum_duration = 250000         # 50ms = 50,000 μs
chunksize = 100000             # CSV 읽기 단위 (행)
# 시간 구간 (μs 단위)
target_start = 0       # 예: 2초
target_end   = 10_000_000       # 예: 3초
# ================================================

colnames = ['x', 'y', 'p', 't']

print("Reading CSV in chunks and filtering by time range...")

# 프레임 누적용 딕셔너리
frames = {}

# 시간 bin 경계 계산
bin_edges = np.arange(target_start, target_end, accum_duration)
num_bins = len(bin_edges) - 1

# bin 인덱스 계산 함수
def bin_index(t):
    return int((t - target_start) // accum_duration)

# CSV를 chunk 단위로 읽고 필요한 구간만 처리
reader = pd.read_csv(csv_file, header=None, names=colnames, chunksize=chunksize)
for chunk in reader:
    # 시간 구간으로 필터링
    chunk = chunk[(chunk["t"] >= target_start) & (chunk["t"] < target_end)]
    if chunk.empty:
        continue

    # bin에 맞춰 누적
    for x, y, p, t in zip(chunk["x"], chunk["y"], chunk["p"], chunk["t"]):
        idx = bin_index(t)
        if idx < 0 or idx >= num_bins:
            continue
        if idx not in frames:
            frames[idx] = np.zeros(img_size, dtype=np.int32)
        frames[idx][y, x] += (2 * p - 1)

# 누적한 프레임들을 리스트로 정렬
sorted_indices = sorted(frames.keys())
frame_imgs = [frames[i] for i in sorted_indices]

print(f"총 {len(frame_imgs)} 개의 프레임 생성 완료.")

# --------------------- 시각화 ---------------------
# 서브플롯으로 한 번에 비교
cols = min(5, len(frame_imgs))  # 한 행당 최대 5개
rows = math.ceil(len(frame_imgs) / cols)
fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
if rows == 1:
    axes = np.array([axes])
axes = axes.flatten()

for i, img in enumerate(frame_imgs):
    vmax = max(1, np.max(np.abs(img)))
    ax = axes[i]
    ax.imshow(img, cmap='seismic', vmin=-vmax, vmax=vmax)
    start_t = bin_edges[sorted_indices[i]] / 1e6
    end_t = bin_edges[sorted_indices[i]+1] / 1e6
    ax.set_title(f"{start_t:.2f}-{end_t:.2f}s")
    ax.axis('off')

# 남는 subplot 칸 비우기
for j in range(len(frame_imgs), len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()
