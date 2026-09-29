import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 회전행렬 → quaternion 변환 함수
def rotmat_to_quat(R):
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]

    trace = m00 + m11 + m22

    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2  # S=4*qw
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S

    return np.array([qw, qx, qy, qz])


# CSV 읽기
pose = pd.read_csv("~/event_mocap/camera_pose/pose/camera_pose_0819_multiLED_v2.csv")

# 회전행렬 리스트 만들기
Rs = []
for _, row in pose.iterrows():
    R = np.array([
        [row['R00'], row['R01'], row['R02']],
        [row['R10'], row['R11'], row['R12']],
        [row['R20'], row['R21'], row['R22']]
    ])
    Rs.append(R)

# R → quaternion 변환
quats = np.array([rotmat_to_quat(R) for R in Rs])  # shape (N, 4)

# 프레임 간 dot product
dots = np.sum(quats[:-1] * quats[1:], axis=1)

# =========================================================
# 1️⃣ Quaternion 연속 프레임 dot product plot
# =========================================================
plt.figure(figsize=(10,4))
plt.plot(dots, label='Quaternion dot(adjacent frames)')
plt.axhline(0, color='r', linestyle='--', label='0 threshold')
plt.title('Frame-to-frame Quaternion Dot Product (from Rotation Matrix)')
plt.xlabel('Frame index')
plt.ylabel('Dot product')
plt.legend()
plt.grid(True)
plt.show()

# =========================================================
# 2️⃣ Quaternion (w,x,y,z) 시간 변화 plot
# =========================================================
plt.figure(figsize=(12,8))

labels = ["w", "x", "y", "z"]
for i in range(4):
    plt.subplot(4, 1, i+1)
    plt.plot(quats[:, i])
    plt.ylabel(labels[i])
    plt.grid(True)
    if i == 0:
        plt.title("Quaternion Components Over Time")

plt.xlabel("Frame index")
plt.tight_layout()
plt.show()


# =========================================================
# 3️⃣ CSV 안에 있는 position x,y,z 변화 plot
# =========================================================
plt.figure(figsize=(12,8))

plt.subplot(3,1,1)
plt.plot(pose['x'])
plt.ylabel('x')
plt.grid(True)
plt.title("Position (x, y, z) over time")

plt.subplot(3,1,2)
plt.plot(pose['y'])
plt.ylabel('y')
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(pose['z'])
plt.ylabel('z')
plt.grid(True)
plt.xlabel("Frame index")

plt.tight_layout()
plt.show()
