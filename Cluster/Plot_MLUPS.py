import matplotlib.pyplot as plt
import numpy as np

'''300x300'''
n900 = 162.71
n676 = 173.17
n400 = 155.199
n256 = 162.57
n144 = 136.40
n64 = 139.53
n36 = 110.388
n16 = 76.923
n4 = 17.4

'''500x500'''
l900 = 426.31
l676 = 448.10
l400 = 371.41
l256 = 344.59
l144 = 275.81
l64 = 231.76
l36 = 162.61
l16 = 80.25
l4 = 16.65

'''plotting'''
fig, ax = plt.subplots()
ncpu = np.array([4, 16, 36, 64, 144, 256, 400, 900])
x = np.array([n4, n16, n36, n64, n144, n256, n400, n900])
y = np.array([l4, l16, l36, l64, l144, l256, l400, l900])
ax.plot(ncpu, x, marker='o', label="300x300")
ax.plot(ncpu, y, marker='o', label="500x500")
# ax = plt.gca()
# ax.set_xlim([-10, 1000])
# ax.set_ylim([0, 1000])
plt.xlabel("No. of processes")
plt.ylabel("Million lattice updates per second (MLUPS)")
plt.legend()
plt.suptitle("MLUPS vs. No. of Processes", fontweight="bold")
plt.savefig(f'C:/Users/rouna/PycharmProjects/HPC/figures/MLUPS.png', bbox_inches='tight')
plt.show()