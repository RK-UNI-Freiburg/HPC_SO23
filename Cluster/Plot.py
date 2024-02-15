import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    u = np.load('ux_300_900.npy')
    v = np.load('uy_300_900.npy')

    L = u[0].size

    fig, ax = plt.subplots()

    x = np.arange(L)
    y = np.arange(L)
    plt.gca().invert_yaxis()

    plt.suptitle("Sliding Lid on BWUniCluster with 900 processes", fontweight="bold")

    norm = plt.Normalize(0, 0.1)
    ax.streamplot(x, y, v, u, color=np.sqrt(u ** 2 + v ** 2) if u.sum() and v.sum() else None, norm=norm, density=2.0)
    plt.savefig(f'C:/Users/rouna/PycharmProjects/HPC/figures/Milestone6Sliding_Lid_Parallel_Cluster_900_300.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
