import numpy as np
import matplotlib.pyplot as plt

from simulation_func.lattice_boltzman import LatticeBoltzmann


def main() -> None:
    """
    The function that simulates Couette Flow. In this milestone we will implement a moving wall. This scenario, in which
    the fluid flows between a fixed and a moving wall, is known as the Couette flow.
    :return: Plots the Couette Flow
    """

    lattice_grid_shape = (100, 100)
    epsilon = 0.1
    # rho0 = 0.5
    time_steps = 50000
    omega_main = 1.3
    X = lattice_grid_shape[0]
    Y = lattice_grid_shape[1]
    rho_nm = np.ones((X, Y))

    lattice = LatticeBoltzmann(grid_x=X, grid_y=Y, omega=omega_main, init_rho=rho_nm)
    # lattice.update(rho=rho_nm)
    wall_vel = 1
    measurement_point = X // 2
    # velocity_field = np.empty((time_steps + 1, X, Y))
    velocity_field = lattice.couette_run(time_steps=time_steps)
    print()

    '''Plotting the Couette Flow'''
    fig, ax = plt.subplots()
    y_data = np.arange(X)
    for t in range(time_steps):
        if t < 2000:
            if t % 500 == 0:
                x_data = velocity_field[t, :, measurement_point]
                ax.plot(x_data, y_data, label=f"$t=${t}")
        elif 2000 <= t <= 5000:
            if t % 1000 == 0:
                x_data = velocity_field[t, :, measurement_point]
                ax.plot(x_data, y_data, label=f"$t=${t}")
        elif 5000 <= t <= 20000:
            if t % 5000 == 0:
                x_data = velocity_field[t, :, measurement_point]
                ax.plot(x_data, y_data, label=f"$t=${t}")
        elif t > 20000:
            if t % 10000 == 0:
                x_data = velocity_field[t, :, measurement_point]
                ax.plot(x_data, y_data, label=f"$t=${t}")

    ax.set_xlabel(f"velocity at x={measurement_point}")
    ax.set_ylabel("y-axis")
    ax.plot(np.flip((y_data + 1) / X * wall_vel), y_data, "k-.", label="analytical\nsolution")
    plt.legend(loc=1)
    plt.savefig(f'figures/Milestone4/Couette_Flow2.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
