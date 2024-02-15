import numpy as np
import matplotlib.pyplot as plt

from simulation_func.lattice_boltzman import LatticeBoltzmann


def main() -> None:
    """
    The function that simulates the Sliding Lid problem. In this milestone we will implement a moving wall. This scenario, in which
    the fluid flows between a fixed and a moving wall, is known as the Couette flow.
    :return: Plots the Couette Flow
    """

    lattice_grid_shape = (100, 100)
    epsilon = 0.1
    # rho0 = 0.5
    time_steps = 20000
    omega_main = 1.3
    X = lattice_grid_shape[0]
    Y = lattice_grid_shape[1]
    rho_nm = np.ones((X, Y))
    L = X
    wall_vel = 0.5

    "calculating analytical viscosity"
    viscosity = 1 / 3 * ((1 / omega_main) - 0.5)

    "Calculating Reynold's Number"
    ReyNum = (L * wall_vel) / viscosity

    lattice = LatticeBoltzmann(grid_x=X, grid_y=Y, omega=omega_main, init_rho=rho_nm)
    velocity_field = lattice.sliding_serial_run(time_steps=time_steps)

    "Plotting the sliding Lid Visualization"

    fig, ax = plt.subplots()
    x = np.arange(L)
    y = np.arange(L)
    u = velocity_field[1]
    v = velocity_field[0]
    norm = plt.Normalize(0, wall_vel)
    ax.streamplot(x, y, u, v, color=np.sqrt(u ** 2 + v ** 2) if u.sum() and v.sum() else None, norm=norm, density=2.0)
    fig.tight_layout()
    ax.invert_yaxis()
    plt.title(
        f"\n$Re={round(ReyNum)}$, $\\nu={viscosity:.2f}$, $Omega={omega_main:.2f}$, $U_w={wall_vel}$, $t={time_steps}$")
    plt.savefig(f'figures/Milestone6/Sliding_Lid_Serial_1.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
