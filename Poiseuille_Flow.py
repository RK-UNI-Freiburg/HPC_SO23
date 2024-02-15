import numpy as np
import matplotlib.pyplot as plt

from simulation_func.lattice_boltzman import LatticeBoltzmann


def main() -> None:
    """
    The function that simulates Poiseuille Flow. IIn this milestone we simulate the flow in a pipe which is driven by a
    pressure difference between the outlet and the inlet. This category of flows is called Hagen-Poiseuille flow.
    :return: Plots the Poiseuille Flow
    """

    lattice_grid_shape = (50, 50)
    epsilon = 0.1
    time_steps = 15000
    omega_main = 1.0
    X = lattice_grid_shape[0]
    Y = lattice_grid_shape[1] + 2
    rho_nm = np.ones((X, Y))
    rho_in_out = np.array([1.005, 0.995])
    lattice = LatticeBoltzmann(grid_x=X, grid_y=Y, omega=omega_main, init_rho=rho_nm)

    measurement_point = X // 2
    velocity_field = lattice.poiseuille_run(time_steps=time_steps, rho_in_out=rho_in_out)

    viscosity = 1 / 3 * (1 / omega_main - 0.5)

    '''Plotting the pressure derivative and velocity Profile at measurement point at the time steps '''

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.invert_yaxis()
    ax.set_xlabel(f"velocity at x={measurement_point}")
    ax.set_ylabel("y-axis")

    y_data = np.arange(X)

    for t in range(time_steps):
        if t < 2000:
            if t % 300 == 0:
                x_data = velocity_field[t, 1, :, measurement_point]
                ax.plot(x_data, y_data, label=f"$t=${t:,}")
        elif t >= 2000:
            if t % 1000 == 0:
                x_data = velocity_field[t, 1, :, measurement_point]
                ax.plot(x_data, y_data, label=f"$t=${t:,}")

    '''plotting analytical solution'''
    y_data = np.append(y_data, X)

    '''calculating pressure derivative'''
    pressure_der = (rho_in_out[1] - rho_in_out[0]) / 3.0 / Y

    '''calculating dynamic viscosity as per the formula'''
    dynamic_viscosity = viscosity * rho_nm[:, measurement_point].mean()

    analytical = - 0.5 / dynamic_viscosity * pressure_der * y_data * (X - y_data)
    ax.plot(analytical, y_data - 0.5, linestyle="dotted", label="analytical\nsolution")
    ax.legend()
    plt.savefig(f'figures/Milestone5/Poiseuille_Flow.png', bbox_inches='tight')
    plt.show()

    '''Plotting flow fields or streamlines'''
    fig, ax = plt.subplots()
    ax.set_ylabel("y-axis")
    ax.set_xlabel("x-axis")
    x, y, v, u = np.arange(Y), np.arange(X), velocity_field[-1, 0], velocity_field[-1, 1]
    plt.streamplot(x, y, u, v, color=u, density=1.3)
    ax.set_title("flow field at final time step")
    ax.invert_yaxis()
    ax.legend(loc=1)
    plt.savefig(f'figures/Milestone5/Flow_Fields.png', bbox_inches='tight')
    plt.show()

    '''Plotting density gradient across center'''

    plt.figure()
    plt.xlabel("x-axis")
    plt.ylabel("density")
    plt.plot(lattice.get_rho()[X // 2, 1:-1], label=f"density at y={X // 2}, t={time_steps})")
    plt.legend(loc=1)
    plt.savefig(f'figures/Milestone5/Density_Gradient_center.png', bbox_inches='tight')
    plt.show()

    '''Plotting velocity profile and calculating change in area'''

    plt.figure()
    plt.xlabel("x-axis")
    plt.ylabel("area of velocity profile at final time-step")
    x_data = np.arange(1, Y - 1)
    y_data = np.array([np.sum(velocity_field[time_steps, 1, :, x]) for x in x_data])
    plt.plot(x_data, y_data)
    area = (y_data[-1] / y_data[0] - 1) * 100
    plt.text(0.5, 0.95, f"Increase in area: {area:.3f}%", transform=plt.gca().transAxes)
    plt.legend(loc=1)
    plt.savefig(f'figures/Milestone5/Velocity_profile_area_change.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
