import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from simulation_func.lattice_boltzman import LatticeBoltzmann
from utils.experiment_data import sinusoidal_density, sinusoidal_velocity, analytical_decay, viscosity


def main() -> None:
    """
    Main function to try the streaming and collision on sinusoidal density and velocity
    """
    lattice_grid_shape = (100, 100)
    epsilon = 0.01
    rho0 = 0.5
    time_steps = 10000
    omega_main = 1.0
    X = lattice_grid_shape[0]
    Y = lattice_grid_shape[1]
    density_field, velocity_field = sinusoidal_density(lattice_grid_shape=lattice_grid_shape, epsilon=epsilon,
                                                       rho0=rho0)

    densities = np.zeros((time_steps, X))
    max_position = np.argmax(density_field[0][:])
    lattice = LatticeBoltzmann(grid_x=X, grid_y=Y, omega=omega_main)
    lattice.update(rho=density_field, u=velocity_field)
    densities = lattice.run(time_steps=time_steps, get_rho=True, dens=densities)

    # Plotting analytical decay vs simulated density
    figure, ax = plt.subplots(figsize=(15, 8))
    x_axis = np.arange(time_steps)

    ax.plot(x_axis, densities[:, max_position], label="Simulated density")
    ax.plot(x_axis, rho0 + analytical_decay(x_axis, max_position, X, omega_main, epsilon), label="Analytical decay",
            linestyle="dashed", lw=1.5, color="red")
    ax.plot(x_axis, rho0 - analytical_decay(x_axis, max_position, X, omega_main, epsilon), linestyle="dashed", lw=1.5,
            color="red")
    plt.title("omega: " + str(omega_main))
    plt.xlabel("timestamps")
    plt.ylabel("a(t)")
    ax.grid()
    ax.legend()
    plt.savefig(f'figures/Milestone3/ShearWaveDecay2.png', bbox_inches='tight')
    plt.show()

    # Plotting Multiple Omegas Shear wave Decay

    multiple_omegas = np.arange(0.2, 2.2, 0.4)
    x_axis = np.arange(time_steps)
    multiple_densities = np.zeros((multiple_omegas.shape[0], time_steps, X))
    # plt.close()

    plt.style.use('classic')
    plt.figure(figsize=(18, 10))
    for idx, omegas_any in enumerate(multiple_omegas):
        densities = np.zeros((time_steps, X))
        lattice = LatticeBoltzmann(grid_x=X, grid_y=Y, omega=omegas_any)
        lattice.update(rho=density_field, u=velocity_field)
        densities = lattice.run(time_steps=time_steps, get_rho=True, dens=densities)
        multiple_densities[idx, ...] = densities
        # get absolut function for more function points
        absolut_function = np.abs(densities[:, max_position] - rho0)
        # print("I0", idx, absolut_function)
        # normalizing
        absolut_function = absolut_function / epsilon
        # print("I1", idx, absolut_function)
        maxima_s = argrelextrema(absolut_function, np.greater, mode='wrap')
        plt.plot(x_axis, (analytical_decay(x_axis, max_position, X, omegas_any, epsilon) / epsilon),
                 label="theoretical" if idx == 0 else "", linestyle="dashed", lw=1.5, color="red")
        plt.plot(maxima_s[0][::2], absolut_function[maxima_s[0]][::2], 'x', label=round(omegas_any, 2), markersize=8,
                 linewidth=4)
        # print("I2", idx, maxima_s[0][::2])
        # print("I3", idx, absolut_function[maxima_s[0]][::2])

    plt.title("Shear Wave decay with multiple omegas")
    plt.xlabel("Time steps")
    plt.ylabel("a(t)/a(0)")
    plt.ylim(-0.25, 1.25)
    plt.grid()
    plt.legend(title="omega", loc='upper right', fancybox=True, numpoints=1)
    plt.savefig(f'figures/Milestone3/Shear_Wave_Decay_Multiple_Omega3.png', bbox_inches='tight')
    plt.show()

    # Plotting Kinematic Viscosity
    theo_viscosity = np.zeros((multiple_omegas.shape[0]))
    measured_viscosity = np.zeros((multiple_omegas.shape[0]))
    for idx, omega in enumerate(multiple_omegas):
        # theoretical viscosity
        theo_viscosity[idx] = viscosity(omega)
        # measured viscosity
        vals = multiple_densities[idx, ...][:, max_position]
        densities = np.array(np.abs(vals - rho0))
        # get absolut function for more function points
        peaks = argrelextrema(densities, np.greater, mode='wrap')[0]
        densities = densities[peaks]
        measured_viscosity[idx] = curve_fit(lambda t, v: epsilon * np.exp(-v * t * (2 * np.pi / X) ** 2),
                                            xdata=peaks,
                                            ydata=densities)[0][0]
    plt.figure(figsize=(15, 9))
    plt.title("Kinematic Viscosity for multiple omegas")
    plt.xlabel("Omega ω")
    plt.ylabel("Kinematic Viscosity")
    plt.grid()
    plt.plot(multiple_omegas, theo_viscosity, 'b', label="Theoretical Viscosity")
    plt.plot(multiple_omegas, measured_viscosity, 'r', label="Measured Viscosity")
    plt.legend(title="legend", loc='upper right')
    plt.savefig(f'figures/Milestone3/KinematicViscosity2.png', bbox_inches='tight')
    plt.show()

    # Plotting Velocity profile
    density_field_1, velocity_field_1 = sinusoidal_velocity(lattice_grid_shape=lattice_grid_shape, epsilon=epsilon)
    velocities = np.zeros((time_steps, Y))
    max_position_1 = np.argmax(velocity_field_1[1][:][0])
    lattice1 = LatticeBoltzmann(grid_x=X, grid_y=Y, omega=omega_main)
    lattice1.update(rho=density_field_1, u=velocity_field_1)
    velocities = lattice1.run(time_steps=time_steps, get_vel=True, vels=velocities, max_pos_vel=max_position_1)
    print("velocities", velocities.shape)
    print("velocities", velocities)
    plt.figure(figsize=(8, 8))
    velo_timestamps = np.arange(0, 2600, 500)
    for time in velo_timestamps:
        plt.plot(velocities[time, :], range(Y), label=time)
    plt.legend(title="timestamps", loc='upper right')
    plt.title("velocity Field with Sinus Omega " + str(omega_main))
    plt.xlabel("velocity")
    plt.ylabel("Y dimension")
    plt.grid()
    plt.xlim(-epsilon * 1.2, epsilon * 1.2)
    plt.savefig(f'figures/Milestone3/VelocityOverTime.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
