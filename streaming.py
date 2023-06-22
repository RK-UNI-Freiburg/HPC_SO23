import numpy as np
import matplotlib.pyplot as plt
from simulation_func.lattice_boltzman import LatticeBoltzmann


'''def streaming(f_inm, c_ai):
    """
    Streaming step of the LBM
    :param f_inm: f_inm[i] is the distribution function of the i-th velocity at the m-th node
    :param c_ai: c_ai.T[i] is the i-th velocity vector
    """
    for i in np.arange(1, 9):
        f_inm[i] = np.roll(f_inm[i], shift=c_ai.T[i], axis=[0, 1])
    return f_inm'''


def main() -> None:
    """
    Main function of the streaming.py module which to initialize the grid and run the simulation
    """
    x_grid, y_grid = 25, 10
    time_steps = 10

    field = LatticeBoltzmann(grid_x=x_grid, grid_y=y_grid)
    field.update(f_new=1.01, i=x_grid//2, j=y_grid//2)
    for i in range(time_steps):
        field.streaming()
        field.update_rho()
        field.update_u()
        field.plot()


if __name__ == '__main__':
    main()


