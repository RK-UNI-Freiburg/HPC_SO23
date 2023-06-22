import numpy as np
from typing import Optional
import matplotlib.pyplot as plt


class LatticeBoltzmann:
    def __init__(self, grid_x: int, grid_y: int, omega: float = 0.5,
                 init_f: Optional[np.ndarray] = None,
                 init_rho: Optional[np.ndarray] = None,
                 init_u: Optional[np.ndarray] = None,
                 ):
        """
        :param grid_x: Number of grid points in x direction
        :param grid_y: Number of grid points in y direction
        :param omega: Relaxation parameter
        """
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.u = init_u
        self.rho = init_rho
        self.f = init_f
        self.omega = omega
        self.u = np.zeros((2, self.grid_x, self.grid_y))
        self.w_i = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
        self.c_ai = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],
                              [0, 0, 1, 0, -1, 1, 1, -1, -1]])
        self.f = np.einsum('i,jk->ijk', self.w_i, np.ones((self.grid_x, self.grid_y)))

    def update(self, rho: Optional[np.ndarray] = None, u: Optional[np.ndarray] = None):
        """
            Update the distribution function, density and velocity fields
            :param rho: Updated density
            :param u: Updated velocity
        """
        self.rho = rho
        self.u = u

    def update_f(self, f_new: Optional[float] = None, i: Optional[int] = None, j: Optional[int] = None):
        """
            Update the distribution function
            :param j:jth index of the distribution function
            :param i:ith index of the distribution function
            :param f_new: Updated distribution function
        """

        self.f[:, i, j] = f_new * self.f[:, i, j]

    def streaming(self) -> None:
        """
            Streaming step of the LBM
        """

        for i in np.arange(1, 9):
            self.f[i] = np.roll(self.f[i], shift=self.c_ai.T[i], axis=[0, 1])

    def update_rho(self) -> None:
        """
            Update the density field
        """

        self.rho = np.einsum('ijk->jk', self.f)

    def update_u(self) -> None:
        """
            Update the velocity field
        """

        self.u = np.einsum('ij,jkl->ikl', self.c_ai, self.f) / self.rho

    def plot(self) -> None:
        """
            Plot the density field
        """

        plt.imshow(self.rho)
        plt.show()

    def equilibrium_dist(self) -> np.ndarray:
        """
            Calculate the equilibrium distribution function
        """
        cu = np.einsum('ai,anm->inm', self.c_ai, self.u)
        sq_cu = cu ** 2
        u2 = np.einsum('ijk,ijk->jk', self.u, self.u)
        w_rho = np.einsum('i,jk->ijk', self.w_i, self.rho)
        feq_inm = w_rho * (1 + 3 * cu + 4.5 * sq_cu - 1.5 * u2)
        return feq_inm

    def collision(self) -> np.ndarray:
        """
            Collision step of the LBM
        """
        feq_inm = self.equilibrium_dist()
        return self.omega * (feq_inm - self.f)

    def run(self, time_steps: int, get_rho: bool = False, get_vel: bool = False, dens: Optional[np.ndarray] = None,
            vels: Optional[np.ndarray] = None, max_pos_vel: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
            Run the simulation
            :param max_pos_vel: position of max velocity
            :param vels: Optional parameter to store the certain velocity field values
            :param get_vel: If the velocity field is to be returned
            :param dens: Optional parameter to store the certain density field values
            :param get_rho: If the density field is to be returned
            :param time_steps: Number of time steps
        """
        self.f = self.equilibrium_dist()
        if get_rho:
            for i in range(time_steps):
                self.f = self.f + self.collision()
                self.streaming()
                self.update_rho()
                self.update_u()
                dens[i] = self.get_rho()[0]
        elif get_vel:
            for i in range(time_steps):
                self.f = self.f + self.collision()
                self.streaming()
                self.update_rho()
                self.update_u()
                vels[i] = self.get_u()[1, :, max_pos_vel]
        else:
            for i in range(time_steps):
                self.f = self.f + self.collision()
                self.streaming()
                self.update_rho()
                self.update_u()

        if get_rho:
            return dens
        elif get_vel:
            return vels

    def get_rho(self):
        """
        :return: The density field
        """
        return self.rho

    def get_u(self):
        """
        :return: The velocity field
        """
        return self.u
