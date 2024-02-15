import math

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
        self.rho_avg = 0
        self.u = np.zeros((2, self.grid_x, self.grid_y))
        self.w_i = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
        # self.c_ai = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],
        #                     [0, 0, 1, 0, -1, 1, 1, -1, -1]])
        self.c_ai = np.array([[0, 0, -1, 0, 1, -1, -1, 1, 1],
                              [0, 1, 0, -1, 0, 1, -1, -1, 1]])
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

        for i in range(9):
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

    def update_u_wo_rho(self) -> None:
        """
            Update the velocity field without dividing by rho
        """

        self.u = np.einsum('ij,jkl->ikl', self.c_ai, self.f)

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
        # print(u2)
        feq_inm = w_rho * (1 + 3 * cu + 4.5 * sq_cu - 1.5 * u2)
        return feq_inm

    def equilibrium_dist_pdfparam(self, rho, u) -> np.ndarray:
        """
            Calculate the equilibrium distribution function
            :param u: Contains the Velocity
            :param rho: Contains the Density
        """
        '''cu = np.einsum('ai,anm->inm', self.c_ai, u)
        sq_cu = cu ** 2
        u2 = np.einsum('ijk,ijk->jk', u, u)
        w_rho = np.einsum('i,jk->ijk', self.w_i, rho)
        feq_inm = w_rho * (1 + 3 * cu + 4.5 * sq_cu - 1.5 * u2)
        return feq_inm'''

        cu_nm = (u.T @ self.c_ai).T
        sqcu_nm = cu_nm ** 2
        usq_nm = np.linalg.norm(u, axis=0) ** 2
        return self.w_i[..., np.newaxis] * rho[np.newaxis, ...] * (
                1. + 3. * cu_nm + 4.5 * sqcu_nm - 1.5 * usq_nm[np.newaxis, ...])

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

    def get_rho_avg(self):
        """
        :return: The average velocity
        """
        return self.rho_avg

    def boundary_conditions(self):
        """
        The function returns the results in case of a moving boundary.There is periodic boundary conditions in
        the x direction and bounce-back boundary conditions on the lower wall. The top boundary move to the right with a
        prescribed velocity. Applied boundary-conditions to the top wall.
        """

        '''The rigid_index[] contains the directions bottom, bot-left and bot-right and the mirrored directions in opp[] array'''
        rigid_index = np.array([4, 7, 8])
        rigid_opp = np.array([2, 5, 6])

        '''The moving_index[] contains the directions top, top-left and top-right and the mirrored directions in opp[] array'''
        moving_index = np.array([2, 5, 6])
        moving_opp = np.array([4, 7, 8])

        '''wall velocity, which is an array of (2, ) '''
        u_w = np.array([0, 1])
        c_s = 1 / math.sqrt(3)

        '''calc avg value of the density matrix'''
        total = 0
        for i in self.rho:
            for j in i:
                total += j
        self.rho_avg = total / self.rho.size

        '''Copying probability density function to a dummy variable used in the calculations and to preserve the initial 
        density function'''
        f_dummy = np.empty_like(self.f)
        for i in range(9):
            for j in range(self.grid_x):
                for k in range(self.grid_y):
                    f_dummy[i, j, k] = self.f[i, j, k]

        '''Initial Streaming to update the probability density function rho'''
        self.streaming()
        self.update_rho()

        for i in range(9):

            '''Check whether we are in channels 4, 7 or 8'''
            if i in rigid_index:

                '''rigid wall condition'''
                self.f[rigid_opp[np.where(i == rigid_index)[0][0]], -1, :] = f_dummy[i, -1, :]

            elif i in moving_index:

                '''moving wall boundary condition'''
                self.f[moving_opp[np.where(i == moving_index)[0][0]], 0, :] = f_dummy[i, 0, :] - 2 * self.w_i[
                    i] * self.rho_avg * ((self.c_ai[:, i] @ u_w).T / (c_s ** 2))

        '''Calculate the density and velocity after boundary conditions applied '''
        self.update_rho()
        self.update_u_wo_rho()

    def couette_run(self, time_steps: int) -> np.ndarray:
        """
        Runs the simulation for Couette Flow

        :param time_steps: Time steps for which the simulation is run
        :return: The updated velocity field
        """
        velocity_field = np.empty((time_steps + 1, self.grid_x, self.grid_y))
        '''print(velocity_field)
        print("ABCDSDSSDS")
        print(self.u)'''
        velocity_field[0] = self.get_u()[1]

        '''Calculating initial equilibrium pdf and density'''
        self.f = self.f + self.collision()
        self.update_rho()

        '''Applying Boundary conditions once'''
        self.boundary_conditions()

        for i in range(time_steps):
            '''calculating subsequent collision pdf'''
            self.f = self.f + self.collision()

            '''Deriving density matrix after collision'''
            self.update_rho()

            '''applying boundary conditions'''
            self.boundary_conditions()

            '''copying velocity into new matrix'''
            velocity_field[i + 1] = self.get_u()[1]

        return velocity_field

    def pressure_gradient(self, rho_in_out: np.ndarray):
        """
        The function describes the boundary conditions for the Poiseuille Flow and applies the boundary conditions and
        the pressure gradient.
        :param rho_in_out: The density at the inlet and the outlet
        """

        '''derive density matrix and initialize boundary densities'''
        self.update_rho()
        rho_inlet = np.full(self.grid_x, rho_in_out[0])
        rho_outlet = np.full(self.grid_x, rho_in_out[1])

        '''storing indices for each direction'''
        bottom_index = np.array([4, 7, 8])
        bottom_opp = np.array([2, 5, 6])
        top_index = np.array([2, 5, 6])
        top_opp = np.array([4, 7, 8])

        '''checking velocity at boundaries'''
        u_in = self.get_u()[:, :, 1]
        u_out = self.get_u()[:, :, -2]

        '''calculate inlet pdf'''
        f_eq_in = self.equilibrium_dist_pdfparam(rho_inlet, u_out)
        f_int_beg = self.f[:, :, -2] - self.equilibrium_dist_pdfparam(self.get_rho()[:, -2], u_out)
        self.f[:, :, 0] = f_eq_in + f_int_beg

        '''calculating outlet pdf'''
        f_eq_out = self.equilibrium_dist_pdfparam(rho_outlet, u_in)
        f_int_end = self.f[:, :, 1] - self.equilibrium_dist_pdfparam(self.get_rho()[:, 1], u_in)
        self.f[:, :, -1] = f_eq_out + f_int_end

        '''copying probability density function to a separate variable'''
        f_dummy = np.empty_like(self.f)

        for i in range(9):
            for j in range(self.grid_x):
                for k in range(self.grid_y):
                    f_dummy[i, j, k] = self.f[i, j, k]

        '''performing an initial streaming operation'''
        self.streaming()

        '''rigid wall condition'''
        for i in range(9):

            '''checking if we are in channels 4, 7, or 8'''
            if i in bottom_index:
                '''rigid wall boundary condition'''
                self.f[bottom_opp[np.where(i == bottom_index)[0][0]], -1, :] = f_dummy[i, -1, :]

            '''checking if we are in channels 2, 5, or 6'''
            if i in top_index:
                '''rigid wall boundary condition'''
                self.f[top_opp[np.where(i == top_index)[0][0]], -1, :] = f_dummy[i, 0, :]

        '''Calculating density and velocity using pdf'''
        self.update_rho()
        self.update_u_wo_rho()

    def poiseuille_run(self, time_steps: int, rho_in_out: np.ndarray) -> np.ndarray:
        """
        Runs the simulation for Poiseuille Flow
        :param rho_in_out: The density at the inlet and the outlet
        :param time_steps: Time steps for which the simulation is run
        :return: The updated velocity field
        """

        velocity_field = np.empty((time_steps + 1, 2, self.grid_x, self.grid_y))
        velocity_field[0] = self.get_u()

        '''Calculating initial equilibrium pdf and density'''
        self.f = self.equilibrium_dist()
        self.update_rho()
        self.update_u()
        self.f = self.f + self.collision()
        '''self.f = self.f + self.collision()
        self.update_rho()
        self.update_u()'''

        '''Applying Pressure Gradient once'''
        self.pressure_gradient(rho_in_out=rho_in_out)

        for i in range(time_steps):
            self.update_u()

            '''calculating subsequent collision pdf'''
            self.f = self.f + self.collision()

            velocity_field[i + 1] = self.get_u()

            '''Deriving density matrix after collision'''
            self.update_rho()

            '''applying Pressure Gradient'''
            self.pressure_gradient(rho_in_out=rho_in_out)

        return velocity_field

    def boundary_sliding_lid(self):
        """
        It applies the boundary checks and conditions for the sliding lid problem and updates the pdf with the same.
        """
        c_s = 1 / math.sqrt(3)

        "Fixed index markings in all the four directions are stored under xyz_index (xyz exchangeable)"

        south_index = np.array([4, 7, 8])
        north_index = np.array([2, 5, 6])
        west_index = np.array([3, 6, 7])
        east_index = np.array([1, 8, 5])

        "wall velocity"
        u_w = np.array([0, 0.1])

        "Avg of density matrix"
        rho_avg = np.mean(self.rho)

        "copying probability density function to a dummy variable"
        f_dummy = np.empty_like(self.f)
        np.copyto(f_dummy, self.f)

        "Initial Streaming and update the probability density function rho"
        self.streaming()
        self.update_rho()

        for i in range(9):

            "applying fixed/rigid wall at the south boundary"
            if i in south_index:
                self.f[north_index[np.where(i == south_index)[0][0]], -1, :] = f_dummy[i, -1, :]

            "applying fixed/rigid wall at the west boundary"
            if i in west_index:
                self.f[east_index[np.where(i == west_index)[0][0]], :, 0] = f_dummy[i, :, 0]

            "applying fixed/rigid wall at the east boundary"
            if i in east_index:
                self.f[west_index[np.where(i == east_index)[0][0]], :, -1] = f_dummy[i, :, -1]

            "applying moving/ sliding wall on the north boundary"
            if i in north_index:
                self.f[south_index[np.where(i == north_index)[0][0]], 0, :] = f_dummy[i, 0, :] - 2 * self.w_i[
                    i] * rho_avg * ((self.c_ai[:, i] @ u_w).T / (c_s ** 2))

    def sliding_serial_run(self, time_steps: int) -> np.ndarray:
        """
        Runs the simulation for sliding lid problem
        :param time_steps: Time steps for which the simulation is run
        :return: The updated velocity field
        """

        velocity_field = np.empty_like(self.u)

        "Calculating initial equilibrium pdf"
        self.f = self.equilibrium_dist()
        self.update_rho()
        self.update_u()
        self.f = self.f + self.collision()

        "Calculating density after collision"
        self.update_rho()

        "Applying the Sliding Lid Boundary conditions"
        self.boundary_sliding_lid()

        for i in range(time_steps):

            self.update_rho()
            self.update_u()
            "calculating subsequent collision pdf"
            self.f = self.f + self.collision()

            "Applying the Sliding Lid Boundary conditions"
            self.boundary_sliding_lid()

        velocity_field = self.u

        return velocity_field


