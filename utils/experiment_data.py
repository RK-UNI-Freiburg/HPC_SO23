import numpy as np
from typing import Tuple


def sinusoidal_density(lattice_grid_shape: Tuple[int, int], epsilon: float,
                       rho0: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return initial values according to
    rho(0, x, y) = rho0 + eps*sin(2PI x/X)
    and v(o, x, y) = 0

    Args:
        :param rho0: The offset of density
        :param epsilon: amplitude of swinging
        :param lattice_grid_shape: shape of the lattice grid

    Returns:
        density := rho(x, y, 0) = rho0 + eps * sin(2PI x/X)
        v(x, y, 0) = 0
    """
    assert rho0 + epsilon < 2
    assert rho0 - epsilon > 0
    X, Y = lattice_grid_shape
    x = np.arange(X)
    y = np.arange(Y)
    X, Y = np.meshgrid(x, y)
    vel = np.zeros((2, *lattice_grid_shape))
    density = rho0 + epsilon * np.sin(2 * np.pi * X / lattice_grid_shape[0])

    return density, vel


def sinusoidal_velocity(lattice_grid_shape: Tuple[int, int], epsilon: float
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return initial values according to
    rho(0, x, y) = 1 and v(0, x, y) = eps * sin(2PI * y/Y)

    Args:
        :param lattice_grid_shape: shape of the lattice grid
        :param epsilon: amplitude of swinging

    Returns:
        density := rho(x, y, 0) = 1
        u(0, x, y) = eps * sin(2PI * y/Y)

    Note:
        constraint |u| < 0.1

    """
    assert abs(epsilon) < 0.1

    X, Y = lattice_grid_shape
    density = np.ones(lattice_grid_shape)
    vel = np.zeros((2, *lattice_grid_shape))
    x = np.arange(X)
    y = np.arange(Y)
    X, Y = np.meshgrid(x, y)
    vel[1, ...] = epsilon * np.sin(2 * np.pi * Y / lattice_grid_shape[1])

    return density, vel


def viscosity(omega: float) -> float:
    """
    Return the viscosity of the fluid
    """
    return (1.0 / 3.0) * (1.0 / omega - 0.5)


def analytical_decay(times: np.ndarray, x: np.ndarray, x_total: int, omega: float, epsilon: float) -> float:
    """
    Return the analytical viscosity of the fluid
    """
    vis = viscosity(omega)
    lx = 2 * np.pi / x_total
    ana_vis = epsilon * np.exp(-vis * lx ** 2 * times) * np.sin(lx * x)
    return ana_vis
