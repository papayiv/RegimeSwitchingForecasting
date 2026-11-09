"""Finite-difference solver (Crank-Nicolson + a few implicit Euler start steps) for the
coupled CDF equation in pde/__init__.py."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

DT_DAY = 1 / 252


def solve_cdf_fd(params, horizons, z_max=1.0, nz=4001, steps_per_day=50, n_rannacher=4):
    """Returns z grid (nz,) and dict h -> array (m, nz) with F_i(h trading days, z)."""
    m = params.m
    z = np.linspace(-z_max, z_max, nz); dz = z[1] - z[0]
    n_in = nz - 2
    dt = DT_DAY / steps_per_day
    blocks, ups = [], []
    for i in range(m):
        a, b = 0.5 * params.sigma[i] ** 2, -params.drift[i]
        lo, di, up = a / dz**2 - b / (2 * dz), -2 * a / dz**2, a / dz**2 + b / (2 * dz)
        blocks.append(sp.diags([lo, di, up], [-1, 0, 1], shape=(n_in, n_in)))
        ups.append(up)
    L = (sp.block_diag(blocks) + sp.kron(sp.csr_matrix(params.Q), sp.eye(n_in))).tocsc()
    I = sp.eye(m * n_in, format="csc")
    bnd = np.zeros((m, n_in))
    for i in range(m):
        bnd[i, -1] = ups[i] * 1.0                 # right boundary F = 1, left boundary F = 0
    bnd = bnd.ravel()

    imp = spla.factorized(I - dt * L)
    cn = spla.factorized(I - 0.5 * dt * L)
    step = (z[1:-1] > 0).astype(float)
    step[np.isclose(z[1:-1], 0.0)] = 0.5             # value at the jump = 1/2 (much smaller error)
    u = np.tile(step, m)
    out, steps_done = {}, 0
    for h in sorted(horizons):
        while steps_done < h * steps_per_day:
            if steps_done < n_rannacher:
                u = imp(u + dt * bnd)
            else:
                u = cn(u + 0.5 * dt * (L @ u) + dt * bnd)
            steps_done += 1
        F = np.zeros((m, nz)); F[:, -1] = 1.0
        F[:, 1:-1] = u.reshape(m, n_in)
        out[h] = np.clip(F, 0, 1)
    return z, out
