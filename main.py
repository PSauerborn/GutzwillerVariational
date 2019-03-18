from scipy.optimize import curve_fit, least_squares
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.optimize import minimize
from scipy.optimize import fmin_cg
from pyprind import ProgBar
import matplotlib.cm as cm
import threading
from multiprocessing import Process
from gutzwiller_functions import *
from timer import timer


style.use('bmh')


def solve_system(N_sites, V, nrows=10, ncols=10, mu=1, z=6, N=4, inhomogenous=False, vary_mu=False, U=None, target=plot_phase, total_iterations=200, CG=False):
    """Function that uses multiprocessing unit to split parameter space into quadrants and solve quadrants in parallel, resulting is much superior calculation time

    Parameters
    ----------
    N_sites: int
        number of sites
    V: float
        nearest neighbour interaction strength
    nrows: int
        number of rows in quadrant split
    ncols: int
        number of columns in quadrant split
    mu: float
        chemical potential
    z: int
        coordination number
    N: int
        maximum occupation number
    inhomogenous: boolean
        solves inhomogenous system if set to True
    vary_mu: boolean
        varies mu if set to True; else, U is varied
    target: func object
        target function for thread; one of ['plot_phase', 'plot_n', 'plot_uncertainty']

    """

    # note that the x-axis domain is significantly shorter for the inhomgenous system as the coordiantion number is not used


    ytop = N - 1 if N - 1 < 4 else 3

    if not inhomogenous:
        quadrants = construct_quadrants(nrows=nrows, ncols=ncols, yrange=(0.01, N-1), xrange=(0.01, 0.2))
    else:
        quadrants = construct_quadrants(nrows=nrows, ncols=ncols, xrange=(0.01, 0.45), yrange=(0.01, ytop))


    nquads = len(quadrants)

    iters = int(np.ceil(total_iterations / nrows))

    threads = []
    processes = []


    for n in quadrants.keys():

        quad, range = quadrants[n]

        xrange, yrange = range

        if __name__ == '__main__':

            p = Process(target=target, args=(iters, mu, z, N, N_sites, V, xrange, yrange, quad, inhomogenous, vary_mu, U, CG))

            processes.append(p)

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    return nrows, ncols, nquads, iters

def plot_data(nrows, ncols, mu=1, N=5, z=6, iters=50, inhomogenous=False, V=0, U=None, uncertainty=False):
    """Function used to assemble the quadrants of parameter space to then plot the resulting phase diagram

    Parameters
    ----------
    nrows: int
        number of rows in parameter space
    ncols: int
        number of columns in parameter space
    mu: float
        chemical potential
    z: int
        coordination number
    iters: int
        number of iterations in each quadrant

    """

    # the individual quadrants must first be reassembled in the correct order

    data = None

    for i in reversed(range(nrows)):
        row = None

        for j in range(ncols):

            if not inhomogenous:
                if not uncertainty:
                    path = 'C:/directory_python/research/bose_hubbard/Gutzwiller/extensions/threaded_application/data/phase_diagram_nmax_{}_mu_{}_iter_{}_z_{}_quad_{},{}.npy'.format(N, mu, iters, z, i, j)
                else:
                    path = 'C:/directory_python/research/bose_hubbard/Gutzwiller/extensions/threaded_application/data/uncertainty/phase_diagram_nmax_{}_mu_{}_iter_{}_z_{}_quad_{},{}.npy'.format(N, mu, iters, z, i, j)

            else:
                path = 'C:/directory_python/research/bose_hubbard/Gutzwiller/extensions/threaded_application/data/inhomogenous/phase_diagram_nmax_{}_mu_{}_iter_{}_quad_{},{}_V_{}_U_{}.npy'.format(N, mu, iters, i, j, V, U)


            if row is None:
                row = np.load(path)
            else:
                loaded_row = np.load(path)
                row = np.concatenate((row, loaded_row), axis=1)

        if data is None:
            data = row
        else:
            data = np.concatenate((data, row), axis=0)

    # the results can then be plotted as one concurrent matrix

    # np.save(r'C:\directory_python\research\bose_hubbard\Gutzwiller\plot_settings\data\homogenous\sample.npy', data)

    import seaborn as sns

    fig, ax = plt.subplots()

    from scipy.interpolate import interp2d

    x = np.linspace(0, 0.2, iters*ncols)
    y = np.linspace(0, 4, iters*ncols)

    func = interp2d(x, y, data)

    xx, yy = np.linspace(0,0.2,1000), np.linspace(0, 4, 1000)

    data_inter = func(xx, yy)

    plot = ax.imshow(data_inter,
                    interpolation='spline16', cmap=cm.gist_heat)

    cbar = plt.colorbar(plot)
    cbar.set_label(r'$|\psi|$', rotation='horizontal',
                    fontsize=20, labelpad=20)

    arrow = ax.arrow(y=2.2, x=1, dx=2, dy=3)

    ax.set_xlabel(r'$\frac{wz}{U}$', fontsize=35, labelpad=30)
    ax.set_ylabel(r'$\frac{\mu}{U}$', fontsize=35, rotation='horizontal', labelpad=30)

    # ax.set_title(r'$Phase \ Diagram \ for \ U \ = \ 1 \ at \ n_{max} \ = \ 6$', fontsize=30, pad=30)

    ax.set_xticks(np.linspace(0, 1000, 5))
    ax.set_yticks(np.linspace(0, 1000, 4))

    ticks = np.linspace(0, 0.45, 5)
    # ticks = [0, 0.05, 0.1, 0.15, 0.2]

    ax.set_xticklabels(ticks, rotation=0)
    ax.set_yticklabels([i for i in reversed(range(4))], rotation=0)

    fig.tight_layout()

    plt.show()


# nrows, ncols, nquads, iters = solve_system(N_sites=9, N=5, z=6, mu=1, nrows=4, ncols=4, V=0., target=plot_n)

# plot_data(nrows=5, ncols=5, iters=80, N=5, mu=1, seaborn=False)

class Lattice():

    def __init__(self, func, expected_phase):

        self.func = func
        self.expected_phase = expected_phase
        self.coeffs = self.get_coeffs()

        self.delta_dw, self.delta_sf = 0, 0

        for site in range(1, self.N_sites + 1):
            self.delta_dw += (-1)**site * self.n(self.coeffs[site - 1, :])
            self.delta_sf += (-1)**site * self.b(self.coeffs[site - 1, :])

        self.delta_dw = abs(self.delta_dw) / self.N_sites
        self.delta_sf = abs(self.delta_sf) / self.N_sites


    def __getattr__(self, attr):
        return getattr(self.func, attr)

    def check_norm(self):

        for i in range(self.N_sites):
            print(np.linalg.norm(self.coeffs[i, :]))

    def __str__(self):

        observables = self.observables

        print('\n')

        rep = 'psi: {:.2f} phase: {}\n'.format(observables['<b>'], self.expected_phase)
        rep += 'x: {:.2f} y: {:.2f} zV: {}\n'.format(((self.z * self.w) / self.U), (self.mu / self.U), self.z*self.V)
        rep += 'Delta_DW: {0.delta_dw:.2f} Delta_SF: {0.delta_sf:.2f}\n\n'.format(self)

        for site in range(N_sites):
            rep += '{:.2f}   '.format(self.n(self.coeffs[site, :]))

        return rep

def view_lattice(U, N, z, N_sites, V):


    phases = ['CDW', 'MOTT', 'SS', 'SF']
    points = [(0.08, 2.5), (0.08, 1.5), (0.15, 0.9), (0.45, 0.8)]

    for phase, point in zip(phases, points):

        mu = point[1] * U
        w = (point[0] * U) / z

        func = GutzwillerWaveFunction(mu=mu, U=U, w=w, z=z, N_sites=N_sites, N=N, V=V, speed=False, inhomogenous=True)

        a = Lattice(func, expected_phase=phase)

        print(a)

#
# # #
# U = 1
# z = 4
# N_sites = 9
# N = 6
# V = 0.45


# view_lattice(U=U, z=z, N_sites=N_sites, N=N, V=V)


# nrows, ncols, nquads, iters = solve_system(nrows=2, ncols=2, N_sites=9, N=8, z=4, U=1, V=0.225, vary_mu=True, inhomogenous=True, total_iterations=100)

# plot_data(nrows=2, ncols=2, iters=50, V=0.225, U=1, N=8, z=4, inhomogenous=True)
#
x = 0.2
U = 1
N = 10
z = 4
mu = 0.3
V = 0.225
w = x / z
N_sites = 9

#
func1 = GutzwillerWaveFunction(w=w, U=U, mu=mu, z=z, N=N, N_sites=N_sites, V=V, inhomogenous=True, speed=False)


print(func1.observables['<b>'])
