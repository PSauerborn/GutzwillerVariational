from scipy.optimize import curve_fit, least_squares
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.optimize import minimize
from scipy.optimize import fmin_cg
from pyprind import ProgBar
import matplotlib.cm as cm

style.use('bmh')


class GutzwillerWaveFunction():
    """Object defining a Gutzwiller Wave Function function

    Parameters
    ----------
    X: tuple/array
        Gutzwiller Coefficients inputed into the function. Consits of N values, where N = n_max + 1
    w: float
        tunneling parameter
    z: int
        coordination number
    mu: float
        chemical potential
    U: float
        interaction strength
    N: int
        maximum occupation number

    Returns
    -------
    energy of the system with given parameters/inputs

    """

    def __init__(self, w, z, mu, U, N, random_state=1, sites_total=10000, method='slsqp', normalize=False, speed=True):

        self.w = w
        self.z = z
        self.mu = mu
        self.U = U


        # Note; self.N does not refer to the occupation number. self.N is used for the loops

        self.N = N + 1
        self.sites_total = sites_total
        self.normalize = normalize

        self.rand = np.random.RandomState(random_state)
        self.method = method

        # an initial guess is made for the coefficients

        x0 = [0.0 for i in range(self.N)]

        # the coefficients are then evaluated

        self.coefficients = self.find_coefficients(x0)
        self.coefficients_ = {'f{}'.format(i): x for i, x in zip(
            range(self.N), self.coefficients)}

        # a series of observables are then evaluated. Note that <b> gives the superfluid parameter

        if not speed:
            self.observables = {'<E>': self.E(
                self.coefficients) * sites_total, '<b>': self.b(self.coefficients), '<n>': self.n(self.coefficients), '<n^2>': self.n2(self.coefficients)}
        else:
            self.observables = {'<b>': self.b(self.coefficients)}

    def E(self, X):
        """Function to be minimized with respect to the coefficients

        Parameters
        ----------
        X: iterable object
            iterator containg the Gutzwiller Coefficients

        """

        x1, x2 = 0, 0

        for n in range(1, self.N):
            for m in range(1, self.N):
                x1 += X[n] * X[n - 1] * X[m - 1] * X[m] * np.sqrt(n * m)

        x1 = -(self.z * self.w) * x1

        for n in range(self.N):
            x2 += abs(X[n])**2 * ((self.U / 2) * n * (n - 1) - self.mu * n)

        return x2 + x1

    def b(self, X):
        """Function used to evaluate the expectation value of the lowering operator i.e. the superfluid parameter

        Parameters
        ----------
        X: iterable
            Gutzwiller Coefficients

        """

        val = 0

        for n in range(self.N - 1):
            val += X[n] * X[n + 1] * np.sqrt(n + 1)

        return val

    def n(self, X):
        """Function used to evaluate the expectation value of the number operator i.e. the superfluid parameter

        Parameters
        ----------
        X: iterable
            Gutzwiller Coefficients

        """

        val = 0

        for n in range(self.N):
            val += (abs(X[n])**2) * n

        return val

    def n2(self, X):

        val = 0

        for n in range(self.N):
            val += abs((X[n])**2) * (n**2)

        return val


    def find_coefficients(self, x0):
        """Function used to minimize the ground state energy to find the Gutzwiller Coefficients

        Parameters
        ----------
        x0: array-like
            initial guesses for coefficients

        Returns
        -------
        gutzwiller_coefficients: array
            array containing the coefficients

        """

        constraints = {'type': 'eq', 'fun': lambda X: np.sum(abs(X**2)) - 1}

        bounds = [(-1,1) for k in x0]

        solutions = minimize(
            self.E, x0=x0, method=self.method, constraints=constraints, bounds=bounds)

        gutzwiller_coefficients = solutions['x']

        if self.normalize:
            gutzwiller_coefficients = self.norm_coeff(gutzwiller_coefficients)

        return gutzwiller_coefficients


def get_n(x, y, mu=2, z=6, N=5):
    """Function used to determine the value of the SF parameter for a given system setup

    Parameters
    ----------
    x: float
        x is given by x = (z*w) / U
    y: float:
        y is given by y = mu / U

    mu: float
        value of the chemical potential of the system
    z: int
        coordination number of the system. Controls the number of dimensions
    N: int
        maximum number of particles per site

    Returns
    -------
    psi: float
        superfluid parameter

    """

    U = mu / y
    w = (x * U) / z

    func = GutzwillerWaveFunction(w=w, z=z, mu=mu, U=U, N=N, speed=False)

    n = func.observables['<n>']

    return n

def plot_n(iters=100, mu=2, N=5, save=False, plot=True):
    """Function used to get the particle count and produce a surface plot of the expectation value of the number operator n

    Parameters
    ----------
    iters: int
        number of points in the meshgrid, Note that meshgrid is square i.e. if iters=100, the data matrix consists of 100x100 elements
    mu: float
        value of the chemical potential
    N: int
        maximum occupation number
    save: boolean
        if True, the data and the plot are saved to file. If False, the plot is displayed

    """

    from matplotlib import style
    style.use('bmh')

    portrait = np.zeros((iters, iters))

    bar = ProgBar(iters**2)

    for i, w in enumerate(np.linspace(0.01, 0.2, iters)):
        for j, U in enumerate(np.linspace(0.01, N - 1, iters)):

            # note that the vallues of psi are clipped at 2 to avoid large superfluid parameter values

            n = np.clip(get_n(w, U, N=N, mu=mu), 0, N)

            portrait[j, i] = abs(n)

            bar.update()

    from mpl_toolkits.mplot3d import Axes3D

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

    x = np.linspace(0.01, 0.2, iters)
    y = np.linspace(0.01, 4, iters)

    X, Y = np.meshgrid(x, y)

    ax.plot_surface(X, Y, portrait[::-1, :])

    ax.set_title(
        r'$Particle \ Number \ Expectation \ for \ \mu = 1  \ and \ n_{max} = 5$', pad=20)

    ax.set_xlabel(r'$\frac{wz}{U}$', fontsize=20)
    ax.set_ylabel(r'$\frac{\mu}{U}$', fontsize=20)
    ax.set_zlabel(r'$<\bar{n}>$', fontsize=20, rotation='vertical')

    if save:
        plt.savefig(
            'C:/directory_python/research/bose_hubbard/figures/number_plot_nmax_{}_mu_{}_iter_{}.png'.format(N, mu, iters))
        np.save('C:/directory_python/research/bose_hubbard/Gutzwiller/data/number/number_plot_nmax_{}_mu_{}_iter_{}.npy'.format(N,
                                                                             mu, iters), portrait[::-1, :])

    if plot:
        plt.show()

def get_psi(x, y, mu=2, z=6, N=5):
    """Function used to determine the value of the SF parameter for a given system setup

    Parameters
    ----------
    x: float
        x is given by x = (z*w) / U
    y: float:
        y is given by y = mu / U

    mu: float
        value of the chemical potential of the system
    z: int
        coordination number of the system. Controls the number of dimensions
    N: int
        maximum number of particles per site

    Returns
    -------
    psi: float
        superfluid parameter

    """

    U = mu / y

    w = (x * U) / z

    func = GutzwillerWaveFunction(w=w, z=z, mu=mu, U=U, N=N)

    psi = func.observables['<b>']

    return psi

def plot_phase(iters=100, mu=1, N=5, z=6, save=False, plot=True):
    """Function used to get the phase diagram and produce an implot image

    Parameters
    ----------
    iters: int
        number of points in the meshgrid, Note that meshgrid is square i.e. if iters=100, the data matrix consists of 100x100 elements
    mu: float
        value of the chemical potential
    N: int
        maximum occupation number
    save: boolean
        if True, the data and the plot are saved to file. If False, the plot is displayed

    """

    from matplotlib import style
    style.use('bmh')

    portrait = np.zeros((iters, iters))

    bar = ProgBar(iters**2)

    for i, x in enumerate(np.linspace(0.01, 0.2, iters)):
        for j, y in enumerate(np.linspace(0.01, N - 1, iters)):

            # note that the vallues of psi are clipped at 2 to avoid large superfluid parameter values

            psi = np.clip(get_psi(x, y, N=N, mu=mu, z=z), -2, 2)

            portrait[j, i] = abs(psi)

            bar.update()

    fig, ax = plt.subplots()

    ax.set_title(
        r'$Phase \ Diagram \ for \ \mu = 1  \ and \ n_{max} = 5$', pad=20)

    # note that the matrix needs to be plotted in reverse (i.e. from the bottom up) since the values are stored from top to bottom

    plot = ax.imshow(portrait[::-1, :],
                     interpolation='spline16', cmap=cm.inferno)

    cbar = plt.colorbar(plot)
    cbar.set_label(r'$|\psi|$', rotation='horizontal',
                   fontsize=20, labelpad=20)

    ax.set_xlabel(r'$\frac{wz}{U}$', fontsize=20)
    ax.set_ylabel(r'$\frac{\mu}{U}$', fontsize=20)

    ax.set_xticks(np.linspace(0, 200, 5))
    ax.set_yticks(np.linspace(0, 200, 5))

    ax.set_xticklabels([0, 0.05, 0.10, 0.15, 0.2])
    ax.set_yticklabels([i for i in reversed(range(5))])

    fig.tight_layout()

    if save:
        plt.savefig(
            'C:/directory_python/research/bose_hubbard/figures/phase_diagram_nmax_{}_mu_{}_iter_{}_z_{}.png'.format(N, mu, iters, z))
        np.save('C:/directory_python/research/bose_hubbard/Gutzwiller/data/phase_diagram_nmax_{}_mu_{}_iter_{}_z_{}.npy'.format(N,
                                                                        mu, iters, z), portrait[::-1, :])

    if plot:
        plt.show()

def get_exponent_data(z=6, full=False):
    """Convenience Function used to retrive the value of the mean field parameter psi around the phase boundary at the tip of the mott phase

    Parameters
    ----------
    z: int
        coordination number
    full: boolean
        if set to True, the value of the condensate density is evaluated through the entire lobe. If set to False, the condensate density is evaluated only at the critical point

    Returns
    -------
    data: array
        data with form [w/u, psi**2]

    """

    mu = 1
    U = 2

    if not full:
        x = np.linspace(0.02781, 0.0308, 500)
    else:
        x = np.linspace(0.0001, 0.035, 1000)


    data = pd.DataFrame(np.zeros((x.shape[0], 2)), columns=['w/U', 'psi'])
    data['w/U'] = x

    bar = ProgBar(x.shape[0])

    for i, x0 in enumerate(x):

        w = (x0 * U)

        func = GutzwillerWaveFunction(mu=mu, U=U, w=w, z=z, N=5)
        psi = func.observables['<b>']

        data.loc[i, 'psi'] = abs(psi)

        bar.update()

    np.save('./data/critical_exponent_data_z_{}_mu_{}_U_{}'.format(z, mu, U), data.values)

    return data.values

def evaluate_exponent(data, plot=True, full=False):
    """Function used to evaluate the critical exponent of a data set

    Parameters
    ----------
    data: array
        array containin the data of w/U and corresponding psi**2 values
    plot: boolean
        plots the data if set to True
    full: boolean
        if set to True, the value of the condensate density is evaluated through the entire lobe. If set to False, the condensate density is evaluated only at the critical point

    """

    def f(x, a, nu):
        """Model Function used to obtain critical exponent at the first mott lobe tip

        Params
        ------
        x: float
            value of (w/U) - (w/U)_c
        a: float
            constant to be found
        nu: float
            critical exponent

        """
        return a * (x)**(nu)

    if not full:

        x = data[:, 0] - 0.0278

        y = data[:, 1]

        params, params_cov = curve_fit(f, x, y, p0=[12, 0.5], method='dogbox')
        a, nu = params

        print('Calculated Critical Exponent: {:.2f}'.format(nu))

        if plot:

            fig, ax = plt.subplots()

            ax.plot(x, y, lw=1, c='r', label='Data')
            ax.plot(x, f(x, a=a, nu=nu), ls='--', lw=1, c='black', label='Fitted Line')
            ax.set_title(r'$SF \ Parameter \ Value \ \mu = 1 \ U = 2$', pad=20)
            ax.set_xlabel(r'$\frac{w}{U} - (\frac{w}{U})_{c}$', fontsize=15)
            ax.set_ylabel(r'$|\psi|^{2}$', fontsize=15, rotation='horizontal', labelpad=15)
            ax.legend(loc='upper left')

            plt.savefig('C:/directory_python/research/bose_hubbard/figures/critical_exponent')

            plt.show()

        return nu

    else:

        fig, ax = plt.subplots()

        ax.plot(data[:, 0], data[:, 1], lw=1, c='r')
        ax.set_title(r'$Evolution \ of \ |\psi|^{2} \ through \ Mott \ Lobe \ \mu = 1 \ U = 2$', pad=20)
        ax.set_xlabel(r'$\frac{w}{U}$', fontsize=15, labelpad=15)
        ax.set_ylabel(r'$|\psi|^{2}$', fontsize=15, labelpad=15, rotation='horizontal')

        plt.savefig('C:/directory_python/research/bose_hubbard/figures/critical_exponent_full')

        plt.show()


def get_uncertainty(x, y, mu=1, z=6, N=5):

    U = mu / y

    w = (x * U) / z

    func = GutzwillerWaveFunction(w=w, z=z, mu=mu, U=U, N=N, speed=False)

    n = func.observables['<n>']
    n_square = func.observables['<n^2>']

    return n_square - n**2

def plot_uncertainty(iters=100, mu=1, N=5, z=6, save=False, plot=True):

    from matplotlib import style
    style.use('bmh')

    data = np.zeros((iters, iters))

    bar = ProgBar(iters**2)

    for i, x in enumerate(np.linspace(0.01, 0.2, iters)):
        for j, y in enumerate(np.linspace(0.01, N - 1, iters)):

            # note that the vallues of psi are clipped at 2 to avoid large superfluid parameter values

            uncertainty = get_uncertainty(x,y, mu=mu, z=z, N=N)

            data[j, i] = uncertainty

            bar.update()

    fig, ax = plt.subplots()

    ax.set_title(
        r'$Number \ Uncertainty \ for \ \mu = 1  \ and \ n_{max} = 5$', pad=20)

    # note that the matrix needs to be plotted in reverse (i.e. from the bottom up) since the values are stored from top to bottom

    plot = ax.imshow(data[::-1, :],
                     interpolation='spline16', cmap=cm.inferno)

    cbar = plt.colorbar(plot)
    cbar.set_label(r'$|\psi|$', rotation='horizontal',
                   fontsize=20, labelpad=20)

    ax.set_xlabel(r'$\frac{wz}{U}$', fontsize=20)
    ax.set_ylabel(r'$\frac{\mu}{U}$', fontsize=20)

    ax.set_xticks(np.linspace(0, 200, 5))
    ax.set_yticks(np.linspace(0, 200, 5))

    ax.set_xticklabels([0, 0.05, 0.10, 0.15, 0.2])
    ax.set_yticklabels([i for i in reversed(range(5))])

    fig.tight_layout()

    if save:
        plt.savefig(
            'C:/directory_python/research/bose_hubbard/figures/number_uncertainty_nmax_{}_mu_{}_iter_{}_z_{}.png'.format(N, mu, iters, z))
        np.save('C:/directory_python/research/bose_hubbard/Gutzwiller/data/number_uncertainty_nmax_{}_mu_{}_iter_{}_z_{}.npy'.format(N,
                                                                        mu, iters, z), data[::-1, :])

    if plot:
        plt.show()


data = get_exponent_data(z=6, full=False)

evaluate_exponent(data)
