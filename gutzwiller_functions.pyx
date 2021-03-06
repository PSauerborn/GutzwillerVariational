import numpy as np
from pyprind import ProgBar
from scipy.optimize import minimize, NonlinearConstraint
from libcpp cimport bool
cimport numpy as np



cdef class GutzwillerWaveFunction():
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
    N_sites: int
        total number of boson sites
    V: float
        nearest neighbour interaction strength
    method: str
        minimization method
    speed: boolean
        only SF parameter is evaluated if set tot True
    inhomogenouos: boolean
        solves inhomogenous if set to True

    """

    # the attributes of the object are first statically defined with the public keyword

    cdef public double w, mu, U, V
    cdef public int N, z, random_state, N_sites
    cdef public str method
    cdef public list x0
    cdef public dict observables, lattice_struct

    # note; numpy arrays need to be statically defined as memoryviews

    cdef public double[:] coefficients


    def __init__(self, w, mu, U, N, z=6, N_sites=10, V=0.01, method='slsqp', speed=True, inhomogenous=False, CG=False, square=False):

        self.w = w
        self.z = z
        self.mu = mu
        self.U = U
        self.V = V

        # Note; self.N does not refer to the occupation number. self.N is used for the loops

        self.N = N + 1
        self.N_sites = N_sites

        self.method = method

        # an initial guess is made for the coefficients

        x0 = [0.0 for i in range(self.N)]


        x1 = np.zeros(self.N * self.N_sites)


        # the coefficients are then evaluated

        if not inhomogenous:
            if CG:
                self.coefficients = self.find_coefficients_CG(x0)
            else:
                self.coefficients = self.find_coefficients(x0)


        else:
            self.coefficients = self.find_coefficients_inhomogenous(x1)
            coeffs = np.asarray(self.coefficients).reshape(3,3, self.N)


        # a series of observables are then evaluated. Note that <b> gives the superfluid parameter

        if not inhomogenous:
            if not speed:
                self.observables = {'<E>': self.E(
                    self.coefficients) * self.N_sites, '<b>': self.b(self.coefficients), '<n>': self.n(self.coefficients), '<n^2>': self.n2(self.coefficients)}
            else:
                self.observables = {'<b>': self.b(self.coefficients)}
        else:
            # self.observables = {'<b>': self.b_average(coeffs), '<n>': self.n_average(coeffs)}
                self.observables = {'<b>': self.b_average2d(coeffs), '<E>': self.E3(coeffs)}

    def get_coeffs(self):
        return np.asarray(self.coefficients).reshape(self.N_sites, self.N)

    def __call__(self, *args, **kwargs):
        X = self.get_coeffs()
        return self.b_average(X)

    # @cython.infer_types(True)
    cpdef double E(self, X):
        """Function to be minimized with respect to the coefficients

        Parameters
        ----------
        X: iterable object
            iterator containg the Gutzwiller Coefficients

        """

        cdef double x1=0, x2=0, x3=0, E=0
        cdef int n=0, m=0

        for n in range(1, self.N):
            for m in range(1, self.N):
                x1 += X[n] * X[n - 1] * X[m - 1] * X[m] * np.sqrt(n * m)

        x1 = -(self.z * self.w) * x1

        for n in range(self.N):
            x2 += abs(X[n])**2 * ((self.U / 2) * n * (n - 1) - self.mu*n)


        E = x1 + x2 + x3

        return E

    # @cython.infer_types(True)
    cpdef double E2(self, X):
        """Inhomogenous energy function to minimize; note that this is strictly for the 1D case

        Parameters
        ----------
        X: numpy array
            matrix containing state vectors

        """


        X = np.array(X).reshape(self.N_sites, self.N)

        # note that Matrix is padded with a zero matrix at both ends

        # The total energy is split into 4 contributions (tunneling, nearest neighbour, on-site repulsion, chemical potential), which are evaluated seperately

        cdef double e1=0, e2=0, e3=0, e4=0, E=0
        cdef int n=0, site=0
        cdef double[:] adj1_vector, adj2_vector, site_vector

        for site in range(self.N_sites):

            # first, the local site vector and the two adjacent vectors are determined. Note that periodic boundary conditions are applied

            if site == 0:
                site_vector = X[site, :]
                adj1_vector = X[self.N_sites - 1, :]
                adj2_vector = X[site + 1, :]

            elif site == self.N_sites - 1:
                site_vector = X[site, :]
                adj1_vector = X[site - 1, :]
                adj2_vector = X[0, :]

            else:

                site_vector = X[site, :]
                adj1_vector = X[site - 1, :]
                adj2_vector = X[site + 1, :]


            x1, x2, x3 = 0, 0, 0

            # The tunneling contribution is first evaluated

            for n in range(self.N - 1):
                x1 += site_vector[n+1]*site_vector[n]*np.sqrt(n+1)
                x2 += adj1_vector[n]*adj1_vector[n+1]*np.sqrt(n+1)
                x3 += adj2_vector[n]*adj2_vector[n+1]*np.sqrt(n+1)

            e1 += x1*(x2+x3)


            # followed by the nearest neighbour interation

            x1, x2, x3 = 0, 0, 0

            for n in range(self.N):
                x1 += (site_vector[n]**2)*n
                x3 += (adj2_vector[n]**2)*n
                x2 += (adj1_vector[n]**2)*n

            e2 += x1*(x3 + x2)

            # Followd by the on-site interaction term with the chemical potential

            x1 = 0

            for n in range(self.N):
                x1 += (site_vector[n]**2)*((self.U/2)*n*(n-1) - self.mu*n)

            e3 += x1

        # the energy terms are then multiplied by their various constants

        e1 = (-self.w*e1)

        e2 = e2 * (self.V / 2)

        # and the result is summed and then returned

        E = e1 + e2 + e3

        return E

    cpdef double E3(self, X):
        """Inhomogenous energy function to minimize; note that this is strictly for the 1D case

        Parameters
        ----------
        X: numpy array
            matrix containing state vectors

        """



        X = np.array(X).reshape(3, 3, self.N)

        # note that Matrix is padded with a zero matrix at both ends

        # The total energy is split into 4 contributions (tunneling, nearest neighbour, on-site repulsion, chemical potential), which are evaluated seperately

        cdef double e1=0, e2=0, e3=0, e4=0, E=0
        cdef double x1=0, x2=0, x3=0, x4=0, x5=0
        cdef int n=0, site=0
        cdef dict lattice
        cdef double[:] adj1_vector, adj2_vector, site_vector

        lattice = {0: (X[0,0,:], [X[2,0,:], X[1,0,:], X[0,2,:], X[0,1,:]]),
                   1: (X[0,1,:], [X[0,0,:], X[0,2,:], X[1,1,:], X[2,1,:]]),
                   2: (X[0,2,:], [X[0,0,:], X[0,1,:], X[1,2,:], X[2,2,:]]),
                   3: (X[1,0,:], [X[0,0,:], X[1,1,:], X[2,0,:], X[1,2,:]]),
                   4: (X[1,1,:], [X[1,0,:], X[0,1,:], X[1,2,:], X[2,1,:]]),
                   5: (X[1,2,:], [X[1,0,:], X[1,1,:], X[2,1,:], X[0,2,:]]),
                   6: (X[2,0,:], [X[1,0,:], X[2,1,:], X[0,0,:], X[2,2,:]]),
                   7: (X[2,1,:], [X[2,0,:], X[1,1,:], X[2,2,:], X[0,1,:]]),
                   8: (X[2,2,:], [X[2,1,:], X[1,2,:], X[2,0,:], X[0,2,:]])}



        for site in range(self.N_sites):

            # first, the local site vector and the two adjacent vectors are determined. Note that periodic boundary conditions are applied

            x1, x2, x3, x4, x5 = 0, 0, 0, 0, 0


            # The tunneling contribution is first evaluated

            for n in range(self.N - 1):
                x1 += lattice[site][0][n+1]*lattice[site][0][n]*np.sqrt(n+1)
                x2 += lattice[site][1][0][n]*lattice[site][1][0][n+1]*np.sqrt(n+1)
                x3 += lattice[site][1][1][n]*lattice[site][1][1][n+1]*np.sqrt(n+1)
                x4 += lattice[site][1][2][n]*lattice[site][1][2][n+1]*np.sqrt(n+1)
                x5 += lattice[site][1][3][n]*lattice[site][1][3][n+1]*np.sqrt(n+1)

            e1 += x1*(x2+x3+x4+x5)


            # followed by the nearest neighbour interation

            x1, x2, x3, x4, x5 = 0, 0, 0, 0, 0

            for n in range(self.N):
                x1 += (lattice[site][0][n]**2)*n
                x2 += (lattice[site][1][0][n]**2)*n
                x3 += (lattice[site][1][1][n]**2)*n
                x4 += (lattice[site][1][2][n]**2)*n
                x5 += (lattice[site][1][3][n]**2)*n


            # print(x1, x2, x3, x4, x5)

            e2 += x1*(x2+x3+x4+x5)

            # Followd by the on-site interaction term with the chemical potential

            x1 = 0

            for n in range(self.N):
                x1 += (lattice[site][0][n]**2)*((self.U/2)*n*(n-1) - self.mu*n)

            e3 += x1

        # the energy terms are then multiplied by their various constants

        e1 = e1 * -self.w

        e2 = e2 * (self.V / 2)

        # and the result is summed and then returned

        E = e1 + e2 + e3

        return E


    # @cython.infer_types(True)
    cpdef double b(self, X):
        """Function used to evaluate the expectation value of the lowering operator i.e. the superfluid parameter

        Parameters
        ----------
        X: iterable
            Gutzwiller Coefficients

        Returns
        -------
        val: float
            SF parameter

        """

        cdef double val=0
        cdef int n=0

        val = 0

        for n in range(1, self.N):
            val += X[n] * X[n - 1] * np.sqrt(n)

        return abs(val)

    # @cython.infer_types(True)
    cpdef double b_average(self, X):
        """Function used to evaluate the average SF parameter by evaluating the SF parameter at every site and then taking the mean value

        Parameters
        ----------
        X: array
            gutzwiller coefficients

        Returns
        -------
        b_mean: float
            average SF Parameter

        """

        cdef double b_mean=0, psi=0
        cdef int site=0

        b_mean = 0

        for site in range(X.shape[0]):

            psi = self.b(X[site, :])

            b_mean += psi

        b_mean = b_mean / self.N_sites

        return b_mean

    cpdef double b_average2d(self, X):
        """Function used to evaluate the average SF parameter by evaluating the SF parameter at every site and then taking the mean value

        Parameters
        ----------
        X: array
            gutzwiller coefficients

        Returns
        -------
        b_mean: float
            average SF Parameter

        """

        cdef double b_mean=0, psi=0
        cdef int site=0, i=0, j=0

        b_mean = 0

        for i in range(3):
            for j in range(3):

                psi = self.b(X[i,j,:])

                b_mean += psi

        b_mean = b_mean / self.N_sites

        return b_mean

    cpdef double n_average(self, X):
        """Function used to evaluate the average SF parameter by evaluating the SF parameter at every site and then taking the mean value

        Parameters
        ----------
        X: array
            gutzwiller coefficients

        Returns
        -------
        b_mean: float
            average SF Parameter

        """

        cdef double n_mean=0, n=0
        cdef int site=0

        n_mean = 0

        for site in range(X.shape[0]):

            n = self.n(X[site, :])

            n_mean += n

        n_mean = n_mean / self.N_sites

        return n_mean

    # @cython.infer_types(True)
    cpdef double n(self, X):
        """Function used to evaluate the expectation value of the number operator

        Parameters
        ----------
        X: iterable
            Gutzwiller Coefficients

        Returns
        -------
        val: float
            expectation value of number operator

        """

        cdef double val=0
        cdef int n=0

        val = 0

        for n in range(self.N):
            val += (abs(X[n])**2) * n

        return val

    cpdef double n2(self, X):
        """Function used to evaluate <n^2>

        Parameters
        ----------
        X: array
            gutzwiller coefficients

        Returns
        -------
        val: float
            expectation value of n^2

        """

        cdef double val=0
        cdef int n=0

        for n in range(self.N):
            val += abs((X[n])**2) * (n**2)

        return val


    def constraint1(self, X):
        return np.sum(abs(X**2)) - 1

    # @cython.infer_types(True)
    cpdef find_coefficients(self, x0):
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

        cdef dict constraints
        cdef list bounds

        constraints = {'type': 'eq', 'fun': self.constraint1}

        bounds = [(-1, 1) for k in x0]

        solutions = minimize(
            self.E, x0=x0, method=self.method, constraints=constraints, bounds=bounds)

        gutzwiller_coefficients = solutions['x']

        return gutzwiller_coefficients

    cpdef constraint2(self, X):
        """Function used to define constraint. Note that coefficients are given as 1D flattened array and hence must first be reshaped

        Parameters
        ----------
        X: 1D array
            array of coefficients

        """

        cdef int site=0

        X = X.reshape(self.N_sites, self.N)

        norms = np.zeros(shape=(self.N_sites))

        for site in range(X.shape[0]):
            norms[site] = np.linalg.norm(X[site, :])

        return norms - 1

    cpdef constraint3d(self, X):
        """Function used to define constraint. Note that coefficients are given as 1D flattened array and hence must first be reshaped

        Parameters
        ----------
        X: 1D array
            array of coefficients

        """

        cdef int site=0

        X = X.reshape(3, 3 , self.N)

        norms = np.zeros(shape=(self.N_sites))

        index = 0

        for i in range(3):
            for j in range(3):

                norms[index] = np.linalg.norm(X[i, j, :])

                index += 1

        return norms - 1


    cpdef find_coefficients_inhomogenous(self, x0):
        """Function used to minimize the ground state energy to find the Gutzwiller Coefficients for the inhomogenous system. Note that the constraint now becomes more complex, since each
        site has its own independent state vector, and each individual state vector must be self consistently normalized. Additionally, the normalization takes a flattened 1D vector as its input.
        The matrix must then be properly reshaped in order to evaluate the calculation

        Parameters
        ----------
        x0: array-like
            initial guesses for coefficients

        Returns
        -------
        gutzwiller_coefficients: array
            array containing the coefficients

        """

        cdef dict constraints
        cdef list bounds
        cdef double[:] gutzwiller_coefficients

        # the constrains is then defined

        constraints = {'type': 'eq', 'fun': self.constraint3d}

        # bounds are set

        bounds = [(-1, 1) for k in x0]

        # the energy2 function is minimized

        solutions = minimize(
            self.E3, x0=x0, method=self.method, constraints=constraints, bounds=bounds, options={'maxiter': 1000})

        gutzwiller_coefficients = solutions['x']

        return gutzwiller_coefficients

    cpdef double fconstraint(self, X):

        cdef int i=0
        cdef double x=0

        for i in range(X.shape[0]):
            x += X[i]**2

        return x

    cpdef con_jac(self, X):

        con_jacobian = np.array([2*x for x in X])

        return con_jacobian

    cpdef func_jac(self, X):

        cdef int i=0, m=0, n=0
        cdef int x1=0, x2=0, x3=0, x4=0
        cdef double val=0
        cdef double[:] jacobian

        jacobian = np.zeros(self.N)

        for i in range(self.N):

            val = 0

            for n in range(1, self.N):
                for m in range(1, self.N):

                    x1 = 1 if n == i else 0
                    x2 = 1 if n - 1 == i else 0
                    x3 = 1 if m == i else 0
                    x4 = 1 if m - 1 == i else 0

                    val += (x1*X[n-1]*X[m]*X[m-1] + x2*X[n]*X[m]*X[m-1] + x3*X[n]*X[n-1]*X[m-1] + x4*X[n]*X[n-1]*X[m])*np.sqrt(n*m)*(-self.w*self.z)

                val += 2*X[n]*((self.U/2)*n*(n-1) - self.mu*n)*x1

            jacobian[i] = val


        return jacobian


    cpdef find_coefficients_CG(self, x0):
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



        constraint = NonlinearConstraint(self.fconstraint, 1, 1, jac=self.con_jac, hess='3-point')

        solutions = minimize(
            self.E, x0=[0.5 for k in x0], method='trust-constr', constraints=constraint, jac=self.func_jac, hess='3-point', bounds=[(-1,1) for k in x0])

        gutzwiller_coefficients = solutions['x']


        return gutzwiller_coefficients



# @cython.infer_types(True)
cpdef double get_psi(x, y, V=0.15, mu=2,U=1, N_sites=9, z=6, N=4, inhomogenous=False, vary_mu=False, CG=False):
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
    V: float
        NN interaction Strength
    N_sites: int
        total number of sites
    inhomogenous: boolean
        solves inhomogenous system if set to True
    vary_mu: boolean
        varies mu if set to True, else U is varied

    Returns
    -------
    psi: float
        superfluid parameter

    """

    cdef double psi


    mu = U * y

    w = (x * U) / z


    func = GutzwillerWaveFunction(w=w, z=z, mu=mu, U=U, N=N, V=V, N_sites=N_sites, inhomogenous=inhomogenous, CG=CG)

    psi = func.observables['<b>']

    return psi

cpdef double get_uncertainty(x, y, V=0.15, mu=2,U=1, N_sites=9, z=6, N=4, inhomogenous=False, vary_mu=False):
    """Function used to determine the uncertatinty in the number operator for a given system setup

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
    V: float
        NN interaction Strength
    N_sites: int
        total number of sites
    inhomogenous: boolean
        solves inhomogenous system if set to True
    vary_mu: boolean
        varies mu if set to True, else U is varied


    Returns
    -------
    delta: float
        uncertainty in n

    """


    cdef double n, n_square, delta

    if vary_mu:
        mu = U * y

    else:
        U = mu / y

    w = (x * U) / z

    func = GutzwillerWaveFunction(w=w, z=z, mu=mu, U=U, N=N, V=V, N_sites=N_sites, inhomogenous=inhomogenous, speed=False)

    n = func.observables['<n>']
    n_square = func.observables['<n^2>']

    delta = n_square - n**2

    return delta

cpdef double get_n(x, y, V=0.15, mu=2,U=1, N_sites=9, z=6, N=4, inhomogenous=False, vary_mu=False):
    """Function used to determine the expectation value of the number operator for a given system setup

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
    V: float
        NN interaction Strength
    N_sites: int
        total number of sites
    inhomogenous: boolean
        solves inhomogenous system if set to True
    vary_mu: boolean
        varies mu if set to True, else U is varied


    Returns
    -------
    delta: float
        uncertainty in n

    """

    cdef double n

    if vary_mu:
        mu = U * y

    else:
        U = mu / y

    w = (x * U) / z

    func = GutzwillerWaveFunction(w=w, z=z, mu=mu, U=U, N=N, V=V, N_sites=N_sites, inhomogenous=inhomogenous, speed=False)

    n = func.observables['<n>']

    return n

# @cython.infer_types(True)
cpdef plot_phase(iters, mu, z, N, N_sites, V, xrange, yrange, quad, inhomogenous=False, vary_mu=False, U=None, CG=False):
    """Function used to iterate over parameter space and evaluate psi at every point to map phase diagram

    Parameters
    ----------
    iters: int
        number of points in the meshgrid, Note that meshgrid is square i.e. if iters=100, the data matrix consists of 100x100 elements
    mu: float
        value of the chemical potential
    N: int
        maximum occupation number
    V: float
        NN interaction strength
    xrange: tuple
        tuple of form (xmin, xmax) to determine what portion of paramter space is being solved
    yrange: tuple
        tuple of form (ymin, ymax) to determine what portion of paramter space is being solved
    quad: tuple
        tuple of form (i,j) which indexes column; used to reconstruct matrix after solving
    inhomogenous: boolean
        solves inhomogenous system if set to True
    vary_mu: boolean
        varies mu if set to True, else U is varied
    U: float
        On-site repulsion

    """

    # iters, mu, z, N, xrange, yrange, quad = args

    cdef double x, y, psi
    cdef int i, j
    cdef str path

    portrait = np.zeros((iters, iters))

    bar = ProgBar(iters**2)

    for i, x in enumerate(np.linspace(xrange[0], xrange[1], iters)):
        for j, y in enumerate(np.linspace(yrange[0], yrange[1], iters)):

            # note that the vallues of psi are clipped at 2 to avoid large superfluid parameter values

            psi = get_psi(x, y, N_sites=N_sites, V=V, N=N, mu=mu, z=z, inhomogenous=inhomogenous, vary_mu=vary_mu, U=U, CG=CG)

            portrait[j, i] = abs(psi)

            bar.update()

    if not inhomogenous:
        path = 'C:/directory_python/research/bose_hubbard/Gutzwiller/extensions/threaded_application/data/phase_diagram_nmax_{}_mu_{}_iter_{}_z_{}_quad_{},{}.npy'.format(
            N, mu, iters, z, quad[0], quad[1])
    else:
        path = 'C:/directory_python/research/bose_hubbard/Gutzwiller/extensions/threaded_application/data/inhomogenous/phase_diagram_nmax_{}_mu_{}_iter_{}_quad_{},{}_V_{}_U_{}.npy'.format(
            N, mu, iters, quad[0], quad[1], V, U)


    np.save(path, portrait[::-1, :])

cpdef plot_uncertainty(iters, mu, z, N, N_sites, V, xrange, yrange, quad, inhomogenous=False, vary_mu=False, U=None):
    """Function used to iterate over parameter space quadrant and evaluate the uncertainty in the number operator at every point in parameter space

    Parameters
    ----------
    iters: int
        number of points in the meshgrid, Note that meshgrid is square i.e. if iters=100, the data matrix consists of 100x100 elements
    mu: float
        value of the chemical potential
    N: int
        maximum occupation number
    V: float
        NN interaction strength
    xrange: tuple
        tuple of form (xmin, xmax) to determine what portion of paramter space is being solved
    yrange: tuple
        tuple of form (ymin, ymax) to determine what portion of paramter space is being solved
    quad: tuple
        tuple of form (i,j) which indexes column; used to reconstruct matrix after solving
    inhomogenous: boolean
        solves inhomogenous system if set to True
    vary_mu: boolean
        varies mu if set to True, else U is varied
    U: float
        On-site repulsion

    """

    cdef double x, y, uncertainty
    cdef int i=0, j=0
    cdef str path

    portrait = np.zeros((iters, iters))

    bar = ProgBar(iters**2)

    for i, x in enumerate(np.linspace(xrange[0], xrange[1], iters)):
        for j, y in enumerate(np.linspace(yrange[0], yrange[1], iters)):

            # note that the vallues of psi are clipped at 2 to avoid large superfluid parameter values

            uncertainty = abs(get_uncertainty(x, y, N_sites=N_sites, V=V, N=N, mu=mu, z=z, inhomogenous=inhomogenous, vary_mu=vary_mu, U=U))

            portrait[j, i] = uncertainty

            bar.update()

    # note that the matrix needs to be plotted in reverse (i.e. from the bottom up) since the values are stored from top to bottom

    if not inhomogenous:
        path = 'C:/directory_python/research/bose_hubbard/Gutzwiller/extensions/threaded_application/data/uncertainty/phase_diagram_nmax_{}_mu_{}_iter_{}_z_{}_quad_{},{}.npy'.format(
            N, mu, iters, z, quad[0], quad[1])
    else:
        path = 'C:/directory_python/research/bose_hubbard/Gutzwiller/extensions/threaded_application/data/uncertainty/inhomogenous/phase_diagram_nmax_{}_mu_{}_iter_{}_quad_{},{}_V_{}_U_{}.npy'.format(
            N, mu, iters, quad[0], quad[1], V, U)


    np.save(path, portrait[::-1, :])

cpdef plot_n(iters, mu, z, N, N_sites, V, xrange, yrange, quad, inhomogenous=False, vary_mu=False, U=None):
    """Function used to iterate over parameter space quadrant and evaluate the expecation value of the number operator at every point in parameter space

    Parameters
    ----------
    iters: int
        number of points in the meshgrid, Note that meshgrid is square i.e. if iters=100, the data matrix consists of 100x100 elements
    mu: float
        value of the chemical potential
    N: int
        maximum occupation number
    V: float
        NN interaction strength
    xrange: tuple
        tuple of form (xmin, xmax) to determine what portion of paramter space is being solved
    yrange: tuple
        tuple of form (ymin, ymax) to determine what portion of paramter space is being solved
    quad: tuple
        tuple of form (i,j) which indexes column; used to reconstruct matrix after solving
    inhomogenous: boolean
        solves inhomogenous system if set to True
    vary_mu: boolean
        varies mu if set to True, else U is varied
    U: float
        On-site repulsion

    """

    cdef double x, y, n
    cdef int i=0, j=0
    cdef str path

    portrait = np.zeros((iters, iters))

    bar = ProgBar(iters**2)

    for i, x in enumerate(np.linspace(xrange[0], xrange[1], iters)):
        for j, y in enumerate(np.linspace(yrange[0], yrange[1], iters)):

            # note that the vallues of psi are clipped at 2 to avoid large superfluid parameter values

            n = abs(get_n(x, y, N_sites=N_sites, V=V, N=N, mu=mu, z=z, inhomogenous=inhomogenous, vary_mu=vary_mu, U=U))

            portrait[j, i] = n

            bar.update()

    # note that the matrix needs to be plotted in reverse (i.e. from the bottom up) since the values are stored from top to bottom

    if not inhomogenous:
        path = 'C:/directory_python/research/bose_hubbard/Gutzwiller/extensions/threaded_application/data/n/phase_diagram_nmax_{}_mu_{}_iter_{}_z_{}_quad_{},{}.npy'.format(
            N, mu, iters, z, quad[0], quad[1])
    else:
        path = 'C:/directory_python/research/bose_hubbard/Gutzwiller/extensions/threaded_application/data/n/inhomogenous/phase_diagram_nmax_{}_mu_{}_iter_{}_quad_{},{}_V_{}_U_{}.npy'.format(
            N, mu, iters, quad[0], quad[1], V, U)


    np.save(path, portrait[::-1, :])


# @cython.infer_types(True)
cpdef dict construct_quadrants(nrows=3, ncols=3, xrange=(0.01, 0.2), yrange=(0.01,4)):
    """Function that constructs the quandrants in parameter space

    Parmeters
    ---------
    nrows: int
        number of rows
    ncols: int
        number of columns
    xrange: tuple
        tuple specifying domain in x
    yrange: tuple
        tuple specifying domain in y

    Returns
    -------
    quadrants: dict
        dictionary of quadrants

    """

    cdef dict quadrants={}
    cdef double delta_x, delta_y, y0, x0
    cdef int i, j, n

    delta_x = (xrange[1] - xrange[0]) / ncols
    delta_y = (yrange[1] - yrange[0]) / nrows

    n = 1

    for i in range(nrows):
        for j in range(ncols):

            y0 = yrange[0] + i*delta_y
            x0 = xrange[0] + j*delta_x

            quadx = (x0, x0 + delta_x)
            quady = (y0, y0 + delta_y)

            quadrants['{}'.format(n)] = ((i,j), (quadx, quady))

            n += 1

    return quadrants
