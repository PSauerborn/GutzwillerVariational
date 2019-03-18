
import time
import numpy as np
from pyprind import ProgBar



def timer(*args, **kwargs):

    def wrapper(func):
        def closure(*fargs, **fkwargs):

            times = []

            # bar = ProgBar(kwargs['N'])

            for N in range(kwargs['N']):

                start = time.time()

                func(*fargs, **fkwargs)

                end = time.time()

                # bar.update()

                times.append(end-start)

            # print('Average {} Run Time: {:.2f}'.format(func.__name__, np.mean(times)))

            return np.mean(times)

        return closure
    return wrapper
