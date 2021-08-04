#! /usr/bin/python
"""
Spyder Editor.

This is a temporary script file.
"""
import numpy as np
import random
from collections import defaultdict
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt


class EmpiricalDistribution:
    """ Duck coded distribution.

        Ensures that the returned value is non negative.
    """

    def __init__(self, kde):
        """
        Parameters
        ----------
        kde
            scipy.stats.gaussian_kde instance

        Returns
        -------
        EmpiricalDistribution
        """
        self.kde = kde

    def resample(self, n):
        x = self.kde.resample(1)[0]

        while x < 0.0:
            x = self.kde.resample(1)[0]

        return x


class EmpiricalDistributionInt:
    """
    Duck coded distribution.

    Sample values from histogram.
    """

    def __init__(self, data):
        """
        Parameters
        ----------
        data
            numpy array containing integer numbers

        Returns
        -------
        EmpiricalDistribution
        """
        v, c = np.unique(data.astype(int), return_counts=True)

        self.vals = v
        self.counts = c / c.sum()

    def resample(self, n):
        """Return a sample."""
        return np.array([np.random.choice(self.vals, p=self.counts)])


class UniformKdeFloat:
    """Duck coded KDE. Helper."""

    def __init__(self, lower=0.0, upper=2.0*np.pi):
        self.a = lower
        self.b = upper

    def resample(self, n):
        """KDE resample method."""
        return np.random.uniform(self.a, self.b)


class UniformKdeInt:
    """Duck coded KDE. Helper."""

    def __init__(self, lower=0, upper=10):
        self.a = lower
        self.b = upper

    def resample(self, n):
        """KDE resample method."""
        return np.random.randint(self.a, self.b)


class RandomWalkerLattice:
    """Random Walker on a lattice."""

    def __init__(self, x=0, y=0, time=0.0,
                 time_resolution=0.1):

        self.x = int(x)
        self.y = int(y)
        self.t = time
        self.dt = time_resolution
        # self.waiting_time_kde = waiting_time_kde

        self.squared_displacement = self.get_squared_displacement()

    def set_time(self, new_time):
        """Set new time."""
        self.t = new_time

    def copy(self):
        """Copy."""
        return RandomWalkerLattice(self.x, self.y, self.t,
                                   self.dt)

    def step(self):
        """Random walk step."""
        # dx = self.jump_kde.resample(1)
        # dy = self.jump_kde.resample(1)
        lattice_steps = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        dx, dy = random.choice(lattice_steps)

        self.x += dx
        self.y += dy
        self.squared_displacement = self.get_squared_displacement()
        self.t += self.dt

    def get_squared_displacement(self):
        """Return squared displacement."""
        return self.x ** 2 + self.y ** 2

    def sample_trajectory(self, n=100):
        """Perform a random walk over n steps."""
        x, y, r2 = [], [], []
        for i in range(n):
            self.step()
            x.append(self.x)
            y.append(self.y)
            r2.append(self.get_squared_displacement())

        return x, y, r2


class RandomWalker2D:
    """Physical random walk in 2D."""

    def __init__(self, jump_kde, x=0.0, y=0.0, time=0,
                 orientation=None, angle_kde=None):

        self.x = x
        self.y = y
        self.jump_kde = jump_kde
        self.t = time
        # self.dt = time_resolution

        if orientation:
            self.orientation = orientation
        else:
            self.orientation = np.random.uniform(0.0, 2.0*np.pi)
        if angle_kde:
            self.angle_kde = angle_kde
        else:
            self.angle_kde = UniformKdeFloat(lower=0.0, upper=2.0*np.pi)

        self.squared_displacement = self.get_squared_displacement()

    def set_time(self, new_time):
        """
        Setter for new time.

        This is relevant for CTRWs.
        """
        self.t = new_time

    def copy(self):
        """Copy."""
        return RandomWalker2D(self.jump_kde, self.x, self.y, self.t,
                              self.orientation, self.angle_kde)

    def step(self):
        """Step."""
        stepsize = self.jump_kde.resample(1)[0][0]
        course = self.angle_kde.resample(1)

        dy = np.sin(course) * stepsize
        dx = np.cos(course) * stepsize

        self.x += dx
        self.y += dy
        self.squared_displacement = self.get_squared_displacement()
        self.t += 1

    def get_squared_displacement(self):
        """Squared displacement."""
        return self.x ** 2 + self.y ** 2

    def sample_trajectory(self, n=100):
        """Random walk over n steps."""
        assert type(n) == int, "Number of steps must be of integer type."

        x, y, r2 = [], [], []
        for i in range(n):
            self.step()
            x.append(self.x)
            y.append(self.y)
            r2.append(self.get_squared_displacement())

        return x, y, r2


class CTRandomWalk:
    """Continuous time random walk."""

    def __init__(self, Walker, waiting_time_kde, n_walkers=1,
                 init_time=0.0,
                 branching_probability=None,
                 annihilation_probability=None):

        self.waiting_time_kde = waiting_time_kde
        self.p_branch = branching_probability
        self.p_annih = annihilation_probability

        self.walker_list = [Walker.copy() for _ in range(n_walkers)]

        self.t = init_time
        self.MSD = [np.mean([x.get_squared_displacement()
                            for x in self.walker_list])]

    @staticmethod
    def __get_time_jump(time):
        """
        Return random number between 0 and time/2.

        This is a temporary method. Waiting time distribution should be set on
        instantiation. Waiting times should not be greater than the observation
        time.

        Parameters
        ----------
        time : float or int
            The maximum time window.

        Returns
        -------
        int
            Random number.

        """
        return np.random.randint(time/2)

    @staticmethod
    def __fill_states(states):
        filled_states = []

        state = states.pop(0)
        t = state[2]

        while states:
            nextstate = states.pop(0)
            for i in range(state[2], nextstate[2]):
                filled_states.append((state[0], state[1], t, state[3]))
                t += 1
            state = nextstate

        return filled_states

    # def __time_lt_maxtime(self, maxtime):
    #     return any([w.t < maxtime for w in self.walker_list])

    def run(self, maxtime=100, verbose=False):
        """Perform a CTRW."""
        verboseprint = print if verbose else lambda *a, **k: None

        walker_states = []

        for i, w in enumerate(self.walker_list):
            states = [(w.x, w.y, w.t, w.get_squared_displacement())]
            verboseprint("Walker ", i, end=" ")

            while w.t < maxtime:
                # verboseprint("Processing t=", w.t, " of ", maxtime, "with ",
                #             len(self.walker_list), " walkers.")

                self.__update_walker_list(self.p_branch, self.p_annih)

                w.step()
                # t = w.t + self.__get_time_jump(80)
                t = w.t + round(self.waiting_time_kde.resample(1)[0])
                w.set_time(t)

                states.append((w.x, w.y, w.t, w.get_squared_displacement()))

            walker_states.append(self.__fill_states(states))

        walker_states = self.__equalize_lengths(walker_states)
        # print(walker_states)

        self.MSD = self.average_walker_MSDs(walker_states)

    @staticmethod
    def __equalize_lengths(states):
        """Equalize the lengths of multiple random walk trajectories."""
        min_length = len(min(states, key=len))
        equalized_length = []
        for walker in states:
            equalized_length.append(walker[:min_length])

        return equalized_length

    @staticmethod
    def average_walker_MSDs(states):
        """Average trajectories over all walkers."""
        average_over_walkers = np.mean(states, axis=0)
        return [m[3] for m in average_over_walkers]

    def __update_walker_list(self, p_branch=0.01, p_annihilate=0.01):
        """Random walker duplicates itself during the random walk."""
        if p_branch:
            walkers = self.walker_list[:]

            for w in walkers:
                if np.random.random() < p_branch:
                    # print("Walker is at: ", w.x, w.y)
                    self.walker_list.append(w.copy())
                    # print([(z.x, z.y) for z in self.walker_list])

        if p_annihilate:
            walkers = self.walker_list[:]
            annihil_list = []

            for w in walkers:
                if np.random.random() < p_annihilate:
                    # print("Walker is at: ", w.x, w.y)
                    annihil_list.append(w)
            for w in annihil_list:
                self.walker_list.remove(w)

    def sample_trajectory(self, maxtime=1000):
        """
        Perform a single random walk (CTRW).

        This method is for demonstration purposes and the states of the walkers
        in the walker list are not changed.

        Parameters
        ----------
        maxtime : int, optional
            Maximum runtime. The default is 1000.

        Returns
        -------
        x : list
            x-coordinates vs. time.
        y : list
            y-coordinates vs. time.
        r2 : list
            squared displacement vs. time.

        """
        assert type(maxtime) == int, "Time must be of integer type."

        states = list()
        t = 0
        w = self.walker_list[0].copy()

        states.append((w.x, w.y, w.t, w.get_squared_displacement()))

        while t < maxtime:
            w.step()
            t = w.t + round(self.waiting_time_kde.resample(1)[0])
            w.set_time(t)

            states.append((w.x, w.y, w.t, w.get_squared_displacement()))

        states = self.__fill_states(states)
        x, y, _, r2 = zip(*states)

        return x, y, r2

    def estimate_D(self):
        """
        D is the diffusion constant.

        For a normal random walk: MSD = 4 * D * t.
        Method returns D.
        """
        X = np.arange(len(self.MSD)).reshape((-1, 1))
        y = self.MSD

        LM = LinearRegression()

        LM.fit(X, y)
        return LM.coef_[0]/4.0


class RandomWalk:
    """Random Walk."""

    def __init__(self, Walker, n_walkers=1, init_time=0.0,
                 branching_probability=None,
                 annihilation_probability=None):

        self.p_branch = branching_probability
        self.p_annih = annihilation_probability

        self.walker_list = [Walker.copy() for _ in range(n_walkers)]

        self.t = init_time
        # self.squared_displacements = []
        self.MSD = [np.mean([x.get_squared_displacement()
                            for x in self.walker_list])]

    def run(self, steps=10, verbose=False):
        """Perform random walk."""
        verboseprint = print if verbose else lambda *a, **k: None

        for i in range(steps):
            verboseprint("Processing t=", i+1, " of ", steps, "with ",
                         len(self.walker_list), " walkers.")

            self.__update_walker_list(self.p_branch, self.p_annih)

            if not self.walker_list:
                break

            for w in self.walker_list:
                w.step()

            self.MSD.append(np.mean([x.get_squared_displacement()
                                     for x in self.walker_list]))

    def __update_walker_list(self, p_branch=0.01, p_annihilate=0.01):
        """Random walker duplicates itself during the random walk."""
        if p_branch:
            walkers = self.walker_list[:]

            for w in walkers:
                if np.random.random() < p_branch:
                    # print("Walker is at: ", w.x, w.y)
                    self.walker_list.append(w.copy())
                    # print([(z.x, z.y) for z in self.walker_list])

        if p_annihilate:
            walkers = self.walker_list[:]
            annihil_list = []

            for w in walkers:
                if np.random.random() < p_annihilate:
                    # print("Walker is at: ", w.x, w.y)
                    annihil_list.append(w)
            for w in annihil_list:
                self.walker_list.remove(w)

    def estimate_D(self):
        """
        D is the diffusion constant.

        For a normal random walk: MSD = 4 * D * t.
        Method returns D.
        """
        X = np.arange(len(self.MSD)).reshape((-1, 1))
        y = self.MSD

        LM = LinearRegression()

        LM.fit(X, y)
        return LM.coef_[0]/4.0


class DataPrep:
    """Prepare data."""

    def __init__(self, fname):
        self.t_xy = self.read_data(fname)
        self.incidence_sparse = self.get_incidence_sparse()

    def new_cases_per_day(self):
        """Pass."""
        pass

    def get_incidence_sparse(self):
        """
        Return the incidence per day.

        Returns
        -------
        dict
            keys-time, values-incidence.

        """
        return {t: len(self.t_xy[t]) for t in self.t_xy}

    def get_incidence(self):
        """
        Return incidence as array.

        Returns
        -------
        x : np.array
            Incidence.

        """
        x = np.zeros(max(self.incidence_sparse.keys())+1, dtype=np.int)
        for key in self.incidence_sparse:
            x[key] = self.incidence_sparse[key]

        return x

    @staticmethod
    def read_data(fname):
        """Read the Data."""
        coords = np.loadtxt(fname, delimiter=",", usecols=(0, 1))
        times = np.loadtxt(fname, delimiter=",", usecols=(2,), dtype=np.int)

        t_xy = defaultdict(list)
        for i in range(len(times)):
            t_xy[times[i]].append((coords[i][0], coords[i][1]))

        return t_xy

    def shuffle_inputs(self):
        """Pass."""
        pass

    def get_position_sequence(self):
        """Pass."""
        pass

    def get_jump_dist(self):
        """Pass."""
        pass

    def get_waiting_time_dist(self):
        """
        pass.

        Returns
        -------
        None.

        """
        pass


if __name__ == "__main__":
    jump_measurements = [1.5, 1.5, 1.3, 1.6, 2.3, 4., 3.2, 1.8, 1.45]
    jump_kde = gaussian_kde(jump_measurements, bw_method=1e-1)
    waiting_kde = UniformKdeInt(0, 10)
    # jump_kde = gaussian_kde(x)
    # d = DataPrep("test.txt")
    # print(d.t_xy)
    # print(d.get_incidence())

    W = RandomWalker2D(jump_kde)
    R = CTRandomWalk(W, waiting_time_kde=waiting_kde, n_walkers=1)
    R.run(10)
    mse = R.MSD
    print("Diffusion constant =", R.estimate_D())
    # plt.plot(mse)

#    W = RandomWalker_lattice()
#    R = RandomWalk(W, n_walkers=100, branching_probability=None,
#                   annihilation_probability=0.1)
#    R.run(steps=100, verbose=True)
#
#    print("Diffusion constant =", R.estimate_D())

#    plt.plot(R.MSD)
    x, y, r2 = R.sample_trajectory(1000)
    plt.plot(x, y)
