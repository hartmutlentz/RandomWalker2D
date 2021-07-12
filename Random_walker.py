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

    def read_data(self, fname):
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


class Uniform_kde():
    """Duck coded KDE. Helper."""

    def __init__(self, lower=0.0, upper=2.0*np.pi):
        self.a = lower
        self.b = upper

    def resample(self, n):
        """KDE resample method."""
        return np.random.uniform(self.a, self.b)


# class Lattice_step_kde():
#    def __init__(self, stay=True):
#        """Duck coded KDE. Helper."""
#        self.stay = stay
#
#    def resample(self, n):
#        if self.stay:
#            return np.random.choice([-1, 0, 1])
#        else:
#            return np.random.choice([-1, 1])


class RandomWalker_lattice():
    """Random Walker on a lattice."""

    def __init__(self, x=0, y=0, time=0.0,
                 time_resolution=0.1, waiting_time_kde=None):

        self.x = int(x)
        self.y = int(y)
        self.t = time
        self.dt = time_resolution
        self.waiting_time_kde = waiting_time_kde

        self.squared_displacement = self.get_squared_displacement()

    def set_time(self, new_time):
        """Set new time."""
        self.t = new_time

    def copy(self):
        """Copy."""
        return RandomWalker_lattice(self.x, self.y, self.t,
                                    self.dt, self.waiting_time_kde)

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


class RandomWalker_2D():
    """Physical random walk in 2D."""

    def __init__(self, jump_kde, x=0.0, y=0.0, time=0, waiting_time_kde=None,
                 orientation=None, angle_kde=None):

        self.x = x
        self.y = y
        self.jump_kde = jump_kde
        self.t = time
        # self.dt = time_resolution

        self.waiting_time_kde = waiting_time_kde
        if orientation:
            self.orientation = orientation
        else:
            self.orientation = np.random.uniform(0.0, 2.0*np.pi)
        if angle_kde:
            self.angle_kde = angle_kde
        else:
            self.angle_kde = Uniform_kde()

        self.squared_displacement = self.get_squared_displacement()

    def set_time(self, new_time):
        """Set new time."""
        self.t = new_time

    def copy(self):
        """Copy."""
        return RandomWalker_2D(self.jump_kde, self.x, self.y, self.t,
                               self.waiting_time_kde, self.orientation,
                               self.angle_kde)

    # def CTRW(self, maxtime=100):
    #     """Continuous time random walk."""
    #     def get_time_jump():
    #         return np.random.randint(100)

    #     def fill_states(states):
    #         #
    #         filled_states = []

    #         state = states.pop(0)
    #         t = state[2]

    #         while states:
    #             nextstate = states.pop(0)
    #             for i in range(state[2], nextstate[2]):
    #                 filled_states.append((state[0], state[1], t, state[3]))
    #                 t += 1
    #             state = nextstate

    #         return filled_states

    #     t = 0
    #     states = [(self.x, self.y, t, self.get_squared_displacement())]

    #     while t < maxtime:
    #         t += get_time_jump()
    #         self.step()
    #         states.append((self.x, self.y, t,
    #                       self.get_squared_displacement()))

    #     return fill_states(states)

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
        x, y, r2 = [], [], []
        for i in range(n):
            self.step()
            x.append(self.x)
            y.append(self.y)
            r2.append(self.get_squared_displacement())

        return x, y, r2


# class CTRW_Walker_2D():
#     """Physical continuous time random walk in 2D."""

#     def __init__(self, jump_kde, x=0.0, y=0.0, time=0.0,
#                  time_resolution=0.1, waiting_time_kde=None,
#                  orientation=None, angle_kde=None):

#         self.x = x
#         self.y = y
#         self.jump_kde = jump_kde
#         self.t = time
#         self.dt = time_resolution

#         self.waiting_time_kde = waiting_time_kde
#         if orientation:
#             self.orientation = orientation
#         else:
#             self.orientation = np.random.uniform(0.0, 2.0*np.pi)
#         if angle_kde:
#             self.angle_kde = angle_kde
#         else:
#             self.angle_kde = Uniform_kde()

#         self.squared_displacement = self.get_squared_displacement()

#     def copy(self):
#         """Copy."""
#         return CTRW_Walker_2D(self.jump_kde, self.x, self.y, self.t, self.dt,
#                               self.waiting_time_kde, self.orientation,
#                               self.angle_kde)

#     def __get_time_jump(self):
#         return np.random.randint(100)

#     def __fill_states(self, states):
#         """pass."""
#         filled_states = []

#         state = states.pop(0)
#         t = state[2]

#         while states:
#             nextstate = states.pop(0)
#             for i in range(state[2], nextstate[2]):
#                 filled_states.append((state[0], state[1], t, state[3]))
#                 t += 1
#             state = nextstate

#         return filled_states

#     def CTRW(self, maxtime=100):
#         """Continuous time random walk."""
#         t = 0
#         states = [(self.x, self.y, t, self.get_squared_displacement())]

#         while t < maxtime:
#             t += self.__get_time_jump()
#             self.step()
#             states.append((self.x, self.y, t,
#                           self.get_squared_displacement()))

#         return self.__fill_states(states)

#     def step(self):
#         """Step."""
#         stepsize = self.jump_kde.resample(1)[0][0]
#         course = self.angle_kde.resample(1)

#         dy = np.sin(course) * stepsize
#         dx = np.cos(course) * stepsize

#         self.x += dx
#         self.y += dy
#         self.squared_displacement = self.get_squared_displacement()
#         self.t += self.dt

#     def get_squared_displacement(self):
#         """Squared displacement."""
#         return self.x ** 2 + self.y ** 2

#     def sample_trajectory(self, n=100):
#         """Random walk over n steps."""
#         x, y, r2 = [], [], []
#         for i in range(n):
#             self.step()
#             x.append(self.x)
#             y.append(self.y)
#             r2.append(self.get_squared_displacement())

#         return x, y, r2


class CTRandomWalk():
    """Continuous time random walk."""

    def __init__(self, Walker, n_walkers=1, init_time=0.0,
                 branching_probability=None,
                 annihilation_probability=None):

        self.p_branch = branching_probability
        self.p_annih = annihilation_probability

        self.walker_list = [Walker.copy() for i in range(n_walkers)]

        self.t = init_time
        # self.squared_displacements = []
        self.MSD = [np.mean([x.get_squared_displacement()
                            for x in self.walker_list])]

    def __get_time_jump(self):
        return np.random.randint(100)

    def __fill_states(self, states):
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

    def __time_lt_maxtime(self, maxtime):
        return any([w.t < maxtime for w in self.walker_list])

    def run(self, maxtime=100, verbose=False):
        """Perform a CTRW."""
        verboseprint = print if verbose else lambda *a, **k: None

        walker_states = []
        for w in self.walker_list:
            states = [(w.x, w.y, w.t, w.get_squared_displacement())]
            while w.t < maxtime:
                verboseprint("Processing t=", w.t, " of ", maxtime, "with ",
                             len(self.walker_list), " walkers.")

                self.__update_walker_list(self.p_branch, self.p_annih)

                w.step()
                t = w.t + self.__get_time_jump()
                w.set_time(t)

                states.append((w.x, w.y, w.t, w.get_squared_displacement()))

            walker_states.append(self.__fill_states(states))
        print(walker_states)

        self.MSD = self.average_walker_MSDs(walker_states)

    def average_walker_MSDs(self, states):
        """Average trajectories over all walkers."""
        pass

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


class RandomWalk():
    """Random Walk."""

    def __init__(self, Walker, n_walkers=1, init_time=0.0,
                 branching_probability=None,
                 annihilation_probability=None):

        self.p_branch = branching_probability
        self.p_annih = annihilation_probability

        self.walker_list = [Walker.copy() for i in range(n_walkers)]

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


if __name__ == "__main__":
    jump_measurements = [1.5, 1.5, 1.3, 1.6, 2.3, 4., 3.2, 1.8, 1.45]
    jump_kde = gaussian_kde(jump_measurements, bw_method=1e-1)
    # jump_kde = gaussian_kde(x)
    # d = DataPrep("test.txt")
    # print(d.t_xy)
    # print(d.get_incidence())

    W = RandomWalker_2D(jump_kde)
    # tra = W.CTRW(1000)
    R = CTRandomWalk(W)
    R.run(100)
    print(R.MSD)

    # plt.plot(t, mse)

#    W = RandomWalker_lattice()
#    R = RandomWalk(W, n_walkers=100, branching_probability=None,
#                   annihilation_probability=0.1)
#    R.run(steps=100, verbose=True)
#
#    print("Diffusion constant =", R.estimate_D())

#    plt.plot(R.MSD)
#    x, y, _ = W.sample_trajectory()
#    plt.plot(x, y)
