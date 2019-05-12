import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.neighbors import KNeighborsRegressor


np.random.seed(42)
# Hyper Parameters
input_size = 2 + 2
K = 5
# Instantiate a Gaussian Process model
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
n_restarts_optimizer = 0
gp_bias = 0
# gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
# value_gps = [GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer) for k in range(K)]
value_gps = [KNeighborsRegressor(n_neighbors=5) for k in range(K)]



class Env:
    def __init__(self):
        self.pos_min = 0.0
        self.pos_max = 1.0
        dx = 0.05
        dy = 0.05
        self.dx = dx
        self.dy = dy
        self.noise = 0.0
        s2 = np.sqrt(2)*dx
        self.action_vec = np.array([[dx, 0], [-dx, 0], [0, dy], [0, -dy], [s2, s2], [s2, -s2], [-s2, s2], [-s2, -s2]])
        self.num_actions = self.action_vec.shape[0]

    def generate_data(self, num_samples):
        pos = np.random.rand(num_samples, 2)
        actions = np.random.randint(0, high=self.num_actions, size=(num_samples))
        d_pos = self.action_vec[actions] + self.noise * np.random.randn(num_samples, 2)
        new_pos = np.clip(pos + d_pos, self.pos_min, self.pos_max)
        return pos, actions, new_pos


def cost(state, next_state):
    obstacle_region = 0.25
    obstacle_center = [0.5, 0.5]
    dist = np.linalg.norm(state - next_state, axis=1)
    dist_to_obs = np.linalg.norm(next_state - obstacle_center, axis=1)
    dist_to_obs2 = np.linalg.norm(state - obstacle_center, axis=1)
    penalty = 5.0
    c = penalty * (dist_to_obs < obstacle_region) + penalty * (dist_to_obs2 < obstacle_region) + dist
    return c


def traj_split(data, value_gps, k_max, iters=1000):
    # first stage - learn V for k=0 using supervised learning
    # we give c=1 for transitions, c=0 for self transition, and c=10 for non-transition states
    costs = cost(data[0], data[2])
    full_states = np.concatenate((data[0], data[2]), axis=1)
    self_states = np.concatenate((data[0], data[0]), axis=1)
    all_rand_states = data[0][np.random.permutation(range(num_samples))]
    all_goals = data[0][np.random.permutation(range(num_samples))]
    non_trans_states = np.concatenate([data[0], all_rand_states], axis=1)
    # value_gps[0].fit(np.concatenate([full_states, self_states],axis=0), np.concatenate([costs + gp_bias, 0.0*costs + gp_bias],axis=0))
    value_gps[0].fit(np.concatenate([full_states, non_trans_states, self_states], axis=0),
                     np.concatenate([costs + gp_bias, 0.0 * costs + 2.0 + gp_bias, 0.0 * costs + gp_bias], axis=0))
    # value_gps[0].fit(full_states, costs + gp_bias)
    plot_values(value_gps[0], np.array([0.8, 0.8]))
    plt.pause(0.1)
    # second stage - learn V for k>0 using traj split update
    for k in range(1, k_max):
        rand_states = data[0][np.random.permutation(range(num_samples))]
        rand_goals = data[0][np.random.permutation(range(num_samples))]
        print('finding mid points level ', k)
        mid_costs = np.array([traj_split_min(value_gps[k-1], start, goal)[0] for start, goal in zip(rand_states, rand_goals)])
        print('fitting GP level ', k)
        self_states = np.concatenate((rand_states , rand_states), axis=1)
        full_states = np.concatenate((rand_states, rand_goals), axis=1)
        # value_gps[k].fit(np.concatenate((full_states, self_states), axis=0), np.concatenate((mid_costs + gp_bias, 0.0*mid_costs + gp_bias), axis=0))
        value_gps[k].fit(full_states, mid_costs + gp_bias)
        plot_values(value_gps[k], np.array([0.8, 0.8]))
        plt.pause(0.1)
    return value_gps


def predict_values(states, goals, gp):
    # import pdb; pdb.set_trace()
    # states = torch.tensor(data).float()
    # goals = torch.tensor(goals).float()
    # state_values, state_values_sigma = gp.predict(np.concatenate((states, goals), axis=1), return_std=True)
    state_values = gp.predict(np.concatenate((states, goals), axis=1))
    state_values -= gp_bias
    return state_values


def plot_values(gp, goal):
    x = np.linspace(0, 1)
    y = np.linspace(0, 1)
    X, Y = np.meshgrid(x, y)
    xy = np.stack((X.reshape(-1), Y.reshape(-1))).T
    goals = np.tile(goal, (xy.shape[0], 1))
    z = predict_values(xy, goals, gp).reshape(X.shape)
    plt.clf()
    plt.pcolor(X, Y, z)
    plt.colorbar()
    plt.grid()
    # plt.draw()


def traj_split_min(gp, start, goal):
    x = np.linspace(0, 1, num=25)
    y = np.linspace(0, 1, num=25)
    X, Y = np.meshgrid(x, y)
    mid_points = np.stack((X.reshape(-1), Y.reshape(-1))).T
    goals = np.tile(goal, (mid_points.shape[0], 1))
    starts = np.tile(start, (mid_points.shape[0], 1))
    to_mid = predict_values(starts, mid_points, gp).reshape(X.shape)
    from_mid = predict_values(mid_points, goals, gp).reshape(X.shape)
    min_mid = np.min(to_mid + from_mid)
    mid_point = mid_points[np.argmin(to_mid + from_mid)]
    return max(min_mid, 0.0), mid_point


def get_traj_split(value_gps, start, goal, k):
    if k == 0:
        return [start, goal]
    else:
        mid_point = traj_split_min(value_gps[k-1], start, goal)[1]
        path_to_mid = get_traj_split(value_gps, start, mid_point, k - 1)
        path_from_mid = get_traj_split(value_gps, mid_point, goal, k - 1)
        return path_to_mid[0:-1] + path_from_mid


def plot_traj(ax, value_gps, start, goal, k_max, color='r'):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    traj = np.array(get_traj_split(value_gps, start, goal, k_max))
    ax.plot(traj[:, 0], traj[:, 1], color)
    circle1 = plt.Circle(goal, 0.05, color='m')
    circle1 = plt.Circle([0.5, 0.5], 0.25, color='r')
    ax.add_artist(circle1)
    plt.show()


def plot_trajs(value_gps, K):
    fig, ax = plt.subplots()
    plot_traj(ax, value_gps, np.array([0.1, 0.1]), np.array([0.8, 0.8]), K, 'r')
    plot_traj(ax, value_gps, np.array([0.9, 0.1]), np.array([0.8, 0.8]), K, 'b')
    plot_traj(ax, value_gps, np.array([0.1, 0.9]), np.array([0.8, 0.8]), K, 'g')

plt.ion()  # enable interactivity
env = Env()
num_samples = 30 * 250
data = env.generate_data(num_samples)
goal = np.array([0.7, 0.7])


policy_net = traj_split(data, value_gps, K, iters=1000)
import pdb; pdb.set_trace()
plot_trajs(value_gps, K)
import pdb; pdb.set_trace()



