import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
from sklearn.neighbors import KNeighborsRegressor


np.random.seed(42)
# Hyper Parameters
input_size = 2 + 2
K = 8
num_actions = 8
action_distance = 100


class Obstacle:
    def __init__(self, top_left, bottom_right):
        self.top_left = top_left
        self.bottom_right = bottom_right

    def in_collision(self, point):
        return self.top_left[0] <= point[0] <= self.bottom_right[0] and \
               self.bottom_right[1] <= point[1] <= self.top_left[1]

    def get_patch(self):
        # Create a Rectangle patch
        rect = matplotlib.patches.Rectangle((self.top_left[0], self.bottom_right[1]), self.bottom_right[0]-self.top_left[0],
                                            self.top_left[1] - self.bottom_right[1], linewidth=1, edgecolor='k',
                                            hatch='x', facecolor='none')
        return rect


class Env:
    def __init__(self):
        self.pos_min = 0.0
        self.pos_max = 1.0
        dx = 0.025
        dy = 0.025
        self.dx = dx
        self.dy = dy
        self.noise = 0.0
        s2 = 1/np.sqrt(2)*dx
        self.action_vec = np.array([[dx, 0], [-dx, 0], [0, dy], [0, -dy], [s2, s2], [s2, -s2], [-s2, s2], [-s2, -s2]])
        self.num_actions = num_actions # self.action_vec.shape[0]
        # self.obstacles = [Obstacle([10,9],[10,9])]
        self.obstacles = [Obstacle([0.2,0.8],[0.4,0]), Obstacle([0.6,1],[0.8,0.2])]

    def generate_data(self, num_samples):
        pos = np.random.rand(num_samples, 2)
        actions = np.random.randint(0, high=self.num_actions, size=(num_samples))
        d_pos = self.action_vec[actions] + self.noise * np.random.randn(num_samples, 2)
        new_pos = np.clip(pos + d_pos, self.pos_min, self.pos_max)
        return pos, actions, new_pos

    def in_collision(self, point):
        return np.any([x.in_collision(point) for x in self.obstacles])

    def get_free_point(self):
        max_tries = 100
        point = np.random.rand(2)
        tries = 0
        while self.in_collision(point):
            point = np.random.rand(2)
            tries += 1
            if tries >= max_tries:
                return None
        return point


    def get_trajectory(self, x0, y0, net, goal, len=25):
        # x0 = 0.1
        # y0 = 0.1
        x = x0
        y = y0
        goal_region = 0.15
        traj = np.zeros((len, 2))
        for i in range(len):
            traj[i] = [x, y]
            state = np.array([x, y, goal[0], goal[1]])
            action = np.argmin([net.predict(np.concatenate((state, [action_distance*a]),axis=0).reshape(1,-1)) for a in range(num_actions)])
            # import pdb; pdb.set_trace()
            d_pos = self.action_vec[action] + self.noise * np.random.randn(1, 2)
            new_state = np.clip(np.array([x,y]) + d_pos, self.pos_min, self.pos_max)
            dist = np.linalg.norm(np.array([x,y]) - goal)
            x = new_state[0][0]
            y = new_state[0][1]
            if dist < goal_region:
                len = i+1
                print('reached goal in %d step', len, state)
                break
        return traj[:len]

    def get_trajectory_im(self, x0, y0, im_net, goal, len=25):
        # x0 = 0.1
        # y0 = 0.1
        x = x0
        y = y0
        goal_region = 0.15
        traj = np.zeros((len, 2))
        for i in range(len):
            traj[i] = [x, y]
            state = np.array([x, y, goal[0], goal[1]])
            action = im_net.predict(state.reshape(1,-1))
            d_pos = self.action_vec[action] + self.noise * np.random.randn(1, 2)
            new_state = np.clip(np.array([x,y]) + d_pos, self.pos_min, self.pos_max)
            dist = np.linalg.norm(np.array([x,y]) - goal)
            x = new_state[0][0]
            y = new_state[0][1]
            if dist < goal_region:
                len = i+1
                print('reached goal in %d step', len, state)
                break
        return traj[:len]

    def draw_image(self):
        for x in self.obstacles:
            ax = plt.gca()
            ax.add_patch(x.get_patch())
        return



def cost(state, next_state, goal, env):
    goal_region = 0.05
    dist = np.maximum(np.linalg.norm(state - next_state, axis=1), 0.025)
    dist_to_goal = np.linalg.norm(state - goal, axis=1)
    next_dist_to_goal = np.linalg.norm(next_state - goal, axis=1)
    penalty = 10.0
    goal_bonus = -0.0
    collisions = np.array([env.in_collision(state[i]) for i in range(state.shape[0])])
    next_collisions = np.array([env.in_collision(next_state[i]) for i in range(state.shape[0])])
    c = penalty * collisions + penalty * next_collisions + dist + goal_bonus * np.array(next_dist_to_goal < goal_region)
    # term = dist_to_goal < goal_region
    term = next_dist_to_goal < goal_region
    # c = penalty * collisions + penalty * next_collisions + dist + goal_bonus * np.array(dist_to_goal < goal_region) \
    #     + goal_bonus * np.array(next_dist_to_goal < goal_region)
    # term = np.logical_or(dist_to_goal < goal_region, next_dist_to_goal < goal_region)
    # import pdb; pdb.set_trace()
    return c, term


def fitted_q(data, q_net, iters, env):
    # data[0] = state, data[1] = actions, data[2] = next states
    # r = reward, term = termination signal
    # net = Q function
    # return labels for fitted Q training
    # states = Variable(torch.from_numpy(data[0]).float())
    # actions = Variable(torch.from_numpy(data[1]).float())
    # next_states = Variable(torch.from_numpy(data[2]).float())
    discount = 0.9
    for i in range(iters):
        goals = np.tile(np.array([0.8,0.8]),[data[0].shape[0], 1])
        costs, term = cost(data[0], data[2], goals, env)
        states = data[0]
        full_states = np.concatenate([states, goals], axis=1)
        actions = data[1]
        next_states = data[2]
        full_next_states = np.concatenate([next_states, goals], axis=1)
        if i == 0:
            q_net.fit(np.concatenate((full_states, actions.reshape(-1,1)), axis=1), costs)
        next_q = np.array([q_net.predict(np.concatenate([full_next_states, action_distance*a*np.ones((full_next_states.shape[0], 1))], axis=1))
                  for a in range(num_actions)])
        next_min_q = np.min(next_q.T, axis=1)
        q_update = costs + discount * np.array([0.0 if term[i] else next_min_q[i] for i in range(next_min_q.shape[0])])
        goals_only = np.concatenate((goals, goals, action_distance*actions.reshape(-1,1)), axis=1)
        states_actions = np.concatenate((full_states, action_distance*actions.reshape(-1,1)), axis=1)
        # q_net.fit(np.concatenate((states_actions, goals_only), axis=0), np.concatenate((q_update, 0.0*q_update),axis=0))
        q_net.fit(states_actions, q_update)
        if i % 1 == 0:
            print('iteration', i)
            plot_values(q_net, np.array([0.8,0.8]))
            plot_trajs(q_net)
            plt.show()
            plt.pause(0.1)
    return q_net


def predict_values(states, goals, net):
    state_action_values = np.array(
        [net.predict(np.concatenate([states, goals, a * action_distance *np.ones((states.shape[0], 1))], axis=1))
         for a in range(num_actions)])
    state_values = np.min(state_action_values.T, axis=1)
    return state_values


def plot_trajs(net):
    plt.subplot(1, 2, 2)
    plot_traj(net, np.array([0.1, 0.1]), np.array([0.8, 0.8]), 'r')
    plot_traj(net, np.array([0.9, 0.3]), np.array([0.8, 0.8]), 'b')
    plot_traj(net, np.array([0.1, 0.9]), np.array([0.8, 0.8]), 'g')


def plot_values(net, goal):
    x = np.linspace(0, 1)
    y = np.linspace(0, 1)
    X, Y = np.meshgrid(x, y)
    xy = np.stack((X.reshape(-1), Y.reshape(-1))).T
    goals = np.tile(goal, (xy.shape[0], 1))
    z = predict_values(xy, goals, net).reshape(X.shape)
    plt.clf()
    plt.subplot(1,2,1)
    plt.pcolor(X, Y, z)
    plt.colorbar()
    plt.grid()


def plot_traj(net, start, goal, color='r'):
    ax = plt.gca()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    traj = env.get_trajectory(x0=start[0], y0=start[1], net=net, goal=goal, len=150)
    ax.plot(traj[:, 0], traj[:, 1], color)


if __name__ == "__main__":
    plt.ion()  # enable interactivity
    env = Env()
    num_samples = 5 * 2500
    data = env.generate_data(num_samples)
    goal = np.array([0.7, 0.1])

    q_net = KNeighborsRegressor(n_neighbors=5)
    q_net = fitted_q(data, q_net, iters=100, env=env)
    import pdb; pdb.set_trace()
    plot_values(q_net, goal)
    plot_traj(q_net, goal)
    import pdb; pdb.set_trace()



