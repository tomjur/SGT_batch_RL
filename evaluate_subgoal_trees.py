import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from universal_fitted_Q_hard import Env
from traj_split_KNN_hard import get_traj_split
import pickle


np.random.seed(42)
# Hyper Parameters
input_size = 2 + 2
K = 8
num_actions = 8
action_distance = 100


def predict_subgoal_tree(env, start, goal, q_net, sg_net, K, k_min=0):
    subgoals = np.array(get_traj_split(sg_net, start, goal, K, k_min))
    curr_state = start
    traj = np.array([start])
    for i in range(1, subgoals.shape[0]):
        curr_segment = env.get_trajectory_im(x0=curr_state[0], y0=curr_state[1], im_net=q_net, goal=subgoals[i])
        traj = np.concatenate((traj, curr_segment), axis=0)
        curr_state = curr_segment[-1]
    return traj, subgoals


def predict_subgoal_tree_q(env, start, goal, q_net, sg_net, K, k_min=0):
    subgoals = np.array(get_traj_split(sg_net, start, goal, K, k_min))
    curr_state = start
    traj = np.array([start])
    for i in range(1, subgoals.shape[0]):
        curr_segment = env.get_trajectory(x0=curr_state[0], y0=curr_state[1], net=q_net, goal=subgoals[i])
        traj = np.concatenate((traj, curr_segment), axis=0)
        curr_state = curr_segment[-1]
    return traj, subgoals


def train_inverse_model(data, im_net):
    states = np.array(data[0])
    next_states = np.array(data[2])
    actions = np.array(data[1])
    im_net.fit(np.concatenate((states, next_states), axis=1), actions)
    return im_net

plt.ion()  # enable interactivity
env = Env()
# import pdb; pdb.set_trace()
data = pickle.load(open("data_maze_baseline.p", "rb"))

starts = np.array([[0.7, 0.1], [0.1, 0.7]])
goals = np.array([[0.1, 0.1], [0.7, 0.1]])

q_net = pickle.load(open("q_knn.p", "rb"))
sg_net = pickle.load(open("values_knn.p", "rb"))

im_net = KNeighborsClassifier(n_neighbors=5)
im_net = train_inverse_model(data, im_net)
# import pdb; pdb.set_trace()


traj_im, _ = predict_subgoal_tree(env, starts[0], goals[0], im_net, sg_net, K=8, k_min=1)
traj, subgoals = predict_subgoal_tree_q(env, starts[0], goals[0], q_net, sg_net, K=8, k_min=1)
plt.subplot(1, 2, 2)
ax = plt.gca()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.plot(traj[:, 0], traj[:, 1], 'r')
ax.plot(traj_im[:, 0], traj_im[:, 1], 'g')
ax.plot(subgoals[:, 0], subgoals[:, 1], 'b')
plt.show()
# import pdb; pdb.set_trace()
# plot_values(q_net, goal)
# plot_traj(q_net, goal)
import pdb; pdb.set_trace()



