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


def predict_subgoal_tree(env, start, goal, im_net, sg_net, K, k_min=0):
    # use sub-goal tree as waypoints, inverse model as controller
    subgoals = np.array(get_traj_split(sg_net, start, goal, K, k_min))
    curr_state = start
    traj = np.array([start])
    for i in range(1, subgoals.shape[0]):
        curr_segment = env.get_trajectory_im(x0=curr_state[0], y0=curr_state[1], im_net=im_net, goal=subgoals[i], len=10)
        traj = np.concatenate((traj, curr_segment), axis=0)
        curr_state = curr_segment[-1]
    return traj, subgoals


def predict_subgoal_tree_q(env, start, goal, q_net, sg_net, K, k_min=0):
    # use sub-goal tree as waypoints, Q network as controller
    subgoals = np.array(get_traj_split(sg_net, start, goal, K, k_min))
    curr_state = start
    traj = np.array([start])
    for i in range(1, subgoals.shape[0]):
        curr_segment = env.get_trajectory(x0=curr_state[0], y0=curr_state[1], net=q_net, goal=subgoals[i], len=10)
        traj = np.concatenate((traj, curr_segment), axis=0)
        curr_state = curr_segment[-1]
    return traj, subgoals


def predict_subgoal_tree_q_baseline(env, start, goal, q_net):
    # use only Q network as controller (fixed goal)
    traj = env.get_trajectory(x0=start[0], y0=start[1], net=q_net, goal=goal, len=100)
    return traj

def train_inverse_model(data, im_net):
    states = np.array(data[0])
    next_states = np.array(data[2])
    actions = np.array(data[1])
    im_net.fit(np.concatenate((states, next_states), axis=1), actions)
    return im_net


if __name__ == "__main__":
    plt.ion()  # enable interactivity
    env = Env()
    data = pickle.load(open("data_maze_baseline.p", "rb"))

    num_points = 3
    starts = np.array([env.get_free_point() for i in range(num_points)])
    goals = np.array([env.get_free_point() for i in range(num_points)])

    q_net = pickle.load(open("q_knn.p", "rb"))
    sg_net = pickle.load(open("values_knn.p", "rb"))

    im_net = KNeighborsClassifier(n_neighbors=5)
    im_net = train_inverse_model(data, im_net)

    plt.figure(figsize=[6.4, 6.4])
    K = 7

    final_dist_baseline = []
    final_dist_im = []
    final_dist_q = []
    success_baseline = []
    success_im = []
    success_q = []

    for i in range(num_points):
        traj_baseline = predict_subgoal_tree_q_baseline(env, starts[i], goals[i], q_net)
        traj_im, _ = predict_subgoal_tree(env, starts[i], goals[i], im_net, sg_net, K=K, k_min=1)
        traj, subgoals = predict_subgoal_tree_q(env, starts[i], goals[i], q_net, sg_net, K=K, k_min=1)

        success_baseline.append(np.any(np.array([env.in_collision(traj_baseline[j]) for j in range(traj_baseline.shape[0])])))
        success_im.append(np.any(np.array([env.in_collision(traj_im[j]) for j in range(traj_im.shape[0])])))
        success_q.append(np.any(np.array([env.in_collision(traj[j]) for j in range(traj.shape[0])])))

        final_dist_baseline.append(np.linalg.norm(goals[i] - traj_baseline[-1]))
        final_dist_im.append(np.linalg.norm(goals[i] - traj_im[-1]))
        final_dist_q.append(np.linalg.norm(goals[i] - traj[-1]))

        if i >= 0:
            plt.clf()
            ax = plt.gca()
            env.draw_image()
            plt.axis('square')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.plot(traj[:, 0], traj[:, 1], 'r', marker='.')
            ax.plot(traj_im[:, 0], traj_im[:, 1], 'g', marker='.')
            ax.plot(traj_baseline[:, 0], traj_baseline[:, 1], 'c', marker='.')
            ax.plot(subgoals[:, 0], subgoals[:, 1], 'b', marker='.')
            plt.show()
            plt.pause(0.1)

    import pdb; pdb.set_trace()



