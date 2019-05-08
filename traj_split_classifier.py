import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt


# np.random.seed(42)
np.random.seed(43)
# Hyper Parameters
input_size = 2 + 2
hidden_size = [128, 128, 128]
num_actions = 8
batch_size = 100
learning_rate = 0.001


# Neural Network Model (1 hidden layer)
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.fc4 = nn.Linear(hidden_size[2], num_classes)
        self.nl = torch.nn.Tanh()
        # self.nl = torch.nn.ReLU()


    def forward(self, x):
        out = self.fc1(x)
        out = self.nl(out)
        out = self.fc2(out)
        out = self.nl(out)
        out = self.fc3(out)
        out = self.nl(out)
        out = self.fc4(out)
        return out

class VNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(VNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.fc4 = nn.Linear(hidden_size[2], num_classes)
        self.nl = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()


    def forward(self, x):
        out = self.fc1(x)
        out = self.nl(out)
        out = self.fc2(out)
        out = self.nl(out)
        out = self.fc3(out)
        out = self.nl(out)
        out = self.fc4(out)
        out = torch.abs(out)
        return out

K = 5
value_nets = [Net(input_size, hidden_size, 1) for k in range(K)]
classifier_nets = [Net(input_size, hidden_size, 1) for k in range(K)]
value_optimizers = [optim.Adam(value_nets[k].parameters(), lr=learning_rate) for k in range(K)]
classifier_optimizers = [optim.Adam(classifier_nets[k].parameters(), lr=learning_rate) for k in range(K)]
criterion = nn.BCEWithLogitsLoss()
sig = nn.Sigmoid()
threshold = 0.5


class Env:
    def __init__(self):
        self.pos_min = 0.0
        self.pos_max = 1.0
        dx = 0.05
        dy = 0.05
        self.dx = dx
        self.dy = dy
        self.noise = 0.05
        s2 = 1.0/np.sqrt(2)*dx
        self.action_vec = np.array([[dx, 0], [-dx, 0], [0, dy], [0, -dy], [s2, s2], [s2, -s2], [-s2, s2], [-s2, -s2]])
        self.num_actions = self.action_vec.shape[0]

    def generate_data(self, num_samples):
        pos = np.random.rand(num_samples, 2)
        actions = np.random.randint(0, high=self.num_actions, size=(num_samples))
        d_pos = self.action_vec[actions] + self.noise * np.random.randn(num_samples, 2)
        new_pos = np.clip(pos + d_pos, self.pos_min, self.pos_max)
        return pos, actions, new_pos

    def get_trajectory(self, x0, y0, net, goal):
        # x0 = 0.1
        # y0 = 0.1
        x = x0
        y = y0
        goal_region = 0.05
        len = 1000
        traj = np.zeros((len, 2))
        for i in range(len):
            traj[i] = [x, y]
            state = torch.tensor([x, y, goal[0], goal[1]]).float()
            action = net(state).max(0)[1]
            d_pos = self.action_vec[action.data] * (1 + self.noise * np.random.randn(1, 2))
            new_state = np.clip(np.array([x,y]) + d_pos, self.pos_min, self.pos_max)
            dist = np.linalg.norm(np.array([x,y]) - goal)
            x = new_state[0][0]
            y = new_state[0][1]
            if dist < goal_region:
                len = i+1
                print('reached goal in %d step', len, state)
                break
        return traj[:len]


def cost(state, next_state):
    obstacle_region = 0.25
    obstacle_center = [0.5, 0.5]
    dist = np.linalg.norm(state - next_state, axis=1)
    dist_to_obs = np.linalg.norm(next_state - obstacle_center, axis=1)
    c = 3.0 * (dist_to_obs < obstacle_region) + dist
    return c


def traj_split(data, value_nets, classifier_nets, value_optimizers, classifier_optimizers, k_max, iters=1000):
    batches = int(data[0].shape[0] / batch_size)
    # first stage - learn V for k=0 using supervised learning
    # we give c=1 for transitions, c=0 for self transition, and c=10 for non-transition states
    for i in range(iters):
        all_rand_states = data[0][np.random.permutation(range(num_samples))]
        all_costs = cost(data[0], data[2])
        print_loss = 0.0
        for b in range(batches):
            states = torch.tensor(data[0][b * batch_size:(b + 1) * batch_size]).float()
            next_states = torch.tensor(data[2][b * batch_size:(b + 1) * batch_size]).float()
            rand_states = torch.tensor(all_rand_states[b * batch_size:(b + 1) * batch_size]).float()
            full_states = torch.cat([states, next_states], dim=1)
            # costs for observed transitions:
            pred_costs = torch.squeeze(value_nets[0](full_states))
            # costs = low_cost * torch.ones(batch_size)
            costs = torch.tensor(all_costs[b * batch_size:(b + 1) * batch_size]).float()
            loss_trans = F.smooth_l1_loss(pred_costs, costs)
            # costs for self transitions:
            self_states = torch.cat([states, states], dim=1)
            pred_self_costs = torch.squeeze(value_nets[0](self_states))
            self_costs = torch.zeros(batch_size)
            loss_self = F.smooth_l1_loss(pred_self_costs, self_costs)
            # classifier costs for non transitions:
            non_trans_states = torch.cat([states, rand_states], dim=1)
            pred_non_trans = torch.squeeze(classifier_nets[0](non_trans_states))
            pred_trans = torch.squeeze(classifier_nets[0](full_states))
            classifier_pred = torch.cat([pred_non_trans, pred_trans], dim=0)
            classifier_labels = torch.cat([torch.zeros(batch_size), torch.ones(batch_size)])
            loss_classifier = criterion(classifier_pred, classifier_labels)
            # loss = loss_trans + loss_self
            loss = loss_trans
            print_loss = loss.data.numpy()
            # loss = loss_trans + loss_non_trans
            # Optimize the model
            value_optimizers[0].zero_grad()
            loss.backward()
            value_optimizers[0].step()

            classifier_optimizers[0].zero_grad()
            loss_classifier.backward()
            classifier_optimizers[0].step()
        if i % 50 == 0:
            # print(i)
            print(i, print_loss)
            plot_values(value_nets[0], classifier_nets[0], np.array([0.8, 0.8]))
            plt.pause(0.1)
    # second stage - learn V for k>0 using traj split update
    for k in range(1, k_max):
        all_rand_states = data[0][np.random.permutation(range(num_samples))]
        all_goals = data[0][np.random.permutation(range(num_samples))]
        mid_costs = np.array([traj_split_min(value_nets[k-1], classifier_nets[k-1], start, goal)[0] for start, goal in zip(all_rand_states, all_goals)])
        mid_class = np.array([0.0 if mid_costs[j] is None else 1.0 for j in range(num_samples)])
        print_loss = 0.0
        for i in range(iters):
            rand_perm = np.random.permutation(range(num_samples))
            all_rand_states = all_rand_states[rand_perm]
            all_goals = all_goals[rand_perm]
            mid_costs = mid_costs[rand_perm]
            mid_class = mid_class[rand_perm]
            for b in range(batches):
                states = torch.tensor(all_rand_states[b * batch_size:(b + 1) * batch_size]).float()
                goals = torch.tensor(all_goals[b * batch_size:(b + 1) * batch_size]).float()
                full_states = torch.cat([states, goals], dim=1)
                classes = torch.tensor(mid_class[b * batch_size:(b + 1) * batch_size]).float()
                feasible_idx = np.array([mid_costs[b * batch_size:(b + 1) * batch_size][j] is not None for j in range(batch_size)])
                feasible_mask = torch.ByteTensor(feasible_idx.astype(int))
                if feasible_mask.any():
                    pred_costs = torch.squeeze(value_nets[k](torch.masked_select(full_states, feasible_mask.reshape(-1,1)).reshape(-1, 4)))
                    costs = torch.tensor(mid_costs[b * batch_size:(b + 1) * batch_size][feasible_idx].astype(float)).float()
                    loss = F.smooth_l1_loss(pred_costs, costs)
                    print_loss = loss.data.numpy()
                    # Optimize the model
                    value_optimizers[k].zero_grad()
                    loss.backward()
                    value_optimizers[k].step()

                    pred_class = torch.squeeze(classifier_nets[k](full_states))
                    loss_classifier = criterion(pred_class, classes)
                    classifier_optimizers[k].zero_grad()
                    loss_classifier.backward()
                    classifier_optimizers[k].step()

            if i % 50 == 0:
                # print(i)
                print(i, print_loss)
                plot_values(value_nets[k], classifier_nets[k], np.array([0.8, 0.8]))
                plt.pause(0.1)
    return value_nets


def predict_values(data, goals, net):
    # import pdb; pdb.set_trace()
    states = torch.tensor(data).float()
    goals = torch.tensor(goals).float()
    state_values = net(torch.cat([states, goals], dim=1))
    return state_values


def plot_values(net, classifier, goal):
    x = np.linspace(0, 1)
    y = np.linspace(0, 1)
    X, Y = np.meshgrid(x, y)
    xy = np.stack((X.reshape(-1), Y.reshape(-1))).T
    goals = np.tile(goal, (xy.shape[0], 1))
    z = predict_values(xy, goals, net).data.numpy().reshape(X.shape)
    c = sig(predict_values(xy, goals, classifier)).data.numpy().reshape(X.shape)
    z_th = z * (c > threshold)
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.pcolor(X, Y, z_th)
    plt.colorbar()
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.pcolor(X, Y, c)
    plt.colorbar()
    plt.grid()
    # plt.draw()


def traj_split_min(net, classifier, start, goal):
    x = np.linspace(0, 1)
    y = np.linspace(0, 1)
    X, Y = np.meshgrid(x, y)
    mid_points = np.stack((X.reshape(-1), Y.reshape(-1))).T
    goals = np.tile(goal, (mid_points.shape[0], 1))
    starts = np.tile(start, (mid_points.shape[0], 1))
    to_mid = predict_values(starts, mid_points, net).data.numpy().squeeze()#.reshape(X.shape)
    to_mid_class = sig(predict_values(starts, mid_points, classifier)).data.numpy().squeeze()#.reshape(X.shape)
    from_mid = predict_values(mid_points, goals, net).data.numpy().squeeze()#.reshape(X.shape)
    from_mid_class = sig(predict_values(mid_points, goals, classifier)).data.numpy().squeeze()#.reshape(X.shape)
    feasible_mid_points_idx = np.logical_and((to_mid_class > threshold) ,(from_mid_class > threshold))
    # min_mid = np.min(to_mid + from_mid)
    feasible_min_mid = None
    mid_point = None
    if feasible_mid_points_idx.any():
        feasible_min_mid = np.min(to_mid[feasible_mid_points_idx] + from_mid[feasible_mid_points_idx])
        feasible_mid_points = mid_points[feasible_mid_points_idx]
        mid_point = feasible_mid_points[np.argmin(to_mid[feasible_mid_points_idx] + from_mid[feasible_mid_points_idx])]
        feasible_min_mid = max(feasible_min_mid, 0.0)
    return feasible_min_mid, mid_point


def get_traj_split(value_nets, classifier_nets, start, goal, k):
    if k == 0:
        return [start, goal]
    else:
        mid_point = traj_split_min(value_nets[k-1], classifier_nets[k-1], start, goal)[1]
        # import pdb; pdb.set_trace()
        assert mid_point is not None
        path_to_mid = get_traj_split(value_nets, classifier_nets, start, mid_point, k - 1)
        path_from_mid = get_traj_split(value_nets, classifier_nets, mid_point, goal, k - 1)
        return path_to_mid[0:-1] + path_from_mid


def plot_traj(ax, value_nets, classifier_nets, start, goal, k_max, color='r'):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    traj = np.array(get_traj_split(value_nets, classifier_nets, start, goal, k_max))
    ax.plot(traj[:, 0], traj[:, 1], color)
    circle1 = plt.Circle(goal, 0.05, color='m')
    circle2 = plt.Circle([0.5, 0.5], 0.25, color='r')
    ax.add_artist(circle1)
    ax.add_artist(circle2)
    plt.show()


def plot_trajs(value_nets, classifier_nets, K):
    fig, ax = plt.subplots()
    plot_traj(ax, value_nets, classifier_nets, np.array([0.9, 0.3]), np.array([0.8, 0.8]), 3, 'r')
    plot_traj(ax, value_nets, classifier_nets, np.array([0.4, 0.2]), np.array([0.8, 0.8]), 3, 'b')
    plot_traj(ax, value_nets, classifier_nets, np.array([0.4, 0.9]), np.array([0.8, 0.8]), 2, 'g')
    plot_traj(ax, value_nets, classifier_nets, np.array([0.2, 0.3]), np.array([0.8, 0.8]), 3, 'm')
    plot_traj(ax, value_nets, classifier_nets, np.array([0.95, 0.95]), np.array([0.8, 0.8]), 2, 'c')
    plot_traj(ax, value_nets, classifier_nets, np.array([0.2, 0.2]), np.array([0.8, 0.8]), 3, 'k')

plt.ion()  # enable interactivity
env = Env()
num_samples = 2 * 2500
data = env.generate_data(num_samples)
goal = np.array([0.7, 0.7])

PATH = './model_large_obstacle.pt'
checkpoint = torch.load(PATH)
for k in range(K):
    value_nets[k].load_state_dict(checkpoint['model_state_dict'][k])
    value_optimizers[k].load_state_dict(checkpoint['optimizer_state_dict'][k])
    classifier_nets[k].load_state_dict(checkpoint['classifier_state_dict'][k])
    classifier_optimizers[k].load_state_dict(checkpoint['classifier_optimizer_state_dict'][k])

plot_trajs(value_nets, classifier_nets, K)
import pdb; pdb.set_trace()


policy_net = traj_split(data, value_nets, classifier_nets, value_optimizers, classifier_optimizers, K, iters=1000)
import pdb; pdb.set_trace()

torch.save({
            'epoch': 5000,
            'model_state_dict': [value_nets[k].state_dict() for k in range(K)],
            'classifier_state_dict': [classifier_nets[k].state_dict() for k in range(K)],
            'optimizer_state_dict': [value_optimizers[k].state_dict() for k in range(K)],
            'classifier_optimizer_state_dict': [classifier_optimizers[k].state_dict() for k in range(K)],
            }, PATH)


