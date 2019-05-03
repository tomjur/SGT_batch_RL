import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt


np.random.seed(42)
# Hyper Parameters
input_size = 2 + 2
hidden_size = [32, 32]
num_actions = 8
# num_epochs = 5
batch_size = 100
learning_rate = 0.001
TARGET_UPDATE = 50


# Neural Network Model (1 hidden layer)
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], num_classes)
        self.nl = torch.nn.Tanh()
        # self.nl = torch.nn.ReLU()


    def forward(self, x):
        out = self.fc1(x)
        out = self.nl(out)
        out = self.fc2(out)
        out = self.nl(out)
        out = self.fc3(out)
        return out


policy_net = Net(input_size, hidden_size, 1)
policy_net1 = Net(input_size, hidden_size, 1)
target_net = Net(input_size, hidden_size, 1)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
optimizer1 = optim.Adam(policy_net1.parameters(), lr=learning_rate)


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
        self.action_vec = np.array([[dx, 0], [-dx, 0], [0, dy], [0, -dy], [s2, s2], [s2, -s2], [-s2, s2], [-s2, s2]])
        self.num_actions = self.action_vec.shape[0]

    def generate_data(self, num_samples):
        pos = np.random.rand(num_samples, 2)
        actions = np.random.randint(0, high=self.num_actions, size=(num_samples))
        d_pos = self.action_vec[actions] * (np.ones((num_samples, 2)) + self.noise * np.random.randn(num_samples, 2))
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


def reward(pos, goal):
    goal_region = 0.15
    eps = 0.05
    dist = np.linalg.norm(pos - goal, axis=1)
    r = 1.0 * (dist < goal_region) - eps * (dist >= goal_region)
    term = dist < goal_region
    return r, term


def traj_split(data, value_nets, k):
    iters = 5000
    batches = int(data[0].shape[0] / batch_size)
    high_cost = 10
    low_cost = 1
    # first stage - learn V for k=0 using supervised learning
    # we give c=1 for transitions, c=0 for self transition, and c=100 for non-transition states

    # train value for k=0
    for i in range(iters):
        all_rand_states = data[0][np.random.permutation(range(num_samples))]
        all_goals = data[0][np.random.permutation(range(num_samples))]
        r, term = reward(data[0], all_goals)
        for k in range(batches):
            states = torch.tensor(data[0][k*batch_size:(k+1)*batch_size]).float()
            next_states = torch.tensor(data[2][k*batch_size:(k+1)*batch_size]).float()
            rand_states = torch.tensor(all_rand_states[k * batch_size:(k + 1) * batch_size]).float()
            full_states = torch.cat([states, next_states], dim=1)
            # costs for observed transitions:
            pred_costs = torch.squeeze(value_nets[0](full_states))
            costs = low_cost * torch.ones(batch_size)
            loss_trans = F.smooth_l1_loss(pred_costs, costs)
            # costs for self transitions:
            self_states = torch.cat([states, states], dim=1)
            pred_self_costs = torch.squeeze(value_nets[0](self_states))
            self_costs = torch.zeros(batch_size)
            loss_self = F.smooth_l1_loss(pred_self_costs, self_costs)
            # costs for non transitions:
            non_trans_states = torch.cat([states, rand_states], dim=1)
            pred_non_trans_costs = torch.squeeze(value_nets[0](non_trans_states))
            non_trans_costs = high_cost * torch.ones(batch_size)
            loss_non_trans = F.smooth_l1_loss(pred_non_trans_costs, non_trans_costs)
            loss = loss_trans + loss_self + loss_non_trans
            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if i % 50 == 0:
            print(i)
            plot_values(value_nets[0], np.array([0.5, 0.5]))
            plt.pause(0.1)
    # train value for k=1
    # import pdb; pdb.set_trace()
    all_rand_states = data[0][np.random.permutation(range(num_samples))]
    all_goals = data[0][np.random.permutation(range(num_samples))]
    mid_costs = np.array([traj_split_min(value_nets[0], start, goal) for start, goal in zip(all_rand_states, all_goals)])
    # import pdb; pdb.set_trace()
    for i in range(iters):
        for k in range(batches):
            states = torch.tensor(all_rand_states[k*batch_size:(k+1)*batch_size]).float()
            goals = torch.tensor(all_goals[k*batch_size:(k+1)*batch_size]).float()
            # rand_states = torch.tensor(all_rand_states[k * batch_size:(k + 1) * batch_size]).float()
            full_states = torch.cat([states, goals], dim=1)
            # costs for observed transitions:
            pred_costs = torch.squeeze(value_nets[1](full_states))
            costs = torch.tensor(mid_costs[k*batch_size:(k+1)*batch_size]).float()
            loss = F.smooth_l1_loss(pred_costs, costs)
            # Optimize the model
            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()
        if i % 50 == 0:
            print(i)
            plot_values(value_nets[1], np.array([0.5, 0.5]))
            plt.pause(0.1)
    return value_nets


def fitted_q(data, policy_net, target_net, iters):
    # data[0] = state, data[1] = actions, data[2] = next states
    # net = Q function
    # return labels for fitted Q training
    # states = Variable(torch.from_numpy(data[0]).float())
    # actions = Variable(torch.from_numpy(data[1]).float())
    # next_states = Variable(torch.from_numpy(data[2]).float())
    batches = int(data[0].shape[0] / batch_size)
    for i in range(iters):
        all_goals = data[0][np.random.permutation(range(num_samples))]
        r, term = reward(data[0], all_goals)
        for k in range(batches):
            states = torch.tensor(data[0][k*batch_size:(k+1)*batch_size]).float()
            goals = torch.tensor(all_goals[k*batch_size:(k+1)*batch_size]).float()
            full_states = torch.cat([states, goals], dim=1)
            actions = torch.tensor(data[1][k*batch_size:(k+1)*batch_size]).int()
            next_states = torch.tensor(data[2][k*batch_size:(k+1)*batch_size]).float()
            full_next_states = torch.cat([next_states, goals], dim=1)
            rewards = torch.tensor(r[k*batch_size:(k+1)*batch_size]).float()
            state_action_values = policy_net(full_states).gather(1, actions.long().unsqueeze(1))
            next_state_values = torch.zeros(data[0][k * batch_size:(k + 1) * batch_size].shape[0])
            # non_final_mask = torch.tensor((~term).astype(int))
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not True, term[k * batch_size:(k + 1) * batch_size].tolist())), dtype=torch.uint8)
            # import pdb; pdb.set_trace()
            next_state_values[non_final_mask] = target_net(full_next_states[non_final_mask]).max(1)[0].detach()
            # Compute the expected Q values
            expected_state_action_values = 0.9 * next_state_values + rewards
            # Compute Huber loss
            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            # for param in policy_net.parameters():
            #     param.grad.data.clamp_(-1, 1)
            optimizer.step()
        if i % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            print('iteration', i)
    return policy_net, target_net


def predict_values(data, goals, net):
    # import pdb; pdb.set_trace()
    states = torch.tensor(data).float()
    goals = torch.tensor(goals).float()
    state_values = net(torch.cat([states, goals], dim=1))
    return state_values


def plot_values(net, goal):
    x = np.linspace(0, 1)
    y = np.linspace(0, 1)
    X, Y = np.meshgrid(x, y)
    xy = np.stack((X.reshape(-1), Y.reshape(-1))).T
    goals = np.tile(goal, (xy.shape[0], 1))
    z = predict_values(xy, goals, net).data.numpy().reshape(X.shape)
    plt.clf()
    plt.pcolor(X, Y, z)
    plt.colorbar()
    # plt.draw()


def traj_split_min(net, start, goal):
    x = np.linspace(0, 1)
    y = np.linspace(0, 1)
    X, Y = np.meshgrid(x, y)
    mid_points = np.stack((X.reshape(-1), Y.reshape(-1))).T
    goals = np.tile(goal, (mid_points.shape[0], 1))
    starts = np.tile(start, (mid_points.shape[0], 1))
    to_mid = predict_values(starts, mid_points, net).data.numpy().reshape(X.shape)
    from_mid = predict_values(mid_points, goals, net).data.numpy().reshape(X.shape)
    min_mid = np.min(to_mid + from_mid)
    return min_mid


def plot_traj(net, goal):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    traj = env.get_trajectory(x0=0.1, y0=0.2, net=net, goal=goal)
    ax.plot(traj[:, 0], traj[:, 1], 'r')
    traj = env.get_trajectory(x0=0.5, y0=0.1, net=net, goal=goal)
    ax.plot(traj[:, 0], traj[:, 1], 'b')
    traj = env.get_trajectory(x0=0.4, y0=0.4, net=net, goal=goal)
    ax.plot(traj[:, 0], traj[:, 1], 'g')
    circle1 = plt.Circle(goal, 0.05, color='r')
    ax.add_artist(circle1)
    plt.show()

plt.ion()  # enable interactivity
env = Env()
num_samples = 2500
data = env.generate_data(num_samples)
# goals = data[0][np.random.permutation(range(num_samples))]
# r, term = reward(data[0], goals)
goal = np.array([0.7, 0.7])
policy_net = traj_split(data, [policy_net, policy_net1], 5000)
# plot_values(policy_net, goal)
# plot_traj(policy_net, goal)
import pdb; pdb.set_trace()
