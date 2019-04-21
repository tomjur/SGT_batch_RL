import networkx as nx
import numpy as np

def FW(w):
    G = nx.DiGraph(w)
    path = dict(nx.all_pairs_dijkstra(G, weight='weight'))
    true_len = np.array([[path[i][0][j] for i in range(N)] for j in range(N)])
    return true_len


def VI(w, iter, verbose=False):
    if verbose:
        true_len = FW(w)
    n = w.shape[0]
    # v = np.zeros([n, n])
    v = 50 * np.ones([n, n])
    for i in range(iter):
        for s in range(n):
            for g in range(n):
                update = [my_mat[s, s_next] + v[s_next, g] for s_next in range(n)]
                v[s, g] = np.min(update) if s is not g else 0
        if verbose:
            print('iteration ', i)
            print(np.max(np.abs(v.T - true_len)))
    return v


def TS(w, max_splits, verbose=False):
    if verbose:
        true_len = FW(w)
    n = w.shape[0]
    v = np.zeros([n, n, max_splits])
    for s in range(n):
        for g in range(n):
            v[s, g, 0] = my_mat[s, g] if s is not g else 0
    for split in range(1, max_splits):
        for s in range(n):
            for g in range(n):
                update = [v[s, s_next, split - 1] + v[s_next, g, split - 1] for s_next in range(n)]
                v[s, g, split] = np.min(update) if s is not g else 0
        if verbose:
            print('split ', split)
        print(np.max(np.abs(v[:, :, split].T - true_len)))
    return v



N = 25
my_mat = np.matrix([[1, 2, 5, 4, 2], [1, 1, 2, 5, 3], [5, 3, 1, 6, 1], [4, 3, 2, 2, 4], [3, 3, 4, 2, 4]])
my_mat = np.abs(np.random.randn(N,N))
true_len = FW(my_mat)
print(true_len)


# value iteration
print('Value Iteration')
Iter = N
v = VI(my_mat, Iter, verbose=True)
print('Incremental Traj Split')
max_splits = 7
vs = TS(my_mat, max_splits, verbose=True)
# print(vs[:,:,-1].T)



# # traj split
# print('Traj Split')
# Iter = 7
# # v = np.zeros([N,N])
# v = 5 * np.ones([N,N])
# # v = my_mat.copy()
# for i in range(Iter):
#     for s in range(N):
#         for g in range(N):
#             # update = [[my_mat[s, s_next] + my_mat[s_next, s_next_next] + v[s_next_next, g] for s_next in range(N)] for s_next_next in range(N)]
#             update = [v[s, s_next] + v[s_next, g] for s_next in range(N)]
#             v[s, g] = np.min(np.array([np.min(update), my_mat[s, g]])) if s is not g else 0

#
#
# # Incremental new split
# print('Incremental Traj Split')
# Iter = 3
# max_splits = 3
# # v = np.zeros([N,N])
# v = np.zeros([N,N,max_splits])
# # v = my_mat.copy()
# for i in range(Iter):
#     for s in range(N):
#         for g in range(N):
#             v[s, g, 0] = my_mat[s, g] if s is not g else 0
#     for split in range(1, max_splits):
#         for s in range(N):
#             for g in range(N):
#                 update = [v[s, s_next, split-1] + v[s_next, g, split-1] for s_next in range(N)]
#                 v[s, g, split] = np.min(update) if s is not g else 0
#                 # v[s, g] = np.min(np.array([np.min(update), my_mat[s, g]])) if s is not g else 0
#                 # import pdb; pdb.set_trace()
#     print('iteration ', i)
#     print(np.max(np.abs(v[:,:,-1].T - length)))

# print(v[:,:,-1].T)