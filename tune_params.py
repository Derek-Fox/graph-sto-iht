# -*- coding: utf-8 -*-

"""
References:
    [1] An Effective Hard Thresholding Method Based on Stochastic Variance Reduction for Nonconvex Sparse Learning - Guannan Liang, Qianqian Tong, Chunjiang Zhu, Jinbo Bi
    [2] Stochastic Iterative Hard Thresholding for Graph-structured Sparsity Optimization - Baojian Zhou,  Feng Chen, Yiming Ying
"""

import os
import sys
import time
import random
import pickle
import multiprocessing
from itertools import product
import numpy as np

try:
    import sparse_module

    try:
        from sparse_module import wrap_head_tail_bisearch
    except ImportError:
        print('cannot find wrap_head_tail_bisearch method in sparse_module')
        sparse_module = None
        exit(0)
except ImportError:
    print('\n'.join([
        'cannot find the module: sparse_module',
        'try run: \'python setup.py build_ext --inplace\' first! ']))


def algo_head_tail_bisearch(edges, x, costs, g, root, s_low, s_high, max_num_iter, verbose):
    """ This is the wrapper of head/tail-projection proposed in [2].
    :param edges:           edges in the graph.
    :param x:               projection vector x.
    :param costs:           edge costs in the graph.
    :param g:               the number of connected components.
    :param root:            root of subgraph. Usually, set to -1: no root.
    :param s_low:           the lower bound of the sparsity.
    :param s_high:          the upper bound of the sparsity.
    :param max_num_iter:    the maximum number of iterations used in
                            binary search procedure.
    :param verbose: print out some information.
    :return:            1.  the support of the projected vector
                        2.  the projected vector
    """
    prizes = x * x
    # to avoid too large upper bound problem.
    if s_high >= len(prizes) - 1:
        s_high = len(prizes) - 1
    re_nodes = wrap_head_tail_bisearch(
        edges, prizes, costs, g, root, s_low, s_high, max_num_iter, verbose)
    proj_w = np.zeros_like(x)
    proj_w[re_nodes[0]] = x[re_nodes[0]]
    return re_nodes[0], proj_w


def simu_grid_graph(width, height, rand_weight=False):
    """ Generate a grid graph with size, width x height. Totally there will be
        width x height number of nodes in this generated graph.
    :param width:       the width of the grid graph.
    :param height:      the height of the grid graph.
    :param rand_weight: the edge costs in this generated grid graph.
    :return:            1.  list of edges
                        2.  list of edge costs
    """
    np.random.seed()
    if width < 0 and height < 0:
        print('Error: width and height should be positive.')
        return [], []
    width, height = int(width), int(height)
    edges, weights = [], []
    index = 0
    for i in range(height):
        for j in range(width):
            if (index % width) != (width - 1):
                edges.append((index, index + 1))
                if index + width < int(width * height):
                    edges.append((index, index + width))
            else:
                if index + width < int(width * height):
                    edges.append((index, index + width))
            index += 1
    edges = np.asarray(edges, dtype=int)
    # random generate costs of the graph
    if rand_weight:
        weights = []
        while len(weights) < len(edges):
            weights.append(random.uniform(1., 2.0))
        weights = np.asarray(weights, dtype=np.float64)
    else:  # set unit weights for edge costs.
        weights = np.ones(len(edges), dtype=np.float64)
    return edges, weights


def sensing_matrix(n, x, norm_noise=0.0):
    """ Generate sensing matrix (design matrix). This generated sensing
        matrix is a Gaussian matrix, i.e., each entry ~ N(0,\sigma/\sqrt(n)).
        Please see more details in equation (1.2) shown in reference [6].
    :param n:           the number of measurements required.
    :param x:           the input signal.
    :param norm_noise:  plus ||norm_noise|| noise on the measurements.
    :return:            1.  the design matrix
                        2.  the vector of measurements
                        3.  the noised vector.
    """
    p = len(x)
    x_mat = np.random.normal(loc=0.0, scale=1.0, size=(n * p)) / np.sqrt(n)
    x_mat = x_mat.reshape((n, p))
    y_tr = np.dot(x_mat, x)
    noise_e = np.random.normal(loc=0.0, scale=1.0, size=len(y_tr))
    y_e = y_tr + (norm_noise / np.linalg.norm(noise_e)) * noise_e
    return x_mat, y_tr, y_e


def random_walk(edges, s, init_node=None, restart=0.0):
    """ The random walk on graphs. Please see details in reference [5].
    :param edges:       the edge list of the graph.
    :param s:           the sparsity ( number of nodes) in the true subgraph.
    :param init_node:   initial point of the random walk.
    :param restart:     with restart.
    :return:            1. list of nodes walked.
                        2. list of edges walked.
    """
    np.random.seed()
    adj, nodes = dict(), set()
    for edge in edges:  # construct the adjacency matrix.
        uu, vv = int(edge[0]), int(edge[1])
        nodes.add(uu)
        nodes.add(vv)
        if uu not in adj:
            adj[uu] = set()
        adj[uu].add(vv)
        if vv not in adj:
            adj[vv] = set()
        adj[vv].add(uu)
    if init_node is None:
        # random select an initial node.
        rand_start_point = random.choice(list(nodes))
        init_node = list(adj.keys())[rand_start_point]
    if init_node not in nodes:
        print('Error: the initial_node is not in the graph!')
        return [], []
    if not (0.0 <= restart < 1.0):
        print('Error: the restart probability not in (0.0,1.0)')
        return [], []
    if not (0 <= s <= len(nodes)):
        print('Error: the number of nodes not in [0,%d]' % len(nodes))
        return [], []
    subgraph_nodes, subgraph_edges = set(), set()
    next_node = init_node
    subgraph_nodes.add(init_node)
    if s <= 1:
        return subgraph_nodes, subgraph_edges
    # get a connected subgraph with s nodes.
    while len(subgraph_nodes) < s:
        next_neighbors = list(adj[next_node])
        rand_nei = random.choice(next_neighbors)
        subgraph_nodes.add(rand_nei)
        subgraph_edges.add((next_node, rand_nei))
        subgraph_edges.add((rand_nei, next_node))
        next_node = rand_nei  # go to next node.
        if random.random() < restart:
            next_node = init_node
    return list(subgraph_nodes), list(subgraph_edges)


def algo_graph_svrg_iht(
        x_mat, y_tr, max_epochs, lr, x_star, x0, tol_algo, edges, costs, s, b,
        g=1, root=-1, gamma=0.1, proj_max_num_iter=50, verbose=0):
    """ Graph Stochastic Iterative Hard Thresholding with Variance Reduction (SVRG).
    :param x_mat:       the design matrix.
    :param y_tr:        the array of measurements.
    :param max_epochs:  the maximum epochs (iterations) allowed.
    :param lr:          the learning rate (should be 1.0).
    :param x_star:      the true signal.
    :param x0:          x0 is the initial point.
    :param tol_algo:    tolerance parameter for early stopping.
    :param edges:       edges in the graph.
    :param costs:       edge costs
    :param s:           sparsity
    :param b: the block size
    :param g:           number of connected component in the true signal.
    :param root:        the root included in the result (default -1: no root).
    :param gamma:       to control the upper bound of sparsity.
    :param proj_max_num_iter: maximum number of iterations of projection.
    :param verbose: print out some information.
    :return:            1.  the final estimation error,
                        2.  number of epochs(iterations) used,
                        3.  and the run time.
    TODO: Figure out why this diverges instead of converges. Gradient calculations wrong?
    """
    np.random.seed()
    start_time = time.time()
    x_hat = np.copy(x0)

    # graph projection para
    h_low = int(len(x0) / 2)
    h_high = int(h_low * (1. + gamma))
    t_low = int(s)
    t_high = int(s * (1. + gamma))

    (n, p) = x_mat.shape
    # if block size is larger than n,
    # just treat it as a single block (batch)
    b = n if n < b else b
    num_blocks = int(n) / int(b)

    num_epochs = 0

    loss_list = []

    for epoch_i in range(max_epochs):
        num_epochs += 1
        outer_grad = calc_grad(x_mat, y_tr, x_hat, range(n))
        x_nil = np.copy(x_hat)
        for _ in range(num_blocks):
            # for _ in range(n * 2):  As described in [2]
            block = get_block(b, num_blocks)
            inner_grad_1 = calc_grad(x_mat, y_tr, x_nil, block)
            if epoch_i < 1:
                gradient = inner_grad_1
            else:
                inner_grad_2 = calc_grad(x_mat, y_tr, x_hat, block)
                gradient = inner_grad_1 - inner_grad_2 + outer_grad
            head_nodes, proj_grad = algo_head_tail_bisearch(
                edges, gradient, costs, g, root, h_low, h_high,
                proj_max_num_iter, verbose)
            bt = x_nil - lr * proj_grad
            tail_nodes, proj_bt = algo_head_tail_bisearch(
                edges, bt, costs, g, root,
                t_low, t_high, proj_max_num_iter, verbose)
            x_nil = proj_bt
        x_hat = x_nil
        x_hat_norm = np.linalg.norm(x_hat)
        residual_norm = np.linalg.norm(y_tr - np.dot(x_mat, x_hat))
        x_err = np.linalg.norm(x_hat - x_star)
        print("Epoch: %d, Residual norm: %.6f, x_hat norm: %.6f, x_err: %.6f" % (
            epoch_i + 1, residual_norm, x_hat_norm, x_err))
        if x_hat_norm >= 1e5:  # diverge cases.
            break
        if residual_norm <= tol_algo:
            break
        loss_list.append((epoch_i, residual_norm))
    x_err = np.linalg.norm(x_hat - x_star)
    run_time = time.time() - start_time
    return loss_list, x_err, num_epochs, run_time


def algo_graph_scsg_iht(
        x_mat, y_tr, max_epochs, lr, x_star, x0, tol_algo, edges, costs, s, b, mini_block_size=1,
        g=1, root=-1, gamma=0.1, proj_max_num_iter=50, verbose=0):
    """ Graph Stochastic Iterative Hard Thresholding with Variance Reduction (SVRG).
    :param x_mat:       the design matrix.
    :param y_tr:        the array of measurements.
    :param max_epochs:  the maximum epochs (iterations) allowed.
    :param lr:          the learning rate (should be 1.0).
    :param x_star:      the true signal.
    :param x0:          x0 is the initial point.
    :param tol_algo:    tolerance parameter for early stopping.
    :param edges:       edges in the graph.
    :param costs:       edge costs
    :param s:           sparsity
    :param b: the block size
    :param mini_block_size: the mini block size
    :param g:           number of connected component in the true signal.
    :param root:        the root included in the result (default -1: no root).
    :param gamma:       to control the upper bound of sparsity.
    :param proj_max_num_iter: maximum number of iterations of projection.
    :param verbose: print out some information.
    :return:            1.  the final estimation error,
                        2.  number of epochs(iterations) used,
                        3.  and the run time.
    """
    np.random.seed()
    start_time = time.time()
    x_hat = np.copy(x0)

    # graph projection para
    h_low = int(len(x0) / 2)
    h_high = int(h_low * (1. + gamma))
    t_low = int(s)
    t_high = int(s * (1. + gamma))

    (n, p) = x_mat.shape
    # if block size is larger than n,
    # just treat it as a single block (batch)
    b = n if n < b else b
    num_blocks = int(n) / int(b)

    num_epochs = 0

    loss_list = []

    for epoch_i in range(max_epochs):
        num_epochs += 1
        block = get_block(b, num_blocks)
        outer_grad = calc_grad(x_mat, y_tr, x_hat, block)
        x_nil = np.copy(x_hat)
        for _ in range(b / mini_block_size):  # option 2 from [1].
            mini_block = get_block(mini_block_size, int(n / mini_block_size))
            inner_grad_1 = calc_grad(x_mat, y_tr, x_nil, mini_block)
            if epoch_i < 1:
                gradient = inner_grad_1
            else:
                inner_grad_2 = calc_grad(x_mat, y_tr, x_hat, mini_block)
                gradient = inner_grad_1 - inner_grad_2 + outer_grad
            head_nodes, proj_grad = algo_head_tail_bisearch(
                edges, gradient, costs, g, root, h_low, h_high,
                proj_max_num_iter, verbose)
            bt = x_nil - lr * proj_grad
            tail_nodes, proj_bt = algo_head_tail_bisearch(
                edges, bt, costs, g, root,
                t_low, t_high, proj_max_num_iter, verbose)
            x_nil = proj_bt
        x_hat = x_nil
        x_hat_norm = np.linalg.norm(x_hat)
        residual_norm = np.linalg.norm(y_tr - np.dot(x_mat, x_hat))
        x_err = np.linalg.norm(x_hat - x_star)
        print("Epoch: %d, Residual norm: %.6f, x_hat norm: %.6f, x_err: %.6f" % (
            epoch_i + 1, residual_norm, x_hat_norm, x_err))
        if x_hat_norm >= 1e5:  # diverge cases.
            break
        if residual_norm <= tol_algo:
            break
        loss_list.append((epoch_i, residual_norm))
    x_err = np.linalg.norm(x_hat - x_star)
    run_time = time.time() - start_time
    return loss_list, x_err, num_epochs, run_time


def calc_grad(x_mat, y_tr, x_hat, block):
    """ Calculate the gradient w.r.t. the block of data, at x_hat.
    :param x_mat:   the design matrix.
    :param y_tr:    the array of measurements.
    :param x_hat:   the current estimation.
    :param block:   the block as range.
    :return:        the gradient.
    """
    x_tr_t = np.transpose(x_mat)
    xtx = np.dot(x_tr_t[:, block], x_mat[block])
    xty = np.dot(x_tr_t[:, block], y_tr[block])
    return -2. * (xty - np.dot(xtx, x_hat))


def get_block(blk_size, num_blks):
    """ Get a range object corresponding to the block of data to use for a gradient.
    :param blk_size: size of the block
    :param num_blks: number of blocks
    :return: block as range
    """
    block_idx = np.random.randint(0, num_blks)
    return range(blk_size * block_idx, blk_size * (block_idx + 1))


def print_helper(method, trial_i, s, n, num_epochs, err, run_time):
    print('%15s trial_%03d s: %02d n: %03d epochs: %03d '
          'rec_error: %.4e run_time: %.4e' %
          (method, trial_i, s, n, num_epochs, err, run_time))


def display_results(results):
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from pylab import rcParams

    for method in methods:
        name = methods[method]
        x = [data[0] for data in results[name]]
        y = [data[1] for data in results[name]]
        plt.plot(x, y, linestyle='-', marker='.', label=name)
        plt.legend()
    dim, s, eta = results['params']

    plt.title("Dimension: %d, Sparsity: %d, Learn Rate: %.2e" % (dim, s, eta))
    plt.xlabel('Epoch Number')
    plt.ylabel('Residual Norm (Loss)')
    test_name = 'tune_params_s=%d_eta=%.1e' % (s, eta) + '.png'
    # plt.savefig('results/' + test_name, dpi=600, bbox_inches='tight', pad_inches=0,
    #             format='png')
    plt.show()



methods = {
    algo_graph_svrg_iht: 'GraphSVRG-IHT',
    algo_graph_scsg_iht: 'GraphSCSG-IHT',
}


def main():
    s_list = [256, 128, 64, 32]
    lr_list = [1e-0, 1e-1, 1e-2, 1e-3]

    for (sparsity, learn_rate) in product(s_list, lr_list):
        run_test(sparsity, learn_rate)


def run_test(sparsity=256, learn_rate=1e-3):
    np.random.seed()
    # Params:
    height, width = 16, 16
    dimension = height * width
    algo_tolerance = 1e-7
    max_epochs = 250
    block_size = sparsity  # to be tuned
    print('Starting test...')
    print('Grid graph: %d x %d' % (height, width))
    print('Sparsity: %d' % sparsity)
    print('Max epochs: %d' % max_epochs)
    print('Learning rate: %.2e' % learn_rate)
    print('Block size: %d' % block_size)
    edges, costs = simu_grid_graph(width, height)
    init_node = (height / 2) * width + width / 2
    sub_graph = random_walk(edges, sparsity, init_node, 0)
    x_star = np.zeros(dimension)
    x_star[sub_graph[0]] = np.random.normal(loc=0.0, scale=1.0, size=sparsity)
    x_mat, y_tr, _ = sensing_matrix(dimension, x_star, 0.0)
    x_0 = np.zeros(dimension)
    g = sparsity
    results = {'params': (dimension, sparsity, learn_rate)}
    for method in methods:
        name = methods[method]
        print('Running %s...' % name)
        loss_list, err, num_epochs, run_time = method(
            x_mat, y_tr, max_epochs, learn_rate, x_star, x_0,
            algo_tolerance, edges, costs, sparsity, block_size, g)
        print_helper(name, 0, sparsity, dimension, num_epochs, err, run_time)
        results[name] = loss_list
    display_results(results)


if __name__ == '__main__':
    main()
