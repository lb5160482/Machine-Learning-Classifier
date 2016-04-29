import numpy as np
import collections
import random
import sys
import copy

def g(x):
    "Returns activation function."
    t = np.exp(-x)
    return 1.0/(1.0+t)

def dg(x):
    "Return the derivative of g."
    f = g(x)
    return f*(1.0-f)

def alpha():
    "Return learning rate."
    return 0.1

def threshold(v):
    return 1.0 if v > 0.5 else 0.0

def init_weight(network):
    weights = network["weights"]
    for l in range(network["num_layers"]-1):
        N = network["num_nodes"][l]
        n = 1.0/np.sqrt(N)
        for i in range(network["num_nodes"][l]):
            for j in range(network["num_nodes"][l+1]):
                # weights[l][i, j] = random.uniform(-0.01, 0.01)
                weights[l][i, j] = random.uniform(-n, n)
    return

def compute_error(network, ex):
    _, _, _, in_value = forward_propagation(network, ex)
    in_value = in_value[-1]
    N = network["num_labels"]

    expected = [0.0] * N
    expected[np.int64(ex[0])] = 1.0
    s = 0
    for j in range(N):
        s += np.square(expected[j]-g(in_value[j]))
    return s

def back_prop_learning(examples, network):
    "Initializing weights."

    # a = []
    # for i in range(network["num_layers"]):
    #     a.append([0] * network["num_nodes"][i])
    # in_value = []
    # for i in range(network["num_layers"]):
    #     in_value.append([0] * network["num_nodes"][i])
    #
    delta = []
    for i in range(network["num_layers"]):
        delta.append([0.0] * network["num_nodes"][i])

    L = network["num_layers"]
    N = network["num_labels"]
    num_nodes = network["num_nodes"]
    eps = 1e-10

    count_iter = 0
    weights = network["weights"]
    last_error = 1e20
    while True:
        for ex in examples:

            _, _, a, in_value = forward_propagation(network, ex)

            expected = [0.0] * N
            expected[np.int64(ex[0])] = 1.0

            for j in range(N):
                delta[L-1][j] = dg(in_value[L-1][j]) * (expected[j] - a[L-1][j])

            for l in reversed(range(L-1)):
                for i in range(num_nodes[l]):
                    if i == num_nodes[l] and l > 0:
                        delta[l][i] = 0
                    else:
                        s = 0
                        for j in range(num_nodes[l+1]):
                            s += weights[l][i, j] * delta[l+1][j]
                        delta[l][i] = dg(in_value[l][i]) * s

            for l in range(L-1):
                for i in range(num_nodes[l]):
                    for j in range(num_nodes[l+1]):
                        weights[l][i, j] += alpha() * a[l][i] * delta[l+1][j]
            pass

        sum_error = 0
        for ex in examples:
            sum_error += compute_error(network, ex)
        # print count_iter, last_error, sum_error
        # if abs(last_error - sum_error) < eps:
        #     break
        last_error = sum_error

        count_iter += 1
        if count_iter > 1000:
            break
    return

def train_neural_net(params, training_data):
    "Initialize network."

    params["network"] = {}
    network = params["network"]

    network["num_layers"] = 4
    network["num_nodes"] = [0] * network["num_layers"]
    network["num_nodes"][0] = training_data.shape[1]
    network["num_nodes"][1] = 2 * network["num_nodes"][0] +1
    network["num_nodes"][2] = 2 * network["num_nodes"][0] +1
    # network["num_nodes"][3] = 2 * network["num_nodes"][0] +1
    network["num_labels"] = len(np.unique(training_data[:, 0]))
    network["num_nodes"][-1] = network["num_labels"]
    network["weights"] = []
    for i in range(network["num_layers"] -1):
        shape = (network["num_nodes"][i], network["num_nodes"][i+1])
        network["weights"].append(np.zeros(shape))
    init_weight(network)

    back_prop_learning(training_data, network)
    print network["weights"]
    # params["network"] = network
    print "training = ", training_data[:, 0].reshape((1, -1))[0]
    return

def forward_propagation(network, ex):
    """ Do a forward propagation to one example and return outputs.
        Input: params: training params
        ex: a single example
        Returns: the label and the predicted values for labels
    """

    weights = network["weights"]

    a = []
    for i in range(network["num_layers"]):
        a.append([0] * network["num_nodes"][i])
    in_value = []
    for i in range(network["num_layers"]):
        in_value.append([0] * network["num_nodes"][i])

    L = network["num_layers"]
    N = network["num_labels"]
    num_nodes = network["num_nodes"]

    for i, x in enumerate(ex[1:]):
        a[0][i] = x
    "Set dummy value."
    a[0][num_nodes[0]-1] = 1.0
    for l in range(1, L):
        for j in range(num_nodes[l]):
            "Setting dummy values. Note that the output layer has no dummy values."
            if j == num_nodes[l]-1 and l != L-1:
                a[l][j] = 1.0
            else:
                in_value[l][j] = 0.0
                for i in range(num_nodes[l-1]):
                    in_value[l][j] += weights[l-1][i, j] * a[l-1][i]
                a[l][j] = threshold(g(in_value[l][j]))

    return [np.argmax(a[-1]), a[-1], a, in_value]

def predict(params, data):
    print data[0]
    print forward_propagation(params["network"], data[0])
    print data[1]
    print forward_propagation(params["network"], data[1])
    print data[2]
    print forward_propagation(params["network"], data[2])
    return map(lambda x: forward_propagation(params["network"], x)[0], data)

def test(params, test_data):
    predicted = predict(params, test_data)
    expected =  test_data[:, 0].reshape((1, -1))[0]
    print predicted
    print expected
    pos = np.equal(predicted, expected)
    print "Accuracy = ", float(np.count_nonzero(pos))/len(predicted)
    return