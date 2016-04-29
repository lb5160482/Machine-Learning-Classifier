import numpy as np
import random
import copy
from compute_statistics import compute_statistics

def g(x):
    "Returns activation function."
    t = np.exp(-x)
    return 1.0/(1.0+t)

def dg(x):
    "Return the derivative of g, the activation function."
    f = g(x)
    return f*(1.0-f)

def threshold(v):
    return 1.0 if v > 0.5 else 0.0

def init_weight(network):
    "Initializes the weights for neural network."
    weights = network["weights"]
    for l in range(network["num_layers"]-1):
        N = network["num_nodes"][l]
        if network["option"] == "shallow":
            range_random = 0.01
        else:
            # range_random = 1.0/np.sqrt(N)
            range_random = np.sqrt(6)/np.sqrt(network["num_nodes"][l] + network["num_nodes"][l+1])
        for i in range(network["num_nodes"][l]):
            for j in range(network["num_nodes"][l+1]):
                weights[l][i, j] = random.uniform(-range_random, range_random)
    return

def compute_error(network, ex):
    "Computes error for a network."
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
    "Backward propagation to train neural network model."
    delta = []
    for i in range(network["num_layers"]):
        delta.append([0.0] * network["num_nodes"][i])

    L = network["num_layers"]
    N = network["num_labels"]
    num_nodes = network["num_nodes"]
    eps = 1e-6
    count_iter = 0
    weights = network["weights"]
    last_error = 1e20
    rho = 0.9
    alpha = 0.1
    while True:
        # Adaptively change rho the decay factor
        rho = 0.9 * float(2000 - count_iter)/2000

        for ex in examples:
            "This loop iteratively updates network weights."

            "First do a forward propagation."
            _, _, a, in_value = forward_propagation(network, ex)

            "Second compute errors(deltas) for output layer."
            expected = [0.0] * N
            expected[np.int64(ex[0])] = 1.0

            for j in range(N):
                delta[L-1][j] = dg(in_value[L-1][j]) * (expected[j] - a[L-1][j])

            "Compute errors backwards."
            for l in reversed(range(L-1)):
                for i in range(num_nodes[l]):
                    if i == num_nodes[l] and l > 0:
                        delta[l][i] = 0
                    else:
                        s = 0
                        for j in range(num_nodes[l+1]):
                            s += weights[l][i, j] * delta[l+1][j]
                        delta[l][i] = dg(in_value[l][i]) * s
            "Finally use the errors to update weights."
            for l in range(L-1):
                for i in range(num_nodes[l]):
                    for j in range(num_nodes[l+1]):
                        "Compute momentum, rho is decay factor."
                        momentum = 0.0
                        if network["prev_prev_weights"] is not None:
                            momentum = rho * (network["prev_weights"][l][i, j] - network["prev_prev_weights"][l][i, j])

                        weights[l][i, j] += alpha * a[l][i] * delta[l+1][j] + (1 - alpha) * momentum

            network["prev_prev_weights"] = copy.deepcopy(network["prev_weights"])
            network["prev_weights"] = copy.deepcopy(network["weights"])
            # print network["prev_prev_weights"]
            # print network["prev_weights"]

        "After each iteration, evaluate the errors and stop on convergence."
        sum_error = 0
        for ex in examples:
            sum_error += compute_error(network, ex)
        if abs(last_error - sum_error)/examples.shape[0] < eps:
            break
        last_error = sum_error

        "Automated stops after a number of iterations."
        count_iter += 1
        if count_iter > 1000:
            break
    return

def train_neural_net(params, training_data):
    "Initialize network layers, number of nodes and weights."
    params["network"] = {}
    network = params["network"]

    if "option" in params:
        network["option"] = params["option"]
    else:
        network["option"] = "shallow"

    if network["option"] == "shallow":
        network["num_layers"] = 3
    else:
        network["num_layers"] = 5

    network["num_nodes"] = [0] * network["num_layers"]
    network["num_nodes"][0] = training_data.shape[1]
    for i in range(1, network["num_layers"] -1):
        network["num_nodes"][i] = 2 * network["num_nodes"][0] +1

    network["num_labels"] = len(np.unique(training_data[:, 0]))
    network["num_nodes"][-1] = network["num_labels"]
    network["weights"] = []
    network["prev_weights"] = None
    network["prev_prev_weights"] = None
    for i in range(network["num_layers"] -1):
        shape = (network["num_nodes"][i], network["num_nodes"][i+1])
        network["weights"].append(np.zeros(shape))
    init_weight(network)

    "Call back_prop_learning to train the neural network."
    back_prop_learning(training_data, network)
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

    a[0][num_nodes[0]-1] = 1.0
    for l in range(1, L):
        for j in range(num_nodes[l]):
            "Setting dummy values to 1. Note that the output layer has no dummy values."
            if j == num_nodes[l]-1 and l != L-1:
                a[l][j] = 1.0
            else:
                in_value[l][j] = 0.0
                for i in range(num_nodes[l-1]):
                    in_value[l][j] += weights[l-1][i, j] * a[l-1][i]
                a[l][j] = threshold(g(in_value[l][j]))

    return [np.argmax(a[-1]), a[-1], a, in_value]

def predict(params, data):
    return map(lambda x: forward_propagation(params["network"], x)[0], data)


def test(params, test_data):
    labels = np.unique(test_data[:, 0])
    predicted = np.array(predict(params, test_data)).reshape((1, -1))
    expected =  test_data[:, 0].reshape((1, -1))
    compute_statistics(labels, predicted, expected)
    # print params
    return