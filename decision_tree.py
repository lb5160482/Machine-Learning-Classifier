import numpy as np
import collections
from scipy.stats import mode # Just for computation of statisticmodes
import sys
from compute_statistics import compute_statistics
"A tree structure. Each internal node is a dict. Each leaf is value."

value_attrs = []
IG_thresh = 0.05

def Tree():
    "This is the structure for a node of the tree."
    return collections.defaultdict(Tree)

def h(examples):
    "returns entropy"
    labels = examples[:, 0]
    value_labels = np.unique(labels)
    ent = 0

    for v in value_labels:
        # print np.argwhere(labels == v).flatten()
        prob = float(len(np.argwhere(labels == v).flatten()))/len(examples)
        ent += prob * np.log2(prob)
    ent = -ent
    return ent

def gain(examples, attribute):
    "Returns the gain function for one attribute."
    ent = h(examples)

    remainder = 0
    values = examples[:, attribute]
    for v in value_attrs[attribute]:
        exs_idx = np.argwhere(values == v).flatten()
        count_v = len(exs_idx)
        exs = examples[exs_idx]
        if count_v != 0:
            remainder += float(count_v)/len(examples) * h(exs)
    return ent - remainder

def gain_ratio(examples, attribute):
    "Returns the gain function for one attribute."
    ent = h(examples)

    remainder = 0
    values = examples[:, attribute]
    for v in value_attrs[attribute]:
        exs_idx = np.argwhere(values == v).flatten()
        count_v = len(exs_idx)
        exs = examples[exs_idx]
        if count_v != 0:
            remainder += float(count_v)/len(examples) * h(exs)
    ig = ent - remainder

    iv = 0
    for v in value_attrs[attribute]:
        exs_idx = np.argwhere(values == v).flatten()
        count_v = len(exs_idx)
        exs = examples[exs_idx]
        if count_v != 0:
            iv -= float(count_v)/len(examples) * np.log2(float(count_v)/len(examples))
    if iv > 1e-6:
        return ig/iv
    else:
        return ig/1e-6


# def pretty(d, indent=0):
#     "Downloaded from stackoverflow."
#     for key, value in d.iteritems():
#        print '\t' * indent + str(key)
#        if isinstance(value, dict):
#           pretty(value, indent+1)
#        else:
#           print '\t' * (indent+1) + str(value)

def plurality_value(examples):
    "Returns the most common value."
    return np.int64(mode(examples[:, 0])[0][0])

def data_to_exmpale(data):
    "Convert the continuous attributes to discrete."
    g = np.vectorize(lambda x: x if x is np.int64 else np.int64(x))
    return g(data)

def train_decision_tree(params, train_data):

    def recursive_tree_learning(examples, attributes, parent_examples):
        "Recursively returns a decision tree."
        "First, check if empty."
        if examples.shape[0] == 0:
            return plurality_value(parent_examples)
        else:
            "Check if same label."
            labels = examples[:, 0]
            first_label = examples[0][0]
            "same_label is a boolean for whether the labels are the same."
            same_label = (len(labels) == len(np.where(labels == first_label)))
            if same_label:
                return first_label
            else:
                "If there are no attributes, return plurality value."
                if len(attributes) == 0:
                    return plurality_value(examples)
                else:
                    "Compute IG/IGR."
                    A = 0
                    if params["option"] == "IG":
                        A = np.argmax(map(lambda x: gain(examples, x), attributes))
                    elif params["option"] == "IGR":
                        A = np.argmax(map(lambda x: gain_ratio(examples, x), attributes))

                    "If IG/IGR is too small then just return plurality and avoid overfitting."
                    if A < IG_thresh:
                        return plurality_value(examples)

                    "Else, return a decision subtree."
                    values = examples[:, attributes[A]]

                    tree = Tree()
                    att = np.delete(attributes, A)
                    for v in value_attrs[attributes[A]]:
                        idx = np.argwhere(values == v).flatten()
                        exs = examples[np.argwhere(values == v).flatten()]
                        tree[(attributes[A], v)] = recursive_tree_learning(exs, att, examples)
                    return tree

    "Preprocess data, i.e. convert continuous values to discrete ones."
    examples = data_to_exmpale(train_data)

    global value_attrs
    value_attrs = []
    columns = np.arange(examples.shape[1])
    for c in columns:
        value_attrs.append(np.unique(examples[:, c]))

    attributes = np.arange(1, examples.shape[1])
    tree = recursive_tree_learning(examples = examples, attributes = attributes, parent_examples = None)
    params["tree"] = tree
    return tree

def rec_predict(d, node):

    if node.__class__ != collections.defaultdict:
        return np.int64(node)
    else:
        keys = node.keys()
        if len(keys) == 0:
            return 0
        else:
            # print keys
            k = keys[0][0]
            v = d[k]
            return rec_predict(d, node[(k, v)])

def predict(params, data):
    examples = data_to_exmpale(data)
    tree = params["tree"]
    res = []
    for ex in examples:
        r = rec_predict(ex, tree)
        res.append(r)
    res = np.array(res).reshape((-1,1))
    return res


def test(params, test_data):
    examples = data_to_exmpale(test_data)
    predicted = predict(params, examples).reshape((1, -1))
    expected = examples[:, 0].reshape((1, -1))
    labels = value_attrs[0]

    compute_statistics(labels, predicted, expected)
    return
