import numpy as np
import collections
from scipy.stats import mode # Just for computation of statisticmodes
import sys
"A tree structure. Each internal node is a dict. Each leaf is value."

def compute_statistics(labels, predicted, expected):
    pos = np.equal(predicted, expected)

    num_labels = len(labels)
    label_ids = {}
    for i,label in enumerate(labels):
        label_ids[label] = i

    M = np.zeros((num_labels, num_labels))
    for i in range(predicted.shape[1]):
        lbl_e, lbl_p = expected[0, i], predicted[0, i]
        idx_e, idx_p = label_ids[lbl_e], label_ids[lbl_p]
        M[idx_e][idx_p] += 1

    M1 = np.sum(M, axis = 0)
    M2 = np.sum(M, axis = 1)
    print "Labels are: ", labels
    print "Accuracy = ", float(np.count_nonzero(pos))/predicted.shape[1]
    print "Precision = ", map(lambda x: float(M[x,x])/M1[x], label_ids)
    print "Recall = ", map(lambda x: float(M[x,x])/M2[x], label_ids)
    return
