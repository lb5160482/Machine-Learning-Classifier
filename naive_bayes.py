import numpy as np
from math import log
import load_data
import copy
from compute_statistics import compute_statistics
# trainData = load_data.load_iris(0.6)[0]
# testData = load_data.load_iris(0.6)[1]

def test(params, testData):
    likelihood = params["likelihood"]
    numTest = len(testData)
    labels = getLabels(testData)
    expected = []
    actual = []
    numCorrect = 0.0
    for testVec in testData:
        expected.append(predict(params, testVec))
        actual.append(int(testVec[0]))
    expected = np.array(expected).reshape((1, -1))
    actual = np.array(actual).reshape((1, -1))
    compute_statistics(labels, expected, actual)
    # print labels
    # print expected
    # print actual
    # return labels,expected,actual
    return


def predict(params, testVec):
    likelihood = params["likelihood"]
    numFeat = len(testVec)
    numTrain = likelihood[-1]
    priorProb = likelihood[-2]
    pMax = -10000.0
    for label in range(len(likelihood)-2):
        plh = log(1)#likelihood probability initialization
        for feat in range(1,numFeat):
            if testVec[feat] in likelihood[label][feat].keys():# if trainning data has this feature value,
                plh += log(likelihood[label][feat][testVec[feat]]+1) - log(priorProb[label]*numTrain+1)
            else:# probability 0/numInClass ==> log(1) - log (numInClass)
                plh += log(1) - log(priorProb[label]*numTrain+1)
            p = plh + log(priorProb[label])# likelihood * prior probability
        if p > pMax:
            pMax = p
            labelIndex = label
    return labelIndex

def priorProbility(data):
    """get the prior probability of each class e.g. [0.1,0.4,0.5]"""
    numTrain = len(data)
    label = []
    for sample in data:
        label.append(sample[0])
    label = list(set(label))#get unique label list
    numEachClass = np.zeros(len(label))
    for sample in data:
        numEachClass[sample[0]] += 1
    pClass = numEachClass / numTrain#p(ci)
    return pClass

def train_naive_bayes(params, trainData):
    numTrain = len(trainData)
    numFeat = len(trainData[0])
    labels = getLabels(trainData)

    likelihood = []
    featvalue_inital = []
    """likelihood initialization, e.g. likelihood = [[],[],[]], each sub list represents a class"""
    for i in range(len(labels)):
        likelihood.append([])
    featvalue_inital = copy.copy(likelihood)
    numEachLabel = getNumEachLabel(trainData)#number of each class
    """This will get: e.g. [[0, {}, {}, {}, {}], [0, {}, {}, {}, {}], [0, {}, {}, {}, {}]], each {} is 
    a feature, 0 used to unify index since index 0 means class"""
    for i in range(len(likelihood)):
        likelihood[i] = [0]
        for j in range(1,numFeat):
            likelihood[i].append({})

    """This will get: e.g. for each feature: {0:0,1:0,2:0}, initialize the number of different feature values as 0"""
    for feat in range(1,numFeat):
        featVal = copy.deepcopy(featvalue_inital)
        for sample in trainData:
            label = int(sample[0])
            if sample[feat] not in featVal[label]:
                likelihood[label][feat][sample[feat]] = 0
            featVal[label].append(sample[feat])
    """train: num of each feature value e.g. {0:43,1:43,2:23}"""
    for feat in range(1,numFeat):
        for sample in trainData:
            label = int(sample[0])
            likelihood[label][feat][sample[feat]] += 1
    likelihood.append(priorProbility(trainData))#likelihood[-2],prior probability
    likelihood.append(len(trainData))#lkelihood[-1],number of train datas
    params["likelihood"] = likelihood
    # print likelihood[-1]
    # return likelihood

def getNumEachLabel(data):
    """get  numbers of each class e.g.[12,23,43]"""
    labels = getLabels(data)
    numLabel = np.zeros(len(labels))
    for sample in data:
        numLabel[sample[0]] += 1
    return numLabel

def getLabels(data):
    """get list of labels e.g. [0,1,2]"""
    labels = []
    for sample in data:
        labels.append(int(sample[0]))
    labels = list(set(labels))
    return labels

# likelihood = train(trainData)
# predict(trainData,trainData[90],likelihood)
# print priorProbility(trainData)
# labels,expected,actual = test(testData,likelihood)
# compute_statistics(labels, expected, actual)
# print getLabels(trainData)