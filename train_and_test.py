import numpy as np
import sys
import load_data
from classifier import Classifier

"""
This is the main python method that will be run.
You should determine what sort of command line arguments
you want to use. But in this module you will need to 
1) initialize your classifier and its params 
2) load training/test data 
3) train the algorithm
4) test it and output the desired statistics.
"""

def main(c = "decision_tree", option = "IG", dataset = "iris", ratio = 0.8):

	classifier_types = {0: "decision_tree", 1: "naive_bayes", 2: "neural_net"}
	options = {0:["IG", "IGR"], 1:["normal"], 2:["shallow", "medium"]}

	ratio = float(ratio)

	if dataset == "monks":
		(training, test) = load_data.load_monks(ratio)
	elif dataset == "congress":
		(training, test) = load_data.load_congress_data(ratio)
	elif dataset == "iris":
		(training, test) = load_data.load_iris(ratio)
	else:
		print "Error: Cannot find dataset name."
		return

	print "Training... Please hold."
	# classifier_types = {0: "decision_tree", 2: "neural_net"}
	# options = {0:["IG", "IGR"], 2:["shallow", "medium"]}
	# (training, test) = load_data.load_iris(0.8)
	# nn_classifier = Classifier(classifier_type="neural_net", option = "medium")
	# nn_classifier.train(training)
	# nn_classifier.test(test)

	# print test
	# (training, test) = load_data.load_congress_data(0.8)
	# print test
	# (training, test) = load_data.load_monks(1)
	# print test	

	# (training, test) = load_data.load_iris(0.8)
	# print training
	# "option = IG/IGR"
	# dt_classifier = Classifier(classifier_type="decision_tree", weights=[], option="IG")
	# dt_classifier.train(training)
	# dt_classifier.test(test)
	# for i, c in classifier_types.iteritems():
	# 	for option in options[i]:
	print "                                                                 "
	print "================================================================="
	print "Dataset    = ", dataset
	print "Classifier = ", c
	print "Option     = ", option
	classifier = Classifier(classifier_type=c, weights = [], option = option)
	classifier.train(training)
	classifier.test(test)
	print "================================================================="
	print "                                                                 "
	# option value could be either shallow(3 layers) or medium(5)
	# nn_classifier = Classifier(classifier_type="neural_net", option = "medium")
	# nn_classifier.train(training)
	# nn_classifier.test(test)
	return 


if __name__ == '__main__':
	if len(sys.argv) == 1:
		main()	
	elif len(sys.argv) == 5:
		main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])