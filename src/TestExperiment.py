import unittest
import numpy as np
import matplotlib.pyplot as plt
from signal_detection import SignalDetection
from Experiment import Experiment

class TestExperiment(unittest.TestCase):
    def test_compute_auc()(self):
	"tests if function produces expected AUC values for known test cases, such as AUC = 0.5 if there are only two experiments and they fall at (0,0) and (1,1). AUC = 1 if there are three experiments and they fall at (0,0), (0,1), and (1,1)."
	# write test cases here
    def test_sorted_roc_points(self):
        "tests if function correctly returns false alarm rate and hit rate sorted by false alarm rate."
	# write test cases here
    def test_add_condition(self):
        "tests if function correctly stores SignalDetection objects and labels"
	# write test cases here

if __name__ == '__main__':
    unittest.main()
