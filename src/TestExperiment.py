import unittest
from SignalDetection import SignalDetection
from Experiment import Experiment

class TestExperiment(unittest.TestCase):

    def setUp(self):
        """Sets up test cases with SignalDetection objects."""
        self.sdt1 = SignalDetection(40, 10, 20, 30)
        self.sdt2 = SignalDetection(60, 20, 10, 40)
        self.sdt3 = SignalDetection(50, 15, 25, 35)
        self.exp = Experiment()

    def test_add_condition(self):
        """Test that conditions are added correctly."""
        self.exp.add_condition(self.sdt1, label="Condition A")
        self.exp.add_condition(self.sdt2, label="Condition B")
        self.assertEqual(len(self.exp.conditions), 2)
        
        # Check that the conditions are instances of SignalDetection
        for condition, label in self.exp.conditions:
            self.assertIsInstance(condition, SignalDetection)

    def test_sorted_roc_points(self):
        """Test that the ROC points are sorted by false alarm rate."""
        self.exp.add_condition(self.sdt1, label="Condition A")
        self.exp.add_condition(self.sdt2, label="Condition B")
        false_alarm_rate, hit_rate = self.exp.sorted_roc_points()

        # Check if sorted by false alarm rate
        self.assertTrue(all(false_alarm_rate[i] <= false_alarm_rate[i+1] for i in range(len(false_alarm_rate)-1)))

    def test_sorted_roc_points_no_condition(self):
        """Test to see that a ValueError is raised when no conditions are present."""
        with self.assertRaises(ValueError):
            false_alarm_rate, hit_rate = self.exp.sorted_roc_points()

    def test_compute_auc(self):
        """Test that AUC is computed correctly."""
        self.exp.add_condition(self.sdt1, label="Condition A")
        self.exp.add_condition(self.sdt2, label="Condition B")
        auc = self.exp.compute_auc()
        # Check if AUC is a valid number and not zero
        self.assertTrue(auc > 0)
    
    def test_compute_auc_no_condition(self):
        """Test to see that a ValueError is raised when no conditions are present."""
        with self.assertRaises(ValueError):
            auc = self.exp.compute_auc()


    def test_empty_experiment(self):
        """Test for empty experiment conditions."""
        empty_exp = Experiment()
        with self.assertRaises(ValueError):
            empty_exp.compute_auc()

        with self.assertRaises(ValueError):
            empty_exp.sorted_roc_points()

    def test_auc_boundary_conditions(self):
        """Test specific AUC values for known cases."""
        # Case with (0,0) and (1,1) should give AUC = 0.0
        sdt_zero = SignalDetection(0, 0, 0, 0)
        sdt_full = SignalDetection(100, 0, 0, 0)
        exp = Experiment()
        exp.add_condition(sdt_zero)
        exp.add_condition(sdt_full)
        auc = exp.compute_auc()
        self.assertAlmostEqual(auc, 0.0)

        # Case with three points (0,0), (0,1), (1,1) should give AUC = 1
        sdt_zero = SignalDetection(0, 0, 0, 0)
        sdt_half = SignalDetection(100, 0, 0, 0)
        sdt_full = SignalDetection(100, 0, 100, 0)
        exp = Experiment()
        exp.add_condition(sdt_zero)
        exp.add_condition(sdt_half)
        exp.add_condition(sdt_full)
        auc = exp.compute_auc()
        self.assertAlmostEqual(auc, 1.0)

if __name__ == "__main__":
    unittest.main()
