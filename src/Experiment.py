import numpy as np
import matplotlib.pyplot as plt
from SignalDetection import SignalDetection

class Experiment:
    def __init__(self):
        """Initializes an empty list to store SDT objects and their corresponding condition labels."""
        self.conditions = []

    def add_condition(self, sdt_obj: SignalDetection, label: str = None) -> None:
        """Adds a SignalDetection object and an optional label to the experiment."""
        self.conditions.append((sdt_obj, label))

    def sorted_roc_points(self) -> tuple[list[float], list[float]]:
        """Returns sorted false alarm rates and hit rates for plotting the ROC curve."""
        if len(self.conditions) == 0:
            raise ValueError("No conditions in experiment.")
        
        false_alarm_rates = []
        hit_rates = []
        
        for condition, label in self.conditions:
            false_alarm_rate = condition.false_alarm_rate()
            hit_rate = condition.hit_rate()
            false_alarm_rates.append(false_alarm_rate)
            hit_rates.append(hit_rate)

        # Debugging output: print the ROC points before sorting
        # print("Unsorted ROC points:")
        # for f, h in zip(false_alarm_rates, hit_rates):
        #     print(f"False Alarm Rate: {f}, Hit Rate: {h}")

         # Sort by false alarm rate
        sorted_points = sorted(zip(false_alarm_rates, hit_rates))
        false_alarm_rates, hit_rates = zip(*sorted_points)

        # Debugging output: print the sorted ROC points
        # print("Sorted ROC points:")
        # for f, h in zip(false_alarm_rates, hit_rates):
        #     print(f"False Alarm Rate: {f}, Hit Rate: {h}")

        return list(false_alarm_rates), list(hit_rates)

    def compute_auc(self) -> float:
        """Computes the Area Under the Curve (AUC) for the stored SDT conditions."""
        if not self.conditions:
            raise ValueError("No conditions in experiment.")
        
        false_alarm_rates, hit_rates = self.sorted_roc_points()

        if len(false_alarm_rates) < 2:
            raise ValueError("Not enough points to compute AUC.")
        
        # Debugging output: print the ROC points before calculating AUC
        print(f"False Alarm Rates: {false_alarm_rates}")
        print(f"Hit Rates: {hit_rates}")

        # Compute AUC using the trapezoidal rule
        auc = np.trapz(hit_rates, false_alarm_rates)
        print(f"AUC: {auc}")  # Print the AUC value for debugging
        return auc

    def plot_roc_curve(self, show_plot: bool = True):
        """Plots the ROC curve."""
        false_alarm_rates, hit_rates = self.sorted_roc_points()
        plt.plot(false_alarm_rates, hit_rates, marker='o')
        plt.xlabel('False Alarm Rate')
        plt.ylabel('Hit Rate')
        plt.title('ROC Curve')
        if show_plot:
            plt.show()
