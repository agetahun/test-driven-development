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
        if not self.conditions:
            raise ValueError("No conditions in experiment.")
        
        false_alarm_rates = []
        hit_rates = []
        
        for sdt, label in self.conditions:
            false_alarm_rates.append(sdt.false_alarm_rate())
            hit_rates.append(sdt.hit_rate())

        # Sort by false alarm rate
        sorted_indices = np.argsort(false_alarm_rates)
        false_alarm_rates = np.array(false_alarm_rates)[sorted_indices]
        hit_rates = np.array(hit_rates)[sorted_indices]

        return list(false_alarm_rates), list(hit_rates)

    def compute_auc(self) -> float:
        """Computes the Area Under the Curve (AUC) for the stored SDT conditions."""
        if not self.conditions:
            raise ValueError("No conditions in experiment.")
        
        false_alarm_rates, hit_rates = self.sorted_roc_points()

        # Compute AUC using the trapezoidal rule
        auc = np.trapz(hit_rates, false_alarm_rates)
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
