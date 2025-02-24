import scipy.stats

class SignalDetection:
    def __init__(self, hits, misses, falseAlarms, correctRejections):
        self.hits = hits
        self.misses = misses
        self.falseAlarms = falseAlarms
        self.correctRejections = correctRejections

    def hit_rate(self):
        """Calculate hit rate"""
        total_signal_trials = self.hits + self.misses
        if total_signal_trials == 0:
            return 0.0  # No signal trials, so return 0
        return (self.hits) / (total_signal_trials)

    def false_alarm_rate(self):
        """Calculate false alarm rate"""
        total_noise_trials = self.falseAlarms + self.correctRejections
        print(f"False Alarm Trials: {self.falseAlarms}, Correct Rejections: {self.correctRejections}")
        print(f"Total Noise Trials: {total_noise_trials}")
        if total_noise_trials == 0:
            return 0.0  # Treat no noise trials as a zero false alarm rate to avoid division by zero
        return (self.falseAlarms) / (total_noise_trials)

    def d_prime(self):
        """Compute d' using Z(hit rate) - Z(false alarm rate)."""
        z_hit = scipy.stats.norm.ppf(self.hit_rate())
        z_false_alarm = scipy.stats.norm.ppf(self.false_alarm_rate())
	# the scipy functions above used to calculate the z-score s for hit rate and false alarm were found using chatGBT
        return z_hit - z_false_alarm

    def criterion(self):
        """Compute criterion using -0.5 * (Z(hit rate) + Z(false alarm rate))."""
        z_hit = scipy.stats.norm.ppf(self.hit_rate())
        z_false_alarm = scipy.stats.norm.ppf(self.false_alarm_rate())
        # the scipy functions above used to calculate the z-score s for hit rate and false alarm were found using chatGBT
        return -0.5 * (z_hit + z_false_alarm)
