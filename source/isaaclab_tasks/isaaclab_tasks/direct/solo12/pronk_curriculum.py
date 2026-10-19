"""Episode-based acquisition gate, independent of Isaac/Kit."""

from collections import deque
import math


class PronkAcquisitionGate:
    """Advance only after recent complete episodes retain motion and survive.

    Each record is (survived_full_episode, xy_mse, yaw_mse, confirmed_airtime,
    confirmed_cadence). Domain-randomized and nominal cohorts must pass
    separately. Old-level episodes cannot advance a new level.
    """

    def __init__(self, *, window=256, randomized_window=64, survival=0.95,
                 xy_rmse=0.12, yaw_rmse=0.15, airtime=0.08, cadence=0.75):
        if window < 1 or randomized_window < 1:
            raise ValueError("Curriculum windows must be positive.")
        if not 0 < survival <= 1 or not 0 < airtime < 1:
            raise ValueError("Invalid acquisition survival/airtime gates.")
        if not all(math.isfinite(v) and v > 0 for v in (xy_rmse, yaw_rmse, cadence)):
            raise ValueError("Tracking and cadence gates must be positive and finite.")
        self.window = window
        self.randomized_window = randomized_window
        self.thresholds = (survival, xy_rmse**2, yaw_rmse**2, airtime, cadence)
        self.level = 0
        self.records = (deque(maxlen=window), deque(maxlen=randomized_window))

    @property
    def steering_fraction(self):
        return min(self.level / 10, 1.0)

    @property
    def robustness_fraction(self):
        return max((self.level - 10) / 10, 0.0)

    def add(self, records, levels, randomized):
        for record, level, is_randomized in zip(records, levels, randomized):
            if level != self.level:
                continue
            if len(record) != 5 or not all(math.isfinite(float(x)) for x in record):
                raise ValueError("Curriculum records must contain five finite metrics.")
            self.records[int(is_randomized)].append(tuple(float(x) for x in record))

    def statistics(self, cohort):
        values = self.records[cohort]
        if not values:
            return None
        return tuple(sum(row[i] for row in values)/len(values) for i in range(5))

    def _passes(self, cohort, minimum):
        if len(self.records[cohort]) < minimum:
            return False
        s, xy, yaw, air, cadence = self.statistics(cohort)
        ts, tx, ty, ta, tc = self.thresholds
        return s >= ts and xy <= tx and yaw <= ty and air >= ta and cadence >= tc

    def advance(self):
        if self.level >= 20:
            return False
        nominal_required = self.robustness_fraction < 1
        randomized_required = self.robustness_fraction > 0
        if nominal_required and not self._passes(0, self.window):
            return False
        if randomized_required and not self._passes(1, self.randomized_window):
            return False
        self.level += 1
        for values in self.records:
            values.clear()
        return True
