class Discretization:
    def __init__(self, ranges):
        # Ensure the ranges are sorted
        self.ranges = sorted(ranges)

    def discretize(self, value):
        # Discretize the value into bins based on the ranges
        for i, range_val in enumerate(self.ranges):
            if value <= range_val:
                return i
        return len(self.ranges)

    def get_range(self, value):
        # Translate a value to its corresponding range as a tuple
        if value <= self.ranges[0]:
            return None, self.ranges[0]
        for i in range(len(self.ranges) - 1):
            if self.ranges[i] < value <= self.ranges[i + 1]:
                return self.ranges[i], self.ranges[i + 1]
        return self.ranges[-1], None
