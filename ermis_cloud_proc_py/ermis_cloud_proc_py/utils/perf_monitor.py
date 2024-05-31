class PerformanceMonitorErmis():
    def __init__(self):
        self.num_measurements = 0
        self.current_mean = 0

    def update(self, new_measurement):
        self.num_measurements += 1
        self.current_mean = (self.current_mean * (self.num_measurements - 1) + new_measurement) / self.num_measurements

    def get_mean(self):
        return self.current_mean