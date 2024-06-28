import csv

class PerformanceCSVRecorder:
    def __init__(self, filename):
        self.filename = filename
        self.csv_file = open(filename, 'w')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['elapsed_time', 'fps', 'mean_elapsed_time', 'mean_fps'])

    def record(self, elapsed_time, fps, mean_elapsed_time, mean_fps):
        self.csv_writer.writerow([elapsed_time, fps, mean_elapsed_time, mean_fps])

    def close(self):
        self.csv_file.close()