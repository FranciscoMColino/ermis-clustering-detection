import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import open3d as o3d
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import struct
import time
import multiprocessing

import mlpack

class PCPerformanceMonitor():
    def __init__(self):
        self.num_measurements = 0
        self.current_mean = 0

    def update(self, new_measurement):
        self.num_measurements += 1
        self.current_mean = (self.current_mean * (self.num_measurements - 1) + new_measurement) / self.num_measurements

    def get_mean(self):
        return self.current_mean

# TODO: Move this to a separate file

def organize_clusters(points, labels):
    clusters_points = np.zeros(len(np.unique(labels)), dtype=object)
    for label in np.unique(labels):
        #if label == -1:
        #    continue
        cluster_points = np.asarray(points)[labels == label]
        clusters_points[int(label)] = cluster_points
    return clusters_points

def build_pointcloud_clusters(clusters_points, label_colors):
    pcd_list = np.zeros(len(clusters_points), dtype=object)
    for i in range(len(clusters_points)):
        pcd_cluster = o3d.geometry.PointCloud()
        pcd_cluster.points = o3d.utility.Vector3dVector(clusters_points[i])
        pcd_cluster.paint_uniform_color(label_colors[i])
        pcd_list[i] = pcd_cluster
    #pcd_list[-1].paint_uniform_color([1, 0, 0])  # Outliers are red

    return pcd_list


class KmDbscanClustering(Node):
    def __init__(self):
        super().__init__('kmeans_dbscan_ml_o3d')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/zed/zed_node/point_cloud/cloud_registered',
            self.pointcloud_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.vis = o3d.visualization.Visualizer()
        self.pcd = o3d.geometry.PointCloud()
        self.vis.create_window()
        self.first_run = True
        self.pc_performance_monitor = PCPerformanceMonitor()
        self.label_colors = np.random.rand(1000, 3)

        # Initialize z-range parameters for passthrough filter
        self.z_min = -0.05
        self.z_max = 2.0

        # kmeans
        self.kmeans_num_clusters = 4

        # multiprocessing
        self.input_queue = multiprocessing.Queue()
        self.output_queue = multiprocessing.Queue()
        self.processes = []
        self.num_processes = 4
        self.setup_multiprocessing()

    def worker_process(self, input_queue, output_queue):
        while True:
            task = input_queue.get()
            if task is None:
                break
            key, points, eps, min_size = task
            dbscan_labels = self.fit_dbscan_model(points, eps, min_size)
            dbscan_clusters = organize_clusters(points, dbscan_labels)
            output_queue.put((key, dbscan_clusters))

    def setup_multiprocessing(self):
        for i in range(self.num_processes):
            process = multiprocessing.Process(target=self.worker_process, args=(self.input_queue, self.output_queue))
            process.start()
            self.processes.append(process)

    def fit_dbscan_model(self, points, eps=0.35, min_size=5):
        d = mlpack.dbscan(input_=points, epsilon=eps, min_size=min_size)
        return d['assignments']

    def fit_kmeans_model(self, points, clusters=4):
        d = mlpack.kmeans(input_=points, clusters=clusters)
        # return only 4th column of the output
        labels = d['output'][:, 3]
        return labels

    def pointcloud_callback(self, msg):

        #self.vis.clear_geometries()

        start = time.time()

        # Convert ROS PointCloud2 message to numpy array
        pc2_points = pc2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True)
        pc2_points_64 = pc2_points.astype(np.float64)

        # takes about 6 ms extra
        # pc2_points = np.array(list(pc2_points))        

        # Apply z passthrough filter and remove points with NaN or infinite values in one step
        valid_idx = (pc2_points_64[:, 2] >= self.z_min) & (pc2_points_64[:, 2] <= self.z_max) & ~np.isinf(pc2_points_64).any(axis=1)
        pc2_points_64 = pc2_points_64[valid_idx]

        # Update the point cloud
        self.pcd.points = o3d.utility.Vector3dVector(pc2_points_64)

        self.pcd.voxel_down_sample(voxel_size=0.05)

        points = pc2_points_64
        
        kmeans_labels = self.fit_kmeans_model(points, self.kmeans_num_clusters)
        clusters = organize_clusters(points, kmeans_labels)

        for i in range(self.kmeans_num_clusters):
            self.input_queue.put((f'dbscan_cluster_{i}', clusters[i], 0.35, 5))
        
        generated_clusters = np.array([])
        for _ in range(self.kmeans_num_clusters):
            _key, value = self.output_queue.get(block=True)
            generated_clusters = np.concatenate((generated_clusters, value[:-1]))

        end = time.time()

        elapsed_time = end - start
        self.pc_performance_monitor.update(elapsed_time)

        # print current elapsed fps and mean elapsed fps

        print(f'FPS: {(1/elapsed_time):.2f} ; Elapsed time: {(elapsed_time*1000):.2f} ; Mean FPS: {(1/self.pc_performance_monitor.get_mean()):.2f} ; Mean Elapsed time: {(self.pc_performance_monitor.get_mean()*1000):.2f}')

        pcd_list = build_pointcloud_clusters(generated_clusters, self.label_colors)

        self.vis.clear_geometries()

        for pcd_cluster in pcd_list:
            self.vis.add_geometry(pcd_cluster, reset_bounding_box=self.first_run)
        self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5), reset_bounding_box=False)

        if self.first_run:
            self.first_run = False

        # Clear and update the visualizer
        self.vis.poll_events()
        self.vis.update_renderer()
        

def main(args=None):
    rclpy.init(args=args)
    node = KmDbscanClustering()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        node.vis.destroy_window()

if __name__ == '__main__':
    main()
