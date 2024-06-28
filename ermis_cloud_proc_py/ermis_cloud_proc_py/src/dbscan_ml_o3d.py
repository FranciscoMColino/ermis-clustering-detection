import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

import numpy as np
import struct
import time
import argparse  # New import for argument parsing

import open3d as o3d
import mlpack
import yaml

from ermis_cloud_proc_py.src.performance_tools.perf_monitor import PerformanceMonitorErmis
from ermis_cloud_proc_py.src.performance_tools.perf_csv_recorder import PerformanceCSVRecorder

### START - Cluster organization and visualization

def organize_clusters(points, labels):
    clusters_points = np.zeros(len(np.unique(labels)), dtype=object)
    for label in np.unique(labels):
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
    return pcd_list

### END - Cluster organization and visualization

### START - Raw point manipulation

def load_pointcloud_from_ros2_msg(msg):
    pc2_points = pc2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True)
    pc2_points_64 = pc2_points.astype(np.float64)
    return pc2_points_64

def apply_finite_z_passthrough_filter(points, z_min, z_max):
    valid_idx = (points[:, 2] >= z_min) & (points[:, 2] <= z_max) & ~np.isinf(points).any(axis=1)
    return points[valid_idx]

def apply_dbscan_clustering(points, eps=0.35, min_size=10):
    d = mlpack.dbscan(input_=points, epsilon=eps, min_size=min_size)
    labels = d['assignments']
    return labels

### END - Raw point manipulation

### START - Open3D point cloud manipulation

def apply_voxel_downsampling(pcd, voxel_size=0.05):
    pcd_down_samp = pcd.voxel_down_sample(voxel_size=voxel_size)
    return pcd_down_samp.points

def apply_statistical_outlier_removal(pcd, nb_neighbors=10, std_ratio=0.025):
    pcd_res, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd_res.points

### END - Open3D point cloud manipulation

### START - Bounding box generation

def build_pointcloud_aabb(clusters_point_clouds):
    aabb_list = np.zeros(len(clusters_point_clouds), dtype=object)
    for i in range(len(clusters_point_clouds)):
        aabb = clusters_point_clouds[i].get_axis_aligned_bounding_box()
        aabb.color = [1, 0, 0]
        aabb_list[i] = aabb
    return aabb_list

def build_pointcloud_obb(clusters_point_clouds):
    obb_list = np.zeros(len(clusters_point_clouds), dtype=object)
    for i in range(len(clusters_point_clouds)):
        obb = clusters_point_clouds[i].get_oriented_bounding_box()
        obb.color = [0, 1, 0]
        obb_list[i] = obb
    return obb_list

### END - Bounding box generation

# Structs for point cloud processing configurations
class ZFilterConfig:
    def __init__(self, z_min=0.0, z_max=2.0):
        self.z_min = z_min
        self.z_max = z_max

class VoxelDownsampleConfig:
    def __init__(self, voxel_size=0.35):
        self.voxel_size = voxel_size

class StatisticalOutlierRemovalConfig:
    def __init__(self, nb_neighbors=50, std_ratio=1.0):
        self.nb_neighbors = nb_neighbors
        self.std_ratio = std_ratio
    
class DBSCANClusteringConfig:
    def __init__(self, eps=0.35, min_samples=10):
        self.eps = eps
        self.min_samples = min_samples

class BoundingBoxConfig:
    def __init__(self, bounding_box_type="OBB"):
        self.bounding_box_type = bounding_box_type

class PointCloudSubscriber(Node):
    def __init__(self, config_filename=None, recorder_filename=None):
        super().__init__('open3d_pc_viz')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/zed/zed_node/point_cloud/cloud_registered',
            self.pointcloud_callback,
            10)

        self.subscription  # prevent unused variable warning
        self.width = 1200
        self.height = 800
        self.vis = o3d.visualization.Visualizer()
        self.pcd = o3d.geometry.PointCloud()
        self.vis.create_window(window_name='Open3D Point Cloud Visualizer', width=self.width, height=self.height) 
        
        self.pc_performance_monitor = PerformanceMonitorErmis()
        if recorder_filename is not None:
            self.enable_recorder = True
            self.pc_performance_recorder = PerformanceCSVRecorder(recorder_filename)
        else:
            self.enable_recorder = False
            self.pc_performance_recorder = None
        
        self.config_filename = config_filename

        self.setup_processing_configs()
        self.first_run = True

        self.label_colors = np.random.rand(1000, 3)

    def setup_processing_configs(self):
        if self.config_filename is not None:
            with open(self.config_filename, 'r') as file:
                data = yaml.safe_load(file)

                """ Example configuration file:
                z_filter:
                    z_min: 0.0
                    z_max: 2.0

                voxel_downsample:
                    voxel_size: 0.05

                statistical_outlier_removal:
                    nb_neighbors: 50
                    std_ratio: 1.0

                dbscan_clustering:
                    eps: 0.35
                    min_samples: 10

                bounding_box_type: "OBB" # "AABB" or "OBB"
                """
                try:
                    self.z_filter_config = ZFilterConfig(z_min=data['z_filter']['z_min'], z_max=data['z_filter']['z_max'])
                    self.voxel_downsample_config = VoxelDownsampleConfig(voxel_size=data['voxel_downsample']['voxel_size'])
                    self.statistical_outlier_removal_config = StatisticalOutlierRemovalConfig(nb_neighbors=data['statistical_outlier_removal']['nb_neighbors'], std_ratio=data['statistical_outlier_removal']['std_ratio'])
                    self.dbscan_clustering_config = DBSCANClusteringConfig(eps=data['dbscan_clustering']['eps'], min_samples=data['dbscan_clustering']['min_samples'])
                    self.bounding_box_config = BoundingBoxConfig(bounding_box_type=data['bounding_box_type'])
                except KeyError as e:
                    print(f'Error: Configuration file is missing key: {e}')
                    exit(1)
        else:
            print('No configuration file provided. Exiting...')
            exit(1)
        

    def setup_view_control(self):
        view_control = self.vis.get_view_control()
        view_control.rotate(0, -525)
        view_control.rotate(self.width * 0.40, 0)
        self.vis.get_render_option().point_size = 2.0

    def pointcloud_callback(self, msg):
        start = time.time()

        # Raw point cloud pre-processing
        points = load_pointcloud_from_ros2_msg(msg)
        points = apply_finite_z_passthrough_filter(points, 
                                                    z_min=self.z_filter_config.z_min, 
                                                    z_max=self.z_filter_config.z_max)
        
        # Open3D point cloud pre-processing
        self.pcd.points = o3d.utility.Vector3dVector(points)
        self.pcd.points = apply_voxel_downsampling(self.pcd, 
                                                   voxel_size=self.voxel_downsample_config.voxel_size)
        self.pcd.points = apply_statistical_outlier_removal(self.pcd, 
                                                            nb_neighbors=self.statistical_outlier_removal_config.nb_neighbors, 
                                                            std_ratio=self.statistical_outlier_removal_config.std_ratio)

        # DBSCAN clustering
        points = np.asarray(self.pcd.points)
        labels = apply_dbscan_clustering(points, 
                                         eps=self.dbscan_clustering_config.eps,
                                         min_size=self.dbscan_clustering_config.min_samples)

        # Organize clusters and build point clouds
        clusters_points = organize_clusters(points, labels)
        pcd_list = build_pointcloud_clusters(clusters_points, self.label_colors)

        # Build bounding boxes
        if self.bounding_box_config.bounding_box_type == "AABB":
            bb_list = build_pointcloud_aabb(pcd_list)
        elif self.bounding_box_config.bounding_box_type == "OBB":
            bb_list = build_pointcloud_obb(pcd_list)
        else:
            print('Invalid bounding box type. Exiting...')
            exit(1)

        end = time.time()

        elapsed_time = end - start
        self.pc_performance_monitor.update(elapsed_time)

        # print current elapsed fps and mean elapsed fps
        print(f'FPS: {(1/elapsed_time):.2f} ; Elapsed time: {(elapsed_time*1000):.2f} ; Mean FPS: {(1/self.pc_performance_monitor.get_mean()):.2f} ; Mean Elapsed time: {(self.pc_performance_monitor.get_mean()*1000):.2f}')
        if self.enable_recorder:
            self.pc_performance_recorder.record(
                elapsed_time, 1/elapsed_time, 
                self.pc_performance_monitor.get_mean(), 1/self.pc_performance_monitor.get_mean())

        if self.first_run:
            self.vis.add_geometry(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)))
            self.first_run = False
            self.setup_view_control()

        self.vis.clear_geometries()

        for i in range(len(pcd_list)):
            self.vis.add_geometry(pcd_list[i], reset_bounding_box=self.first_run)
            self.vis.add_geometry(bb_list[i], reset_bounding_box=self.first_run)
        self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5), reset_bounding_box=False)

        # Clear and update the visualizer
        self.vis.poll_events()
        self.vis.update_renderer()
        
def main(args=None):
    rclpy.init(args=args)

    # Argument parsing
    parser = argparse.ArgumentParser(description='Open3D Point Cloud Visualizer')
    parser.add_argument('config_fp', type=str, help='Filepath for configuration file')
    parser.add_argument('--record_fp', type=str, default=None, help='Filepath for performance recording')
    parsed_args = parser.parse_args(args=args)

    node = PointCloudSubscriber(config_filename=parsed_args.config_fp, recorder_filename=parsed_args.record_fp)
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
