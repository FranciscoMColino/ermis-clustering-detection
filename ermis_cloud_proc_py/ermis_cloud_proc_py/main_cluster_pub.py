import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header, String, UInt32
from geometry_msgs.msg import Point
from ember_detection_interfaces.msg import EmberCluster, EmberClusterArray, EmberBoundingBox3D
import sensor_msgs_py.point_cloud2 as pc2

import numpy as np
import struct
import time
import argparse  # New import for argument parsing

import open3d as o3d
import mlpack
import yaml

from ermis_cloud_proc_py.performance_tools.perf_monitor import PerformanceMonitorErmis
from ermis_cloud_proc_py.performance_tools.perf_csv_recorder import PerformanceCSVRecorder
from ermis_cloud_proc_py.utils import *
from ermis_cloud_proc_py.visualization.o3d_detect_viz import Open3DClusteringVisualizer

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

class ClusterBboxDetectionWithPoseTransformPublisherNode(Node):
    def __init__(self, config_filename=None, recorder_filename=None):
        super().__init__('cluster_bbox_detection_publisher')
        self.cloud_subscription = self.create_subscription(
            PointCloud2,
            '/zed/zed_node/point_cloud/cloud_registered',
            self.pointcloud_callback,
            10)
        self.pose_subscription = self.create_subscription(
            PoseStamped,
            '/zed/zed_node/pose',
            self.pose_callback,
            10)
        self.cluster_bbox_pub = self.create_publisher(
            EmberClusterArray, 
            '/ember_detection/ember_cluster_array', 
            10)

        self.cloud_subscription  # prevent unused variable warning
        self.pose_subscription  # prevent unused variable warning
        self.current_pose = None

        self.visualizer = Open3DClusteringVisualizer()

        self.pcd = o3d.geometry.PointCloud()
        
        self.pc_performance_monitor = PerformanceMonitorErmis()
        if recorder_filename is not None:
            self.enable_recorder = True
            self.pc_performance_recorder = PerformanceCSVRecorder(recorder_filename)
        else:
            self.enable_recorder = False
            self.pc_performance_recorder = None
        
        self.load_config(config_filename)
        self.label_colors = np.random.rand(1000, 3)

    def load_config(self, config_filename):
        if config_filename is not None:
            with open(config_filename, 'r') as file:
                data = yaml.safe_load(file)
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
            # Clear and update the visualizer
            self.vis.poll_events()
            self.vis.update_renderer()
            exit(1)

    def pose_callback(self, msg):
        self.current_pose = msg

    def pointcloud_callback(self, msg):
        start = time.time()

        # Raw point cloud pre-processing
        points = load_pointcloud_from_ros2_msg(msg)
        points = apply_finite_z_passthrough_filter(points, 
                                                    z_min=self.z_filter_config.z_min, 
                                                    z_max=self.z_filter_config.z_max)
        
        # Open3D point cloud pre-processing
        self.pcd.points = o3d.utility.Vector3dVector(points)

        if self.current_pose is not None:
            transformation_matrix = pose_msg_to_transform_matrix(self.current_pose)
            self.pcd.transform(transformation_matrix)
        
        self.pcd.points = apply_voxel_downsampling(self.pcd, 
                                                   voxel_size=self.voxel_downsample_config.voxel_size)
        self.pcd.points = apply_statistical_outlier_removal(self.pcd, 
                                                            nb_neighbors=self.statistical_outlier_removal_config.nb_neighbors, 
                                                            std_ratio=self.statistical_outlier_removal_config.std_ratio)

        # DBSCAN clustering
        points = np.asarray(self.pcd.points)
        labels, centroids = apply_dbscan_clustering(points, 
                                         eps=self.dbscan_clustering_config.eps,
                                         min_size=self.dbscan_clustering_config.min_samples)

        # Organize clusters and build point clouds
        clusters_points = organize_clusters(points, labels)
        pcd_list = build_pointcloud_clusters(clusters_points, self.label_colors)

        # apply statistical outlier removal to each cluster before building bounding boxes
        # TODO maybe have a different configuration for this
        for pcd in pcd_list:
            # noise removal
            pcd.points = apply_statistical_outlier_removal(pcd, 
                                                            nb_neighbors=self.statistical_outlier_removal_config.nb_neighbors, 
                                                            std_ratio=self.statistical_outlier_removal_config.std_ratio) 

        # Build bounding boxes
        if self.bounding_box_config.bounding_box_type == "AABB":
            bb_list = build_pointcloud_aabb(pcd_list)
        elif self.bounding_box_config.bounding_box_type == "OBB":
            bb_list = build_pointcloud_obb(pcd_list)
        else:
            print('Invalid bounding box type. Exiting...')
            exit(1)

        if len(pcd_list) != len(bb_list):
            print('Error: Number of point clouds and bounding boxes do not match. Exiting...')
            exit(1)

        end = time.time()

        ### transform to ROS2 message

        ember_cluster_array = EmberClusterArray()
        ember_cluster_array.header = msg.header
        ember_cluster_array.header.frame_id = 'cluster_bbox_detection'

        for i in range(len(pcd_list)):
            ember_cluster = EmberCluster()
            cluster_points = np.asarray(pcd_list[i].points)
            cluster_pc2 = pc2.create_cloud_xyz32(ember_cluster_array.header, cluster_points)
            ember_cluster.point_cloud = cluster_pc2
            ember_cluster.centroid = Point(x=centroids[i][0], y=centroids[i][1], z=centroids[i][2])

            ember_bbox = EmberBoundingBox3D()
            ember_bbox.det_label = String(data='default')
            bbox_points = np.asarray(bb_list[i].get_box_points())
            for point in bbox_points:
                ember_bbox.points.append(Point(x=point[0], y=point[1], z=point[2]))
            ember_bbox.points_count = UInt32(data=len(bbox_points))

            ember_cluster.bounding_box = ember_bbox
            ember_cluster_array.clusters.append(ember_cluster)

        self.cluster_bbox_pub.publish(ember_cluster_array)

        ###

        elapsed_time = end - start
        self.pc_performance_monitor.update(elapsed_time)

        # print current elapsed fps and mean elapsed fps
        print(f'FPS: {(1/elapsed_time):.2f} ; Elapsed time: {(elapsed_time*1000):.2f} ; Mean FPS: {(1/self.pc_performance_monitor.get_mean()):.2f} ; Mean Elapsed time: {(self.pc_performance_monitor.get_mean()*1000):.2f}')
        if self.enable_recorder:
            self.pc_performance_recorder.record(
                elapsed_time, 1/elapsed_time, 
                self.pc_performance_monitor.get_mean(), 1/self.pc_performance_monitor.get_mean())

        self.visualizer.reset()
        self.visualizer.draw_clusters(pcd_list)
        self.visualizer.draw_bboxes(bb_list)
        self.visualizer.draw_centroids(centroids)
        self.visualizer.render()
        
def main(args=None):
    rclpy.init(args=args)

    # Argument parsing
    parser = argparse.ArgumentParser(description='Open3D Point Cloud Visualizer')
    parser.add_argument('config_fp', type=str, help='Filepath for configuration file')
    parser.add_argument('--record_fp', type=str, default=None, help='Filepath for performance recording')
    parsed_args = parser.parse_args(args=args)

    node = ClusterBboxDetectionWithPoseTransformPublisherNode(config_filename=parsed_args.config_fp, recorder_filename=parsed_args.record_fp)
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
