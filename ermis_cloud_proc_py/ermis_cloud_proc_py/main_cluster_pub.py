import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header, String, UInt32
from geometry_msgs.msg import Point
from ember_detection_interfaces.msg import EmberCluster, EmberClusterArray, EmberBoundingBox3D
import sensor_msgs_py.point_cloud2 as pc2
import pypatchworkpp

import numpy as np
import struct
import time
import argparse  # New import for argument parsing
import open3d as o3d
import mlpack
import yaml
import multiprocessing
import signal

from ermis_cloud_proc_py.performance_tools.perf_monitor import PerformanceMonitorErmis
from ermis_cloud_proc_py.performance_tools.perf_csv_recorder import PerformanceCSVRecorder
from ermis_cloud_proc_py.utils import *
from ermis_cloud_proc_py.visualization.o3d_detect_viz import Open3DClusteringVisualizer
from ermis_cloud_proc_py.detection_recording.detection_recorder import DetectionRecorder

def clustering_visualizer_worker(clustering_queue):

    global visualize_flag

    if not visualize_flag:
        # Exit if visualization is not enabled, should not be here
        return

    def sigint_handler(sig, frame):
        print('Exiting clustering visualizer...')
        exit(0)

    # sigint exit
    signal.signal(signal.SIGINT, sigint_handler)
    signal.signal(signal.SIGTERM, sigint_handler)

    visualizer = Open3DClusteringVisualizer()
    while True:
        data = clustering_queue.get()
        if data is None:
            break
        visualizer.reset()
        clusters_points_list = data['cluster_points_list']
        bb_points_list = data['bb_points_list']
        centroids = data['centroids']
        label_colors = data['label_colors']
        z_out_points = data['z_out_points']
        xy_out_points = data['xy_out_points']
        z_filter_args = data['z_filter_args']
        xy_filter_args = data['xy_filter_args']
        visualizer.draw_clusters_from_points(clusters_points_list, label_colors)
        visualizer.draw_bboxes_from_points(bb_points_list, bbox_type='AABB') # TODO change according to config
        visualizer.draw_centroids(centroids)
        visualizer.draw_points(z_out_points, color=[0.7, 0.7, 0.7])
        visualizer.draw_points(xy_out_points, color=[0.4, 0.4, 0.4])

        if xy_filter_args is not None:
            # define lines that define a box using z and xy filter arguments
            
            lines = create_box_lines([xy_filter_args[0], xy_filter_args[1]], [xy_filter_args[2], xy_filter_args[3]], [z_filter_args[0], z_filter_args[1]])
            visualizer.draw_lineset_from_points(lines, color=[0, 1, 0])

        visualizer.render()

# Structs for point cloud processing configurations
class ZFilterConfig:
    def __init__(self, z_min=0.0, z_max=2.0):
        self.z_min = z_min
        self.z_max = z_max

class XYFilterConfig:
    def __init__(self, enable, x_min=-10.0, x_max=10.0, y_min=-10.0, y_max=10.0):
        self.enable = enable
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

class VoxelDownsampleConfig:
    def __init__(self, voxel_size=0.35):
        self.voxel_size = voxel_size

class StatisticalOutlierRemovalConfig:
    def __init__(self, nb_neighbors=50, std_ratio=1.0):
        self.nb_neighbors = nb_neighbors
        self.std_ratio = std_ratio

class PatchworkPPConfig:
    def __init__(self, enable=False, verbose=False, enable_RNR=False, min_range=0.0, sensor_height=0.0, 
                 num_iter=5, th_seeds=0.05, th_dist=0.05, uprightness_thr=0.5, adaptive_seed_selection_margin=-0.1):
        self.enable = enable
        self.verbose = verbose
        self.enable_RNR = enable_RNR
        self.min_range = min_range
        self.sensor_height = sensor_height
        self.num_iter = num_iter
        self.th_seeds = th_seeds
        self.th_dist = th_dist
        self.uprightness_thr = uprightness_thr
        self.adaptive_seed_selection_margin = adaptive_seed_selection_margin
    
class DBSCANClusteringConfig:
    def __init__(self, eps=0.35, min_samples=10):
        self.eps = eps
        self.min_samples = min_samples

class BoundingBoxConfig:
    def __init__(self, bounding_box_type="AABB"): # TODO instead of string use a class with enum
        self.bounding_box_type = bounding_box_type

class DetectionRecordingConfig:
    def __init__(self, enable=False, save_dir=None):
        self.enable = enable
        self.save_dir = save_dir

class ClusterBboxDetectionWithPoseTransformPublisherNode(Node):
    def __init__(self, config_filename=None, recorder_filename=None, visualizer_queue=None):
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

        self.pcd = o3d.geometry.PointCloud()
        
        self.pc_performance_monitor = PerformanceMonitorErmis()
        if recorder_filename is not None:
            self.enable_recorder = True
            self.pc_performance_recorder = PerformanceCSVRecorder(recorder_filename)
        else:
            self.enable_recorder = False # TODO rename from enable_recorder to something less generic
            self.pc_performance_recorder = None        
        
        self.load_config(config_filename)
        self.label_colors = np.random.rand(1000, 3)

        self.visualizer_queue = visualizer_queue

        if self.patchwork_pp_config.enable:
            self.patchwork_pp_model = pypatchworkpp.patchworkpp(self.get_patchwork_pp_params())
        
        if self.detection_recording_config.enable:
            self.detection_recorder = DetectionRecorder(self.detection_recording_config.save_dir)

    def get_patchwork_pp_params(self):
        params = pypatchworkpp.Parameters()
        params.enable_RNR=self.patchwork_pp_config.enable_RNR
        params.min_range=self.patchwork_pp_config.min_range
        params.sensor_height=self.patchwork_pp_config.sensor_height
        params.num_iter=self.patchwork_pp_config.num_iter
        params.th_seeds=self.patchwork_pp_config.th_seeds
        params.th_dist=self.patchwork_pp_config.th_dist
        params.uprightness_thr=self.patchwork_pp_config.uprightness_thr
        params.adaptive_seed_selection_margin=self.patchwork_pp_config.adaptive_seed_selection_margin
        return params

    def load_config(self, config_filename):
        if config_filename is not None:
            with open(config_filename, 'r') as file:
                data = yaml.safe_load(file)
                try:
                    self.z_filter_config = ZFilterConfig(z_min=data['z_filter']['z_min'], z_max=data['z_filter']['z_max'])
                    self.xy_filter_config = XYFilterConfig(enable=data['xy_filter']['enable'], x_min=data['xy_filter']['x_min'], x_max=data['xy_filter']['x_max'], y_min=data['xy_filter']['y_min'], y_max=data['xy_filter']['y_max'])
                    self.voxel_downsample_config = VoxelDownsampleConfig(voxel_size=data['voxel_downsample']['voxel_size'])
                    self.statistical_outlier_removal_config = StatisticalOutlierRemovalConfig(nb_neighbors=data['statistical_outlier_removal']['nb_neighbors'], std_ratio=data['statistical_outlier_removal']['std_ratio'])
                    self.dbscan_clustering_config = DBSCANClusteringConfig(eps=data['dbscan_clustering']['eps'], min_samples=data['dbscan_clustering']['min_samples'])
                    self.bounding_box_config = BoundingBoxConfig(bounding_box_type=data['bounding_box_type'])
                    if data['patchwork_pp'] is not None and data['patchwork_pp']['enable']:
                        self.patchwork_pp_config = PatchworkPPConfig(
                            enable=data['patchwork_pp']['enable'],
                            verbose=data['patchwork_pp']['verbose'],
                            enable_RNR=data['patchwork_pp']['enable_RNR'],
                            min_range=data['patchwork_pp']['min_range'],
                            sensor_height=data['patchwork_pp']['sensor_height'],
                            num_iter=data['patchwork_pp']['num_iter'],
                            th_seeds=data['patchwork_pp']['th_seeds'],
                            th_dist=data['patchwork_pp']['th_dist'],
                            uprightness_thr=data['patchwork_pp']['uprightness_thr'],
                            adaptive_seed_selection_margin=data['patchwork_pp']['adaptive_seed_selection_margin']
                        )
                    else:
                        self.patchwork_pp_config = PatchworkPPConfig(enable=False)

                    if data['detection_recording'] is not None and data['detection_recording']['enable']:
                        self.detection_recording_config = DetectionRecordingConfig(
                            enable=data['detection_recording']['enable'],
                            save_dir=data['detection_recording']['save_dir']
                        )
                    else:
                        self.detection_recording_config = DetectionRecordingConfig(enable=False)
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
        # pose stamped message
        self.current_pose = msg

    def pointcloud_callback(self, msg):
        global visualize_flag

        start = time.time()

        # Raw point cloud pre-processing
        points = load_pointcloud_from_ros2_msg(msg)
        points, z_out_points = apply_finite_z_passthrough_filter(points, 
                                                    z_min=self.z_filter_config.z_min, 
                                                    z_max=self.z_filter_config.z_max)
        
        if self.xy_filter_config.enable:
            points, xy_out_points = apply_finite_xy_passthrough_filter(points, 
                                                x_min=self.xy_filter_config.x_min, 
                                                x_max=self.xy_filter_config.x_max, 
                                                y_min=self.xy_filter_config.y_min, 
                                                y_max=self.xy_filter_config.y_max)
        else:
            xy_out_points = np.array([])
        
        # Open3D point cloud pre-processing
        self.pcd.points = o3d.utility.Vector3dVector(points)

        current_pose = None
        if self.current_pose is not None:
            current_pose = self.current_pose.pose
            transformation_matrix = pose_msg_to_transform_matrix(current_pose)
            self.pcd.transform(transformation_matrix)
        
        self.pcd.points = apply_voxel_downsampling(self.pcd, 
                                                   voxel_size=self.voxel_downsample_config.voxel_size)
        self.pcd.points = apply_statistical_outlier_removal(self.pcd, 
                                                            nb_neighbors=self.statistical_outlier_removal_config.nb_neighbors, 
                                                            std_ratio=self.statistical_outlier_removal_config.std_ratio)
        
        points = np.asarray(self.pcd.points)

        if self.patchwork_pp_config.enable:
            # Patchwork++ ground segmentation
            ground, nonground, _, _ = appply_patchwork_pp(self.patchwork_pp_model, points, self.patchwork_pp_config.verbose)

            points = nonground

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
                                                            std_ratio=self.statistical_outlier_removal_config.std_ratio * 0.25)

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

        elapsed_time = end - start
        self.pc_performance_monitor.update(elapsed_time)

        verbose_performance_str = ""

        # print current elapsed fps and mean elapsed fps
        #print(f'FPS: {(1/elapsed_time):.2f} ; Elapsed time: {(elapsed_time*1000):.2f} ; Mean FPS: {(1/self.pc_performance_monitor.get_mean()):.2f} ; Mean Elapsed time: {(self.pc_performance_monitor.get_mean()*1000):.2f}')
        verbose_performance_str += f'DETECTION # Elapsed(ms): {(elapsed_time*1000):.2f} ; FPS: {(1/elapsed_time):.2f} ; Mean Elapsed(ms): {(self.pc_performance_monitor.get_mean()*1000):.2f} ; Mean FPS: {(1/self.pc_performance_monitor.get_mean()):.2f}\n'
        if self.enable_recorder:
            self.pc_performance_recorder.record(
                elapsed_time, 1/elapsed_time, 
                self.pc_performance_monitor.get_mean(), 1/self.pc_performance_monitor.get_mean())

        ### transform to ROS2 message
        ember_cluster_array = build_ember_cluster_array_msg(pcd_list, bb_list, centroids, msg, current_pose)
        self.cluster_bbox_pub.publish(ember_cluster_array)
        ###

        elapsed_time = time.time() - start
        verbose_performance_str += f'ROS2_PUB # Elapsed(ms): {(elapsed_time*1000):.2f} ; FPS: {(1/elapsed_time):.2f}\n'

        if self.detection_recording_config.enable:
            self.detection_recorder.record(bb_list, centroids, msg.header.stamp.sec, msg.header.stamp.nanosec)
            elapsed_time = time.time() - start
            verbose_performance_str += f'RECORD # Elapsed(ms): {(elapsed_time*1000):.2f} ; FPS: {(1/elapsed_time):.2f}\n'

        bbox_points_list = [np.asarray(bb.get_box_points()) for bb in bb_list]
        
        # Visualizer in separate process
        visualizer_data = {
            'cluster_points_list': clusters_points,
            'bb_points_list': bbox_points_list,
            'centroids': centroids,
            'label_colors': self.label_colors,
            # from here on this data is more for debugging purposes
            'z_out_points': z_out_points,
            'xy_out_points': xy_out_points,
            'z_filter_args': (self.z_filter_config.z_min, self.z_filter_config.z_max),
            'xy_filter_args': (self.xy_filter_config.x_min, self.xy_filter_config.x_max, self.xy_filter_config.y_min, self.xy_filter_config.y_max) if self.xy_filter_config.enable else None,
        }

        if visualize_flag:
            self.visualizer_queue.put(visualizer_data)
            elapsed_time = time.time() - start
            verbose_performance_str += f'VISUALIZATION # Elapsed(ms): {(elapsed_time*1000):.2f} ; FPS: {(1/elapsed_time):.2f}\n'

        print(verbose_performance_str)
        
def signal_handler(sig, frame, node, process_1, queue_1):
    global visualize_flag
    print('Exiting via signal handler...')

    if visualize_flag:
        queue_1.put(None)
        process_1.terminate()
        process_1.join()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

def main(args=None):
    # Argument parsing
    parser = argparse.ArgumentParser(description='Open3D Point Cloud Visualizer')
    parser.add_argument('config_fp', type=str, help='Filepath for configuration file')
    parser.add_argument('--record_fp', type=str, default=None, help='Filepath for performance recording')
    # argument, default false, for enabling visualization
    parser.add_argument('--visualize', action='store_true', help='Enable visualization', default=False)
    parsed_args = parser.parse_args(args=args)

    print(f'Arguments: {parsed_args}')

    global visualize_flag
    visualize_flag = parsed_args.visualize

    if visualize_flag:
        visualizer_queue = multiprocessing.Queue()
        visualizer_process = multiprocessing.Process(target=clustering_visualizer_worker, args=(visualizer_queue,))
        visualizer_process.start()
    else :
        visualizer_queue = None
        visualizer_process = None

    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, node, visualizer_process, visualizer_queue))

    rclpy.init(args=args)
    node = ClusterBboxDetectionWithPoseTransformPublisherNode(config_filename=parsed_args.config_fp, recorder_filename=parsed_args.record_fp, visualizer_queue=visualizer_queue)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if visualize_flag:
            visualizer_queue.put(None)
            visualizer_process.terminate()
            visualizer_process.join()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
