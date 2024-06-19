import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
import sensor_msgs_py.point_cloud2 as pc2

import numpy as np
import time

import open3d as o3d

from ermis_cloud_proc_py.src.utils.perf_monitor import PerformanceMonitorErmis

### START - Raw point manipulation

def load_pointcloud_from_ros2_msg(msg):
    pc2_points = pc2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True)
    pc2_points_64 = pc2_points.astype(np.float64)
    return pc2_points_64

def apply_finite_z_passthrough_filter(points, z_min, z_max):
    valid_idx = (points[:, 2] >= z_min) & (points[:, 2] <= z_max) & ~np.isinf(points).any(axis=1)
    return points[valid_idx]

### END - Raw point manipulation

### START - Open3D point cloud manipulation

def apply_voxel_downsampling(pcd, voxel_size=0.05):
    pcd_down_samp = pcd.voxel_down_sample(voxel_size=voxel_size)
    return pcd_down_samp.points

def apply_statistical_outlier_removal(pcd, nb_neighbors=10, std_ratio=0.025):
    pcd_res, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd_res.points

### END - Open3D point cloud manipulation

def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion to a 3x3 rotation matrix.
    """
    x, y, z, w = q
    R = np.array([[1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
                  [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
                  [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]])
    return R

class CloudPoseTransformNode(Node):
    def __init__(self):
        super().__init__('cloud_pose_transform_node')
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
        
        self.cloud_callback_counter = 0
        self.pose_callback_counter = 0
        self.current_pose = None

        self.cloud_subscription  # prevent unused variable warning
        self.width = 1200
        self.height = 800
        self.vis = o3d.visualization.Visualizer()
        self.pcd = o3d.geometry.PointCloud()
        self.vis.create_window(window_name='Open3D Point Cloud Visualizer', width=self.width, height=self.height) 
        
        self.pc_performance_monitor = PerformanceMonitorErmis()
        self.setup_view_control()
        self.label_colors = np.random.rand(1000, 3)
        
    def setup_view_control(self):
        # Add 8 points to initiate the visualizer's bounding box
        points = np.array([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [10, 0, 0],
            [10, 0, 1],
            [10, 1, 0],
            [10, 1, 1]
        ])

        points *= 4

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        self.vis.add_geometry(pcd, reset_bounding_box=True)

        view_control = self.vis.get_view_control()
        view_control.rotate(0, -525)
        view_control.rotate(500, 0)

        # points thinner and lines thicker
        self.vis.get_render_option().point_size = 2.0
        self.vis.get_render_option().line_width = 10.0

    def pose_callback(self, msg):
        self.pose_callback_counter += 1
        print(f'Pose callback counter: {self.pose_callback_counter}')
        print(f'Pose: {msg.pose.position.x}, {msg.pose.position.y}, {msg.pose.position.z}')
        print(f'Orientation: {msg.pose.orientation.x}, {msg.pose.orientation.y}, {msg.pose.orientation.z}, {msg.pose.orientation.w}')
        self.current_pose = msg


    def pointcloud_callback(self, msg):

        self.cloud_callback_counter += 1
        print(f'Cloud callback counter: {self.cloud_callback_counter}')

        start = time.time()

        # Raw point cloud pre-processing
        points = load_pointcloud_from_ros2_msg(msg)
        #points = apply_finite_z_passthrough_filter(points, 
        #                                            z_min=self.z_filter_config.z_min, 
        #                                            z_max=self.z_filter_config.z_max)

        self.pcd.points = o3d.utility.Vector3dVector(points)

        #if self.current_pose is not None:
        #    pose = self.current_pose
        #    self.pcd.transform(np.array([
        #        [pose.pose.orientation.w, -pose.pose.orientation.z, pose.pose.orientation.y, pose.pose.position.x],
        #        [pose.pose.orientation.z, pose.pose.orientation.w, -pose.pose.orientation.x, pose.pose.position.y],
        #        [-pose.pose.orientation.y, pose.pose.orientation.x, pose.pose.orientation.w, pose.pose.position.z],
        #        [0, 0, 0, 1]
        #    ]))
        
        if self.current_pose is not None:
            pose = self.current_pose.pose
            translation = np.array([pose.position.x,
                                    pose.position.y,
                                    pose.position.z])
            quaternion = [pose.orientation.x,
                        pose.orientation.y,
                        pose.orientation.z,
                        pose.orientation.w]
            rotation_matrix = quaternion_to_rotation_matrix(quaternion)
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation_matrix
            transformation_matrix[:3, 3] = translation
            self.pcd.transform(transformation_matrix)


        end = time.time()

        elapsed_time = end - start
        self.pc_performance_monitor.update(elapsed_time)

        # print current elapsed fps and mean elapsed fps
        print(f'FPS: {(1/elapsed_time):.2f} ; Elapsed time: {(elapsed_time*1000):.2f} ; Mean FPS: {(1/self.pc_performance_monitor.get_mean()):.2f} ; Mean Elapsed time: {(self.pc_performance_monitor.get_mean()*1000):.2f}')

        self.vis.clear_geometries()
        self.vis.add_geometry(self.pcd, reset_bounding_box=False)
        self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5), reset_bounding_box=False)

        # Clear and update the visualizer
        self.vis.poll_events()
        self.vis.update_renderer()
        
def main(args=None):
    rclpy.init(args=args)

    node = CloudPoseTransformNode()
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
