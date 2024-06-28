import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import open3d as o3d
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import struct
import time

class PCPerformanceMonitor():
    def __init__(self):
        self.num_measurements = 0
        self.current_mean = 0

    def update(self, new_measurement):
        self.num_measurements += 1
        self.current_mean = (self.current_mean * (self.num_measurements - 1) + new_measurement) / self.num_measurements

    def get_mean(self):
        return self.current_mean

def plane_segmentation_with_axis(input_cloud, distance_threshold=0.05, ransac_n=3, num_iterations=1000, axis=[0, 0, 1], best_plane_iter = 5, angle_threshold=5.0):
    if isinstance(input_cloud, np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(input_cloud)
    else:
        pcd = input_cloud

    best_plane_model = None
    best_inliers = []
    
    for _ in range(best_plane_iter):  # Try multiple times to find the best plane model
        plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                                 ransac_n=ransac_n,
                                                 num_iterations=num_iterations)
        [a, b, c, d] = plane_model

        # Check if the plane normal is approximately perpendicular to the given axis
        plane_normal = np.array([a, b, c])
        axis = np.array(axis)
        angle = np.degrees(np.arccos(np.dot(plane_normal, axis) / (np.linalg.norm(plane_normal) * np.linalg.norm(axis))))

        if angle < angle_threshold or angle > (180 - angle_threshold):
            if len(inliers) > len(best_inliers):
                best_plane_model = plane_model
                best_inliers = inliers

    if best_plane_model is None:
        raise ValueError("Could not find a perpendicular plane within the given constraints.")

    inlier_cloud = pcd.select_by_index(best_inliers)
    outlier_cloud = pcd.select_by_index(best_inliers, invert=True)

    return inlier_cloud, outlier_cloud, best_plane_model

class ClusteringGroundSegment(Node):
    def __init__(self):
        super().__init__('clustering_ground_segment')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/zed/zed_node/point_cloud/cloud_registered',
            self.pointcloud_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.vis = o3d.visualization.Visualizer()
        self.pcd = o3d.geometry.PointCloud()
        self.ground = o3d.geometry.PointCloud()
        self.vis.create_window()
        self.first_run = True
        self.pc_performance_monitor = PCPerformanceMonitor()

    def pointcloud_callback(self, msg):

        #self.vis.clear_geometries()

        start = time.time()

        # Convert ROS PointCloud2 message to numpy array
        pc2_points = pc2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True)
        pc2_points_64 = pc2_points.astype(np.float64)

        # takes about 6 ms extra
        # pc2_points = np.array(list(pc2_points))        

        # takes about .8 ms extra
        inf_idx = np.isinf(pc2_points_64).any(axis=1)
        pc2_points_64 = pc2_points_64[~inf_idx]

        # Update the point cloud
        self.pcd.points = o3d.utility.Vector3dVector(pc2_points_64)

        self.pcd.voxel_down_sample(voxel_size=0.05)

        inlier_cloud, outlier_cloud, plane_model = plane_segmentation_with_axis(
            self.pcd, distance_threshold=0.05, ransac_n=3, num_iterations=1000, 
            axis=[0, 0, 1], best_plane_iter=5, angle_threshold=10.0)

        self.pcd.points = o3d.utility.Vector3dVector(outlier_cloud.points)

        #self.ground.paint_uniform_color([1, 0, 0])

        end = time.time()

        elapsed_time = end - start
        self.pc_performance_monitor.update(elapsed_time)

        # print current elapsed fps and mean elapsed fps

        print(f'FPS: {(1/elapsed_time):.2f} ; Elapsed time: {(elapsed_time*1000):.2f} ; Mean FPS: {(1/self.pc_performance_monitor.get_mean()):.2f} ; Mean Elapsed time: {(self.pc_performance_monitor.get_mean()*1000):.2f}')


        if self.first_run:
            self.vis.add_geometry(self.pcd, reset_bounding_box=True)
            #self.vis.add_geometry(self.ground, reset_bounding_box=False)
            self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5), reset_bounding_box=True)
            self.first_run = False
        else:
            self.vis.update_geometry(self.pcd)
            #self.vis.update_geometry(self.ground)

        # Clear and update the visualizer
        self.vis.poll_events()
        self.vis.update_renderer()
        

def main(args=None):
    rclpy.init(args=args)
    node = ClusteringGroundSegment()
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
