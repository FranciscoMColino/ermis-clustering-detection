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

class PointCloudSubscriber(Node):
    def __init__(self):
        super().__init__('open3d_pc_viz')
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

        end = time.time()

        elapsed_time = end - start
        self.pc_performance_monitor.update(elapsed_time)

        # print current elapsed fps and mean elapsed fps

        print(f'FPS: {(1/elapsed_time):.2f} ; Elapsed time: {(elapsed_time*1000):.2f} ; Mean FPS: {(1/self.pc_performance_monitor.get_mean()):.2f} ; Mean Elapsed time: {(self.pc_performance_monitor.get_mean()*1000):.2f}')


        if self.first_run:
            self.vis.add_geometry(self.pcd, reset_bounding_box=True)
            self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5), reset_bounding_box=True)
            self.first_run = False
        else:
            self.vis.update_geometry(self.pcd)

        # Clear and update the visualizer
        self.vis.poll_events()
        self.vis.update_renderer()
        

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudSubscriber()
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
