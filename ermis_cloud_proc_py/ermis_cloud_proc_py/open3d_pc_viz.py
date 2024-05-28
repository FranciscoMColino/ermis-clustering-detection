import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import open3d as o3d
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np

class PointCloudSubscriber(Node):
    def __init__(self):
        super().__init__('open3d_pc_viz')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/zed/zed_node/point_cloud/cloud_registered',
            self.pointcloud_callback,
            2)
        self.subscription  # prevent unused variable warning
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.first_run = True

    def pointcloud_callback(self, msg):
        # Convert ROS PointCloud2 message to numpy array
        pc2_points = pc2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True)

        self.vis.clear_geometries()
        pcd = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(pc2_points))
        pcd.paint_uniform_color([0.5, 0.5, 0.5])
        self.vis.add_geometry(pcd, reset_bounding_box=self.first_run)

        if self.first_run:
            self.first_run = False

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
