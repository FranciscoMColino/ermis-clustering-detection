import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import open3d as o3d
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import struct

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

    def pointcloud_callback(self, msg):

        #self.vis.clear_geometries()

        # Convert ROS PointCloud2 message to numpy array
        pc2_data = pc2.read_points_numpy(msg, field_names=("x", "y", "z", 'rgb'), skip_nans=True)
        # separate the rgb values
        pc2_data = np.array(list(pc2_data))
        pc2_points = pc2_data[:, :3]

        # Unpack RGB values
        pc2_colors = pc2_data[:, 3]
        colors = np.zeros((pc2_colors.size, 3), dtype=np.float32)

        for i, color in enumerate(pc2_colors):
            packed = struct.unpack('I', struct.pack('f', color))[0]
            r = (packed & 0x00FF0000) >> 16
            g = (packed & 0x0000FF00) >> 8
            b = (packed & 0x000000FF)
            colors[i, :] = [r, g, b]

        colors /= 255.0

        # get indexes where points are infinite
        inf_idx = np.isinf(pc2_points).any(axis=1)

        # remove infinite points from both the point cloud and the colors
        pc2_points = pc2_points[~inf_idx]
        colors = colors[~inf_idx]

        #print(f'Removing {inf_idx.sum()} infinite points.')
        
        # Update the point cloud
        self.pcd.points = o3d.utility.Vector3dVector(pc2_points)
        self.pcd.colors = o3d.utility.Vector3dVector(colors)

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
