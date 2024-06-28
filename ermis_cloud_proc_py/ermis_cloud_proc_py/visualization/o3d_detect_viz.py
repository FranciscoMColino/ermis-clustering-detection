import open3d as o3d
import numpy as np

class Open3DClusteringVisualizer:
    def __init__(self, name='Open3D Clustering Visualizer'):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(name, width=640, height=480)
        self.setup_visualizer()

    def setup_visualizer(self):
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

    def reset(self):
        self.vis.clear_geometries()
        self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5), reset_bounding_box=False)

    def render(self):
        self.vis.poll_events()
        self.vis.update_renderer()

    """
    Draw the clusters
        - clusters_pcds: list of o3d.geometry.PointCloud
    """
    def draw_clusters(self, clusters_pcds):
        for pcd in clusters_pcds:
            self.vis.add_geometry(pcd, reset_bounding_box=False)

    """
    Draw the bounding boxes of the clusters
        - bboxes: list of o3d.geometry.AxisAlignedBoundingBox or o3d.geometry.OrientedBoundingBox
    """
    def draw_bboxes(self, bboxes):
        for bbox in bboxes:
            self.vis.add_geometry(bbox, reset_bounding_box=False)

    """
    Draw the centroids of the clusters
        - centroids: list of 3D points
    """
    def draw_centroids(self, centroids):
        for centroid in centroids:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
            sphere.translate(centroid)
            sphere.paint_uniform_color([0.7, 0, 1])
            self.vis.add_geometry(sphere, reset_bounding_box=False)