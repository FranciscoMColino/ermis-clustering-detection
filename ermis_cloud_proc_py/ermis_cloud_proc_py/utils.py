import numpy as np
import open3d as o3d
import mlpack
import sensor_msgs_py.point_cloud2 as pc2

from std_msgs.msg import Header, String, UInt32
from geometry_msgs.msg import Point
from ember_detection_interfaces.msg import EmberCluster, EmberClusterArray, EmberBoundingBox3D

def organize_clusters(points, labels):
    # DOn't add in the -1 cluster (noise)
    uniq_labels = np.unique(labels)
    uniq_labels = uniq_labels[uniq_labels != -1]
    clusters_points = np.zeros(len(uniq_labels), dtype=object)
    for label in uniq_labels:
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

def load_pointcloud_from_ros2_msg(msg):
    pc2_points = pc2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True)
    pc2_points_64 = pc2_points.astype(np.float64)
    return pc2_points_64

def apply_finite_z_passthrough_filter(points, z_min, z_max):
    valid_idx = (points[:, 2] >= z_min) & (points[:, 2] <= z_max) & ~np.isinf(points).any(axis=1)
    return points[valid_idx]

def appply_patchwork_pp(patchwork_pp_model, points, verbose=False):
    patchwork_pp_model.estimateGround(points)

    ground      = patchwork_pp_model.getGround()
    nonground   = patchwork_pp_model.getNonground()
    time_taken  = patchwork_pp_model.getTimeTaken()

    ground_idx      = patchwork_pp_model.getGroundIndices()
    nonground_idx   = patchwork_pp_model.getNongroundIndices()

    if verbose:
        print_str = ""
        print_str += f"PATCHWORK++: Original Points  #: {points.shape[0]}\n"
        print_str += f"PATCHWORK++: Ground Points    #: {ground.shape[0]}\n"
        print_str += f"PATCHWORK++: Nonground Points #: {nonground.shape[0]}\n"
        print_str += f"PATCHWORK++: Time Taken : {time_taken / 1000000} (sec)"
        print(print_str)

    return ground, nonground, ground_idx, nonground_idx

def apply_dbscan_clustering(points, eps=0.35, min_size=10):
    d = mlpack.dbscan(input_=points, epsilon=eps, min_size=min_size)
    labels = d['assignments']
    centroids = d['centroids']
    return labels, centroids

def apply_voxel_downsampling(pcd, voxel_size=0.05):
    pcd_down_samp = pcd.voxel_down_sample(voxel_size=voxel_size)
    return pcd_down_samp.points

def apply_statistical_outlier_removal(pcd, nb_neighbors=10, std_ratio=0.025):
    pcd_res, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd_res.points

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

def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion to a 3x3 rotation matrix.
    """
    x, y, z, w = q
    R = np.array([[1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
                  [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
                  [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]])
    return R

def pose_msg_to_transform_matrix(pose):
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

    return transformation_matrix

def build_ember_cluster_array_msg(pcd_list, bb_list, centroids, msg, pose = None):
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
        for bbox_point in bbox_points:
            ember_bbox.points.append(Point(x=bbox_point[0], y=bbox_point[1], z=bbox_point[2]))
        ember_bbox.points_count = UInt32(data=len(bbox_points))

        ember_cluster.bounding_box = ember_bbox
        ember_cluster_array.clusters.append(ember_cluster)

    ember_cluster_array.pose = pose

    return ember_cluster_array