from setuptools import find_packages, setup

package_name = 'ermis_cloud_proc_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools',
                      'open3d',
                      'numpy',
                      'sensor_msgs_py',
                      'rclpy'],
    zip_safe=True,
    maintainer='colino',
    maintainer_email='francisco.m.colino@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'open3d_pc_viz = ermis_cloud_proc_py.src.open3d_pc_viz:main',
            'open3d_nocolor_pc_viz = ermis_cloud_proc_py.src.open3d_nocolor_pc_viz:main',
            'dbscan_ml_o3d = ermis_cloud_proc_py.src.dbscan_ml_o3d:main',
            'open3d_passthrough_viz = ermis_cloud_proc_py.src.open3d_passthrough_viz:main',
            'kmeans_dbscan_ml_o3d = ermis_cloud_proc_py.src.kmeans_dbscan_ml_o3d:main',
            'open3d_ground_segment = ermis_cloud_proc_py.src.open3d_ground_segment:main',
            'cluster_bbox_pub = ermis_cloud_proc_py.src.cluster_bbox_pub:main',
            'cloud_pose_transform_viz = ermis_cloud_proc_py.src.cloud_pose_transform_viz:main',
            'cluster_bbox_pose_transform_pub = ermis_cloud_proc_py.src.cluster_bbox_pose_transform_pub:main',
            'main_cluster_pub = ermis_cloud_proc_py.src.main_cluster_pub:main',
        ],
    },
)
