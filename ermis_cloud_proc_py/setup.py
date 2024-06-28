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
            'kmeans_dbscan_ml_o3d = ermis_cloud_proc_py.kmeans_dbscan_ml_o3d:main',
            'open3d_ground_segment = ermis_cloud_proc_py.open3d_ground_segment:main',
            'main_cluster_pub = ermis_cloud_proc_py.main_cluster_pub:main',
        ],
    },
)
