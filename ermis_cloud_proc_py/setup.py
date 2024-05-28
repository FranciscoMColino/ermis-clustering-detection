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
            'open3d_pc_viz = ermis_cloud_proc_py.open3d_pc_viz:main',
        ],
    },
)
