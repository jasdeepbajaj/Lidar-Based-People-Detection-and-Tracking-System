from setuptools import find_packages, setup
import glob

package_name = 'project3'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, glob.glob('launch/*'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jasdeep bajaj',
    maintainer_email='jasdeepbajaj3@tamu.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': ['ObjectDetectionNode = project3.ObjectDetectionNode:main',
        'ObjectTrackingNode = project3.ObjectTrackingNode:main',
        'DetectionAndTracking = project3.DetectionAndTracking:main'
        ],
    },
)
