import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'jet_commander'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='User',
    maintainer_email='user@example.com',
    description='Control package for the F14 Tomcat jet robot',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'circle_flight = jet_commander.circle_flight:main',
            'dino_tracker = jet_commander.dino_tracker:main',
            'target_follower = jet_commander.target_follower:main',
            'target_mover = jet_commander.target_mover:main',
        ],
    },
)
