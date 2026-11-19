# No-hardware deployment validation

This test uses a replayed trajectory as mock ROS 2 joint and IMU topics. It does not enable motors.

## Shell 1: ROS 2 environment

Use system Python 3.10 for ROS 2 Humble.

    cd ~/jlj_robot_repo
    unset PYTHONHOME
    unset PYTHONPATH
    source /opt/ros/humble/setup.bash
    export PYTHONPATH=$PWD:/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages

## Shell 2: mock sensors

    /usr/bin/python3.10 scripts/mock_sensor_replay.py

## Shell 3: SuperMimic shadow node

    /usr/bin/python3.10 scripts/supermimic_ros2_node.py --ros-args -p asset_root:=/home/yuuuuu/jlj_robot_repo -p shadow_mode:=true

The node should log a frame counter, action norm, and target range. Shadow mode does not publish commands.

## Optional command path test

Run the monitor in one shell and the node with shadow_mode=false in another. This only tests ROS message flow and does not drive motors without a live whole_body_controller.

    /usr/bin/python3.10 scripts/monitor_supermimic_commands.py
    /usr/bin/python3.10 scripts/supermimic_ros2_node.py --ros-args -p asset_root:=/home/yuuuuu/jlj_robot_repo -p shadow_mode:=false

## Static checks

    /usr/bin/python3.10 scripts/check_deployment_assets.py

The ROS 2 process uses Python 3.10. The separate MuJoCo and ONNX environment uses Python 3.12. Do not activate the Python 3.12 environment while importing ROS 2 Humble.
