# TD3 PX4 D430 ROS-Base Onboard Planner

English | [中文说明](README_cn.md)

This repository contains a ROS Noetic catkin workspace for a UAV obstacle-avoidance system based on TD3 local planning, optional TSP waypoint ordering, Intel RealSense D430 depth sensing, PX4CTRL, and MAVROS.

It also includes the simulation and training files used during development, so the project can be used in two ways:

1. Onboard runtime in a ROS-base Docker environment.
2. Simulation and TD3 training with the original Diff-Planner environment.

## System Overview

Runtime data flow:

```text
D430 point cloud -> TD3 local planner -> PositionCommand -> PX4CTRL -> MAVROS -> PX4
```

Optional multi-goal mode:

```text
Waypoint set -> TSP ordering -> current local goal -> TD3 local planner -> PositionCommand
```

The TD3 planner consumes odometry, a local depth point cloud, and a target point. It outputs `quadrotor_msgs/PositionCommand` at 20 Hz by default.

## Repository Layout

```text
src/
  airborne_bundle/       Onboard TD3/TSP nodes, launch files, model weights
  px4ctrl/               Low-level controller receiving PositionCommand
  quadrotor_msgs/        Message definitions used by the planner and px4ctrl
  uav_utils/             Utility headers required by px4ctrl

simulation_training/
  training/              TD3 training and actor export scripts
  diff_planner_overlay/  Files to copy into the original Diff-Planner package
```

## Onboard Runtime

### Assumptions

1. PX4 fuses optical flow and GPS internally.
2. MAVROS publishes local odometry on `/mavros/local_position/odom`.
3. The D430 or another depth sensor publishes a point cloud on `/camera/depth/color/points`.
4. The onboard computer communicates with PX4 through serial, such as `/dev/ttyS0:921600` or `/dev/ttyAMA0:921600`.
5. The target platform runs ROS Noetic, usually inside a ROS-base Docker container.

### Dependencies

Inside the ROS container:

```bash
sudo apt update
sudo apt install -y \
  ros-noetic-mavros \
  ros-noetic-mavros-extras \
  ros-noetic-tf \
  ros-noetic-cv-bridge \
  python3-numpy
sudo geographiclib-get-geoids egm96-5
```

If the D430 camera is launched inside the same container, install the RealSense ROS wrapper as well. If another process already publishes `/camera/depth/color/points`, this package only needs to subscribe to that topic.

### Build

Clone this repository as a catkin workspace root:

```bash
git clone <your-repository-url> ~/td3_px4_d430_rosbase
cd ~/td3_px4_d430_rosbase
catkin_make
source devel/setup.bash
```

### Single-Goal TD3 Mode

```bash
roslaunch airborne_bundle airborne_stack.launch \
  fcu_url:=/dev/ttyS0:921600 \
  odom_topic:=/mavros/local_position/odom \
  cloud_topic:=/camera/depth/color/points
```

Publish a local target to `/move_base_simple/goal`. The planner publishes:

```text
/airborne_planner/position_cmd
```

`px4ctrl` receives this command and sends the control output through MAVROS/PX4.

### Multi-Goal TSP + TD3 Mode

```bash
roslaunch airborne_bundle airborne_td3_tsp_d430.launch \
  odom_topic:=/mavros/local_position/odom \
  cloud_topic:=/camera/depth/color/points
```

The TSP manager waits for a trigger from `/move_base_simple/goal`, sorts the configured waypoints, and sends each waypoint to the TD3 local planner. Each local segment is still handled by TD3 obstacle avoidance.

Default waypoints are configured in:

```text
src/airborne_bundle/launch/airborne_td3_tsp_d430.launch
```

## Simulation And Training

The simulation files are stored as an overlay for the original Diff-Planner repository:

```text
simulation_training/diff_planner_overlay/
  scripts/
  launch/
  models/
```

To use them, copy the overlay files into the `diff_planner` ROS package of the original Diff-Planner workspace. In this project, that package corresponds to `src/diff_planner/plan_manage`.

Example:

```bash
cp -a simulation_training/diff_planner_overlay/scripts/* \
  /path/to/Diff-Planner/src/diff_planner/plan_manage/scripts/
cp -a simulation_training/diff_planner_overlay/launch/* \
  /path/to/Diff-Planner/src/diff_planner/plan_manage/launch/
cp -a simulation_training/diff_planner_overlay/models \
  /path/to/Diff-Planner/src/diff_planner/plan_manage/
```

Then rebuild or source the Diff-Planner workspace as needed.

Useful simulation launches:

```bash
roslaunch diff_planner run_td3_forest_depth_demo.launch
roslaunch diff_planner run_td3_tsp_forest_demo.launch
roslaunch diff_planner run_td3_velocity_sim.launch
roslaunch diff_planner run_astar_forest_sim.launch
```

TD3 training scripts:

```text
simulation_training/training/train_td3_velocity_ray.py
simulation_training/training/export_actor_npz.py
```

Example training command:

```bash
cd simulation_training/training
python3 train_td3_velocity_ray.py --episodes 3000 --save-dir runs/td3_velocity_ray
```

After training, export the actor to NumPy weights and replace:

```text
src/airborne_bundle/models/actor_weights.npz
simulation_training/diff_planner_overlay/models/actor_weights.npz
```

The onboard planner uses the NumPy backend by default, so PyTorch is not required on the ARM flight computer.

## Important Topics

```text
/mavros/local_position/odom       nav_msgs/Odometry, input odometry
/camera/depth/color/points        sensor_msgs/PointCloud2, D430 obstacle input
/move_base_simple/goal            geometry_msgs/PoseStamped, goal or trigger input
/airborne_planner/position_cmd    quadrotor_msgs/PositionCommand, planner output
```

## Safety Notes

This repository is intended for controlled research experiments. Before flying on real hardware:

1. Test with propellers removed.
2. Confirm PX4 mode, arming logic, RC failsafe, and emergency stop behavior.
3. Verify that `/mavros/local_position/odom` is stable.
4. Check D430 point cloud direction, frame alignment, and distance scale.
5. Start with low `max_vel`, `max_acc`, and `max_z_vel`.

## Quick Checks

```bash
rostopic hz /mavros/local_position/odom
rostopic hz /camera/depth/color/points
rostopic echo /airborne_planner/position_cmd
rostopic echo /mavros/state
```

