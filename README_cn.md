# TD3 PX4 D430 ROS-Base 机载规划系统

[English](README.MD) | 中文说明

本仓库整理了一套面向无人机机载电脑的 ROS Noetic catkin 工作空间源码，核心功能包括：TD3 局部避障规划、可选 TSP 多目标点排序、Intel RealSense D430 深度点云感知、PX4CTRL 控制接口以及 MAVROS 与 PX4 飞控通信。

仓库同时包含原先开发中使用的仿真和训练部分，因此可以用于两个场景：

1. 在 ROS-base Docker 环境中进行机载部署。
2. 在原 Diff-Planner 仿真环境中进行树林场景展示和 TD3 训练。

## 系统总体流程

机载运行主链路：

```text
D430 深度点云 -> TD3 局部规划器 -> PositionCommand -> PX4CTRL -> MAVROS -> PX4
```

多目标点任务模式：

```text
目标点集合 -> TSP 排序 -> 当前子目标 -> TD3 局部规划器 -> PositionCommand
```

TD3 规划器接收里程计、局部深度点云和目标点，默认以 20 Hz 输出 `quadrotor_msgs/PositionCommand`。

## 仓库结构

```text
src/
  airborne_bundle/       机载 TD3/TSP 节点、launch 文件、模型权重
  px4ctrl/               接收 PositionCommand 的底层控制器
  quadrotor_msgs/        规划器和 px4ctrl 使用的消息定义
  uav_utils/             px4ctrl 依赖的工具头文件

simulation_training/
  training/              TD3 训练与 actor 导出脚本
  diff_planner_overlay/  复制到原 Diff-Planner 工程中的仿真补丁文件
```

## 机载运行

### 运行假设

1. PX4 内部已经融合光流和 GPS。
2. MAVROS 在 `/mavros/local_position/odom` 发布本地里程计。
3. D430 或其他深度相机在 `/camera/depth/color/points` 发布点云。
4. 机载电脑通过串口连接 PX4，例如 `/dev/ttyS0:921600` 或 `/dev/ttyAMA0:921600`。
5. 机载环境使用 ROS Noetic，通常运行在 ROS-base Docker 容器中。

### 依赖安装

在 ROS 容器内安装：

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

如果 D430 也在同一个容器中启动，还需要安装 RealSense ROS wrapper。如果点云已经由宿主机或其他容器发布，本仓库只需要订阅 `/camera/depth/color/points`。

### 编译

将本仓库作为 catkin 工作空间根目录克隆：

```bash
git clone <你的仓库地址> ~/td3_px4_d430_rosbase
cd ~/td3_px4_d430_rosbase
catkin_make
source devel/setup.bash
```

### 单目标 TD3 模式

```bash
roslaunch airborne_bundle airborne_stack.launch \
  fcu_url:=/dev/ttyS0:921600 \
  odom_topic:=/mavros/local_position/odom \
  cloud_topic:=/camera/depth/color/points
```

向 `/move_base_simple/goal` 发布局部目标点后，TD3 规划器会输出：

```text
/airborne_planner/position_cmd
```

`px4ctrl` 订阅该指令，并通过 MAVROS/PX4 控制链路执行。

### 多目标 TSP + TD3 模式

```bash
roslaunch airborne_bundle airborne_td3_tsp_d430.launch \
  odom_topic:=/mavros/local_position/odom \
  cloud_topic:=/camera/depth/color/points
```

TSP 管理节点会等待 `/move_base_simple/goal` 触发，然后对预设目标点进行排序，并依次发送给 TD3 局部规划器。每一段子目标之间的避障仍然由 TD3 完成。

默认目标点配置在：

```text
src/airborne_bundle/launch/airborne_td3_tsp_d430.launch
```

## 仿真与训练

仿真文件以 overlay 形式保存：

```text
simulation_training/diff_planner_overlay/
  scripts/
  launch/
  models/
```

使用时，将 overlay 内容复制到原 Diff-Planner 工程中的 `diff_planner` ROS 包内。本项目中对应位置是 `src/diff_planner/plan_manage`。

示例：

```bash
cp -a simulation_training/diff_planner_overlay/scripts/* \
  /path/to/Diff-Planner/src/diff_planner/plan_manage/scripts/
cp -a simulation_training/diff_planner_overlay/launch/* \
  /path/to/Diff-Planner/src/diff_planner/plan_manage/launch/
cp -a simulation_training/diff_planner_overlay/models \
  /path/to/Diff-Planner/src/diff_planner/plan_manage/
```

然后重新编译或 source 原 Diff-Planner 工作空间。

常用仿真启动命令：

```bash
roslaunch diff_planner run_td3_forest_depth_demo.launch
roslaunch diff_planner run_td3_tsp_forest_demo.launch
roslaunch diff_planner run_td3_velocity_sim.launch
roslaunch diff_planner run_astar_forest_sim.launch
```

TD3 训练脚本：

```text
simulation_training/training/train_td3_velocity_ray.py
simulation_training/training/export_actor_npz.py
```

训练示例：

```bash
cd simulation_training/training
python3 train_td3_velocity_ray.py --episodes 3000 --save-dir runs/td3_velocity_ray
```

训练完成后，将 actor 导出为 NumPy 权重，并替换：

```text
src/airborne_bundle/models/actor_weights.npz
simulation_training/diff_planner_overlay/models/actor_weights.npz
```

机载规划器默认使用 NumPy 后端，因此 ARM 机载电脑上不强制依赖 PyTorch。

## 关键话题

```text
/mavros/local_position/odom       nav_msgs/Odometry，输入里程计
/camera/depth/color/points        sensor_msgs/PointCloud2，D430 障碍物点云输入
/move_base_simple/goal            geometry_msgs/PoseStamped，目标点或触发输入
/airborne_planner/position_cmd    quadrotor_msgs/PositionCommand，规划器输出
```

## 安全注意事项

本仓库用于受控实验研究。真实飞行前请确认：

1. 先拆桨测试所有话题和控制链路。
2. 确认 PX4 模式、解锁逻辑、遥控器失控保护和急停方式。
3. 确认 `/mavros/local_position/odom` 稳定可靠。
4. 检查 D430 点云方向、坐标系对齐和距离尺度。
5. 初次测试时降低 `max_vel`、`max_acc` 和 `max_z_vel`。

## 快速检查

```bash
rostopic hz /mavros/local_position/odom
rostopic hz /camera/depth/color/points
rostopic echo /airborne_planner/position_cmd
rostopic echo /mavros/state
```

