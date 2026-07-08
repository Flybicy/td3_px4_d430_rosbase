#!/usr/bin/env python3
import math
from typing import Optional

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from quadrotor_msgs.msg import PositionCommand
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2


class TD3VelocityPlanner:
    def __init__(self):
        self.drone_id = rospy.get_param("~drone_id", 0)
        self.rate_hz = rospy.get_param("~rate", 20.0)
        self.max_vel = rospy.get_param("~max_vel", 1.5)
        self.max_acc = rospy.get_param("~max_acc", 2.0)
        self.max_z_vel = rospy.get_param("~max_z_vel", 0.5)
        self.world_size = rospy.get_param("~world_size", 20.0)
        self.yaw_rays = rospy.get_param("~yaw_rays", 12)
        self.pitch_rays = rospy.get_param("~pitch_rays", 3)
        self.sensor_range = rospy.get_param("~sensor_range", 5.0)
        self.max_cloud_points = rospy.get_param("~max_cloud_points", 1200)
        self.ray_dirs = self.make_ray_directions(self.yaw_rays, self.pitch_rays)
        self.goal_tolerance = rospy.get_param("~goal_tolerance", 0.25)
        self.z_min = rospy.get_param("~z_min", 0.3)
        self.z_max = rospy.get_param("~z_max", 2.8)
        self.goal_z_min = rospy.get_param("~goal_z_min", self.z_min)
        self.default_goal_z = rospy.get_param("~default_goal_z", 1.0)
        self.keep_current_z_if_goal_low = rospy.get_param("~keep_current_z_if_goal_low", True)
        self.debug = rospy.get_param("~debug", False)
        self.command_mode = rospy.get_param("~command_mode", "position_velocity")
        self.policy_backend = rospy.get_param("~policy_backend", "heuristic")
        self.model_path = rospy.get_param("~model_path", "")

        self.odom: Optional[Odometry] = None
        self.goal: Optional[np.ndarray] = None
        self.cloud_points = np.empty((0, 3), dtype=np.float32)
        self.ref_pos: Optional[np.ndarray] = None
        self.last_vel_cmd = np.zeros(3)
        self.last_stamp = rospy.Time.now()
        self.traj_id = 0
        self.plan_path = Path()
        self.last_plan_path_stamp = rospy.Time(0)

        self.actor = None
        if self.policy_backend == "torchscript":
            self.actor = self._load_torchscript_actor(self.model_path)
        elif self.policy_backend == "numpy":
            self.actor = self._load_numpy_actor(self.model_path)
        elif self.policy_backend != "heuristic":
            raise ValueError("Unsupported policy_backend: {}".format(self.policy_backend))

        self.cmd_pub = rospy.Publisher("position_cmd", PositionCommand, queue_size=20)
        self.plan_path_pub = rospy.Publisher("~planned_path", Path, queue_size=1, latch=True)
        self.odom_sub = rospy.Subscriber("odom", Odometry, self.odom_callback, queue_size=1)
        self.goal_sub = rospy.Subscriber("goal", PoseStamped, self.goal_callback, queue_size=1)
        self.cloud_sub = rospy.Subscriber("obstacle_cloud", PointCloud2, self.cloud_callback, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self.timer_callback)

        rospy.loginfo("[td3_velocity_planner] ready, backend=%s, command_mode=%s",
                      self.policy_backend, self.command_mode)

    def _load_torchscript_actor(self, model_path):
        if not model_path:
            raise ValueError("~model_path is required when policy_backend=torchscript")
        try:
            import torch
        except ImportError as exc:
            raise ImportError("PyTorch is required for policy_backend=torchscript") from exc

        actor = torch.jit.load(model_path, map_location="cpu")
        actor.eval()
        return actor

    def _load_numpy_actor(self, model_path):
        if not model_path:
            raise ValueError("~model_path is required when policy_backend=numpy")
        data = np.load(model_path)
        required = ["w0", "b0", "w1", "b1", "w2", "b2", "max_speed"]
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError("Missing numpy actor weights: {}".format(", ".join(missing)))
        return {key: np.asarray(data[key], dtype=np.float32) for key in required}

    def odom_callback(self, msg):
        self.odom = msg
        if self.ref_pos is None:
            self.ref_pos = np.array([
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ], dtype=np.float64)

    def goal_callback(self, msg):
        goal_z = msg.pose.position.z
        if self.keep_current_z_if_goal_low and goal_z < self.goal_z_min:
            if self.odom is not None:
                goal_z = self.odom.pose.pose.position.z
            else:
                goal_z = self.default_goal_z
        goal_z = np.clip(goal_z, self.z_min, self.z_max)

        self.goal = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            goal_z,
        ], dtype=np.float64)
        self.traj_id += 1
        self.plan_path = Path()
        self.last_plan_path_stamp = rospy.Time(0)
        rospy.loginfo("[td3_velocity_planner] new goal: %.2f %.2f %.2f",
                      self.goal[0], self.goal[1], self.goal[2])

    def cloud_callback(self, msg):
        points = []
        for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            points.append(point)
            if len(points) >= self.max_cloud_points:
                break

        if not points:
            self.cloud_points = np.empty((0, 3), dtype=np.float32)
            return

        self.cloud_points = np.asarray(points, dtype=np.float32)

    def timer_callback(self, _event):
        if self.odom is None:
            return

        now = rospy.Time.now()
        dt = max((now - self.last_stamp).to_sec(), 1.0 / self.rate_hz)
        self.last_stamp = now

        pos = np.array([
            self.odom.pose.pose.position.x,
            self.odom.pose.pose.position.y,
            self.odom.pose.pose.position.z,
        ], dtype=np.float64)
        vel = np.array([
            self.odom.twist.twist.linear.x,
            self.odom.twist.twist.linear.y,
            self.odom.twist.twist.linear.z,
        ], dtype=np.float64)

        if self.goal is None:
            self.ref_pos = pos
            self.publish_cmd(now, pos, np.zeros(3), 0.0)
            return

        goal_delta = self.goal - pos
        dist = np.linalg.norm(goal_delta)
        if dist < self.goal_tolerance:
            self.ref_pos = pos
            self.publish_cmd(now, pos, np.zeros(3), self.compute_yaw(self.last_vel_cmd))
            return

        raw_vel_cmd = self.policy(pos, vel, self.goal)
        raw_vel_cmd = self.limit_vertical_velocity(raw_vel_cmd, pos)
        vel_cmd = self.limit_velocity(raw_vel_cmd)
        vel_cmd = self.limit_acceleration(vel_cmd, self.last_vel_cmd, dt)
        vel_cmd = self.limit_vertical_velocity(vel_cmd, pos)
        self.last_vel_cmd = vel_cmd
        if self.debug:
            ray_dist = self.ray_distances(pos)
            rospy.loginfo_throttle(
                1.0,
                "[td3_velocity_planner] dist=%.2f vel_cmd=(%.2f, %.2f, %.2f) nearest_ray=%.2f",
                dist,
                vel_cmd[0],
                vel_cmd[1],
                vel_cmd[2],
                float(np.min(ray_dist)) if ray_dist.size else self.sensor_range,
            )

        if self.command_mode == "velocity_only":
            cmd_pos = np.array([math.nan, math.nan, math.nan])
        else:
            if self.ref_pos is None or not np.all(np.isfinite(self.ref_pos)):
                self.ref_pos = pos
            self.ref_pos = self.ref_pos + vel_cmd * dt
            self.ref_pos[2] = np.clip(self.ref_pos[2], self.z_min, self.z_max)
            cmd_pos = self.ref_pos

        self.publish_cmd(now, cmd_pos, vel_cmd, self.compute_yaw(vel_cmd))
        if self.command_mode != "velocity_only" and np.all(np.isfinite(cmd_pos)):
            self.publish_planned_path(stamp=now, pos=cmd_pos)

    def policy(self, pos, vel, goal):
        state = self.build_observation(pos, vel, goal)

        if self.policy_backend == "heuristic":
            direction = goal - pos
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                return np.zeros(3)
            return direction / norm * min(self.max_vel, norm)

        if self.policy_backend == "numpy":
            return self.numpy_actor_forward(state)

        import torch
        with torch.no_grad():
            tensor = torch.from_numpy(state).unsqueeze(0)
            action = self.actor(tensor).squeeze(0).cpu().numpy()
        return np.asarray(action, dtype=np.float64)

    def numpy_actor_forward(self, state):
        x = np.asarray(state, dtype=np.float32)
        x = np.maximum(x @ self.actor["w0"].T + self.actor["b0"], 0.0)
        x = np.maximum(x @ self.actor["w1"].T + self.actor["b1"], 0.0)
        x = np.tanh(x @ self.actor["w2"].T + self.actor["b2"])
        return np.asarray(x * float(self.actor["max_speed"]), dtype=np.float64)

    def build_observation(self, pos, vel, goal):
        goal_vec = goal - pos
        goal_dist = float(np.linalg.norm(goal_vec))
        obs = [
            *(pos / self.world_size),
            *(vel / self.max_vel),
            *(goal_vec / self.world_size),
            goal_dist / self.world_size,
        ]

        obs.extend(self.ray_distances(pos) / self.sensor_range)
        return np.asarray(obs, dtype=np.float32)

    def make_ray_directions(self, yaw_rays, pitch_rays):
        pitch_min = math.radians(-45.0)
        pitch_max = math.radians(45.0)
        pitches = np.linspace(pitch_min, pitch_max, pitch_rays)
        yaws = np.linspace(-math.pi, math.pi, yaw_rays, endpoint=False)
        dirs = []
        for pitch in pitches:
            cp = math.cos(float(pitch))
            sp = math.sin(float(pitch))
            for yaw in yaws:
                dirs.append(np.array([
                    cp * math.cos(float(yaw)),
                    cp * math.sin(float(yaw)),
                    sp,
                ], dtype=np.float32))
        return np.asarray(dirs, dtype=np.float32)

    def ray_distances(self, pos):
        distances = np.full(len(self.ray_dirs), self.sensor_range, dtype=np.float32)
        if self.cloud_points.size > 0:
            rel_points = self.cloud_points.astype(np.float64) - pos.reshape(1, 3)
            dists = np.linalg.norm(rel_points, axis=1)
            valid = np.where((dists > 1e-3) & (dists <= self.sensor_range))[0]

            if valid.size > 0:
                rel = rel_points[valid]
                valid_dists = dists[valid]
                unit = rel / valid_dists.reshape(-1, 1)
                ray_scores = unit.dot(self.ray_dirs.T)
                ray_indices = np.argmax(ray_scores, axis=1)

                for point_idx, ray_idx in enumerate(ray_indices):
                    if ray_scores[point_idx, ray_idx] > 0.92:
                        distances[ray_idx] = min(distances[ray_idx], valid_dists[point_idx])

        return distances

    def limit_velocity(self, vel_cmd):
        speed = np.linalg.norm(vel_cmd)
        if speed > self.max_vel:
            vel_cmd = vel_cmd / speed * self.max_vel
        return vel_cmd

    def limit_vertical_velocity(self, vel_cmd, pos):
        vel_cmd = np.asarray(vel_cmd, dtype=np.float64).copy()
        vel_cmd[2] = np.clip(vel_cmd[2], -self.max_z_vel, self.max_z_vel)
        if pos[2] <= self.z_min + 0.05 and vel_cmd[2] < 0.0:
            vel_cmd[2] = 0.0
        if pos[2] >= self.z_max - 0.05 and vel_cmd[2] > 0.0:
            vel_cmd[2] = 0.0
        return vel_cmd

    def limit_acceleration(self, vel_cmd, last_vel_cmd, dt):
        delta = vel_cmd - last_vel_cmd
        max_delta = self.max_acc * dt
        delta_norm = np.linalg.norm(delta)
        if delta_norm > max_delta:
            vel_cmd = last_vel_cmd + delta / delta_norm * max_delta
        return vel_cmd

    def compute_yaw(self, vel_cmd):
        if np.linalg.norm(vel_cmd[:2]) < 0.05:
            return 0.0
        return math.atan2(vel_cmd[1], vel_cmd[0])

    def publish_cmd(self, stamp, pos, vel, yaw):
        cmd = PositionCommand()
        cmd.header.stamp = stamp
        cmd.header.frame_id = "world"
        cmd.trajectory_flag = PositionCommand.TRAJECTORY_STATUS_READY
        cmd.trajectory_id = self.traj_id

        cmd.position.x = float(pos[0])
        cmd.position.y = float(pos[1])
        cmd.position.z = float(pos[2])
        cmd.velocity.x = float(vel[0])
        cmd.velocity.y = float(vel[1])
        cmd.velocity.z = float(vel[2])
        cmd.acceleration.x = 0.0
        cmd.acceleration.y = 0.0
        cmd.acceleration.z = 0.0
        cmd.jerk.x = 0.0
        cmd.jerk.y = 0.0
        cmd.jerk.z = 0.0
        cmd.yaw = float(yaw)
        cmd.yaw_dot = 0.0
        self.cmd_pub.publish(cmd)

    def publish_planned_path(self, stamp, pos):
        if (stamp - self.last_plan_path_stamp).to_sec() < 0.10:
            return

        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = "world"
        pose.pose.position.x = float(pos[0])
        pose.pose.position.y = float(pos[1])
        pose.pose.position.z = float(pos[2])
        pose.pose.orientation.w = 1.0

        self.plan_path.header = pose.header
        self.plan_path.poses.append(pose)
        if len(self.plan_path.poses) > 600:
            self.plan_path.poses = self.plan_path.poses[-600:]
        self.plan_path_pub.publish(self.plan_path)
        self.last_plan_path_stamp = stamp


if __name__ == "__main__":
    rospy.init_node("td3_velocity_planner")
    TD3VelocityPlanner()
    rospy.spin()
