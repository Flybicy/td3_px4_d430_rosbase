#!/usr/bin/env python3
import math
from typing import List, Optional

import numpy as np
import rospy
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker


class TD3TSPGoalManager:
    def __init__(self):
        self.goal_frame_id = str(rospy.get_param("~goal_frame_id", "world"))
        self.arrival_tolerance = float(rospy.get_param("~arrival_tolerance", 0.35))
        self.arrival_hold = float(rospy.get_param("~arrival_hold", 0.25))
        self.republish_period = float(rospy.get_param("~republish_period", 1.0))
        self.wait_for_trigger = bool(rospy.get_param("~wait_for_trigger", True))
        self.start_on_launch = bool(rospy.get_param("~start_on_launch", False))
        self.exact_tsp_limit = int(rospy.get_param("~exact_tsp_limit", 12))

        self.odom: Optional[Odometry] = None
        self.route_points: List[np.ndarray] = []
        self.route_order: List[int] = []
        self.preview_order: List[int] = []
        self.preview_points: List[np.ndarray] = []
        self.current_goal_idx = 0
        self.route_active = False
        self.start_requested = self.start_on_launch
        self.goal_reached_stamp: Optional[rospy.Time] = None
        self.last_goal_pub_stamp = rospy.Time(0)

        self.targets = self.load_targets()
        if not self.targets:
            rospy.logwarn("[td3_tsp_goal_manager] no targets configured")

        self.goal_pub = rospy.Publisher("goal_out", PoseStamped, queue_size=1, latch=True)
        self.route_vis_pub = rospy.Publisher("~route_marker", Marker, queue_size=16, latch=True)
        self.odom_sub = rospy.Subscriber("odom", Odometry, self.odom_callback, queue_size=1)
        self.trigger_sub = None
        if self.wait_for_trigger:
            self.trigger_sub = rospy.Subscriber("nav_goal", PoseStamped, self.trigger_callback, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(0.1), self.timer_callback)

        rospy.loginfo(
            "[td3_tsp_goal_manager] targets=%d, wait_for_trigger=%s, start_on_launch=%s",
            len(self.targets), self.wait_for_trigger, self.start_on_launch)
        self.publish_idle_markers()

    def load_targets(self) -> List[np.ndarray]:
        point_num = int(rospy.get_param("~point_num", 0))
        targets = []
        for idx in range(point_num):
            x = self.get_target_param(idx, "x")
            y = self.get_target_param(idx, "y")
            z = self.get_target_param(idx, "z")
            targets.append(np.array([x, y, z], dtype=np.float64))
        return targets

    @staticmethod
    def get_target_param(idx: int, axis: str) -> float:
        target_key = "~target{}_{}".format(idx, axis)
        waypoint_key = "~waypoint{}_{}".format(idx, axis)
        if rospy.has_param(target_key):
            return float(rospy.get_param(target_key))
        return float(rospy.get_param(waypoint_key))

    def odom_callback(self, msg: Odometry):
        self.odom = msg
        if not self.route_active:
            self.publish_idle_markers()
        if self.start_requested and not self.route_active:
            self.start_requested = False
            self.start_route()

    def trigger_callback(self, _msg: PoseStamped):
        if self.odom is None:
            self.start_requested = True
            rospy.logwarn("[td3_tsp_goal_manager] trigger received, waiting for odom")
            return
        self.start_route()

    def timer_callback(self, _event):
        if not self.route_active or self.odom is None or not self.route_points:
            return

        now = rospy.Time.now()
        goal = self.route_points[self.current_goal_idx]
        pos = np.array([
            self.odom.pose.pose.position.x,
            self.odom.pose.pose.position.y,
            self.odom.pose.pose.position.z,
        ], dtype=np.float64)
        dist = np.linalg.norm(goal - pos)

        if (now - self.last_goal_pub_stamp).to_sec() >= self.republish_period:
            self.publish_current_goal()

        if dist > self.arrival_tolerance:
            self.goal_reached_stamp = None
            return

        if self.goal_reached_stamp is None:
            self.goal_reached_stamp = now
            return

        if (now - self.goal_reached_stamp).to_sec() < self.arrival_hold:
            return

        self.goal_reached_stamp = None
        if self.current_goal_idx + 1 < len(self.route_points):
            self.current_goal_idx += 1
            rospy.loginfo(
                "[td3_tsp_goal_manager] switching to route waypoint %d/%d",
                self.current_goal_idx + 1, len(self.route_points))
            self.publish_current_goal()
        else:
            self.route_active = False
            rospy.loginfo("[td3_tsp_goal_manager] route finished")

    def start_route(self):
        if self.odom is None:
            self.start_requested = True
            self.publish_idle_markers()
            return
        if not self.targets:
            rospy.logwarn("[td3_tsp_goal_manager] no targets to start")
            return

        start = np.array([
            self.odom.pose.pose.position.x,
            self.odom.pose.pose.position.y,
            self.odom.pose.pose.position.z,
        ], dtype=np.float64)

        self.route_order = self.solve_order(start, self.targets)
        self.route_points = [self.targets[idx] for idx in self.route_order]
        self.current_goal_idx = 0
        self.route_active = True
        self.goal_reached_stamp = None

        rospy.loginfo(
            "[td3_tsp_goal_manager] optimized order=%s",
            " -> ".join(str(idx) for idx in self.route_order))
        self.publish_route_markers(start)
        self.publish_current_goal()

    def solve_order(self, start: np.ndarray, targets: List[np.ndarray]) -> List[int]:
        if len(targets) <= 1:
            return list(range(len(targets)))
        if len(targets) <= self.exact_tsp_limit:
            return self.solve_exact_open_tsp(start, targets)
        rospy.logwarn(
            "[td3_tsp_goal_manager] target count %d exceeds exact_tsp_limit %d, fallback to greedy 2-opt",
            len(targets), self.exact_tsp_limit)
        return self.solve_greedy_two_opt(start, targets)

    def solve_exact_open_tsp(self, start: np.ndarray, targets: List[np.ndarray]) -> List[int]:
        pts = np.asarray(targets, dtype=np.float64)
        n = pts.shape[0]
        full_mask = 1 << n

        start_cost = np.linalg.norm(pts - start.reshape(1, 3), axis=1)
        pair_cost = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)

        dp = np.full((full_mask, n), np.inf, dtype=np.float64)
        parent = np.full((full_mask, n), -1, dtype=np.int32)

        for idx in range(n):
            dp[1 << idx, idx] = start_cost[idx]

        for mask in range(full_mask):
            for last in range(n):
                if not (mask & (1 << last)):
                    continue
                prev_mask = mask ^ (1 << last)
                if prev_mask == 0:
                    continue
                best_cost = np.inf
                best_prev = -1
                for prev in range(n):
                    if not (prev_mask & (1 << prev)):
                        continue
                    new_cost = dp[prev_mask, prev] + pair_cost[prev, last]
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_prev = prev
                dp[mask, last] = best_cost
                parent[mask, last] = best_prev

        final_mask = full_mask - 1
        last = int(np.argmin(dp[final_mask]))
        order = []
        mask = final_mask
        while last >= 0:
            order.append(last)
            prev = int(parent[mask, last])
            mask ^= (1 << last)
            last = prev
        order.reverse()
        return order

    def solve_greedy_two_opt(self, start: np.ndarray, targets: List[np.ndarray]) -> List[int]:
        pts = np.asarray(targets, dtype=np.float64)
        remaining = set(range(len(targets)))
        order = []
        current = start

        while remaining:
            next_idx = min(remaining, key=lambda idx: np.linalg.norm(pts[idx] - current))
            order.append(next_idx)
            remaining.remove(next_idx)
            current = pts[next_idx]

        improved = True
        while improved:
            improved = False
            for i in range(len(order) - 2):
                for j in range(i + 2, len(order)):
                    candidate = order[:i + 1] + list(reversed(order[i + 1:j + 1])) + order[j + 1:]
                    if self.path_length(start, pts, candidate) + 1e-6 < self.path_length(start, pts, order):
                        order = candidate
                        improved = True
                        break
                if improved:
                    break
        return order

    @staticmethod
    def path_length(start: np.ndarray, pts: np.ndarray, order: List[int]) -> float:
        if not order:
            return 0.0
        total = float(np.linalg.norm(pts[order[0]] - start))
        for idx in range(1, len(order)):
            total += float(np.linalg.norm(pts[order[idx]] - pts[order[idx - 1]]))
        return total

    def estimate_preview_start(self) -> np.ndarray:
        if self.odom is not None:
            return np.array([
                self.odom.pose.pose.position.x,
                self.odom.pose.pose.position.y,
                self.odom.pose.pose.position.z,
            ], dtype=np.float64)

        if self.targets:
            z_mean = float(np.mean([target[2] for target in self.targets]))
        else:
            z_mean = 1.0
        return np.array([0.0, 0.0, z_mean], dtype=np.float64)

    def publish_idle_markers(self):
        if not self.targets:
            return
        start = self.estimate_preview_start()
        self.preview_order = self.solve_order(start, self.targets)
        self.preview_points = [self.targets[idx] for idx in self.preview_order]
        self.publish_route_markers(start, self.preview_points, active_index=None)

    def publish_current_goal(self):
        goal = self.route_points[self.current_goal_idx]
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.goal_frame_id
        msg.pose.position.x = float(goal[0])
        msg.pose.position.y = float(goal[1])
        msg.pose.position.z = float(goal[2])
        msg.pose.orientation.w = 1.0
        self.goal_pub.publish(msg)
        self.last_goal_pub_stamp = msg.header.stamp
        rospy.loginfo(
            "[td3_tsp_goal_manager] publish goal %d/%d: %.2f %.2f %.2f",
            self.current_goal_idx + 1, len(self.route_points),
            goal[0], goal[1], goal[2])

    def make_point(self, xyz: np.ndarray) -> Point:
        pt = Point()
        pt.x = float(xyz[0])
        pt.y = float(xyz[1])
        pt.z = float(xyz[2])
        return pt

    def publish_route_markers(self, start: np.ndarray, route_points: List[np.ndarray], active_index: Optional[int] = None):
        stamp = rospy.Time.now()
        delete_marker = Marker()
        delete_marker.header.stamp = stamp
        delete_marker.header.frame_id = self.goal_frame_id
        delete_marker.action = Marker.DELETEALL
        self.route_vis_pub.publish(delete_marker)

        ordered_points = [start] + list(route_points)

        line_marker = Marker()
        line_marker.header.stamp = stamp
        line_marker.header.frame_id = self.goal_frame_id
        line_marker.ns = "td3_tsp_route"
        line_marker.id = 0
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.pose.orientation.w = 1.0
        line_marker.scale.x = 0.10
        line_marker.color.r = 0.05
        line_marker.color.g = 0.55
        line_marker.color.b = 1.0
        line_marker.color.a = 0.95
        line_marker.points = [self.make_point(pt) for pt in ordered_points]
        self.route_vis_pub.publish(line_marker)

        if route_points:
            current_marker = Marker()
            current_marker.header.stamp = stamp
            current_marker.header.frame_id = self.goal_frame_id
            current_marker.ns = "td3_tsp_current_leg"
            current_marker.id = 10
            current_marker.type = Marker.LINE_STRIP
            current_marker.action = Marker.ADD
            current_marker.pose.orientation.w = 1.0
            current_marker.scale.x = 0.16
            current_marker.color.r = 1.0
            current_marker.color.g = 0.10
            current_marker.color.b = 0.10
            current_marker.color.a = 0.95

            leg_idx = 0 if active_index is None else max(0, min(active_index, len(route_points) - 1))
            leg_start = start if leg_idx == 0 else route_points[leg_idx - 1]
            leg_goal = route_points[leg_idx]
            current_marker.points = [self.make_point(leg_start), self.make_point(leg_goal)]
            self.route_vis_pub.publish(current_marker)

        point_marker = Marker()
        point_marker.header.stamp = stamp
        point_marker.header.frame_id = self.goal_frame_id
        point_marker.ns = "td3_tsp_points"
        point_marker.id = 1
        point_marker.type = Marker.SPHERE_LIST
        point_marker.action = Marker.ADD
        point_marker.pose.orientation.w = 1.0
        point_marker.scale.x = 0.28
        point_marker.scale.y = 0.28
        point_marker.scale.z = 0.28
        point_marker.color.r = 1.0
        point_marker.color.g = 0.55
        point_marker.color.b = 0.05
        point_marker.color.a = 0.95
        point_marker.points = [self.make_point(pt) for pt in route_points]
        self.route_vis_pub.publish(point_marker)

        for seq, pt in enumerate(route_points, start=1):
            text_marker = Marker()
            text_marker.header.stamp = stamp
            text_marker.header.frame_id = self.goal_frame_id
            text_marker.ns = "td3_tsp_labels"
            text_marker.id = 100 + seq
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position = self.make_point(pt)
            text_marker.pose.position.z += 0.45
            text_marker.pose.orientation.w = 1.0
            text_marker.scale.z = 0.45
            text_marker.color.r = 0.05
            text_marker.color.g = 0.05
            text_marker.color.b = 0.05
            text_marker.color.a = 0.95
            text_marker.text = str(seq)
            self.route_vis_pub.publish(text_marker)

        start_marker = Marker()
        start_marker.header.stamp = stamp
        start_marker.header.frame_id = self.goal_frame_id
        start_marker.ns = "td3_tsp_start"
        start_marker.id = 200
        start_marker.type = Marker.SPHERE
        start_marker.action = Marker.ADD
        start_marker.pose.position = self.make_point(start)
        start_marker.pose.orientation.w = 1.0
        start_marker.scale.x = 0.34
        start_marker.scale.y = 0.34
        start_marker.scale.z = 0.34
        start_marker.color.r = 0.15
        start_marker.color.g = 0.85
        start_marker.color.b = 0.15
        start_marker.color.a = 0.95
        self.route_vis_pub.publish(start_marker)

        start_label = Marker()
        start_label.header.stamp = stamp
        start_label.header.frame_id = self.goal_frame_id
        start_label.ns = "td3_tsp_start_label"
        start_label.id = 201
        start_label.type = Marker.TEXT_VIEW_FACING
        start_label.action = Marker.ADD
        start_label.pose.position = self.make_point(start)
        start_label.pose.position.z += 0.50
        start_label.pose.orientation.w = 1.0
        start_label.scale.z = 0.42
        start_label.color.r = 0.0
        start_label.color.g = 0.35
        start_label.color.b = 0.0
        start_label.color.a = 0.95
        start_label.text = "START"
        self.route_vis_pub.publish(start_label)


if __name__ == "__main__":
    rospy.init_node("td3_tsp_goal_manager")
    TD3TSPGoalManager()
    rospy.spin()
