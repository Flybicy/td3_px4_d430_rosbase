#!/usr/bin/env python3
import math
from typing import Optional

import numpy as np
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header

try:
    import cv2
except ImportError:
    cv2 = None


class DepthCloudColorizer:
    def __init__(self):
        self.width = int(rospy.get_param("~cam_width", 640))
        self.height = int(rospy.get_param("~cam_height", 480))
        self.fx = float(rospy.get_param("~cam_fx", 387.229248046875))
        self.fy = float(rospy.get_param("~cam_fy", 387.229248046875))
        self.cx = float(rospy.get_param("~cam_cx", 321.04638671875))
        self.cy = float(rospy.get_param("~cam_cy", 243.44969177246094))
        self.min_depth = float(rospy.get_param("~min_depth", 0.3))
        self.max_depth = float(rospy.get_param("~max_depth", 5.0))
        self.publish_rate = float(rospy.get_param("~publish_rate", 10.0))
        self.max_map_points = int(rospy.get_param("~max_map_points", 180000))
        self.max_project_points = int(rospy.get_param("~max_project_points", 90000))
        self.latch_first_map = bool(rospy.get_param("~latch_first_map", True))
        self.color_map = str(rospy.get_param("~color_map", "jet")).lower()
        self.frame_id = str(rospy.get_param("~frame_id", "camera"))
        self.display_dilate = int(rospy.get_param("~display_dilate", 3))

        # Same camera-to-body rotation used by local_sensing/pcl_render_node.cpp.
        self.cam_to_body = np.array([
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ], dtype=np.float64)

        self.points_world: Optional[np.ndarray] = None
        self.odom: Optional[Odometry] = None

        self.depth_pub = rospy.Publisher("depth", Image, queue_size=1)
        self.color_pub = rospy.Publisher("colordepth", Image, queue_size=1)
        self.cloud_pub = rospy.Publisher("rendered_pcl", PointCloud2, queue_size=1)
        self.map_sub = rospy.Subscriber("global_cloud", PointCloud2, self.map_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber("odom", Odometry, self.odom_callback, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.publish_rate), self.timer_callback)

        rospy.loginfo(
            "[depth_cloud_colorizer] publishing virtual depth image %dx%d, range %.2f-%.2fm",
            self.width, self.height, self.min_depth, self.max_depth)

    def map_callback(self, msg):
        if self.latch_first_map and self.points_world is not None:
            return

        points = []
        for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            points.append((point[0], point[1], point[2]))
            if len(points) >= self.max_map_points:
                break

        if not points:
            rospy.logwarn("[depth_cloud_colorizer] received empty global cloud")
            return

        self.points_world = np.asarray(points, dtype=np.float64)
        rospy.loginfo("[depth_cloud_colorizer] cached %d map points", self.points_world.shape[0])

    def odom_callback(self, msg):
        self.odom = msg

    def timer_callback(self, _event):
        if self.points_world is None or self.odom is None:
            return

        pos = np.array([
            self.odom.pose.pose.position.x,
            self.odom.pose.pose.position.y,
            self.odom.pose.pose.position.z,
        ], dtype=np.float64)
        q = self.odom.pose.pose.orientation
        body_to_world = self.quat_to_rot(q.x, q.y, q.z, q.w)
        cam_to_world = body_to_world.dot(self.cam_to_body)

        depth, camera_points = self.project_depth(pos, cam_to_world)
        stamp = rospy.Time.now()
        self.depth_pub.publish(self.make_depth_msg(depth, stamp))
        self.color_pub.publish(self.make_color_msg(depth, stamp))
        if camera_points.shape[0] > 0:
            self.cloud_pub.publish(self.make_cloud_msg(camera_points, stamp))

    def project_depth(self, camera_pos, cam_to_world):
        points = self.points_world
        rel = points - camera_pos.reshape(1, 3)

        dist2 = np.einsum("ij,ij->i", rel, rel)
        near_mask = dist2 <= (self.max_depth * self.max_depth)
        if not np.any(near_mask):
            return np.zeros((self.height, self.width), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

        rel = rel[near_mask]
        if rel.shape[0] > self.max_project_points:
            stride = int(math.ceil(float(rel.shape[0]) / float(self.max_project_points)))
            rel = rel[::stride]

        # Row-vector form of world-to-camera: p_cam = (p_world - t) * R_cam_to_world.
        points_cam = rel.dot(cam_to_world)
        z = points_cam[:, 2]
        valid = (z > self.min_depth) & (z < self.max_depth)
        if not np.any(valid):
            return np.zeros((self.height, self.width), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

        points_cam = points_cam[valid]
        z = z[valid]
        u = np.rint(self.fx * points_cam[:, 0] / z + self.cx).astype(np.int32)
        v = np.rint(self.fy * points_cam[:, 1] / z + self.cy).astype(np.int32)

        inside = (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        if not np.any(inside):
            return np.zeros((self.height, self.width), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

        u = u[inside]
        v = v[inside]
        z = z[inside].astype(np.float32)
        points_cam = points_cam[inside].astype(np.float32)

        depth = np.full((self.height, self.width), np.inf, dtype=np.float32)
        flat = depth.reshape(-1)
        pixel_index = v * self.width + u
        np.minimum.at(flat, pixel_index, z)
        depth[~np.isfinite(depth)] = 0.0
        return depth, points_cam

    def make_depth_msg(self, depth, stamp):
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.height = self.height
        msg.width = self.width
        msg.encoding = "32FC1"
        msg.is_bigendian = 0
        msg.step = self.width * 4
        msg.data = depth.astype(np.float32, copy=False).tobytes()
        return msg

    def make_color_msg(self, depth, stamp):
        valid = depth > 0.0
        scaled = np.zeros((self.height, self.width), dtype=np.uint8)
        if np.any(valid):
            norm = (depth[valid] - self.min_depth) / max(self.max_depth - self.min_depth, 1e-6)
            # Near obstacles are warm, far surfaces are cool.
            scaled[valid] = np.clip((1.0 - norm) * 255.0, 0, 255).astype(np.uint8)

        if cv2 is not None:
            if self.display_dilate > 1:
                kernel = np.ones((self.display_dilate, self.display_dilate), dtype=np.uint8)
                scaled = cv2.dilate(scaled, kernel, iterations=1)
                valid = cv2.dilate(valid.astype(np.uint8), kernel, iterations=1).astype(bool)
            cmap = cv2.COLORMAP_RAINBOW if self.color_map == "rainbow" else cv2.COLORMAP_JET
            color = cv2.applyColorMap(scaled, cmap)
            color[~valid] = 0
        else:
            color = self.simple_colormap(scaled, valid)

        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.height = self.height
        msg.width = self.width
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = self.width * 3
        msg.data = color.astype(np.uint8, copy=False).tobytes()
        return msg

    def make_cloud_msg(self, camera_points, stamp):
        header = Header()
        header.stamp = stamp
        header.frame_id = self.frame_id
        return pc2.create_cloud_xyz32(header, camera_points.tolist())

    @staticmethod
    def simple_colormap(scaled, valid):
        x = scaled.astype(np.float32) / 255.0
        b = np.clip(1.5 - 4.0 * np.abs(x - 0.0), 0, 1)
        g = np.clip(1.5 - 4.0 * np.abs(x - 0.5), 0, 1)
        r = np.clip(1.5 - 4.0 * np.abs(x - 1.0), 0, 1)
        color = np.dstack((b, g, r))
        color[~valid] = 0
        return (color * 255.0).astype(np.uint8)

    @staticmethod
    def quat_to_rot(x, y, z, w):
        n = x * x + y * y + z * z + w * w
        if n < 1e-12:
            return np.eye(3)
        s = 2.0 / n
        xx, yy, zz = x * x * s, y * y * s, z * z * s
        xy, xz, yz = x * y * s, x * z * s, y * z * s
        wx, wy, wz = w * x * s, w * y * s, w * z * s
        return np.array([
            [1.0 - yy - zz, xy - wz, xz + wy],
            [xy + wz, 1.0 - xx - zz, yz - wx],
            [xz - wy, yz + wx, 1.0 - xx - yy],
        ], dtype=np.float64)


if __name__ == "__main__":
    rospy.init_node("depth_cloud_colorizer")
    DepthCloudColorizer()
    rospy.spin()
