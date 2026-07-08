#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PointStamped, PoseStamped


class ClickedPointGoalBridge:
    def __init__(self):
        self.default_z = float(rospy.get_param("~default_z", 1.2))
        self.min_z = float(rospy.get_param("~min_z", 0.2))
        self.frame_id = str(rospy.get_param("~frame_id", "world"))
        self.goal_pub = rospy.Publisher("goal", PoseStamped, queue_size=1)
        self.clicked_sub = rospy.Subscriber("clicked_point", PointStamped, self.clicked_callback, queue_size=1)
        rospy.loginfo("[clicked_point_goal_bridge] /clicked_point -> /goal, default_z=%.2f", self.default_z)

    def clicked_callback(self, msg):
        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = msg.header.frame_id or self.frame_id
        goal.pose.position.x = msg.point.x
        goal.pose.position.y = msg.point.y
        goal.pose.position.z = msg.point.z if msg.point.z >= self.min_z else self.default_z
        goal.pose.orientation.w = 1.0
        self.goal_pub.publish(goal)
        rospy.logwarn("[clicked_point_goal_bridge] goal from clicked point: %.2f %.2f %.2f",
                      goal.pose.position.x, goal.pose.position.y, goal.pose.position.z)


if __name__ == "__main__":
    rospy.init_node("clicked_point_goal_bridge")
    ClickedPointGoalBridge()
    rospy.spin()
