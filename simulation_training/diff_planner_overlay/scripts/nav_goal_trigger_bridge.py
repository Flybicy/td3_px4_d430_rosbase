#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped


class NavGoalTriggerBridge:
    def __init__(self):
        self.repeat = int(rospy.get_param("~repeat", 2))
        self.interval = float(rospy.get_param("~interval", 0.2))
        self.frame_id = str(rospy.get_param("~frame_id", "world"))

        self.trigger_pub = rospy.Publisher("traj_start_trigger", PoseStamped, queue_size=1)
        self.goal_sub = rospy.Subscriber("nav_goal", PoseStamped, self.goal_callback, queue_size=1)

        rospy.loginfo("[nav_goal_trigger_bridge] /move_base_simple/goal -> /traj_start_trigger")

    def goal_callback(self, msg):
        trigger = PoseStamped()
        trigger.header.stamp = rospy.Time.now()
        trigger.header.frame_id = msg.header.frame_id or self.frame_id
        trigger.pose = msg.pose

        for idx in range(max(1, self.repeat)):
            trigger.header.stamp = rospy.Time.now()
            self.trigger_pub.publish(trigger)
            if idx + 1 < self.repeat:
                rospy.sleep(self.interval)

        rospy.logwarn("[nav_goal_trigger_bridge] start trigger sent from 2D Nav Goal")


if __name__ == "__main__":
    rospy.init_node("nav_goal_trigger_bridge")
    NavGoalTriggerBridge()
    rospy.spin()
