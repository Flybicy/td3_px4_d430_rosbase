#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped


def main():
    rospy.init_node("auto_traj_start_trigger")

    delay = float(rospy.get_param("~delay", 3.0))
    repeat = int(rospy.get_param("~repeat", 3))
    interval = float(rospy.get_param("~interval", 0.5))

    pub = rospy.Publisher("/traj_start_trigger", PoseStamped, queue_size=1, latch=True)

    rospy.sleep(delay)

    msg = PoseStamped()
    msg.header.frame_id = "world"

    for _ in range(max(1, repeat)):
      if rospy.is_shutdown():
        return
      msg.header.stamp = rospy.Time.now()
      pub.publish(msg)
      rospy.sleep(interval)

    rospy.loginfo("[auto_traj_start_trigger] start trigger published")


if __name__ == "__main__":
    main()
