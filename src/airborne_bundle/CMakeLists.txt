cmake_minimum_required(VERSION 3.0.2)
project(airborne_bundle)
find_package(catkin REQUIRED COMPONENTS
  rospy
  geometry_msgs
  nav_msgs
  sensor_msgs
  visualization_msgs
  quadrotor_msgs
)
catkin_package()
catkin_install_python(PROGRAMS
  scripts/td3_velocity_planner.py
  scripts/td3_tsp_goal_manager.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY launch config models
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
