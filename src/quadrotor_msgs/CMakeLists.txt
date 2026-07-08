cmake_minimum_required(VERSION 3.0.2)
project(quadrotor_msgs)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  geometry_msgs
  nav_msgs
  message_generation
)
include_directories(include ${catkin_INCLUDE_DIRS})
add_message_files(FILES
  PositionCommand.msg
  TakeoffLand.msg
  Px4ctrlDebug.msg
  Px4ctrlState.msg
)
generate_messages(DEPENDENCIES geometry_msgs nav_msgs)
catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS geometry_msgs nav_msgs message_runtime
)
install(DIRECTORY include/${PROJECT_NAME}/ DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})
