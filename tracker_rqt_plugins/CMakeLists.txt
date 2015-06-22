cmake_minimum_required(VERSION 2.8.3)

project(tracker_rqt_plugins)
# Load catkin and all dependencies required for this package
find_package(catkin REQUIRED COMPONENTS rqt_gui rqt_gui_cpp image_transport sensor_msgs cv_bridge rosconsole)

find_package(Qt4 COMPONENTS QtCore QtGui REQUIRED)

include(${QT_USE_FILE})

set(rqt_image_view_SRCS
  src/rqt_image_view/image_view.cpp
  src/rqt_image_view/ratio_layouted_frame.cpp
)

set(rqt_image_view_HDRS
  include/rqt_image_view/image_view.h
  include/rqt_image_view/ratio_layouted_frame.h
)

set(rqt_image_view_UIS
  src/rqt_image_view/image_view.ui
)

set(rqt_image_cropping_SRCS
  src/rqt_image_cropping/image_cropper.cpp
  src/rqt_image_cropping/ratio_layouted_frame.cpp
)

set(rqt_image_cropping_HDRS
  include/rqt_image_cropping/image_cropper.h
  include/rqt_image_cropping/ratio_layouted_frame.h
)

set(rqt_image_cropping_UIS
  src/rqt_image_cropping/image_cropper.ui
)

set(rqt_flipper_control_SRCS
  src/rqt_flipper_control/flipper_control.cpp
  src/rqt_flipper_control/ratio_layouted_frame.cpp
)

set(rqt_flipper_control_HDRS
  include/rqt_flipper_control/flipper_control.h
  include/rqt_flipper_control/ratio_layouted_frame.h
)

set(rqt_flipper_control_UIS
  src/rqt_flipper_control/flipper_control.ui
)

set(tracker_rqt_plugins_INCLUDE_DIRECTORIES
  include
  ${CMAKE_CURRENT_BINARY_DIR}
)

catkin_package(
	INCLUDE_DIRS include
	LIBRARIES ${PROJECT_NAME} 
	CATKIN_DEPENDS rqt_gui rqt_gui_cpp image_transport sensor_msgs cv_bridge rosconsole
)
catkin_python_setup()

# show additional files in qtcreator
execute_process(COMMAND find ${PROJECT_SOURCE_DIR} -type f
  OUTPUT_VARIABLE FILES_STRING)
string(REPLACE "\n" ";" FILES_LIST ${FILES_STRING})
add_custom_target(additional_files_${PROJECT_NAME} SOURCES
  ${FILES_LIST}
  ${PROJECT_SOURCE_DIR}/../README.md
)

qt4_wrap_cpp(rqt_image_view_MOCS ${rqt_image_view_HDRS})
qt4_wrap_ui(rqt_image_view_UIS_H ${rqt_image_view_UIS})

qt4_wrap_cpp(rqt_image_cropping_MOCS ${rqt_image_cropping_HDRS})
qt4_wrap_ui(rqt_image_cropping_UIS_H ${rqt_image_cropping_UIS})

qt4_wrap_cpp(rqt_flipper_control_MOCS ${rqt_flipper_control_HDRS})
qt4_wrap_ui(rqt_flipper_control_UIS_H ${rqt_flipper_control_UIS})

include_directories(${tracker_rqt_plugins_INCLUDE_DIRECTORIES} ${catkin_INCLUDE_DIRS})
add_library(${PROJECT_NAME}
		${rqt_image_view_SRCS} ${rqt_image_view_MOCS} ${rqt_image_view_UIS_H}
                ${rqt_image_cropping_SRCS} ${rqt_image_cropping_MOCS} ${rqt_image_cropping_UIS_H}
                ${rqt_flipper_control_SRCS} ${rqt_flipper_control_MOCS} ${rqt_flipper_control_UIS_H})
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${QT_QTCORE_LIBRARY} ${QT_QTGUI_LIBRARY})

find_package(class_loader)
class_loader_hide_library_symbols(${PROJECT_NAME})

install(FILES plugin.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})
 
install(PROGRAMS scripts/rqt_image_enhancer
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/
  DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION}
)
