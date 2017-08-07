CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
PROJECT(AnimeFace)

add_library(AnimeFace SHARED animeface.c)
set_target_properties(AnimeFace PROPERTIES PREFIX "" INSTALL_RPATH_USE_LINK_PATH TRUE)

find_package(Ruby REQUIRED)
include_directories("${RUBY_INCLUDE_DIRS}")
target_link_libraries(AnimeFace "${RUBY_LIBRARY}")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../install/include")
get_filename_component(NVXS_LIB_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../install/lib" ABSOLUTE)
target_link_libraries(AnimeFace "${NVXS_LIB_PATH}/libnvxs.so")

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH "Install path prefix, prepended onto install directories." FORCE)
endif()
install(TARGETS AnimeFace LIBRARY DESTINATION ".")
