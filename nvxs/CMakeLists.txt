cmake_minimum_required(VERSION 3.0)
project(nvxs)

add_library(nvxs SHARED
    nv_face/nv_face_analyze.cpp
    nv_face/nv_face_haarlike.cpp
    nv_face/nv_face_detect.cpp
    nv_face/nv_face_mlp_static.cpp
    nv_core/nv_core_matrix.cpp
    nv_core/nv_core_util.cpp
    nv_ml/nv_gaussian.cpp
    nv_ml/nv_kmeans.cpp
    nv_ml/nv_mlp.cpp
    nv_num/nv_cov.cpp
    nv_ip/nv_euclidean_color.cpp
    nv_ip/nv_integral.cpp
)

set(NVXS_PUBLIC_HEADER
    nv_face/nv_face.h
    nv_face/nv_face_feature.h
    nv_face/nv_face_mlp_static.h
    nv_face/nv_face_analyze.h
    nv_face/nv_face_detect_internal.h
    nv_face/nv_face_detect.h
    nv_core/nv_core.h
    nv_core/nv_core_matrix.h
    nv_core/nv_core_util.h
    nv_core/nv_portable.h
    nv_ip/nv_ip.h
    nv_ip/nv_ip_euclidean_color.h
    nv_ip/nv_ip_integral.h
    nv_num/nv_num.h
    nv_num/nv_num_cov.h
    nv_ml/nv_ml.h
    nv_ml/nv_ml_gaussian.h
    nv_ml/nv_ml_kmeans.h
    nv_ml/nv_ml_mlp.h
)

set_target_properties(nvxs PROPERTIES PUBLIC_HEADER "${NVXS_PUBLIC_HEADER}")

include_directories(nv_face nv_core nv_ip nv_num nv_ml)

find_package(OpenCV REQUIRED)
target_link_libraries(nvxs "${OpenCV_LIBS}")

find_package(Eigen3 REQUIRED)
target_link_libraries(nvxs Eigen3::Eigen)

find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_executable ( animeface animeface/animeface.cpp )
target_link_libraries ( animeface nvxs )


install(TARGETS nvxs
    RUNTIME DESTINATION lib
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)
