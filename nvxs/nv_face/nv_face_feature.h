#ifndef __NV_FACE_FEATURE_H
#define __NV_FACE_FEATURE_H
#include "nv_core.h"

#ifdef __cplusplus

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#define NV_FACE_HAARLIKE_DIM 1152

#define NV_INTEGRAL_V(sum, x, y, xw, yh) \
(NV_MAT3D_V((sum), (yh), (xw), 0) \
- NV_MAT3D_V((sum), (yh), (x), 0) \
- (NV_MAT3D_V((sum), (y), (xw), 0) - NV_MAT3D_V((sum), (y), (x), 0))) 

typedef enum {
	NV_NORMALIZE_NONE,
	NV_NORMALIZE_MAX,
	NV_NORMALIZE_NORM
} nv_face_haarlike_normalize_e;

void nv_face_haarlike(nv_face_haarlike_normalize_e type,
	Eigen::Ref<Eigen::Matrix<float, NV_FACE_HAARLIKE_DIM, 1> > feature,
	const nv_matrix_t *sum,
	cv::Rect roi);

#endif

#endif
