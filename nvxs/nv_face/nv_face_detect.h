#ifndef __NV_FACE_DETECTION_H
#define __NV_FACE_DETECTION_H
#include "nv_core.h"
#include "nv_ml.h"
#include <opencv2/opencv.hpp>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	float likelihood;
	cv::Rect face;
	cv::Rect left_eye;
	cv::Rect right_eye;
	cv::Rect nose;
	cv::Rect mouth;
	cv::Rect chin;
} nv_face_position_t;

int nv_face_detect(nv_face_position_t *face_pos, 
				   int maxface,
				   const nv_matrix_t *gray_integral, 
				   const nv_matrix_t *edge_integral, 
				   const cv::Rect *image_size,
				   const nv_mlp_t *dir_mlp,
				   const nv_mlp_t *detector_mlp,
				   const nv_mlp_t **bagging_mlp, int bagging_mlps,
				   const nv_mlp_t *parts_mlp,
				   float step,
				   float scale_factor,
				   float min_window_size,
				   float threshold
				   );

#if NV_ENABLE_CUDA
#include "nv_face_detect_gpu.h"
#endif

#ifdef __cplusplus
}
#endif

#endif
