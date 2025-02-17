#include <algorithm>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "nv_core.h"
#include "nv_ml.h"
#include "nv_ip.h"
#include "nv_num.h"
#include "nv_face_detect.h"
#include "nv_face_feature.h"
#include "nv_face_detect_internal.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#define NV_FACE_DETECT_MAX_CANDIDATES 8192

int 
nv_face_detect(nv_face_position_t *face_pos, 
			   int maxface,
			   const nv_matrix_t *gray_integral, 
			   const nv_matrix_t *edge_integral, 
			   const CvRect *image_size,
			   const nv_mlp_t *dir_mlp,
			   const nv_mlp_t *detector_mlp,
			   const nv_mlp_t **bagging_mlp, int bagging_mlps,
			   const nv_mlp_t *parts_mlp,
			   float step,
			   float scale_factor,
			   float min_window_size,
			   float threshold
			   )
{
	nv_candidate candidates[NV_FACE_DETECT_MAX_CANDIDATES] = {}; // max
	const float stride = step / 32.F;
	float scale = min_window_size / 32.0F;
	int ncandidate = 0;
	int nface;
#ifdef _OPENMP
	int threads = omp_get_num_procs();
#else
	int threads = 1;
#endif
	while (std::min(image_size->width, image_size->height) / scale > min_window_size) {
		float window = (32.0F * scale);
		int ye = cvRound((image_size->height - window) / (stride * window));
		int xe = cvRound((image_size->width - window) / (stride * window));

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1)
#endif
		for (int i = 0; i < ye * xe; ++i) {
#ifdef _OPENMP
			int thread_idx = omp_get_thread_num();
#else
			int thread_idx = 0;
#endif
			int yi = i / xe;
			int xi = i % xe;
			float y = yi * (image_size->height - window) / ye;
			float x = xi * (image_size->width - window) / xe;
			cv::Rect roi(cvRound(x), cvRound(y), cvRound(x + window) - cvRound(x), cvRound(y + window) - cvRound(y));
			Eigen::Matrix<float, NV_FACE_HAARLIKE_DIM, 1> haar;

			// エッジで枝刈り
			float area = NV_MAT3D_V(edge_integral, roi.y + roi.height, roi.x + roi.width, 0) + NV_MAT3D_V(edge_integral, roi.y, roi.x, 0)
				- NV_MAT3D_V(edge_integral, roi.y + roi.height, roi.x, 0) - NV_MAT3D_V(edge_integral, roi.y, roi.x + roi.width, 0);
			if (area > window*6266.88F - 104448.F || area <= 52224.F || area <= (window - 64)*2611.2F) {
				continue;
			}

			// 特徴量抽出
			nv_face_haarlike(NV_NORMALIZE_MAX, haar, gray_integral, roi);

			// 顔方向判定
			int label = nv_mlp_predict_label(dir_mlp, haar);
			if (!(label == 0 )) {
				continue; // 0 = -30°〜30° 以外だったらはじく
			}

			// 顔判別1
			double z;
			double z1 = nv_mlp_predict_d(detector_mlp, haar);
			if (z1 > 0.1) {
				if (bagging_mlps == 0) {
					z = z1;
				} else {
					// 顔判別2
					z = nv_mlp_bagging_predict_d(bagging_mlp, bagging_mlps, haar);
				}
				if (z > threshold) {
					// 顔
#ifdef _OPENMP
#pragma omp critical
#endif
					if (NV_FACE_DETECT_MAX_CANDIDATES > ncandidate) {
						candidates[ncandidate].rect = roi;
						candidates[ncandidate].z = z;
						candidates[ncandidate].flag = 1;
						candidates[ncandidate].parts = nullptr;
						++ncandidate;
					}
				}
			}
			
		}
		scale *= scale_factor;
	}

	// 重複領域の除去
	qsort(candidates, ncandidate, sizeof(nv_candidate), nv_candidate_cmp);
	for (int i = ncandidate-1; i >= 0; --i) {
		if (candidates[i].flag == 0) {
			continue;
		}
		for (int j = i - 1; j >= 0; --j) {
			if (candidates[j].flag == 0) {
				continue;
			}

			float intersect = (candidates[i].rect & candidates[j].rect).area() / static_cast<float>(candidates[j].rect.area());
			if (intersect > 0.1F) {
				if (candidates[i].z == candidates[j].z) {
					if (candidates[i].rect.width > candidates[j].rect.width) {
						candidates[i].flag = 0;
						break;
					} 
						candidates[j].flag = 0;
					
				} else {
					if (candidates[i].z <= candidates[j].z) {
						candidates[i].flag = 0;
						break;
					} 
						candidates[j].flag = 0;
					
				}
			}
		}
	}

	// 部品推定
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1)
#endif
	for (int i = 0; i < std::min(ncandidate, maxface); ++i) {
		if (candidates[i].flag != 0) {
#ifdef _OPENMP
			int thread_idx = omp_get_thread_num();
#else
			int thread_idx = 0;
#endif
			Eigen::Matrix<float, NV_FACE_HAARLIKE_DIM, 1> haar;
			nv_face_haarlike(NV_NORMALIZE_NORM, haar, gray_integral, candidates[i].rect);
			candidates[i].parts = nv_matrix_alloc(parts_mlp->output, 1);
			nv_mlp_regression(parts_mlp, haar, candidates[i].parts);
		}
	}

	// 結果作成
	nface = 0;
	for (int i = 0; i < ncandidate && i < maxface; ++i) {
		if (candidates[i].flag != 0) {
			float d = candidates[i].rect.width;

			face_pos[nface].likelihood = static_cast<float>(candidates[i].z);
			face_pos[nface].face = cvRect(candidates[i].rect);
			face_pos[nface].right_eye.x = cvRound(candidates[i].rect.x + d * (NV_MAT_V(candidates[i].parts, 0, 0) - (NV_MAT_V(candidates[i].parts, 0, 2))));
			face_pos[nface].right_eye.y = cvRound(candidates[i].rect.y + d * (NV_MAT_V(candidates[i].parts, 0, 1) - (NV_MAT_V(candidates[i].parts, 0, 3))));
			face_pos[nface].right_eye.width = cvRound(d * NV_MAT_V(candidates[i].parts, 0, 2) * 2.0F);
			face_pos[nface].right_eye.height = cvRound(d * NV_MAT_V(candidates[i].parts, 0, 3) * 2.0F);
			face_pos[nface].left_eye.x = cvRound(candidates[i].rect.x + d * (NV_MAT_V(candidates[i].parts, 0, 4) - (NV_MAT_V(candidates[i].parts, 0, 6))));
			face_pos[nface].left_eye.y = cvRound(candidates[i].rect.y + d * (NV_MAT_V(candidates[i].parts, 0, 5) - (NV_MAT_V(candidates[i].parts, 0, 7))));
			face_pos[nface].left_eye.width = cvRound(d * NV_MAT_V(candidates[i].parts, 0, 6) * 2.0F);
			face_pos[nface].left_eye.height = cvRound(d * NV_MAT_V(candidates[i].parts, 0, 7) * 2.0F);
			face_pos[nface].nose.x = cvRound(candidates[i].rect.x + d * (NV_MAT_V(candidates[i].parts, 0, 8) - 1.0F * (1.0F / 32.0F)));
			face_pos[nface].nose.y = cvRound(candidates[i].rect.y + d * (NV_MAT_V(candidates[i].parts, 0, 9) - 1.0F * (1.0F / 32.0F)));
			face_pos[nface].nose.width = cvRound(d / 16.F);
			face_pos[nface].nose.height = cvRound(d / 16.F);
			face_pos[nface].mouth.x = cvRound(candidates[i].rect.x + d * (NV_MAT_V(candidates[i].parts, 0, 10) - (NV_MAT_V(candidates[i].parts, 0, 12))));
			face_pos[nface].mouth.y = cvRound(candidates[i].rect.y + d * (NV_MAT_V(candidates[i].parts, 0, 11) - (NV_MAT_V(candidates[i].parts, 0, 13))));
			face_pos[nface].mouth.width = cvRound(d * NV_MAT_V(candidates[i].parts, 0, 12) * 2.0F);
			face_pos[nface].mouth.height = cvRound(d * NV_MAT_V(candidates[i].parts, 0, 13) * 2.0F);
			face_pos[nface].chin.x = cvRound(candidates[i].rect.x + d * (NV_MAT_V(candidates[i].parts, 0, 14) - 1.0F * (1.0F / 32.0F)));
			face_pos[nface].chin.y = cvRound(candidates[i].rect.y + d * (NV_MAT_V(candidates[i].parts, 0, 15) - 1.0F * (1.0F / 32.0F)));
			face_pos[nface].chin.width = cvRound(d / 16.F);
			face_pos[nface].chin.height = cvRound(d / 16.F);
			++nface;
		}
		if (candidates[i].parts != nullptr) {
			nv_matrix_free(&candidates[i].parts);
		}
	}

	return nface;
}
