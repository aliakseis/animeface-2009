#include <algorithm>
#include <opencv2/opencv.hpp>
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
	const float stride = step / 32.f;
	float scale = min_window_size / 32.0f;
	int ncandidate = 0;
	int nface;
#ifdef _OPENMP
	int threads = omp_get_num_procs();
#else
	int threads = 1;
#endif
	nv_matrix_t **haar = (nv_matrix_t **)malloc(sizeof(nv_matrix_t *) * threads);

	for (int i = 0; i < threads; ++i) {
		haar[i] = nv_matrix_alloc(NV_FACE_HAARLIKE_DIM, 1);
	}
	while (std::min(image_size->width, image_size->height) / scale > min_window_size) {
		float window = (32.0f * scale);
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

			// エッジで枝刈り
			float area = NV_MAT3D_V(edge_integral, roi.y + roi.height, roi.x + roi.width, 0) + NV_MAT3D_V(edge_integral, roi.y, roi.x, 0)
				- NV_MAT3D_V(edge_integral, roi.y + roi.height, roi.x, 0) - NV_MAT3D_V(edge_integral, roi.y, roi.x + roi.width, 0);
			if (area > window*6266.88f - 104448.f || area <= 52224.f || area <= (window - 64)*2611.2f) {
				continue;
			}

			// 特徴量抽出
			nv_face_haarlike(
				NV_NORMALIZE_MAX,
				haar[thread_idx], 0, 
				gray_integral,
				roi.x, roi.y, roi.width, roi.height);

			// 顔方向判定
			int label = nv_mlp_predict_label(dir_mlp, haar[thread_idx]);
			if (!(label == 0 )) {
				continue; // 0 = -30°〜30° 以外だったらはじく
			}

			// 顔判別1
			double z, z1 = nv_mlp_predict_d(detector_mlp, haar[thread_idx]);//
			if (z1 > 0.1) {
				if (bagging_mlps == 0) {
					z = z1;
				} else {
					// 顔判別2
					z = nv_mlp_bagging_predict_d(bagging_mlp, bagging_mlps, haar[thread_idx]);
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
						candidates[ncandidate].parts = NULL;
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
		if (!candidates[i].flag) {
			continue;
		}
		for (int j = i - 1; j >= 0; --j) {
			if (!candidates[j].flag) {
				continue;
			}

			float intersect = (candidates[i].rect & candidates[j].rect).area() / (float)candidates[j].rect.area();
			if (intersect > 0.1f) {
				if (candidates[i].z == candidates[j].z) {
					if (candidates[i].rect.width > candidates[j].rect.width) {
						candidates[i].flag = 0;
						break;
					} else {
						candidates[j].flag = 0;
					}
				} else {
					if (candidates[i].z <= candidates[j].z) {
						candidates[i].flag = 0;
						break;
					} else {
						candidates[j].flag = 0;
					}
				}
			}
		}
	}

	// 部品推定
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1)
#endif
	for (int i = 0; i < std::min(ncandidate, maxface); ++i) {
		if (candidates[i].flag) {
#ifdef _OPENMP
			int thread_idx = omp_get_thread_num();
#else
			int thread_idx = 0;
#endif
			nv_face_haarlike(
				NV_NORMALIZE_NORM,
				haar[thread_idx], 0,
				gray_integral,
				candidates[i].rect.x, candidates[i].rect.y, candidates[i].rect.width, candidates[i].rect.height);
			candidates[i].parts = nv_matrix_alloc(parts_mlp->output, 1);
			nv_mlp_regression(parts_mlp, haar[thread_idx], candidates[i].parts);
		}
	}

	// 結果作成
	nface = 0;
	for (int i = 0; i < ncandidate && i < maxface; ++i) {
		if (candidates[i].flag) {
			float d = candidates[i].rect.width;

			face_pos[nface].likelihood = (float)candidates[i].z;
			face_pos[nface].face = candidates[i].rect;
			face_pos[nface].right_eye.x = cvRound(candidates[i].rect.x + d * (NV_MAT_V(candidates[i].parts, 0, 0) - (NV_MAT_V(candidates[i].parts, 0, 2))));
			face_pos[nface].right_eye.y = cvRound(candidates[i].rect.y + d * (NV_MAT_V(candidates[i].parts, 0, 1) - (NV_MAT_V(candidates[i].parts, 0, 3))));
			face_pos[nface].right_eye.width = cvRound(d * NV_MAT_V(candidates[i].parts, 0, 2) * 2.0f);
			face_pos[nface].right_eye.height = cvRound(d * NV_MAT_V(candidates[i].parts, 0, 3) * 2.0f);
			face_pos[nface].left_eye.x = cvRound(candidates[i].rect.x + d * (NV_MAT_V(candidates[i].parts, 0, 4) - (NV_MAT_V(candidates[i].parts, 0, 6))));
			face_pos[nface].left_eye.y = cvRound(candidates[i].rect.y + d * (NV_MAT_V(candidates[i].parts, 0, 5) - (NV_MAT_V(candidates[i].parts, 0, 7))));
			face_pos[nface].left_eye.width = cvRound(d * NV_MAT_V(candidates[i].parts, 0, 6) * 2.0f);
			face_pos[nface].left_eye.height = cvRound(d * NV_MAT_V(candidates[i].parts, 0, 7) * 2.0f);
			face_pos[nface].nose.x = cvRound(candidates[i].rect.x + d * (NV_MAT_V(candidates[i].parts, 0, 8) - 1.0f * (1.0f / 32.0f)));
			face_pos[nface].nose.y = cvRound(candidates[i].rect.y + d * (NV_MAT_V(candidates[i].parts, 0, 9) - 1.0f * (1.0f / 32.0f)));
			face_pos[nface].nose.width = cvRound(d / 16.f);
			face_pos[nface].nose.height = cvRound(d / 16.f);
			face_pos[nface].mouth.x = cvRound(candidates[i].rect.x + d * (NV_MAT_V(candidates[i].parts, 0, 10) - (NV_MAT_V(candidates[i].parts, 0, 12))));
			face_pos[nface].mouth.y = cvRound(candidates[i].rect.y + d * (NV_MAT_V(candidates[i].parts, 0, 11) - (NV_MAT_V(candidates[i].parts, 0, 13))));
			face_pos[nface].mouth.width = cvRound(d * NV_MAT_V(candidates[i].parts, 0, 12) * 2.0f);
			face_pos[nface].mouth.height = cvRound(d * NV_MAT_V(candidates[i].parts, 0, 13) * 2.0f);
			face_pos[nface].chin.x = cvRound(candidates[i].rect.x + d * (NV_MAT_V(candidates[i].parts, 0, 14) - 1.0f * (1.0f / 32.0f)));
			face_pos[nface].chin.y = cvRound(candidates[i].rect.y + d * (NV_MAT_V(candidates[i].parts, 0, 15) - 1.0f * (1.0f / 32.0f)));
			face_pos[nface].chin.width = cvRound(d / 16.f);
			face_pos[nface].chin.height = cvRound(d / 16.f);
			++nface;
		}
		if (candidates[i].parts != NULL) {
			nv_matrix_free(&candidates[i].parts);
		}
	}

	for (int i = 0; i < threads; ++i) {
		nv_matrix_free(&haar[i]);
	}
	free(haar);
	haar = NULL;

	return nface;
}
