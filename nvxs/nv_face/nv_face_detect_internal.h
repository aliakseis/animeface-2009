#ifndef __NV_FACE_DETECTION_INTERNAL_H
#define __NV_FACE_DETECTION_INTERNAL_H

#include <opencv/cv.h>

typedef struct candidate {
	cv::Rect rect;
	double z;
	int flag;
	nv_matrix_t *parts;
} nv_candidate;

static int nv_candidate_cmp(const void *p1, const void *p2)
{
	nv_candidate *e1 = (nv_candidate *)p1;
	nv_candidate *e2 = (nv_candidate *)p2;

	if (e1->rect.width != e2->rect.width) {
		return e1->rect.width - e2->rect.width;
	} else {
		return e1->rect.height - e2->rect.height;
	}
}


static int nv_is_face_edge(int window, float scale, float area) 
{
	float v = area / (255.0f * window * window) * scale * scale * 0.5f;
	if (window < 84.0f) {
		if (0.1f < v && v < window * 0.012f - 0.2f) {
			return 1;
		}
	} else {
		if ((window - 64.0f) * 0.005f < v && v < window * 0.012f - 0.2f) {
			return 1;
		}
	}
	return 0;
}

#endif
