#ifndef __NV_FACE_DETECTION_INTERNAL_H
#define __NV_FACE_DETECTION_INTERNAL_H

#include <opencv/cv.h>
#include <stdio.h>

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

#endif
