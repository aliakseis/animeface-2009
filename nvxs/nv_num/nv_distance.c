#include "nv_core.h"
#include "nv_num.h"
#include "nv_num_distance.h"

// ユークリッド距離^2
float nv_euclidean2(const nv_matrix_t *vec1, int m1, const nv_matrix_t *vec2, int m2)
{
	int n;
	float dist = 0.0f;

	assert(vec1->n == vec2->n);

	for (n = 0; n < vec1->n; ++n) {
		dist += (NV_MAT_V(vec1, m1, n) - NV_MAT_V(vec2, m2, n))
			* (NV_MAT_V(vec1, m1, n) - NV_MAT_V(vec2, m2, n));
	}
	return dist;
}
