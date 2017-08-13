#include <Eigen/Dense>
#include "nv_core.h"
#include "nv_num_distance.h"
#include "nv_num_vector.h"
#if NV_ENABLE_SSE2
#include <emmintrin.h>
#include <xmmintrin.h>
#endif

float nv_vector_dot(const nv_matrix_t *vec1, int m1,
					const nv_matrix_t *vec2, int m2)
{
	assert(vec1->n == vec2->n);
	return Eigen::Map<Eigen::VectorXf>(&vec1->v[m1*vec1->n], vec1->n).dot(Eigen::Map<Eigen::VectorXf>(&vec2->v[m2*vec2->n], vec2->n));
}

int nv_vector_maxsum_m(const nv_matrix_t *v)
{
	int m;
	Eigen::Map<Eigen::MatrixXf>(v->v, v->n, v->m).colwise().sum().maxCoeff(&m);
	return m;
}
