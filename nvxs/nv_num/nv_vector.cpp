#include <Eigen/Dense>
#include "nv_core.h"
#include "nv_num_distance.h"
#include "nv_num_vector.h"

int nv_vector_maxsum_m(const nv_matrix_t *v)
{
	int m;
	Eigen::Map<Eigen::MatrixXf>(v->v, v->n, v->m).colwise().sum().maxCoeff(&m);
	return m;
}
