#include "nv_core.h"
#include "nv_num_eigen.h"
#include <lapacke.h>

#define NV_SYMBOL_N "N"
#define NV_SYMBOL_V "V"
#define NV_LAPACK_SPEC_FAST 1


// 対角行列の固有値,固有ベクトルを求める
int nv_eigen_dm(nv_matrix_t *eigen_vec, 
				nv_matrix_t *eigen_val,
				const nv_matrix_t *dmat)
{
	int nb = 10;
	int lwork;
	int vec_step = eigen_vec->step;
	int vec_n = eigen_vec->n;
	float *work;
	int info = 0;

	assert(eigen_vec->n == dmat->n
		&& eigen_vec->m == dmat->m
		&& eigen_val->n == 1
		&& eigen_val->m == eigen_vec->m);
	nv_matrix_zero(eigen_val);
	nv_matrix_copy(eigen_vec, 0, dmat, 0, dmat->m);
	lwork = (nb + 2) * eigen_vec->n;
	work = (float *)malloc(sizeof(float) * lwork);

	ssyev_("V", "U", &vec_n, eigen_vec->v, &vec_step,
		eigen_val->v, work, &lwork, 
		&info);

	free(work);

	return info;
}
