#include "nv_core.h"
#include "nv_num_matrix.h"
#include "nv_num_vector.h"
#include <cblas.h>

// y = (A, x)
void nv_gemv(nv_matrix_t *y, int ym,
			 nv_matrix_tr_t a_tr,
			 const nv_matrix_t *a,
			 const nv_matrix_t *x,
			 int xm)
{
	float alpha = 1.0f;
	float beta = 0.0f;
	int o = 1;
	int a_step = a->step;
	int a_m = a->m;
	int a_n = a->n;

	nv_matrix_zero(y);
	cblas_sgemv(CblasRowMajor, a_tr == NV_MAT_TR ? CblasTrans:CblasNoTrans,
		a_m, a_n, alpha, a->v, a_step,
		&NV_MAT_V(x, xm, 0), o, beta, &NV_MAT_V(y, ym, 0), o);
}
