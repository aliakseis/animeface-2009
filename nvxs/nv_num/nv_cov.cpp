#include "nv_core.h"
#include "nv_num_cov.h"

// 分散共分散

nv_cov_t *nv_cov_alloc(int n)
{
	nv_cov_t *cov = (nv_cov_t *)malloc(sizeof(nv_cov_t));

	cov->n = n;
	cov->u = nv_matrix_alloc(n, 1);
	cov->cov = nv_matrix_alloc(n, n);

	return cov;
}

void nv_cov_free(nv_cov_t **cov)
{
	nv_matrix_free(&(*cov)->cov);
	nv_matrix_free(&(*cov)->u);

	free(*cov);
	*cov = NULL;
}

void nv_cov_eigen(nv_cov_t *cov, const nv_matrix_t *data)
{
	nv_cov(cov->cov, cov->u, data);
}

void nv_cov(nv_matrix_t *cov,
			nv_matrix_t *u,
			const nv_matrix_t *data)
{
	int m, n, i;
	int alloc_u = 0;
	float factor = 1.0f / data->m;

	if (u == NULL) {
		u = nv_matrix_alloc(cov->n, 1);
		alloc_u =1;
	}
	assert(cov->n == data->n && cov->n == cov->m
		&& u->n == cov->n);

	// 平均
	nv_matrix_zero(u);
	for (m = 0; m < data->m; ++m) {
		for (n = 0; n < data->n; ++n) {
			NV_MAT_V(u, 0, n) += NV_MAT_V(data, m, n) * factor;
		}
	}

	// 上三角 分散共分散行列
	nv_matrix_zero(cov);
	for (n = 0; n < cov->n; ++n) {
		for (m = n; m < cov->m; ++m) {
			float v = 0.0f;
			for (i = 0; i < data->m; ++i) {
				float a = NV_MAT_V(data, i, n) - NV_MAT_V(u, 0, n);
				float b = NV_MAT_V(data, i, m) - NV_MAT_V(u, 0, m);
				v += a * b * factor;
			}
			NV_MAT_V(cov, m, n) = v;
		}
	}

	if (alloc_u) {
		nv_matrix_free(&u);
	}
}
