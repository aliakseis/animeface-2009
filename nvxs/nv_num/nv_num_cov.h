#ifndef __NV_NUM_COV_H
#define __NV_NUM_COV_H

#ifdef __cplusplus
extern "C" {
#endif

// •ªU‹¤•ªUs—ñ
typedef struct {
	int n;
	nv_matrix_t *u;        // •½‹Ï
	nv_matrix_t *cov;      // s•ªUs—ñ
} nv_cov_t;

nv_cov_t *nv_cov_alloc(int n);
void nv_cov(nv_matrix_t *cov,
			nv_matrix_t *u,
			const nv_matrix_t *data);
void nv_cov_eigen(nv_cov_t *cov, const nv_matrix_t *data);
void nv_cov_free(nv_cov_t **cov);

#ifdef __cplusplus
}
#endif


#endif
