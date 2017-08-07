#ifndef __NV_NUM_MATRIX_H
#define __NV_NUM_MATRIX_H

#include "nv_core.h"

#ifdef __cplusplus
extern "C" {
#endif

// blas
typedef enum {
	NV_MAT_TR,
	NV_MAT_NOTR
} nv_matrix_tr_t;

void nv_gemv(nv_matrix_t *y, int ym,
			 nv_matrix_tr_t a_tran,
			 const nv_matrix_t *a,
			 const nv_matrix_t *x,
			 int xm);

#ifdef __cplusplus
}
#endif

#endif
