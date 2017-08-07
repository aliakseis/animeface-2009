#ifndef __NV_MATRIX_H
#define __NV_MATRIX_H

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
	int list;
	unsigned int list_step;
	int n;
	int m;
	int rows;
	int cols;
	int step;
	int alias;
	float *v;
} nv_matrix_t;

#define NV_MAT_M(mat, row, col) ((mat)->cols * (row) + (col))
#define NV_MAT3D_V(mat, row, col, n) ((mat)->v[NV_MAT_M((mat), (row), (col)) * (mat)->step + (n)])
#define NV_MAT_V(mat, m, n) ((mat)->v[(m) * (mat)->step + (n)])

#define NV_MAT_LIST_V(mat, list, m, n) ((mat)->v[(mat->list_step * list) + (m) * (mat)->step + (n)])

// image option
#define NV_CH_B 0
#define NV_CH_G 1
#define NV_CH_R 2

// matrix
nv_matrix_t *nv_matrix_alloc(int n, int m);
nv_matrix_t *nv_matrix3d_alloc(int n, int rows, int cols);
void nv_matrix_zero(nv_matrix_t *mat);
void nv_vector_zero(nv_matrix_t *mat, int m);
void nv_matrix_free(nv_matrix_t **matrix);
void nv_vector_copy(nv_matrix_t *dest, int dm, const nv_matrix_t *src, int sm);
void nv_matrix_copy(nv_matrix_t *dest, int dm, const nv_matrix_t *src, int sm, int count_m);
void nv_matrix_m(nv_matrix_t *mat, int m);

#ifdef __cplusplus
}
#endif

#endif
