#include "nv_core.h"
#include "nv_core_matrix.h"

#if (NV_ENABLE_CUDA && NV_GPU_PIN_MALLOC)
#include <cuda_runtime.h>
#endif


static void *nv_malloc(unsigned long n)
{
	void *mem;
#if (NV_ENABLE_CUDA && NV_GPU_PIN_MALLOC)
	if (nv_gpu_available()) {
		cudaMallocHost(&mem, n);
	} else {
		mem = malloc(n);
	}
#else
	mem = malloc(n);
#endif
	return mem;
}

nv_matrix_t *nv_matrix_alloc(int n, int m)
{
	void *mem;
	
	int step = n * sizeof(float);//(n + 4 - (n & 3)) * sizeof(float); // SSE2
	int mem_size = step * m + sizeof(nv_matrix_t) + 0x10;
	nv_matrix_t *matrix = (nv_matrix_t *)nv_malloc(mem_size);

	if (matrix == NULL) {
		return NULL;
	}
	mem = ((char *)matrix) + sizeof(nv_matrix_t);
	matrix->v = (float *)(((char *)mem) + 0x10 - ((size_t)mem & 0xf));

	matrix->list = 1;

	matrix->n = n;
	matrix->m = m;
	matrix->cols = m;
	matrix->rows = 1;
	matrix->step = step / sizeof(float);
	matrix->alias = 0;
	matrix->list_step = matrix->step * matrix->m;

	return matrix;
}

nv_matrix_t *nv_matrix3d_alloc(int n, int rows, int cols)
{
	nv_matrix_t *mat = nv_matrix_alloc(n, rows * cols);
	mat->rows = rows;
	mat->cols = cols;
	
	return mat;
}

void nv_matrix_zero(nv_matrix_t *mat)
{
	if (mat->list > 1) {
		memset(mat->v, 0, mat->list_step * mat->list * sizeof(float));
	} else {
		memset(mat->v, 0, mat->m * mat->step * sizeof(float));
	}
}

void nv_vector_zero(nv_matrix_t *mat, int m)
{
	memset(&NV_MAT_V(mat, m, 0), 0, mat->step * sizeof(float));
}

void nv_matrix_copy(nv_matrix_t *dest, int dm, const nv_matrix_t *src, int sm, int count_m)
{
	assert(dest->n == src->n);
	memmove(&NV_MAT_V(dest, dm, 0), &NV_MAT_V(src, sm, 0), dest->step * count_m * sizeof(float));
}

void nv_matrix_free(nv_matrix_t **matrix)
{
	if (*matrix != NULL) {
		free(*matrix);
		*matrix = NULL;
	}
}

void nv_vector_copy(nv_matrix_t *dest, int dm, const nv_matrix_t *src, int sm)
{
	assert(dest->n == src->n);

	memmove(&NV_MAT_V(dest, dm, 0), &NV_MAT_V(src, sm, 0), dest->step * sizeof(float));
}

void nv_matrix_m(nv_matrix_t *mat, int m)
{
	if (mat->rows == 1) {
		mat->cols = m;
	} else {
		int diff = mat->m - m;
		if (diff % mat->cols) {
			mat->rows -= (mat->m - m) / mat->cols; 
		} else {
			mat->rows -= (mat->m - m) / mat->cols;
			if (diff > 0) {
				--mat->rows;
			} else {
				++mat->rows;
			}
		}
	}
	mat->m = m;
}
