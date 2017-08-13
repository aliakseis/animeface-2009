#include "nv_core.h"
#include "nv_core_matrix.h"

nv_matrix_t *nv_matrix_alloc(int n, int m)
{
	void *mem;
	
	int step = n * sizeof(float);//(n + 4 - (n & 3)) * sizeof(float); // SSE2
	int mem_size = step * m + sizeof(nv_matrix_t) + 0x10;
	nv_matrix_t *matrix = (nv_matrix_t *)malloc(mem_size);

	if (matrix == NULL) {
		return NULL;
	}
	mem = ((char *)matrix) + sizeof(nv_matrix_t);
	matrix->v = (float *)(((char *)mem) + 0x10 - ((size_t)mem & 0xf));
	matrix->n = n;
	matrix->m = m;
	matrix->cols = m;
	matrix->rows = 1;
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
	memset(mat->v, 0, mat->m * mat->n * sizeof(float));
}

void nv_vector_zero(nv_matrix_t *mat, int m)
{
	memset(&NV_MAT_V(mat, m, 0), 0, mat->n * sizeof(float));
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

	memmove(&NV_MAT_V(dest, dm, 0), &NV_MAT_V(src, sm, 0), dest->n * sizeof(float));
}

void nv_matrix_m(nv_matrix_t *mat, int m)
{
	assert(mat->rows == 1);
	mat->cols = m;
	mat->m = m;
}

nv_matrix_t *nv_from_image(IplImage *img)
{
	nv_matrix_t *ret = (nv_matrix_t*)malloc(sizeof(nv_matrix_t));
	ret->cols = img->width;
	ret->rows = img->height;
	ret->m = ret->cols * ret->rows;
	ret->n = img->nChannels;
	assert(img->depth == IPL_DEPTH_32F);
	ret->v = (float*)img->imageData;
	return ret;
}

IplImage *nv_to_image(nv_matrix_t *mtx)
{
	IplImage *ret = cvCreateImageHeader(cvSize(mtx->cols, mtx->rows), IPL_DEPTH_32F, mtx->n);
	cvSetData(ret, mtx->v, mtx->cols * mtx->n * sizeof(float));
	return ret;
}
