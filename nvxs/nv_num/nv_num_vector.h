#ifndef __NV_NUM_VECTOR_H
#define __NV_NUM_VECTOR_H

#include "nv_core.h"

#ifdef __cplusplus
extern "C" {
#endif

float nv_vector_dot(const nv_matrix_t *vec1, int m1, const nv_matrix_t *vec2, int m2);
int nv_vector_maxsum_m(const nv_matrix_t *v);
#ifdef __cplusplus
}
#endif
#endif
