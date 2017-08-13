#ifndef __NV_ML_MLP_H
#define __NV_ML_MLP_H

#include "nv_core.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
	int input;
	int hidden;
	int output;
	nv_matrix_t *input_w;
	nv_matrix_t *hidden_w;
	nv_matrix_t *input_bias;
	nv_matrix_t *hidden_bias;
 } nv_mlp_t;

int nv_mlp_predict_label(const nv_mlp_t *mlp, const nv_matrix_t *x);
double nv_mlp_predict_d(const nv_mlp_t *mlp, const nv_matrix_t *x);
double nv_mlp_bagging_predict_d(const nv_mlp_t **mlp, int nmlp, const nv_matrix_t *x);

void nv_mlp_regression(const nv_mlp_t *mlp, const nv_matrix_t *x, nv_matrix_t *out);

#ifdef __cplusplus
}
#endif

#endif
