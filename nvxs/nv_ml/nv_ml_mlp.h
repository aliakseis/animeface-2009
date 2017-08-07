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

nv_mlp_t *nv_mlp_alloc(int input, int hidden, int k);
void nv_mlp_free(nv_mlp_t **mlp);

float nv_mlp_sigmoid(float a);

int nv_mlp_predict_label(const nv_mlp_t *mlp, const nv_matrix_t *x, int xm);
float nv_mlp_predict(const nv_mlp_t *mlp, const nv_matrix_t *x, int xm, int cls);
float nv_mlp_bagging_predict(const nv_mlp_t **mlp, int nmlp, 
							 const nv_matrix_t *x, int xm, int cls);
double nv_mlp_predict_d(const nv_mlp_t *mlp,
					   const nv_matrix_t *x, int xm, int cls);
double nv_mlp_bagging_predict_d(const nv_mlp_t **mlp, int nmlp, 
							   const nv_matrix_t *x, int xm, int cls);

void nv_mlp_regression(const nv_mlp_t *mlp, const nv_matrix_t *x, int xm, nv_matrix_t *out, int om);

#ifdef __cplusplus
}
#endif

#endif
