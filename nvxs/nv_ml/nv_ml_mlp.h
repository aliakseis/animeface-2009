#ifndef __NV_ML_MLP_H
#define __NV_ML_MLP_H

#include "nv_core.h"

#ifdef __cplusplus

#include <Eigen/Dense>
#include "nv_face_feature.h"

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


#ifdef __cplusplus
}

int nv_mlp_predict_label(const nv_mlp_t *mlp, Eigen::Ref<Eigen::Matrix<float, NV_FACE_HAARLIKE_DIM, 1> > x);
double nv_mlp_predict_d(const nv_mlp_t *mlp, Eigen::Ref<Eigen::Matrix<float, NV_FACE_HAARLIKE_DIM, 1> > x);
double nv_mlp_bagging_predict_d(const nv_mlp_t **mlp, int nmlp, Eigen::Ref<Eigen::Matrix<float, NV_FACE_HAARLIKE_DIM, 1> > x);
void nv_mlp_regression(const nv_mlp_t *mlp, Eigen::Ref<Eigen::Matrix<float, NV_FACE_HAARLIKE_DIM, 1> > x, nv_matrix_t *out);
#endif

#endif
