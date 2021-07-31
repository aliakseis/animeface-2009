#include <cstdlib>
#include <cmath>
#include <Eigen/Dense>
#include "nv_core.h"
#include "nv_ml_mlp.h"


// 多層パーセプトロン
// 2 Layer

static float nv_mlp_sigmoid(float a) {
	return 1.0F / (1.0F + expf(-a));
}

// クラス分類

int nv_mlp_predict_label(const nv_mlp_t *mlp, const Eigen::Ref<Eigen::Matrix<float, NV_FACE_HAARLIKE_DIM, 1> > x)
{
	Eigen::Map<Eigen::VectorXf> input_bias(mlp->input_bias->v, mlp->hidden);
	Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > input_w(mlp->input_w->v, mlp->hidden, mlp->input);
	Eigen::Map<Eigen::VectorXf> hidden_bias(mlp->hidden_bias->v, mlp->output);
	Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > hidden_w(mlp->hidden_w->v, mlp->output, mlp->hidden);
	Eigen::VectorXf y = hidden_w*(input_w*x + input_bias).unaryExpr(&nv_mlp_sigmoid) + hidden_bias;
	int l;
	y.maxCoeff(&l);
	return (y[l] > 0.F) ? l : -1;
}

double nv_mlp_predict_d(const nv_mlp_t *mlp, const Eigen::Ref<Eigen::Matrix<float, NV_FACE_HAARLIKE_DIM, 1> > x)
{
	Eigen::Map<Eigen::VectorXf> input_bias(mlp->input_bias->v, mlp->hidden);
	Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > input_w(mlp->input_w->v, mlp->hidden, mlp->input);
	float hidden_bias = *mlp->hidden_bias->v;
	Eigen::Map<Eigen::RowVectorXf > hidden_w(mlp->hidden_w->v, mlp->hidden);
	return 1./(1.+exp(-hidden_w.dot((input_w*x + input_bias).unaryExpr(&nv_mlp_sigmoid)) - hidden_bias));
}

double nv_mlp_bagging_predict_d(const nv_mlp_t **mlp, int nmlp, const Eigen::Ref<Eigen::Matrix<float, NV_FACE_HAARLIKE_DIM, 1> > x)
{
	double p = 0.0F;
	double factor = 1.0 / nmlp;
	int i;
	
	for (i = 0; i < nmlp; ++i) {
		p += factor * nv_mlp_predict_d(mlp[i], x);
	}

	return p;
}

// 非線形重回帰

void nv_mlp_regression(const nv_mlp_t *mlp, const Eigen::Ref<Eigen::Matrix<float, NV_FACE_HAARLIKE_DIM, 1> >  x, nv_matrix_t *out)
{
	Eigen::Map<Eigen::VectorXf> input_bias(mlp->input_bias->v, mlp->hidden);
	Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > input_w(mlp->input_w->v, mlp->hidden, mlp->input);
	Eigen::Map<Eigen::VectorXf> hidden_bias(mlp->hidden_bias->v, mlp->output);
	Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > hidden_w(mlp->hidden_w->v, mlp->output, mlp->hidden);
	Eigen::Map<Eigen::VectorXf> y(out->v, out->n);
	y = hidden_w*(input_w*x + input_bias).unaryExpr(&nv_mlp_sigmoid) + hidden_bias;
}
