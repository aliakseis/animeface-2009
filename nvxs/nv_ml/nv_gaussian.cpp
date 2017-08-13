#include <Eigen/Dense>
#include <math.h>
#include "nv_core.h"
#include "nv_num.h"
#include "nv_ml_gaussian.h"


// ガウス分布

float nv_gaussian_log_predict(const nv_cov_t *cov, const nv_matrix_t *x, int xm)
{
	Eigen::VectorXf X = Eigen::Map<Eigen::VectorXf>(&x->v[xm*x->n], x->n) - Eigen::Map<Eigen::VectorXf>(cov->u->v, x->n);
	Eigen::Map<Eigen::MatrixXf> Sigma(cov->cov->v, x->n, x->n);
	return log(1 / sqrt(pow(2 * acos(-1), x->n) * Sigma.determinant())) - X.dot(Sigma.llt().solve(X)) / 2;
}
