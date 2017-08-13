#include <algorithm>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include "nv_core.h"
#include "nv_ml.h"
#include "nv_num.h"

// k-means++
int nv_kmeans(nv_matrix_t *means_mat,  // k
		  nv_matrix_t *count_mat,  // k
		  nv_matrix_t *labels_mat, // data->m
		  const nv_matrix_t *data_mat,
		  const int k,
		  const int max_epoch)
{
	cv::Mat means(means_mat->m, means_mat->n, CV_32F, means_mat->v);
	cv::Mat data(data_mat->m, data_mat->n, CV_32F, data_mat->v);
	cv::Mat labels(data_mat->m, 1, CV_32S);

	Eigen::Map<Eigen::VectorXf> count(count_mat->v, count_mat->m);
	count.setZero();

	cv::kmeans(data, k, labels, cv::TermCriteria(cv::TermCriteria::MAX_ITER, max_epoch, 0), 1, cv::KMEANS_PP_CENTERS, means);

	for (int m = 0; m < data_mat->m; ++m) {
		NV_MAT_V(labels_mat, m, 0) = labels.at<int>(m, 0);
		++count(labels.at<int>(m, 0));
	}
	return k;
}

