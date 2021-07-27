#include "nv_core.h"
#include "nv_face_feature.h"

// haar的なもの
// 現在1レベルの解像度しか使用してない

static float nv_face_haarlike_diagonal_filter(int type,
											  const nv_matrix_t *sum,
											  int px, int py,
											  float xscale, float yscale)
{
	int i = 0;
	float p = 0.0f, p1 = 0.0f, p2 = 0.0f;
	int area1 = 0, area2 = 0;
	int ystep = NV_ROUND_INT(yscale);
	int xstep = NV_ROUND_INT(xscale);

	if (type == 1) {
		// |＼|
		for (i = 0; i < 7; ++i) {
			int ppx = px + NV_ROUND_INT((1.0f + i) * xscale);
			int ppy = py + NV_ROUND_INT(i * yscale);
			int eex = px + NV_ROUND_INT(8.0f * xscale);
			int eey = py + NV_ROUND_INT((i + 1) * yscale);

			//printf("p1: %d, %d, %d, %d¥n", 1+i,8,i,i+1);

			p1 += NV_INTEGRAL_V(sum, ppx, ppy, eex, eey);
			area1 += (eex - ppx) * (eey - ppy);	
		}
		for (i = 1; i < 8; ++i) {
			int ppx = px;
			int ppy = py + NV_ROUND_INT(i * yscale);
			int eex = px + NV_ROUND_INT(i * xscale);
			int eey = py + NV_ROUND_INT((i + 1) * yscale);

			//printf("p2: %d, %d, %d, %d¥n", 0,i,i,i+1);
			p2 += NV_INTEGRAL_V(sum, ppx, ppy, eex, eey);
			area2 += (eex - ppx) * (eey - ppy);	
		}
		p = p1 / (area1 * 255.0f) - p2 / (area2 * 255.0f);
	} else {
		// |/|
		for (i = 0; i < 7; ++i) {
			int ppx = px;
			int ppy = py + NV_ROUND_INT(i * yscale);
			int eex = px + NV_ROUND_INT((7.0f - i) * xscale);
			int eey = py + NV_ROUND_INT((i + 1) * yscale);

			//printf("p1: %d, %d, %d, %d¥n", 0, 7-i, i, i+1);
			
			p1 += NV_INTEGRAL_V(sum, ppx, ppy, eex, eey);
			area1 += (eex - ppx) * (eey - ppy);	
		}
		for (i = 1; i < 8; ++i) {
			int ppx = px + NV_ROUND_INT((8.0f - i) * xscale);
			int ppy = py + NV_ROUND_INT(i * yscale);
			int eex = px + NV_ROUND_INT(8.0f * xscale);
			int eey = py + NV_ROUND_INT((i + 1) * yscale);

			//printf("p2: %d, %d, %d, %d¥n", 8-i, 8, i, i+1);

			p2 += NV_INTEGRAL_V(sum, ppx, ppy, eex, eey);
			area2 += (eex - ppx) * (eey - ppy);	
		}
		p = p1 / (area1 * 255.0f) - p2 / (area2 * 255.0f);
	}

	return p;
}


void nv_face_haarlike(nv_face_haarlike_normalize_e normalize_type,
	Eigen::Ref<Eigen::Matrix<float, NV_FACE_HAARLIKE_DIM, 1> > feature,
	const nv_matrix_t *sum,
	cv::Rect roi)
{
	int ix, iy, n;
	float vmax, vmin;
	float xscale = roi.width / 32.0f;
	float yscale = roi.height / 32.0f;
	float ystep = yscale;
	float xstep = xscale;
	const int hystep = 12;
	int sy = NV_ROUND_INT(4.0f * ystep);
	int sx = NV_ROUND_INT(4.0f * xstep);
	int hy, hx;

	feature.setZero();

	Eigen::Map<Eigen::Matrix<float, 8, 144> > fmat(feature.data());

	// level1
	for (iy = 0, hy = 0; iy < 32 - 8; iy += 2, ++hy) {
		int py = roi.y + NV_ROUND_INT(ystep * iy);
		int ey = py + NV_ROUND_INT(8.0f * ystep);
		const float pty = (ey - py) * 255.0f;
		for (ix = 0, hx = 0; ix < 32 - 8; ix += 2, ++hx) {
			int px = roi.x + NV_ROUND_INT(xstep * ix);
			int ex = px + NV_ROUND_INT(8.0f * xstep);
			float p1, p2, area, ptx;

			// 全エリア
			area = NV_MAT3D_V(sum, ey, ex, 0)
				- NV_MAT3D_V(sum, ey, px, 0)
				- (NV_MAT3D_V(sum, py, ex, 0) - NV_MAT3D_V(sum, py, px, 0));

			// 1
			// [+]
			// [-]
			p1 = NV_MAT3D_V(sum, py + sy, ex, 0)
				- NV_MAT3D_V(sum, py + sy, px, 0)
				- (NV_MAT3D_V(sum, py, ex, 0) - NV_MAT3D_V(sum, py, px, 0));
			p2 = area - p1;
			ptx = (ex - px) * 255.0f;
			p1 /= ((py + sy) - py) * ptx;
			p2 /= (ey - (py + sy)) * ptx;
			if (p1 > p2) {
				fmat(0, hy * hystep + hx) = p1 - p2;
			}
			else {
				fmat(1, hy * hystep + hx) = p2 - p1;
			}

			// 2
			// [+][-]
			p1 = NV_MAT3D_V(sum, ey, px + sx, 0)
				- NV_MAT3D_V(sum, ey, px, 0)
				- (NV_MAT3D_V(sum, py, px + sx, 0) - NV_MAT3D_V(sum, py, px, 0));
			p2 = area - p1;
			p1 /= ((px + sx) - px) * pty;
			p2 /= (ex - (px + sx)) * pty;
			if (p1 > p2) {
				fmat(2, hy * hystep + hx) = p1 - p2;
			}
			else {
				fmat(3, hy * hystep + hx) = p2 - p1;
			}

			// 3
			p1 = nv_face_haarlike_diagonal_filter(1, sum, px, py, xscale, yscale);
			if (p1 > 0.0f) {
				fmat(4, hy * hystep + hx) = p1;
			}
			else {
				fmat(5, hy * hystep + hx) = -p1;
			}

			// 4
			p1 = nv_face_haarlike_diagonal_filter(2, sum, px, py, xscale, yscale);
			if (p1 > 0.0f) {
				fmat(6, hy * hystep + hx) = p1;
			}
			else {
				fmat(7, hy * hystep + hx) = -p1;
			}
		}
	}

	// 正規化
	if (normalize_type == NV_NORMALIZE_MAX) {
		// 最大値=1.0
		vmax = 0.0f;
		vmin = FLT_MAX;
		for (n = 0; n < feature.size(); ++n) {
			if (feature(n) > vmax) {
				vmax = feature(n);
			}
			if (feature(n) != 0.0f && feature(n) < vmin)
			{
				vmin = feature(n);
			}
		}
		if (vmax != 0.0f && vmax > vmin) {
			for (n = 0; n < feature.size(); ++n) {
				if (feature(n) != 0.0f) {
					feature(n) = (feature(n) - vmin) / (vmax - vmin);
				}
			}
		}
	}
	else if(normalize_type==NV_NORMALIZE_NORM)
		// ベクトル NORM=1.0
		feature /= feature.norm();
}
