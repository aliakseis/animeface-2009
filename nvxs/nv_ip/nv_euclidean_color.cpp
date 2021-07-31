#include "nv_core.h"
#include "nv_ip_euclidean_color.h"

// ユークリッド空間で比較する用の色空間
// 中身は変わる可能性あり

// 現在
// C1 = (R + G + B) / 3
// C2 = (R + (255 - B)) / 2
// C3 = (R + 2 * (255 - G) + B) / 4
// B  = (4 * C3 - 6 * C2 + 6 * C1 + 255) / 6
// G  = (-4 * C3 + 3 * C1 + 510) / 3
// R  = (4 * C3 + 6 * C2 + 6 * C1 - 1275) / 6

static float v_0_255(float v)
{
	if (v > 255.0F) {
		return 255.0F;
	}
	if (v < 0.0F) {
		return 0.0F;
	}
	return v;
}

void nv_color_euclidean2bgr_scalar(nv_matrix_t *bgr, int bgr_m, const nv_matrix_t *ec, int ec_m)
{
	float c1 = NV_MAT_V(ec, ec_m, 0);
	float c2 = NV_MAT_V(ec, ec_m, 1);
	float c3 = NV_MAT_V(ec, ec_m, 2);

	assert(ec->n == bgr->n && ec->n == 3);

	NV_MAT_V(bgr, bgr_m, NV_CH_B) = floorf((4.0F * c3 - 6.0F * c2 + 6.0F * c1 + 255.0F) / 6.0F);
	NV_MAT_V(bgr, bgr_m, NV_CH_G) = floorf((-4.0F * c3 + 3.0F * c1 + 510.0F) / 3.0F);
	NV_MAT_V(bgr, bgr_m, NV_CH_R) = floorf((4.0F * c3 + 6.0F * c2 + 6.0F * c1 - 1275.0F) / 6.0F);
	NV_MAT_V(bgr, bgr_m, 0) = v_0_255(NV_MAT_V(bgr, bgr_m, 0));
	NV_MAT_V(bgr, bgr_m, 1) = v_0_255(NV_MAT_V(bgr, bgr_m, 1));
	NV_MAT_V(bgr, bgr_m, 2) = v_0_255(NV_MAT_V(bgr, bgr_m, 2));
}

void nv_color_bgr2euclidean_scalar(nv_matrix_t *ec, int ec_m, const nv_matrix_t *bgr, int bgr_m)
{
	assert(ec->n == bgr->n && ec->n == 3);
	NV_MAT_V(ec, ec_m, 0) = floorf((NV_MAT_V(bgr, bgr_m, NV_CH_R) + NV_MAT_V(bgr, bgr_m, NV_CH_G) + NV_MAT_V(bgr, bgr_m, NV_CH_B)) / 3.0F);
	NV_MAT_V(ec, ec_m, 1) = floorf((NV_MAT_V(bgr, bgr_m, NV_CH_R) + (255.0F - NV_MAT_V(bgr, bgr_m, NV_CH_B))) / 2.0F);
	NV_MAT_V(ec, ec_m, 2) = floorf((NV_MAT_V(bgr, bgr_m, NV_CH_R) + 2.0F * (255.0F - NV_MAT_V(bgr, bgr_m, NV_CH_G)) + NV_MAT_V(bgr, bgr_m, NV_CH_B)) / 4.0F);
}
