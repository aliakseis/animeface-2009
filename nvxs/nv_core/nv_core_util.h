#ifndef __NV_CORE_UTIL_H
#define __NV_CORE_UTIL_H

#ifdef __cplusplus
extern "C" {
#endif

#define NV_PI 3.1415926f

typedef struct {
	float v[4];
} nv_color_t;

float nv_rand(void);

#ifdef __cplusplus
}
#endif


#endif
