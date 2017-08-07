#ifndef __NV_CONFIG_H
#define __NV_CONFIG_H

#ifdef __GNUC__
// gcc
#define NV_ENABLE_CUDA   0  // CUDAを使うか
#ifdef __SSE2__
#define NV_ENABLE_SSE2   1  // SSE2を使うか
#else
#define NV_ENABLE_SSE2   0
#endif
#define NV_XS            1  // Perl用

#else
// VC++
#define NV_ENABLE_CUDA   0  // CUDAを使うか
#define NV_ENABLE_SSE2   1  // SSE2を使うか
#define NV_XS            1  // Perl用

#endif
#endif
