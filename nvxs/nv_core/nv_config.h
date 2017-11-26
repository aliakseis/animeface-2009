#ifndef __NV_CONFIG_H
#define __NV_CONFIG_H

#ifdef __GNUC__
// gcc
#define NV_ENABLE_CUDA   0  // CUDA���g����
#ifdef __SSE2__
#define NV_ENABLE_SSE2   1  // SSE2���g����
#else
#define NV_ENABLE_SSE2   0
#endif
#define NV_XS            1  // Perl�p

#else
// VC++
#define NV_ENABLE_CUDA   0  // CUDA���g����
#define NV_ENABLE_SSE2   1  // SSE2���g����
#define NV_XS            1  // Perl�p

#endif
#endif
