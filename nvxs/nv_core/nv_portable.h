#ifndef __NV_PORTABLE_H
#define __NV_PORTABLE_H

//#ifdef __cplusplus
//extern "C" {
//#endif

// 標準Cヘッダー

//#define _CRTDBG_MAP_ALLOC
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <cctype>
#include <cassert>
#include <cerrno>

// OpenMP 
#if (_MSC_VER >= 1400 && defined(_OPENMP))
#ifdef _DEBUG
    #undef _DEBUG
    #include <omp.h>
    #define _DEBUG
#else
    #include <omp.h>
#endif
#endif

// aligne
#if __GNUC__
#define NV_ALIGNED(typed, variable, align_size) typed variable __attribute__((aligned(align_size)))
#else // MSC
#define NV_ALIGNED(typed, variable, align_size) __declspec(align(align_size)) typed variable 
#endif

// roundf
#define NV_ROUND_INT(x) ((int)(0.5f + x))

//#ifdef __cplusplus
//}
//#endif


#endif
