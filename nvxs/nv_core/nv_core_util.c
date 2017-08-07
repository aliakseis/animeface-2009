#include "nv_core.h"
#include "nv_core_matrix.h"
#include "nv_core_util.h"
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836
#define MASK 123459876

#define IDUM_N 2048
static long idums[IDUM_N] = { 8273 };

float nv_rand(void)
{
	long k;
	float ans;
#ifdef _OPENMP
	int thread_idx = omp_get_thread_num();
#else
	int thread_idx = 0;
#endif
	long idum = idums[thread_idx];
	idum ^= MASK;
	k = idum / IQ;
	idum = IA * (idum - k * IQ) - IR * k;
	if (idum < 0) { 
		idum += IM; 
	}
	ans = (float)(AM * idum);
	idum ^= MASK;
	idums[thread_idx] = idum;

	return ans;
}
