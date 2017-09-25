#include <THC/THC.h>

#include "ball_query_gpu.h"

extern THCState *state;

int ball_query_wrapper(int b, int n, int m, float radius, int nsample,
		       THCudaTensor *dmat_tensor,
		       THCudaLongTensor *idx_tensor) {

	const float *dmat = THCudaTensor_data(state, dmat_tensor);
	long *idx = THCudaLongTensor_data(state, idx_tensor);

	cudaStream_t stream = THCState_getCurrentStream(state);

	query_ball_point_kernel_wrapper(b, n, m, radius, nsample, dmat, idx,
					stream);
	return 1;
}
