#include <THC/THC.h>

#include "gather_points_gpu.h"

extern THCState *state;

int gather_points_wrapper(int b, int n, int c, int npoints,
			  THCudaTensor *points_tensor,
			  THCudaIntTensor *idx_tensor,
			  THCudaTensor *out_tensor) {

	const float *points = THCudaTensor_data(state, points_tensor);
	const int *idx = THCudaIntTensor_data(state, idx_tensor);
	float *out = THCudaTensor_data(state, out_tensor);

	cudaStream_t stream = THCState_getCurrentStream(state);

	gather_points_kernel_wrapper(b, n, c, npoints, points, idx, out,
				     stream);
	return 1;
}
