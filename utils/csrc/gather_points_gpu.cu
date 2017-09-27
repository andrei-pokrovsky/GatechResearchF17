#include <stdio.h>
#include <stdlib.h>

#include "gather_points_gpu.h"

// input: points(b, n, c) idx(b, m)
// output: out(b, m, c)
__global__ void gather_points_kernel(int b, int n, int c, int m,
				     const float *points, const int *idx,
				     float *out) {
	for (int i = blockIdx.x; i < b; i += gridDim.x) {
		for (int j = blockIdx.y * blockDim.x + threadIdx.x; j < m;
		     j += blockDim.x * gridDim.y) {
			int a = idx[i * m + j];
			memcpy(out + (i * m + j) * c, points + (i * n + a) * c,
			       sizeof(float) * c);
		}
	}
}

void gather_points_kernel_wrapper(int b, int n, int c, int npoints,
				  const float *points, const int *idx,
				  float *out, cudaStream_t stream) {

	cudaError_t err;
	gather_points_kernel<<<dim3(2, 8, 1), min(512, npoints), 0, stream>>>(
	    b, n, c, npoints, points, idx, out);

	err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "CUDA kernel failed : %s\n",
			cudaGetErrorString(err));
		exit(-1);
	}
}
