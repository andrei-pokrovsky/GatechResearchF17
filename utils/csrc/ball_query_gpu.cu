#include <stdio.h>
#include <stdlib.h>

#include "ball_query_gpu.h"

// input: dmat(b, m, n)
// output: idx(b, m, nsample)
__global__ void query_ball_point_kernel(int b, int n, int m, float radius,
					int nsample, const float *dmat,
					long *idx) {
	int batch_index = blockIdx.x;
	dmat += n * m * batch_index;
	idx += m * nsample * batch_index;

	int index = threadIdx.x;
	int stride = blockDim.x;
	for (int j = index; j < m; j+=stride) {
		for (int k = 0, cnt = 0; k < n && cnt < nsample; ++k) {
			float d = dmat[j * n + k];
			if (d <= radius) {
				if (cnt == 0) {
					for (int l = 0; l < nsample; ++l) {
						idx[j * nsample + l] = k;
					}
				}
				idx[j * nsample + cnt] = k;
				++cnt;
			}
		}
	}
}

void query_ball_point_kernel_wrapper(int b, int n, int m, float radius,
				     int nsample, const float *dmat, long *idx,
				     cudaStream_t stream) {


	cudaError_t err;
	query_ball_point_kernel<<<b, 256, 0, stream>>>(b, n, m, radius, nsample,
						     dmat, idx);

	err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "CUDA kernel failed : %s\n",
			cudaGetErrorString(err));
		exit(-1);
	}
}
