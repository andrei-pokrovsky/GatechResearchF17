#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "interpolate_gpu.h"

// input: unknown(b, n, 3) known(b, m, 3)
// output: dist2(b, n, 3), idx(b, n, 3)
__global__ void three_nn_kernel(int b, int n, int m, const float *unknown,
				const float *known, float *dist2, int *idx) {
    int batch_index = blockIdx.x;
    unknown += batch_index * n * 3;
    known += batch_index * m * 3;
    dist2 += batch_index * n * 3;
    idx += batch_index * n * 3;

    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int j = index; j < n; j += stride) {
	float ux = unknown[j * 3 + 0];
	float uy = unknown[j * 3 + 1];
	float uz = unknown[j * 3 + 2];

	double best1 = 1e40, best2 = 1e40, best3 = 1e40;
	int besti1 = 0, besti2 = 0, besti3 = 0;
	for (int k = 0; k < m; ++k) {
	    float x = known[k * 3 + 0];
	    float y = known[k * 3 + 1];
	    float z = known[k * 3 + 2];
	    float d =
		(ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);
	    if (d < best1) {
		best3 = best2;
		besti3 = besti2;
		best2 = best1;
		besti2 = besti1;
		best1 = d;
		besti1 = k;
	    } else if (d < best2) {
		best3 = best2;
		besti3 = besti2;
		best2 = d;
		besti2 = k;
	    } else if (d < best3) {
		best3 = d;
		besti3 = k;
	    }
	}
	dist2[j * 3 + 0] = best1;
	dist2[j * 3 + 1] = best2;
	dist2[j * 3 + 2] = best3;

	idx[j * 3 + 0] = besti1;
	idx[j * 3 + 1] = besti2;
	idx[j * 3 + 2] = besti3;
    }
}

void three_nn_kernel_wrapper(int b, int n, int m, const float *unknown,
			     const float *known, float *dist2, int *idx,
			     cudaStream_t stream) {

    cudaError_t err;
    three_nn_kernel<<<b, min(n, 512), 0, stream>>>(b, n, m, unknown, known,
						   dist2, idx);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
	fprintf(stderr, "CUDA kernel "
			"failed : %s\n",
		cudaGetErrorString(err));
	exit(-1);
    }
}

// input: points(b, m, c), idx(b, n, 3), weight(b, n, 3)
// output: out(b, n, c)
__global__ void three_interpolate_kernel(int b, int m, int c, int n,
					 const float *points, const int *idx,
					 const float *weight, float *out) {
    int batch_index = blockIdx.x;
    points += batch_index * m * c;

    idx += batch_index * n * 3;
    weight += batch_index * n * 3;

    out += batch_index * n * c;

    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int j = index; j < n; j += stride) {
	float w1 = weight[j * 3 + 0];
	float w2 = weight[j * 3 + 1];
	float w3 = weight[j * 3 + 2];

	int i1 = idx[j * 3 + 0];
	int i2 = idx[j * 3 + 1];
	int i3 = idx[j * 3 + 2];

	for (int l = 0; l < c; ++l) {
	    out[j * c + l] = points[i1 * c + l] * w1 + points[i2 * c + l] * w2 +
			     points[i3 * c + l] * w3;
	}
    }
}

void three_interpolate_kernel_wrapper(int b, int m, int c, int n,
				      const float *points, const int *idx,
				      const float *weight, float *out,
				      cudaStream_t stream) {

    cudaError_t err;
    three_interpolate_kernel<<<b, min(n, 512), 0, stream>>>(b, m, c, n, points,
							    idx, weight, out);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
	fprintf(stderr, "CUDA kernel "
			"failed : %s\n",
		cudaGetErrorString(err));
	exit(-1);
    }
}

// input: grad_out(b, n, c), idx(b, n, 3), weight(b, n, 3)
// output: grad_points(b, m, c)

__global__ void three_interpolate_grad_kernel(int b, int n, int c, int m,
					      const float *grad_out,
					      const int *idx,
					      const float *weight,
					      float *grad_points) {
    int batch_index = blockIdx.x;
    grad_out += batch_index * n * c;
    idx += batch_index * n * 3;
    weight += batch_index * n * 3;
    grad_points += batch_index * m * c;

    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int j = index; j < n; j += stride) {
	float w1 = weight[j * 3 + 0];
	float w2 = weight[j * 3 + 1];
	float w3 = weight[j * 3 + 2];

	int i1 = idx[j * 3 + 0];
	int i2 = idx[j * 3 + 1];
	int i3 = idx[j * 3 + 2];

	for (int l = 0; l < c; ++l) {
	    atomicAdd(grad_points + i1 * c + l, grad_out[j * c + l] * w1);
	    atomicAdd(grad_points + i2 * c + l, grad_out[j * c + l] * w2);
	    atomicAdd(grad_points + i3 * c + l, grad_out[j * c + l] * w3);
	}
    }
}

void three_interpolate_grad_kernel_wrapper(int b, int n, int c, int m,
					   const float *grad_out,
					   const int *idx, const float *weight,
					   float *grad_points,
					   cudaStream_t stream) {

    cudaError_t err;
    three_interpolate_grad_kernel<<<b, min(n, 512), 0, stream>>>(
	b, n, c, m, grad_out, idx, weight, grad_points);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
	fprintf(stderr, "CUDA kernel "
			"failed : %s\n",
		cudaGetErrorString(err));
	exit(-1);
    }
}
