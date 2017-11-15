#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"
#include "sampling_gpu.h"

// input: points(b, n, c) idx(b, m)
// output: out(b, m, c)
__global__ void gather_points_kernel(int b, int n, int c, int m,
				     const float *__restrict__ points,
				     const int *__restrict__ idx,
				     float *__restrict__ out) {
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
    gather_points_kernel<<<dim3(2, 8, 1), opt_n_threads(npoints) / 4, 0,
			   stream>>>(b, n, c, npoints, points, idx, out);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
	fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
	exit(-1);
    }
}

// Input dataset: (b, n, 3), tmp: (b, n)
// Ouput idxs (b, m)
__global__ void furthest_point_sampling_kernel(
    int b, int n, int m, const float *__restrict__ dataset,
    float *__restrict__ temp, int *__restrict__ idxs) {
    if (m <= 0)
	return;
    const int BlockSize = 512;
    __shared__ float dists[BlockSize];
    __shared__ int dists_i[BlockSize];

    int batch_index = blockIdx.x;
    dataset += batch_index * n * 3;
    temp += batch_index * n;
    idxs += batch_index * m;

    int tid = threadIdx.x;
    int stride = blockDim.x;
    int n_threads = stride;

    int old = 0;
    if (threadIdx.x == 0)
	idxs[0] = old;

    __syncthreads();
    for (int j = 1; j < m; j++) {
	int besti = 0;
	float best = -1;
	float x1 = dataset[old * 3 + 0];
	float y1 = dataset[old * 3 + 1];
	float z1 = dataset[old * 3 + 2];
	for (int k = tid; k < n; k += stride) {
	    float x2, y2, z2;
	    x2 = dataset[k * 3 + 0];
	    y2 = dataset[k * 3 + 1];
	    z2 = dataset[k * 3 + 2];
	    float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) +
		      (z2 - z1) * (z2 - z1);

	    float d2 = min(d, temp[k]);
	    temp[k] = d2;
	    if (d2 > best) {
		best = d2;
		besti = k;
	    }
	}
	dists[tid] = best;
	dists_i[tid] = besti;
	__syncthreads();
	for (int s = n_threads / 2; s > 0; s >>= 1) {
	    if (tid < s) {
		int idx1 = tid;
		int idx2 = idx1 + s;
		if (dists[idx2] > dists[idx1]) {
		    dists[idx1] = dists[idx2];
		    dists_i[idx1] = dists_i[idx2];
		}
	    }
	    __syncthreads();
	}
	old = dists_i[0];
	if (tid == 0)
	    idxs[j] = old;

	/* if (threadIdx.x == 0) {
	    for (int i = 0; i < n_threads; ++i) {
		if (dists[i] > best) {
		    best = dists[i];
		    besti = dists_i[i];
		}
	    }

	    idxs[j] = besti;
	}
	__syncthreads();

	old = idxs[j]; */
    }
}

void furthest_point_sampling_kernel_wrapper(int b, int n, int m,
					    const float *dataset, float *temp,
					    int *idxs, cudaStream_t stream) {

    cudaError_t err;
    furthest_point_sampling_kernel<<<b, max(opt_n_threads(n), 512), 0,
				     stream>>>(b, n, m, dataset, temp, idxs);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
	fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
	exit(-1);
    }
}
