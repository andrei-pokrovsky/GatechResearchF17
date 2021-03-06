
#ifndef _SAMPLING_GPU_H
#define _SAMPLING_GPU_H

#ifdef __cplusplus
extern "C" {
#endif

void gather_points_kernel_wrapper(int b, int n, int c, int npoints,
				  const float *points, const int *idx,
				  float *out, cudaStream_t stream);

void furthest_point_sampling_kernel_wrapper(int b, int n, int m,
					    const float *dataset, float *temp,
					    int *idxs, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
#endif
