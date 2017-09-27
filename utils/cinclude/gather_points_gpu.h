#ifndef _BALL_QUERY_GPU
#define _BALL_QUERY_GPU

#ifdef __cplusplus
extern "C" {
#endif

void gather_points_kernel_wrapper(int b, int n, int c, int npoints,
				  const float *points, const int *idx,
				  float *out, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
#endif
