#ifndef _CUDA_UTILS_H
#define _CUDA_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

inline int opt_n_threads(int work_size) {
	return min(
	    work_size + (work_size % 32 == 0 ? 0 : (32 - work_size % 32)), 512);
}

#ifdef __cplusplus
}
#endif
#endif
