
#ifndef _PIA_WRAPPER_GPU_H
#define _PIA_WRAPPER_GPU_H

#ifdef __cplusplus
extern "C" {
#endif
void pia_kernel_wrapper(int b, int n, int bpp, int nb, const float *preds,
			const float *boxes, float *iou_out,
			cudaStream_t stream);

#ifdef __cplusplus
}
#endif
#endif
