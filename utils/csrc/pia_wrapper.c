#include <THC/THC.h>

#include "pia_wrapper_gpu.h"

extern THCState *state;

int pia_wrapper(int b, int n, int bpp, int nb, THCudaTensor *preds_tensor,
		THCudaTensor *boxes_tensor, /*THCudaTensor *i_obj_tensor, */
		/* THCudaLongTensor *target_box_tensor, */
		THCudaTensor *iou_out_tensor) {

	const float *preds = THCudaTensor_data(state, preds_tensor);
	const float *boxes = THCudaTensor_data(state, boxes_tensor);
	/* float *i_obj = THCudaTensor_data(state, i_obj_tensor); */
	/* long *target_box = THCudaLongTensor_data(state, target_box_tensor);
	 */
	float *iou_out = THCudaTensor_data(state, iou_out_tensor);

	cudaStream_t stream = THCState_getCurrentStream(state);

	pia_kernel_wrapper(b, n, bpp, nb, preds, boxes, iou_out, stream);
	return 1;
}
